"""FP32 reference and device-path emulators for ESP-DL ``Attention``."""

from __future__ import annotations

import math
import numbers
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from esp_ppq.core import DataType, TargetPlatform
from esp_ppq.IR import Operation
from esp_ppq.log import NaiveLogger

logger = NaiveLogger.get_logger('PPQ')


# ``#ifndef`` defaults in esp-dl ``ESP_DL_ATTENTION_MAX_*``. Env overrides here
# must be rebuilt into the firmware with the matching compile definition.
ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM = 64
ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS = 2048


def _resolve_max_head_dim() -> int:
    """Resolve the head_dim ceiling accepted by the fused Attention kernel."""
    raw = os.environ.get("ESPDL_ATTENTION_MAX_HEAD_DIM")
    if raw is None:
        return ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            f"ESPDL_ATTENTION_MAX_HEAD_DIM={raw!r} is not an integer; "
            f"falling back to {ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM}"
        )
        return ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM
    if parsed < 16 or parsed > 128 or (parsed % 16) != 0:
        logger.warning(
            f"ESPDL_ATTENTION_MAX_HEAD_DIM={parsed} out of range [16, 128] "
            f"or not a multiple of 16 (QACC lane count); "
            f"falling back to {ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM}"
        )
        return ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM
    return parsed


def _resolve_max_kv_tokens() -> int:
    """Resolve the key/value sequence length ceiling accepted by the fused Attention kernel."""
    raw = os.environ.get("ESPDL_ATTENTION_MAX_KV_TOKENS")
    if raw is None:
        return ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            f"ESPDL_ATTENTION_MAX_KV_TOKENS={raw!r} is not an integer; "
            f"falling back to {ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS}"
        )
        return ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS
    if parsed < 64 or parsed > 8192:
        logger.warning(
            f"ESPDL_ATTENTION_MAX_KV_TOKENS={parsed} out of range [64, 8192] "
            f"accepted by ESP_DL_ATTENTION_MAX_KV_TOKENS; "
            f"falling back to {ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS}"
        )
        return ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS
    return parsed


ESPDL_ATTENTION_MAX_HEAD_DIM: int = _resolve_max_head_dim()
ESPDL_ATTENTION_MAX_KV_TOKENS: int = _resolve_max_kv_tokens()
ESPDL_ATTENTION_LAYOUTS = frozenset({"BHND", "BNHD"})

# Fused kernel / generic executor may read these. On a static 4-D node a
# positive ``q_num_heads`` / ``kv_num_heads`` must match the layout head axis;
# see ``espdl_attention_head_attrs_ineligible_reason``.
_ATTENTION_FUSED_ATTRS = frozenset({"scale", "is_causal", "layout", "q_num_heads", "kv_num_heads"})
# Bookkeeping stamped during export or ONNX domain rewrite.
_ATTENTION_INFRA_ATTRS = frozenset({"quant_type", "platform", "domain"})
# Spec defaults that are no-ops; presence at this value == absence.
ESPDL_ATTENTION_DEFAULT_NOOP_ATTRS = {
    "softcap": 0.0,
    "qk_matmul_output_mode": 0,
    "left_window_size": -1,
    "right_window_size": -1,
}
_ENV_DEVICE_EMULATE_KEY = "ESPDL_ATTENTION_DEVICE_EMULATE"
_ESPDL_ATTENTION_SIMULATION_ENABLED = False
# P4 kernels the Python emulator models. S3/C stay on the fp32 reference.
_ATTENTION_EMULATED_PLATFORMS = frozenset(
    {
        TargetPlatform.ESPDL_INT8,
        TargetPlatform.ESPDL_INT16,
        TargetPlatform.ESPDL_H_PRE_INT16,
        TargetPlatform.ESPDL_W8A16,
    }
)


def set_espdl_attention_simulation_enabled(enabled: bool) -> None:
    """Force the P4 device emulator on regardless of ``op.platform``."""
    global _ESPDL_ATTENTION_SIMULATION_ENABLED
    _ESPDL_ATTENTION_SIMULATION_ENABLED = bool(enabled)


# Device computes in fp32; round scalars with ``_fp32`` so Python fp64 does not drift 1 ULP.
_TILE_K: int = 64
_FLOAT_EXP_LUT_ENTRIES: int = 1025
_FLOAT_EXP_LUT_MAX_NEG: float = 16.0
_FLOAT_EXP_PRECISE_MAX_R: float = 33.0
_PV_S16_BITS: int = 29


def _fp32(x) -> float:
    """Round a scalar to fp32-nearest then back to Python float."""
    import numpy as _np

    return float(_np.float32(x))


# Device-kernel constants.
_FLOAT_EXP_LUT_STEP: float = _fp32(_fp32(_FLOAT_EXP_LUT_MAX_NEG) / _fp32(_FLOAT_EXP_LUT_ENTRIES - 1))
_FLOAT_EXP_LUT_INV_STEP: float = _fp32(_fp32(1.0) / _fp32(_FLOAT_EXP_LUT_STEP))
_INT8_EXP_SHIFT_MAX: int = 21
_INT32_MIN: int = -(1 << 31)
_MASK_DOTS_CLAMP: float = float(1 << 28)  # MASK_DOTS_CLAMP


_FLOAT_EXP_LUT_CACHE: Optional[torch.Tensor] = None
_FLOAT_EXP_LUT_PRECISE_CACHE: Optional[torch.Tensor] = None
_INT8_EXP_LUT_CACHE: dict = {}


def _float_exp_lut() -> torch.Tensor:
    """Cached fp32 ``exp(-i * step)`` LUT (1025 entries; mirrors C++ ``AttentionFloatExpLut``)."""
    global _FLOAT_EXP_LUT_CACHE
    if _FLOAT_EXP_LUT_CACHE is None:
        import numpy as np

        from .espdl_attention_exp_lut import FLOAT_EXP_LUT_BITS

        bits = np.asarray(FLOAT_EXP_LUT_BITS, dtype=np.uint32)
        _FLOAT_EXP_LUT_CACHE = torch.from_numpy(bits.view(np.float32).copy())
    return _FLOAT_EXP_LUT_CACHE


def _float_exp_lut_lookup(x: torch.Tensor) -> torch.Tensor:
    """Linear-interp ``exp(x)`` for x <= 0 (mirrors ``AttentionFloatExpLut::lookup``)."""
    lut = _float_exp_lut().to(device=x.device)
    zeros = torch.zeros_like(x)
    pos = torch.where(x >= 0, zeros, -x * _FLOAT_EXP_LUT_INV_STEP)
    idx = pos.to(torch.int64)
    max_idx = _FLOAT_EXP_LUT_ENTRIES - 1
    idx_clamped = idx.clamp(0, max_idx - 1)
    frac = pos - idx_clamped.to(pos.dtype)
    a = lut[idx_clamped].to(torch.float64)
    b = lut[idx_clamped + 1].to(torch.float64) - a
    val = (a + b * frac.to(torch.float64)).to(pos.dtype)
    val = torch.where(x >= 0, torch.ones_like(val), val)
    val = torch.where(idx >= max_idx, lut[max_idx].expand_as(val), val)
    return val


def _float_exp_lut_precise() -> torch.Tensor:
    """Extended table used by ``lookup_precise`` (mirrors C++ ``AttentionFloatExpLut``)."""
    global _FLOAT_EXP_LUT_PRECISE_CACHE
    if _FLOAT_EXP_LUT_PRECISE_CACHE is None:
        base = _float_exp_lut()
        steps = _FLOAT_EXP_LUT_ENTRIES - 1
        n = int(_FLOAT_EXP_PRECISE_MAX_R * _FLOAT_EXP_LUT_INV_STEP) + 1
        table = torch.zeros(n, dtype=torch.float32)
        table[:_FLOAT_EXP_LUT_ENTRIES] = base
        exp_m16 = table[steps]
        table[_FLOAT_EXP_LUT_ENTRIES : 2 * steps] = table[1:steps] * exp_m16
        _FLOAT_EXP_LUT_PRECISE_CACHE = table
    return _FLOAT_EXP_LUT_PRECISE_CACHE


def _float_exp_lut_lookup_precise(x: torch.Tensor) -> torch.Tensor:
    """Cubic-corrected ``exp(x)`` for x <= 0 (mirrors ``lookup_precise``)."""
    lut = _float_exp_lut_precise().to(device=x.device)
    r = -x
    idx = (r * _FLOAT_EXP_LUT_INV_STEP).to(torch.int64)
    idx = idx.clamp(0, lut.numel() - 1)
    d = r - idx.to(r.dtype) * _FLOAT_EXP_LUT_STEP
    p = 1.0 - d * (1.0 - d * (0.5 - d * (1.0 / 6.0)))
    val = lut[idx] * p
    val = torch.where(x >= 0, torch.ones_like(val), val)
    val = torch.where(r >= (2.0 * _FLOAT_EXP_LUT_MAX_NEG), torch.zeros_like(val), val)
    return val


def _build_int8_exp_lut(logit_scale: float) -> Tuple[torch.Tensor, int, int]:
    """512-entry int8-path exp LUT (values in [0, 127]) for one ``logit_scale``.

    Mirrors C++ ``build_int8_exp_lut``. Returns ``(table[512], shift, sat)``.
    """
    if logit_scale <= 0.0:
        raise ValueError(f"logit_scale must be positive, got {logit_scale}")
    cache_key = float(logit_scale)
    cached = _INT8_EXP_LUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import numpy as np

    logit_fp32 = _fp32(logit_scale)
    inv_alpha = _fp32((16.0 / 255.0) / logit_fp32)
    shift = 0
    if inv_alpha >= 1.0:
        while shift < _INT8_EXP_SHIFT_MAX and float(1 << (shift + 1)) <= inv_alpha:
            shift += 1
    step_fp32 = _fp32(float(1 << shift) * logit_fp32)

    idx_f32 = np.arange(512, dtype=np.float32)
    xs = torch.from_numpy(-(idx_f32 * np.float32(step_fp32)).copy())
    exp_vals = _float_exp_lut_lookup(xs).detach().cpu().numpy().astype(np.float32)
    raw = np.float32(127.0) * exp_vals + np.float32(0.5)
    q = np.clip(raw.astype(np.int32), 0, 127).astype(np.int8)
    table = torch.from_numpy(q.copy())
    sat = 511 << shift
    _INT8_EXP_LUT_CACHE[cache_key] = (table, shift, sat)
    return table, shift, sat


def _fake_quant_to_int(x_fp32: torch.Tensor, scale: float, qmin: int, qmax: int) -> torch.Tensor:
    """Recover int representation of a fake-quantized fp32 tensor."""
    return torch.round(x_fp32 / scale).clamp(qmin, qmax).to(torch.int32)


def _simulate_espdl_int8_attention(
    q_fp32: torch.Tensor,
    k_fp32: torch.Tensor,
    v_fp32: torch.Tensor,
    scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    o_scale: float,
    layout: str,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bit-exact Python emulator of the P4 int8 SDPA fast path.

    ``attn_mask`` (additive, broadcastable onto ``[batch_size, Hq, query_length, key_length]``) joins the
    int32 dots the way ``attention_mask_dots`` does: ``-inf`` slots are dropped
    from softmax, finite values become ``round_half_even(fp32(mask * (1 / logit_scale)))``
    clamped to ``±2^28``.
    """
    layout = layout.upper()
    if layout not in ESPDL_ATTENTION_LAYOUTS:
        raise ValueError(f"Attention: layout must be one of {sorted(ESPDL_ATTENTION_LAYOUTS)}, got {layout!r}.")

    # Quantize in the caller's layout first, then integer-permute BNHD → BHND.
    # Device gather is an integer permute of the stored int8 (no second round).
    # Fake-quant after an fp32 transpose can flip a rounding boundary.
    q_i8 = _fake_quant_to_int(q_fp32, q_scale, -128, 127)
    k_i8 = _fake_quant_to_int(k_fp32, k_scale, -128, 127)
    v_i8 = _fake_quant_to_int(v_fp32, v_scale, -128, 127)
    permute_back = False
    if layout == "BNHD":
        q_i8 = q_i8.transpose(1, 2).contiguous()
        k_i8 = k_i8.transpose(1, 2).contiguous()
        v_i8 = v_i8.transpose(1, 2).contiguous()
        permute_back = True

    device = q_i8.device
    batch_size, num_heads, query_length, head_dim = q_i8.shape
    num_kv_heads = k_i8.shape[1]
    # GQA/MQA: several query heads share one KV head. Device fast path also
    # walks Q per-head and indexes K/V at h_kv = h // group; repeat_interleave
    # here matches those integer reads (int8 copy, no re-rounding).
    if num_kv_heads != num_heads:
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                f"Attention emulator: Hq={num_heads} must be a positive multiple of num_kv_heads={num_kv_heads}"
            )
        group = num_heads // num_kv_heads
        k_i8 = k_i8.repeat_interleave(group, dim=1).contiguous()
        v_i8 = v_i8.repeat_interleave(group, dim=1).contiguous()
    _, _, key_length, _ = k_i8.shape

    q_scale_f32 = _fp32(q_scale)
    k_scale_f32 = _fp32(k_scale)
    v_scale_f32 = _fp32(v_scale)
    o_scale_f32 = _fp32(o_scale)
    softmax_scale_f32 = _fp32(scale)
    logit_scale_f32 = _fp32(_fp32(softmax_scale_f32 * q_scale_f32) * k_scale_f32)
    table, shift, sat = _build_int8_exp_lut(logit_scale_f32)
    table = table.to(device=device)

    mask_bias = None
    mask_drop = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros(attn_mask.shape, dtype=torch.float32, device=device).masked_fill(
                ~attn_mask.to(device), float("-inf")
            )
        mask_f32 = attn_mask.to(device=device, dtype=torch.float32).broadcast_to(
            (batch_size, num_heads, query_length, key_length)
        )
        mask_drop = torch.isneginf(mask_f32)
        to_dots = torch.tensor(_fp32(1.0 / logit_scale_f32), dtype=torch.float32, device=device)
        scaled = torch.where(mask_drop, torch.zeros_like(mask_f32), mask_f32) * to_dots
        scaled = scaled.clamp(-_MASK_DOTS_CLAMP, _MASK_DOTS_CLAMP)
        mask_bias = torch.round(scaled).to(torch.int64)

    # CUDA has no int32 matmul; fp32+TF32 can lose bits.
    dots_full = torch.matmul(q_i8.to(torch.float64), k_i8.to(torch.float64).transpose(-1, -2)).round().to(torch.int32)

    out_int8 = torch.empty((batch_size, num_heads, query_length, head_dim), dtype=torch.int8, device=device)

    inv_127_f32 = _fp32(1.0 / 127.0)
    v_scale_over_127_f32 = _fp32(v_scale_f32 * inv_127_f32)
    inv_o_scale_f32 = _fp32(1.0 / o_scale_f32)
    logit_scale_tensor = torch.tensor(logit_scale_f32, dtype=torch.float32, device=device)
    v_over_127_tensor = torch.tensor(v_scale_over_127_f32, dtype=torch.float32, device=device)
    inv_o_scale_tensor = torch.tensor(inv_o_scale_f32, dtype=torch.float32, device=device)
    inv_127_f64 = float(inv_127_f32)

    for b in range(batch_size):
        for h in range(num_heads):
            dots_bh = dots_full[b, h]
            value_head = v_i8[b, h].to(torch.int32)

            m_i32 = torch.full((query_length,), _INT32_MIN, dtype=torch.int64, device=device)
            l_sum = torch.zeros((query_length,), dtype=torch.float32, device=device)
            acc = torch.zeros((query_length, head_dim), dtype=torch.int32, device=device)

            for j0 in range(0, key_length, _TILE_K):
                j1 = min(j0 + _TILE_K, key_length)
                dots_tile = dots_bh[:, j0:j1].to(torch.int64)
                drop_tile = None
                if mask_bias is not None:
                    dots_tile = dots_tile + mask_bias[b, h, :, j0:j1]
                    drop_tile = mask_drop[b, h, :, j0:j1]
                    tile_max = torch.where(drop_tile, torch.full_like(dots_tile, _INT32_MIN), dots_tile)
                    tile_max = tile_max.max(dim=-1).values
                else:
                    tile_max = dots_tile.max(dim=-1).values
                m_new = torch.maximum(m_i32, tile_max)

                # no-op rescale by 1.0 would still round-trip int32 through fp32
                rescale_rows = (m_i32 != _INT32_MIN) & (m_new != m_i32)
                if rescale_rows.any():
                    delta_all = (m_i32 - m_new).to(torch.float32) * logit_scale_tensor
                    rescale_all = _float_exp_lut_lookup(delta_all)
                    idx = rescale_rows.nonzero(as_tuple=False).squeeze(-1)
                    rescale_sel = rescale_all[idx]
                    l_sum[idx] = l_sum[idx] * rescale_sel
                    # trunc() is the C cast in ``(int32_t)(float(acc) * rescale)``.
                    acc_sel = acc[idx].to(torch.float32) * rescale_sel.unsqueeze(-1)
                    acc[idx] = acc_sel.trunc().to(torch.int32)

                diff = m_new.unsqueeze(-1) - dots_tile
                diff = diff.clamp_(0, sat)
                idx = (diff >> shift) if shift > 0 else diff
                idx = idx.clamp_(0, 511).to(torch.int64)
                e_i8_tile = table[idx]
                if drop_tile is not None:
                    e_i8_tile = e_i8_tile.masked_fill(drop_tile, 0)

                e_sum = e_i8_tile.to(torch.int32).sum(dim=-1)
                l_sum = (l_sum.to(torch.float64) + e_sum.to(torch.float64) * inv_127_f64).to(torch.float32)

                value_tile = value_head[j0:j1]
                pv_dot = (
                    torch.matmul(
                        e_i8_tile.to(torch.float64),
                        value_tile.to(torch.float64),
                    )
                    .round()
                    .to(torch.int32)
                )
                acc = acc + pv_dot

                m_i32 = m_new

            inv_l = torch.where(
                l_sum > 0,
                v_over_127_tensor / l_sum,
                torch.zeros_like(l_sum),
            )
            val = acc.to(torch.float32) * inv_l.unsqueeze(-1) * inv_o_scale_tensor
            out_int8[b, h] = torch.round(val).clamp(-128, 127).to(torch.int8)

    if permute_back:
        out_int8 = out_int8.transpose(1, 2).contiguous()
    out = out_int8.to(torch.float32) * float(o_scale)
    return out


def _simulate_espdl_int16_attention(
    q_fp32: torch.Tensor,
    k_fp32: torch.Tensor,
    v_fp32: torch.Tensor,
    scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    o_scale: float,
    layout: str,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Device-path emulator of the P4 int16 SDPA fast path.

    Matches integer QK with ``int16_dot_shift(D)`` half-even rounding, then
    online softmax via ``lookup_precise``. P·V uses the recoverable split-exp
    value ``trunc(e * 2^29) / 2^29`` times integer V; it does not replay QACC
    lane reduction order.
    """
    layout = layout.upper()
    if layout not in ESPDL_ATTENTION_LAYOUTS:
        raise ValueError(f"Attention: layout must be one of {sorted(ESPDL_ATTENTION_LAYOUTS)}, got {layout!r}.")

    q_i16 = _fake_quant_to_int(q_fp32, q_scale, -32768, 32767)
    k_i16 = _fake_quant_to_int(k_fp32, k_scale, -32768, 32767)
    v_i16 = _fake_quant_to_int(v_fp32, v_scale, -32768, 32767)
    permute_back = False
    if layout == "BNHD":
        q_i16 = q_i16.transpose(1, 2).contiguous()
        k_i16 = k_i16.transpose(1, 2).contiguous()
        v_i16 = v_i16.transpose(1, 2).contiguous()
        permute_back = True

    device = q_i16.device
    batch_size, num_heads, query_length, head_dim = q_i16.shape
    num_kv_heads = k_i16.shape[1]
    if num_kv_heads != num_heads:
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                f"Attention emulator: Hq={num_heads} must be a positive multiple of num_kv_heads={num_kv_heads}"
            )
        group = num_heads // num_kv_heads
        k_i16 = k_i16.repeat_interleave(group, dim=1).contiguous()
        v_i16 = v_i16.repeat_interleave(group, dim=1).contiguous()
    _, _, key_length, _ = k_i16.shape

    q_scale_f32 = _fp32(q_scale)
    k_scale_f32 = _fp32(k_scale)
    v_scale_f32 = _fp32(v_scale)
    o_scale_f32 = _fp32(o_scale)
    softmax_scale_f32 = _fp32(scale)
    logit_scale_f32 = _fp32(_fp32(softmax_scale_f32 * q_scale_f32) * k_scale_f32)
    shift = 0
    while (1 << shift) < head_dim:
        shift += 1
    dot_scale_f32 = _fp32(logit_scale_f32 * float(1 << shift))
    pv_fixed = float(1 << _PV_S16_BITS)
    pv_scale_f32 = _fp32(v_scale_f32 / pv_fixed)

    mask_f32 = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros(attn_mask.shape, dtype=torch.float32, device=device).masked_fill(
                ~attn_mask.to(device), float("-inf")
            )
        mask_f32 = attn_mask.to(device=device, dtype=torch.float32).broadcast_to(
            (batch_size, num_heads, query_length, key_length)
        )

    dots_full = torch.matmul(q_i16.to(torch.float64), k_i16.to(torch.float64).transpose(-1, -2)).round().to(torch.int64)
    if shift > 0:
        floor_q = dots_full >> shift
        rem = dots_full - (floor_q << shift)
        half = dots_full.new_tensor(1 << (shift - 1), dtype=dots_full.dtype)
        increment = (rem > half) | ((rem == half) & ((floor_q & 1) != 0))
        shifted = floor_q + increment.to(dots_full.dtype)
    else:
        shifted = dots_full
    scores_full = shifted.to(torch.float32) * dot_scale_f32

    out_int16 = torch.empty((batch_size, num_heads, query_length, head_dim), dtype=torch.int16, device=device)
    inv_o_scale_tensor = torch.tensor(_fp32(1.0 / o_scale_f32), dtype=torch.float32, device=device)
    pv_scale_tensor = torch.tensor(pv_scale_f32, dtype=torch.float32, device=device)

    for b in range(batch_size):
        for h in range(num_heads):
            scores_bh = scores_full[b, h]
            value_head = v_i16[b, h].to(torch.int32)
            row_max = torch.full((query_length,), float("-inf"), dtype=torch.float32, device=device)
            l_sum = torch.zeros((query_length,), dtype=torch.float32, device=device)
            acc = torch.zeros((query_length, head_dim), dtype=torch.float32, device=device)

            for j0 in range(0, key_length, _TILE_K):
                j1 = min(j0 + _TILE_K, key_length)
                scores_tile = scores_bh[:, j0:j1].clone()
                if mask_f32 is not None:
                    scores_tile = scores_tile + mask_f32[b, h, :, j0:j1]
                if torch.isneginf(scores_tile).all():
                    continue
                tile_max = scores_tile.amax(dim=-1)
                m_new = torch.maximum(row_max, tile_max)

                rescale_rows = torch.isfinite(row_max) & (m_new != row_max)
                if rescale_rows.any():
                    rescale = _float_exp_lut_lookup_precise(row_max - m_new)
                    acc = torch.where(rescale_rows.unsqueeze(-1), acc * rescale.unsqueeze(-1), acc)
                    l_sum = torch.where(rescale_rows, l_sum * rescale, l_sum)

                delta = scores_tile - m_new.unsqueeze(-1)
                e = _float_exp_lut_lookup_precise(delta)
                e = torch.where(torch.isneginf(scores_tile), torch.zeros_like(e), e)
                e_fixed = torch.trunc(e * pv_fixed)
                l_sum = l_sum + e.to(torch.float32).sum(dim=-1)

                value_tile = value_head[j0:j1]
                pv_dot = torch.matmul(e_fixed.to(torch.float64), value_tile.to(torch.float64)).to(torch.float32)
                acc = acc + pv_dot * pv_scale_tensor

                finite_new = torch.isfinite(m_new)
                row_max = torch.where(finite_new, m_new, row_max)

            inv_l = torch.where(l_sum > 0, 1.0 / l_sum, torch.zeros_like(l_sum))
            val = acc * inv_l.unsqueeze(-1) * inv_o_scale_tensor
            out_int16[b, h] = torch.round(val).clamp(-32768, 32767).to(torch.int16)

    if permute_back:
        out_int16 = out_int16.transpose(1, 2).contiguous()
    return out_int16.to(torch.float32) * float(o_scale)


def _extract_sym_scale(config, num_of_bits: Optional[int] = None) -> Optional[float]:
    """Positive fp32 symmetric per-tensor scale from ``config``, or ``None``."""
    if config is None:
        return None
    try:
        from esp_ppq.core.quant import (
            QuantizationProperty,
            QuantizationStates,
        )
    except ImportError:
        return None
    if getattr(config, "dominated_by", None) is not None:  # follow OVERLAPPED chain
        config = config.dominated_by
    state = getattr(config, "state", None)
    if state is not None and state not in {
        QuantizationStates.ACTIVATED,
        QuantizationStates.PASSIVE,
        QuantizationStates.OVERLAPPED,
        QuantizationStates.BAKED,
        QuantizationStates.PASSIVE_BAKED,
    }:
        return None
    policy = getattr(config, "policy", None)
    if policy is None:
        return None
    if not policy.has_property(QuantizationProperty.LINEAR):
        return None
    if not policy.has_property(QuantizationProperty.PER_TENSOR):
        return None
    if not policy.has_property(QuantizationProperty.SYMMETRICAL):
        return None
    bits = int(getattr(config, "num_of_bits", 0))
    if num_of_bits is not None:
        if bits != num_of_bits:
            return None
    elif bits not in (8, 16):
        return None
    scale = getattr(config, "scale", None)
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            return None
        val = float(scale.item())
    else:
        val = float(scale)
    if not math.isfinite(val) or val <= 0.0:
        return None
    return val


def _decode_layout(raw) -> str:
    """Normalize a ``layout`` attribute (bytes or str) to ASCII upper case."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("ascii")
        except UnicodeDecodeError:
            return ""
    return str(raw).upper()


def _layout_head_seq_axes(layout: str) -> Tuple[int, int]:
    """Return (head_axis, seq_axis) for a 4-D Q/K/V tensor."""
    if str(layout).upper() == "BNHD":
        return 2, 1
    return 1, 2


def _attr_as_number(value):
    """Scalarize an ONNX / PPQ attribute for default-value comparison."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value
        return value.item()
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bytes, str, bool)):
        try:
            return item()
        except Exception:
            return value
    return value


def espdl_attention_attributes_ineligible_reason(attrs) -> str:
    """Return why ``attrs`` are outside the fused Attention subset, or ``''``."""
    attrs = attrs or {}
    if "qkv_hidden_sizes" in attrs:
        return "com.microsoft.Attention keeps its own runtime"
    non_default = []
    unknown = []
    for key, value in attrs.items():
        if key in _ATTENTION_FUSED_ATTRS or key in _ATTENTION_INFRA_ATTRS:
            continue
        if key in ESPDL_ATTENTION_DEFAULT_NOOP_ATTRS:
            try:
                if float(_attr_as_number(value)) == float(ESPDL_ATTENTION_DEFAULT_NOOP_ATTRS[key]):
                    continue
            except (TypeError, ValueError):
                pass
            non_default.append(key)
            continue
        unknown.append(key)
    blocked = sorted(non_default) + sorted(unknown)
    if blocked:
        return f"unsupported attrs {blocked}"
    return ""


_FLOAT32_MIN_NORMAL = 2.0**-126
_FLOAT32_MAX = 3.4028234663852886e38


def espdl_attention_scale_ineligible_reason(scale) -> str:
    """Return why an explicit ``scale`` cannot run on the device, or ``''``."""
    if scale is None:
        return ""
    try:
        value = float(scale)
    except (TypeError, ValueError):
        return f"scale={scale!r} is not a number"
    if not _FLOAT32_MIN_NORMAL <= value <= _FLOAT32_MAX:
        return f"scale={value!r} must be a finite, positive, normal float32"
    return ""


def espdl_attention_logit_scale_ineligible_reason(
    scale,
    q_scale: Optional[float],
    k_scale: Optional[float],
    head_dim: int,
) -> str:
    """Return why ``scale * 2^q_exp * 2^k_exp`` cannot run on the device, or ``''``."""
    if q_scale is None or k_scale is None:
        return ""
    import numpy as np

    if scale is None:
        if head_dim <= 0:
            return f"head_dim={head_dim} cannot form 1/sqrt(D)"
        softmax_scale = float(np.float32(1.0) / np.float32(np.sqrt(np.float32(head_dim))))
    else:
        reason = espdl_attention_scale_ineligible_reason(scale)
        if reason:
            return reason
        softmax_scale = float(scale)
    try:
        q_dequant = float(np.ldexp(np.float32(1.0), int(np.log2(np.asarray(float(q_scale))))))
        k_dequant = float(np.ldexp(np.float32(1.0), int(np.log2(np.asarray(float(k_scale))))))
    except (TypeError, ValueError, OverflowError):
        return "Q/K dequant scales cannot be converted to device exponents"

    with np.errstate(over="ignore"):
        logit = _fp32(_fp32(_fp32(softmax_scale) * _fp32(q_dequant)) * _fp32(k_dequant))
    if not _FLOAT32_MIN_NORMAL <= logit <= _FLOAT32_MAX:
        return f"logit_scale={logit!r} (scale x Q/K dequant scales) must be a finite, positive, normal float32"
    return ""


def espdl_attention_op_logit_scale_ineligible_reason(op: Operation, head_dim: int) -> str:
    """Check the device logit scale when calibrated Q/K scales are available."""
    in_cfg = getattr(op, "input_quant_config", None) or []
    q_scale = _extract_sym_scale(in_cfg[0]) if len(in_cfg) > 0 else None
    k_scale = _extract_sym_scale(in_cfg[1]) if len(in_cfg) > 1 else None
    return espdl_attention_logit_scale_ineligible_reason(op.attributes.get("scale"), q_scale, k_scale, head_dim)


def lower_espdl_attention_attributes(attrs) -> dict:
    """IR → .espdl: only the fields ``Attention::deserialize`` reads."""
    attrs = attrs or {}
    lowered = {}
    if attrs.get("scale") is not None:
        lowered["scale"] = attrs["scale"]
    try:
        if int(attrs.get("is_causal", 0) or 0) != 0:
            lowered["is_causal"] = 1
    except (TypeError, ValueError):
        if attrs.get("is_causal") is not None:
            lowered["is_causal"] = attrs["is_causal"]
    if "layout" in attrs:
        decoded = _decode_layout(attrs["layout"])
        if decoded and decoded != "BHND":
            lowered["layout"] = decoded
        elif not decoded:
            lowered["layout"] = attrs["layout"]
    if "quant_type" in attrs:
        lowered["quant_type"] = attrs["quant_type"]
    return lowered


def espdl_attention_mask_ineligible_reason(
    mask_shape: Optional[Sequence[int]],
    b: int,
    h_q: int,
    lq: int,
    lk: int,
) -> str:
    """Return why ``mask_shape`` cannot broadcast onto ``[b, h_q, lq, lk]``, or ``''``."""
    if mask_shape is None:
        return ""
    dims = list(mask_shape)
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in dims):
        return f"attn_mask has non-static dims {dims}; trace with concrete shapes first"
    if any(d <= 0 for d in dims):
        return f"attn_mask has a non-positive dim: {dims}"
    if len(dims) > 4:
        return (
            f"rank-{len(dims)} attn_mask {dims} cannot broadcast onto 4-D scores "
            f"[B={b}, H={h_q}, Lq={lq}, Lk={lk}]; broadcasting does not lower rank, "
            f"so squeeze the mask to <= 4 dims when exporting"
        )
    bm, hm, lqm, lkm = [1] * (4 - len(dims)) + dims
    if not ((bm == 1 or bm == b) and (hm == 1 or hm == h_q) and (lqm == 1 or lqm == lq) and (lkm == 1 or lkm == lk)):
        return (
            f"attn_mask {dims} (right-aligned to [{bm},{hm},{lqm},{lkm}]) is not "
            f"broadcastable to [B={b}, H={h_q}, Lq={lq}, Lk={lk}]"
        )
    return ""


ESPDL_ATTENTION_MASK_DTYPES = frozenset({DataType.FP32, DataType.BOOL})


def espdl_attention_shapes_eligible(
    qkv_shapes: Sequence[Optional[Sequence[int]]],
    layout: str = "BHND",
    mask_shape: Optional[Sequence[int]] = None,
) -> bool:
    """Shape half of the admission test, callable without an ``Operation``."""
    return not espdl_attention_shapes_ineligible_reason(qkv_shapes, layout=layout, mask_shape=mask_shape)


def espdl_attention_eligible(op: Operation, qkv_shapes: Sequence[Optional[Sequence[int]]]) -> bool:
    """True iff ``op`` maps onto the on-device Attention fast path."""
    return not espdl_attention_ineligible_reason(op, qkv_shapes)


def espdl_attention_shapes_ineligible_reason(
    qkv_shapes: Sequence[Optional[Sequence[int]]],
    layout: str = "BHND",
    mask_shape: Optional[Sequence[int]] = None,
) -> str:
    """Reason paired with espdl_attention_shapes_eligible."""
    if len(qkv_shapes) < 3:
        return f"only {len(qkv_shapes)} of Q/K/V have shape metadata"
    ranks = [None if s is None else len(s) for s in qkv_shapes[:3]]
    if any(r != 4 for r in ranks):
        return f"Q/K/V ranks {ranks} are not all 4-D; pre-reshape required"
    for name, shape in zip("QKV", qkv_shapes[:3]):
        symbolic = [d for d in shape if not isinstance(d, int)]
        if symbolic:
            return f"{name} has non-static dims {list(shape)}; trace with concrete shapes first"
    q_shape, k_shape, v_shape = qkv_shapes[0], qkv_shapes[1], qkv_shapes[2]
    head_dim = q_shape[3]
    if not (1 <= head_dim <= ESPDL_ATTENTION_MAX_HEAD_DIM):
        return (
            f"head_dim={head_dim} outside [1, {ESPDL_ATTENTION_MAX_HEAD_DIM}] supported by the device kernels; "
            f"raise ESP_DL_ATTENTION_MAX_HEAD_DIM and ESPDL_ATTENTION_MAX_HEAD_DIM together, "
            f"or export a model with smaller head_dim"
        )
    if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
        return f"batch mismatch Q={q_shape[0]} K={k_shape[0]} V={v_shape[0]}"
    head_axis, seq_axis = _layout_head_seq_axes(layout)
    if k_shape[head_axis] != v_shape[head_axis] or k_shape[seq_axis] != v_shape[seq_axis]:
        return (
            f"K and V must agree on (Hkv, Lk); got "
            f"K=({k_shape[head_axis]},{k_shape[seq_axis]}) V=({v_shape[head_axis]},{v_shape[seq_axis]})"
        )
    if k_shape[3] != head_dim or v_shape[3] != head_dim:
        return f"head_dim mismatch Q={head_dim} K={k_shape[3]} V={v_shape[3]}; v_head_size != head_size is unsupported"
    h_q, h_kv = q_shape[head_axis], k_shape[head_axis]
    if h_kv < 1 or (h_q % h_kv) != 0:
        return f"Hq must be a positive multiple of Hkv (GQA/MQA); got Hq={h_q} Hkv={h_kv}"
    lk = k_shape[seq_axis]
    if not (1 <= lk <= ESPDL_ATTENTION_MAX_KV_TOKENS):
        return (
            f"Lk={lk} outside [1, {ESPDL_ATTENTION_MAX_KV_TOKENS}] supported by the device kernels; "
            f"raise ESP_DL_ATTENTION_MAX_KV_TOKENS and ESPDL_ATTENTION_MAX_KV_TOKENS together, "
            f"or shorten the key sequence"
        )
    return espdl_attention_mask_ineligible_reason(mask_shape, q_shape[0], h_q, q_shape[seq_axis], lk)


def espdl_attention_head_attrs_ineligible_reason(
    attrs,
    qkv_shapes: Sequence[Optional[Sequence[int]]],
    layout: str = "BHND",
) -> str:
    """Validate ``q_num_heads`` / ``kv_num_heads`` against 4-D Q/K/V head axes."""
    attrs = attrs or {}
    if "q_num_heads" not in attrs and "kv_num_heads" not in attrs:
        return ""
    declared = []
    for key in ("q_num_heads", "kv_num_heads"):
        if key not in attrs or attrs[key] is None:
            declared.append(None)
            continue
        raw_val = attrs[key]
        actual = _attr_as_number(raw_val)
        if isinstance(actual, bool) or not isinstance(actual, numbers.Real):
            return f"{key}={raw_val!r} is not a positive integer"
        if isinstance(actual, numbers.Integral):
            count = int(actual)
        else:
            as_float = float(actual)
            if not as_float.is_integer():
                return f"{key}={raw_val!r} is not a positive integer"
            count = int(as_float)
        if count < 0:
            return f"{key}={count} is not a positive integer"
        declared.append(None if count == 0 else count)
    if not qkv_shapes or len(qkv_shapes) < 3:
        return ""
    shapes = []
    for shape in qkv_shapes[:3]:
        if shape is None or len(shape) != 4:
            return ""
        if any(isinstance(dim, bool) or not isinstance(dim, numbers.Integral) for dim in shape):
            return ""
        shapes.append([int(dim) for dim in shape])
    decoded = _decode_layout(layout) or "BHND"
    head_axis, _ = _layout_head_seq_axes(decoded)
    q_count, kv_count = declared
    if q_count is not None and q_count != shapes[0][head_axis]:
        return (
            f"q_num_heads={q_count} does not match the Q head axis {shapes[0][head_axis]} "
            f"under layout {decoded}; 4-D head counts are read from the shape"
        )
    if kv_count is not None:
        k_heads, v_heads = shapes[1][head_axis], shapes[2][head_axis]
        if kv_count != k_heads or kv_count != v_heads:
            shown = str(k_heads) if k_heads == v_heads else f"K={k_heads}, V={v_heads}"
            return (
                f"kv_num_heads={kv_count} does not match the K/V head axis ({shown}) "
                f"under layout {decoded}; 4-D head counts are read from the shape"
            )
    return ""


def espdl_attention_ineligible_reason_from_parts(
    attrs,
    num_of_input: int,
    qkv_shapes: Sequence[Optional[Sequence[int]]],
    mask_shape: Optional[Sequence[int]] = None,
    mask_dtype: Optional[DataType] = None,
) -> str:
    """Return why an Attention with these parts cannot run on the device, or ``''``."""
    attrs = attrs or {}
    attr_reason = espdl_attention_attributes_ineligible_reason(attrs)
    if attr_reason:
        return attr_reason
    scale_reason = espdl_attention_scale_ineligible_reason(attrs.get("scale"))
    if scale_reason:
        return scale_reason
    if num_of_input > 4:
        return f"{num_of_input} inputs, KV cache is unsupported"
    layout = _decode_layout(attrs.get("layout", "BHND"))
    if layout not in ESPDL_ATTENTION_LAYOUTS:
        return f"layout={layout!r} is unsupported; use BHND or BNHD"
    if mask_dtype is not None and mask_dtype not in ESPDL_ATTENTION_MASK_DTYPES:
        name = getattr(mask_dtype, "name", str(mask_dtype))
        return f"attn_mask dtype {name} is unsupported; export a bool (True = attend) or fp32 additive mask"
    shape_reason = espdl_attention_shapes_ineligible_reason(qkv_shapes, layout=layout, mask_shape=mask_shape)
    if shape_reason:
        return shape_reason
    return espdl_attention_head_attrs_ineligible_reason(attrs, qkv_shapes, layout)


def espdl_attention_ineligible_reason(op: Operation, qkv_shapes: Sequence[Optional[Sequence[int]]]) -> str:
    """Human-readable reason paired with espdl_attention_eligible."""
    mask = op.inputs[3] if op.num_of_input >= 4 else None
    mask_shape = None if mask is None or getattr(mask, "shape", None) is None else list(mask.shape)
    mask_dtype = None if mask is None else mask.dtype
    return espdl_attention_ineligible_reason_from_parts(
        op.attributes,
        op.num_of_input,
        qkv_shapes,
        mask_shape=mask_shape,
        mask_dtype=mask_dtype,
    )


def compose_causal_into_mask(
    attn_mask: "torch.Tensor | None",
    is_causal: bool,
    q: torch.Tensor,
    k: torch.Tensor,
) -> Tuple["torch.Tensor | None", bool]:
    """Fold the causal pattern into ``attn_mask`` so only one reaches F.SDPA.

    torch < 2.3 rejects ``attn_mask`` and ``is_causal`` together; folding
    keeps the ONNX additive composition on every supported torch.
    """
    if not is_causal or attn_mask is None:
        return attn_mask, is_causal
    l_q, l_k = int(q.shape[-2]), int(k.shape[-2])
    keep = torch.ones(l_q, l_k, dtype=torch.bool, device=q.device).tril()
    causal_bias = torch.zeros(l_q, l_k, dtype=attn_mask.dtype, device=q.device).masked_fill(~keep, float("-inf"))
    return attn_mask + causal_bias, False


def _emulator_mask(
    attn_mask: "torch.Tensor | None",
    is_causal: bool,
    l_q: int,
    l_k: int,
    device: torch.device,
) -> "torch.Tensor | None":
    """Fold the causal mask into ``attn_mask`` for the integer emulators."""
    if not is_causal:
        return None if attn_mask is None else attn_mask.contiguous()
    keep = torch.ones(l_q, l_k, dtype=torch.bool, device=device).tril()
    if attn_mask is None:
        return keep
    if attn_mask.dtype == torch.bool:
        additive = torch.zeros(attn_mask.shape, dtype=torch.float32, device=device).masked_fill(
            ~attn_mask.to(device), float("-inf")
        )
    else:
        additive = attn_mask.to(device=device, dtype=torch.float32)
    return torch.where(keep, additive, torch.full_like(additive, float("-inf"))).contiguous()


def _espdl_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: "torch.Tensor | None",
    scale: float,
    is_causal: bool,
    layout: str = "BHND",
) -> torch.Tensor:
    """FP32 reference for the fused SDPA op."""
    if q.ndim != 4:
        raise ValueError(f"Q must be 4D, got {tuple(q.shape)}")
    if k.ndim != 4:
        raise ValueError(f"K must be 4D, got {tuple(k.shape)}")
    if v.ndim != 4:
        raise ValueError(f"V must be 4D, got {tuple(v.shape)}")

    layout = str(layout).upper()
    if layout not in ESPDL_ATTENTION_LAYOUTS:
        raise ValueError(f"Attention: layout must be one of {sorted(ESPDL_ATTENTION_LAYOUTS)}, got {layout!r}.")
    permute_back = False
    if layout == "BNHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        permute_back = True
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"Q/K/V must share batch dim; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"K.Hkv ({k.shape[1]}) must equal V.Hkv ({v.shape[1]})")
    h_q, h_kv = int(q.shape[1]), int(k.shape[1])
    if h_kv <= 0 or h_q % h_kv != 0:
        raise ValueError(f"Hq must be a multiple of Hkv (GQA/MQA); got Hq={h_q}, Hkv={h_kv}")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"K.Lk ({k.shape[2]}) must equal V.Lk ({v.shape[2]})")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError(f"Q/K/V must share head_dim D; got {q.shape[3]}, {k.shape[3]}, {v.shape[3]}")

    if h_kv != h_q:
        group = h_q // h_kv
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    normalized_mask = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            normalized_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(~attn_mask, float("-inf"))
        else:
            normalized_mask = attn_mask.to(q.dtype)

    normalized_mask, sdpa_causal = compose_causal_into_mask(normalized_mask, bool(is_causal), q, k)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=normalized_mask,
        dropout_p=0.0,
        is_causal=sdpa_causal,
        scale=scale,
    )
    if permute_back:
        out = out.transpose(1, 2).contiguous()
    return out


def espdl_attention_forward(
    op: Operation,
    values: List[torch.Tensor],
    ctx=None,
    **kwargs,
) -> List[torch.Tensor]:
    """Fused Attention used by the ESP-DL device path.

    Q, K and V must already be 4-D. A node the device can run and whose
    Q/K/V/O scales are symmetric int8 or int16 uses the tiled emulator;
    otherwise this falls back to the fp32 reference.

    Attributes
        is_causal : int (default is 0)
            When 1, apply an upper-left causal mask (key index j <= query index i)
            before the additive mask.
        kv_num_heads : int
            Number of key and value heads. Query heads must be a positive multiple.
        q_num_heads : int
            Number of query heads.
        scale : float
            Multiplier of QK^T. The default is 1/sqrt(head_dim).
        layout : string (default is BHND)
            BHND or BNHD.
        quant_type : string
            S8, S16 or F32.
        softcap : float (default is 0.0)
            Only the default is accepted.
        qk_matmul_output_mode : int (default is 0)
            Only the default is accepted.

    Inputs (3 - 4)
        Q (differentiable) : T
            Query.
        K (differentiable) : T
            Key.
        V (differentiable) : T
            Value. head_dim must match Q and K.
        attn_mask (optional, differentiable) : T
            Bool mask (True keeps the score) or a float additive mask,
            right-aligned onto [batch, query_heads, query_len, kv_len].

    Outputs
        Y (differentiable) : T
            Same shape and layout as Q. Wrapped in a one-element list.
    """
    if len(values) < 3:
        raise ValueError(f"Attention {op.name} expects at least 3 inputs (Q, K, V), got {len(values)}")
    q, k, v = values[0], values[1], values[2]
    attn_mask = values[3] if len(values) > 3 else None

    qkv_shapes = [list(q.shape), list(k.shape), list(v.shape)]
    reason = espdl_attention_ineligible_reason(op, qkv_shapes)
    if reason:
        raise ValueError(f"Attention {op.name}: {reason}")
    if q.ndim >= 1:
        logit_reason = espdl_attention_op_logit_scale_ineligible_reason(op, int(q.shape[-1]))
        if logit_reason:
            raise ValueError(f"Attention {op.name}: {logit_reason}")

    attr_scale = op.attributes.get('scale', None)
    if attr_scale is None:
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
    else:
        scale = float(attr_scale)
    is_causal = bool(op.attributes.get('is_causal', 0))
    layout = _decode_layout(op.attributes.get('layout', 'BHND')) or 'BHND'

    orig_dtype = q.dtype

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool or attn_mask.is_floating_point():
            pass
        else:
            raise ValueError(
                f"Attention {op.name}: attn_mask dtype {attn_mask.dtype} is unsupported; "
                "export a bool (True = attend) or fp32 additive mask"
            )
        if q.ndim == 4 and k.ndim == 4:
            head_axis, seq_axis = _layout_head_seq_axes(layout)
            mask_reason = espdl_attention_mask_ineligible_reason(
                list(attn_mask.shape),
                int(q.shape[0]),
                int(q.shape[head_axis]),
                int(q.shape[seq_axis]),
                int(k.shape[seq_axis]),
            )
            if mask_reason:
                raise ValueError(f"Attention {op.name}: {mask_reason}")

    use_emulator = (
        getattr(op, "platform", None) in _ATTENTION_EMULATED_PLATFORMS
        or _ESPDL_ATTENTION_SIMULATION_ENABLED
        or os.environ.get(_ENV_DEVICE_EMULATE_KEY, "0") in ("1", "true", "True")
    )
    if use_emulator and q.ndim == 4 and k.ndim == 4 and v.ndim == 4:
        canonical_q, canonical_k = q, k
        head_dim = int(canonical_q.shape[-1])
        head_axis, seq_axis = _layout_head_seq_axes(layout)
        h_q = int(canonical_q.shape[head_axis])
        h_kv = int(canonical_k.shape[head_axis])
        l_q = int(canonical_q.shape[seq_axis])
        l_k = int(canonical_k.shape[seq_axis])
        can_emulate = h_kv > 0 and h_q % h_kv == 0 and 0 < head_dim <= ESPDL_ATTENTION_MAX_HEAD_DIM
        if can_emulate:
            in_cfg = getattr(op, "input_quant_config", None) or []
            out_cfg = getattr(op, "output_quant_config", None) or []

            def _scales(bits: int):
                q_s = _extract_sym_scale(in_cfg[0], bits) if len(in_cfg) > 0 else None
                k_s = _extract_sym_scale(in_cfg[1], bits) if len(in_cfg) > 1 else None
                v_s = _extract_sym_scale(in_cfg[2], bits) if len(in_cfg) > 2 else None
                o_s = _extract_sym_scale(out_cfg[0], bits) if len(out_cfg) > 0 else None
                return q_s, k_s, v_s, o_s

            if attr_scale is None:
                import numpy as np

                head_dim_f32 = np.float32(int(q.shape[-1]))
                scale_for_emulator = float(np.float32(1.0) / np.float32(np.sqrt(head_dim_f32)))
            else:
                scale_for_emulator = scale
            emu_mask = _emulator_mask(attn_mask, is_causal, l_q, l_k, q.device)

            q8, k8, v8, o8 = _scales(8)
            q16, k16, v16, o16 = _scales(16)
            if None not in (q8, k8, v8, o8):
                out = _simulate_espdl_int8_attention(
                    q.float().contiguous(),
                    k.float().contiguous(),
                    v.float().contiguous(),
                    scale_for_emulator,
                    q8,
                    k8,
                    v8,
                    o8,
                    layout=layout,
                    attn_mask=emu_mask,
                )
                return [out.to(dtype=orig_dtype)]
            if None not in (q16, k16, v16, o16):
                out = _simulate_espdl_int16_attention(
                    q.float().contiguous(),
                    k.float().contiguous(),
                    v.float().contiguous(),
                    scale_for_emulator,
                    q16,
                    k16,
                    v16,
                    o16,
                    layout=layout,
                    attn_mask=emu_mask,
                )
                return [out.to(dtype=orig_dtype)]

    out = _espdl_attention_reference(
        q.float().contiguous(),
        k.float().contiguous(),
        v.float().contiguous(),
        None if attn_mask is None else attn_mask.contiguous(),
        scale,
        is_causal,
        layout=layout,
    )
    return [out.to(dtype=orig_dtype)]

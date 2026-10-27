"""ESP-DL RotaryEmbedding reference and integer-path emulator."""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import torch

from esp_ppq.core import (
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
)
from esp_ppq.IR import Operation
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.quantization.qfunction import PPQLinearQuant_toInt
from esp_ppq.utils.round import ppq_tensor_round

from .base import ASSERT_NUM_OF_INPUT, GET_ATTRIBUTE_FROM_OPERATION

_ENV_DEVICE_EMULATE_KEY = "ESPDL_ROPE_DEVICE_EMULATE"
_ESPDL_ROPE_SIMULATION_ENABLED = False
_ROPE_P4_PLATFORMS = frozenset(
    {
        TargetPlatform.ESPDL_INT8,
        TargetPlatform.ESPDL_INT16,
        TargetPlatform.ESPDL_H_PRE_INT16,
        TargetPlatform.ESPDL_W8A16,
    }
)
_ROPE_HALF_UP_PLATFORMS = frozenset(
    {
        TargetPlatform.ESPDL_S3_INT8,
        TargetPlatform.ESPDL_S3_INT16,
        TargetPlatform.ESPDL_S3_H_PRE_INT16,
        TargetPlatform.ESPDL_S3_W8A16,
        TargetPlatform.ESPDL_C_INT8,
        TargetPlatform.ESPDL_C_INT16,
        TargetPlatform.ESPDL_C_H_PRE_INT16,
        TargetPlatform.ESPDL_C_W8A16,
    }
)
_ROPE_EMULATED_PLATFORMS = _ROPE_P4_PLATFORMS | _ROPE_HALF_UP_PLATFORMS
_READY_STATES = frozenset(
    {
        QuantizationStates.ACTIVATED,
        QuantizationStates.PASSIVE,
        QuantizationStates.OVERLAPPED,
        QuantizationStates.BAKED,
        QuantizationStates.PASSIVE_BAKED,
    }
)


def set_espdl_rotary_embedding_simulation_enabled(enabled: bool) -> None:
    """Force the P4 emulator on for tests, irrespective of ``op.platform``."""
    global _ESPDL_ROPE_SIMULATION_ENABLED
    _ESPDL_ROPE_SIMULATION_ENABLED = bool(enabled)


def _simulation_forced() -> bool:
    return _ESPDL_ROPE_SIMULATION_ENABLED or os.environ.get(_ENV_DEVICE_EMULATE_KEY, "0") in {
        "1",
        "true",
        "True",
    }


def _use_device_emulator(op: Operation) -> bool:
    return getattr(op, "platform", None) in _ROPE_EMULATED_PLATFORMS or _simulation_forced()


def _validate_and_normalize(
    op: Operation, values: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    """Validate the device-supported vision subset and normalize cache shape."""
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    interleaved = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="interleaved", default=0)
    num_heads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="num_heads", default=0)
    rotary_embedding_dim = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="rotary_embedding_dim", default=0)

    if interleaved:
        raise ValueError(
            f"RotaryEmbedding {op.name}: interleaved={interleaved} (GPT-J layout) is unsupported; "
            "the ESP-DL vision subset only implements interleaved=0"
        )

    x, cos_cache, sin_cache = values[0], values[1], values[2]
    supported_dtypes = {torch.float32, torch.float64, torch.int8, torch.int16}
    for label, tensor in (("X", x), ("cos_cache", cos_cache), ("sin_cache", sin_cache)):
        if tensor.dtype not in supported_dtypes:
            raise ValueError(
                f"RotaryEmbedding {op.name}: {label} dtype {tensor.dtype} is unsupported; "
                "ESP-DL accepts float32, int8, or int16 only"
            )

    if x.dim() != 4:
        raise ValueError(
            f"RotaryEmbedding {op.name}: X must be 4D [B,H,L,D], got shape {tuple(x.shape)}; "
            "3D [B,L,H*D] is unsupported"
        )
    B, H, L, D = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
    if any(dim <= 0 for dim in (B, H, L, D)):
        raise ValueError(f"RotaryEmbedding {op.name}: X dimensions must be positive, got shape {tuple(x.shape)}")
    if num_heads and int(num_heads) != H:
        raise ValueError(f"RotaryEmbedding {op.name}: num_heads={num_heads} contradicts X's H axis ({H})")
    if D % 2 != 0:
        raise ValueError(f"RotaryEmbedding {op.name}: head_dim D must be even, got {D}")
    if rotary_embedding_dim and int(rotary_embedding_dim) != D:
        raise ValueError(
            f"RotaryEmbedding {op.name}: rotary_embedding_dim={rotary_embedding_dim} != head_dim D={D}; "
            "partial rotation is unsupported"
        )

    D_half = D // 2
    if tuple(cos_cache.shape) != tuple(sin_cache.shape):
        raise ValueError(
            f"RotaryEmbedding {op.name}: cos_cache shape {tuple(cos_cache.shape)} != "
            f"sin_cache shape {tuple(sin_cache.shape)}"
        )
    if cos_cache.dtype != sin_cache.dtype and not (
        cos_cache.dtype in (torch.float32, torch.float64) and sin_cache.dtype in (torch.float32, torch.float64)
    ):
        raise ValueError(
            f"RotaryEmbedding {op.name}: cos_cache dtype {cos_cache.dtype} != sin_cache dtype {sin_cache.dtype}"
        )
    cache_rank = cos_cache.dim()
    if cache_rank not in (2, 3):
        raise ValueError(
            f"RotaryEmbedding {op.name}: cos/sin cache must be 2D or 3D, got rank {cache_rank} "
            f"with shape {tuple(cos_cache.shape)}"
        )
    if int(cos_cache.shape[-1]) != D_half:
        raise ValueError(
            f"RotaryEmbedding {op.name}: cache last dim (cos={cos_cache.shape[-1]}, sin={sin_cache.shape[-1]}) "
            f"!= D/2 ({D_half})"
        )

    if cache_rank == 2:
        if int(cos_cache.shape[0]) != L:
            raise ValueError(
                f"RotaryEmbedding {op.name}: cache sequence length L={int(cos_cache.shape[0])} != X's L={L}"
            )
        cos_cache = cos_cache.unsqueeze(0)
        sin_cache = sin_cache.unsqueeze(0)
    else:
        cache_B = int(cos_cache.shape[0])
        if cache_B not in (1, B):
            raise ValueError(f"RotaryEmbedding {op.name}: cache batch B={cache_B} matches neither X's B={B} nor 1")
        if int(cos_cache.shape[1]) != L:
            raise ValueError(
                f"RotaryEmbedding {op.name}: cache sequence length L={int(cos_cache.shape[1])} != X's L={L}"
            )

    return x, cos_cache, sin_cache, B, H, L, D_half


def rotary_embedding_reference(
    op: Operation,
    values: List[torch.Tensor],
    ctx=None,
    **kwargs,
) -> torch.Tensor:
    """Floating-point ONNX reference for ESP-DL's supported RoPE subset."""
    x, cos_cache, sin_cache, _, _, _, _ = _validate_and_normalize(op, values)
    original_dtype = x.dtype

    # [B,H,L,D] -> [B,L,H,D], with cache [B|1,L,D/2] broadcast over heads.
    inp = x.float().permute(0, 2, 1, 3)
    cos = cos_cache.float().unsqueeze(2)
    sin = sin_cache.float().unsqueeze(2)
    x_first, x_second = torch.chunk(inp, 2, dim=-1)
    real = cos * x_first - sin * x_second
    imag = sin * x_first + cos * x_second
    return torch.cat((real, imag), dim=-1).permute(0, 2, 1, 3).contiguous().to(dtype=original_dtype)


def _power_of_two_scale(config) -> Optional[Tuple[int, float]]:
    """Return an exact power-of-two exponent and scale, or ``None``."""
    if config is None or getattr(config, "state", None) not in _READY_STATES:
        return None
    root = getattr(config, "dominated_by", config)
    policy = getattr(root, "policy", None)
    if policy is None or not (
        policy.has_property(QuantizationProperty.LINEAR)
        and policy.has_property(QuantizationProperty.SYMMETRICAL)
        and policy.has_property(QuantizationProperty.PER_TENSOR)
        and policy.has_property(QuantizationProperty.POWER_OF_2)
    ):
        return None
    bits = int(getattr(config, "num_of_bits", 0))
    if bits not in (8, 16):
        return None
    scale = getattr(root, "scale", None)
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            return None
        scale = float(scale.detach().cpu().item())
    else:
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    mantissa, exponent = math.frexp(scale)
    if mantissa != 0.5:
        return None
    # frexp(scale) == (0.5, exponent), so scale == 2 ** (exponent - 1).
    return exponent - 1, scale


def _extract_integer_contract(op: Operation):
    """Read the four device tensor contracts from a calibrated quantized op."""
    if not isinstance(op, QuantableOperation):
        return None
    in_cfg = list(getattr(op, "input_quant_config", ()) or ())
    out_cfg = list(getattr(op, "output_quant_config", ()) or ())
    if len(in_cfg) < 3 or len(out_cfg) < 1:
        return None
    contracts = [_power_of_two_scale(cfg) for cfg in (*in_cfg[:3], out_cfg[0])]
    if any(contract is None for contract in contracts):
        return None
    x, cos, sin, y = contracts
    x_bits, cos_bits, sin_bits, y_bits = (
        int(in_cfg[0].num_of_bits),
        int(in_cfg[1].num_of_bits),
        int(in_cfg[2].num_of_bits),
        int(out_cfg[0].num_of_bits),
    )
    if cos_bits != sin_bits:
        return None
    return {
        "configs": (in_cfg[0], in_cfg[1], in_cfg[2], out_cfg[0]),
        "exponents": (x[0], cos[0], sin[0], y[0]),
        "scales": (x[1], cos[1], sin[1], y[1]),
        "bits": (x_bits, cos_bits, sin_bits, y_bits),
    }


def _round_shift_half_even(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Round signed integer ``value / 2**shift`` to nearest-even."""
    if shift <= 0:
        return value
    if shift > 63:
        return torch.zeros_like(value)
    if shift == 63:
        # ``1 << 63`` cannot be represented by torch.int64. At this boundary,
        # values outside (-2**62, 2**62) round to +/-1; exact half-way values
        # tie to the even result 0.
        halfway = 1 << 62
        return torch.where(
            value > halfway,
            torch.ones_like(value),
            torch.where(value < -halfway, -torch.ones_like(value), torch.zeros_like(value)),
        )
    divisor = 1 << shift
    quotient = torch.div(value, divisor, rounding_mode="floor")
    remainder = value - quotient * divisor
    halfway = divisor >> 1
    increment = (remainder > halfway) | ((remainder == halfway) & ((quotient & 1) != 0))
    return quotient + increment.to(dtype=quotient.dtype)


def _round_shift_half_up(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Round signed integer ``value / 2**shift`` with ``floor(x + 0.5)``."""
    if shift <= 0:
        return value
    if shift > 63:
        return torch.zeros_like(value)
    if shift == 63:
        halfway = 1 << 62
        return torch.where(
            value >= halfway,
            torch.ones_like(value),
            torch.where(value < -halfway, -torch.ones_like(value), torch.zeros_like(value)),
        )
    divisor = 1 << shift
    quotient = torch.div(value, divisor, rounding_mode="floor")
    remainder = value - quotient * divisor
    halfway = divisor >> 1
    increment = remainder >= halfway
    return quotient + increment.to(dtype=quotient.dtype)


def _rescale_integer(
    value: torch.Tensor,
    shift: int,
    rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
) -> torch.Tensor:
    """Apply the device's signed power-of-two rescale without int64 overflow."""
    if shift > 0:
        if rounding == RoundingPolicy.ROUND_HALF_UP:
            return _round_shift_half_up(value, shift)
        return _round_shift_half_even(value, shift)
    if shift == 0:
        return value

    left_shift = -shift
    int64_min, int64_max = -(1 << 63), (1 << 63) - 1
    if left_shift >= 63:
        return torch.where(
            value > 0,
            torch.full_like(value, int64_max),
            torch.where(value < 0, torch.full_like(value, int64_min), torch.zeros_like(value)),
        )

    factor = 1 << left_shift
    min_before_shift = -(1 << (63 - left_shift))
    max_before_shift = int64_max // factor
    clipped = value.clamp(min=min_before_shift, max=max_before_shift)
    shifted = clipped * factor
    return torch.where(
        value > max_before_shift,
        torch.full_like(value, int64_max),
        torch.where(value < min_before_shift, torch.full_like(value, int64_min), shifted),
    )


def _saturate(value: torch.Tensor, bits: int) -> torch.Tensor:
    return value.clamp(min=-(1 << (bits - 1)), max=(1 << (bits - 1)) - 1)


def _to_integer(tensor: torch.Tensor, config) -> torch.Tensor:
    # PPQ's executor provides fake-quantized fp32 tensors. Reconstructing the
    # stored signed integer through the same PPQ rounding policy makes this
    # model usable before parameter baking as well as after it.
    return PPQLinearQuant_toInt(tensor, config).to(dtype=torch.int64)


def _simd_eligible(bits: Tuple[int, int, int, int], exponents: Tuple[int, int, int, int], D_half: int) -> Optional[int]:
    """Return mac_shift for a P4 SIMD path, otherwise ``None``."""
    x_bits, cos_bits, sin_bits, y_bits = bits
    x_exp, cos_exp, sin_exp, y_exp = exponents
    if cos_exp != sin_exp:
        return None
    mac_shift = y_exp - x_exp - cos_exp
    if x_bits == cos_bits == sin_bits == y_bits == 8:
        return mac_shift if D_half % 16 == 0 and 0 < mac_shift <= 31 else None
    if x_bits == cos_bits == sin_bits == y_bits == 16:
        return mac_shift if D_half % 8 == 0 and 0 < mac_shift <= 63 else None
    if x_bits == y_bits == 8 and cos_bits == sin_bits == 16:
        return mac_shift if D_half % 16 == 0 and 0 < mac_shift <= 31 else None
    return None


def simulate_espdl_rotary_embedding(
    op: Operation,
    values: List[torch.Tensor],
    *,
    force_scalar: bool = False,
) -> Optional[torch.Tensor]:
    """Emulate a calibrated P4 integer RoPE path.

    Returns ``None`` if the op has no complete integer contract, allowing the
    caller to use :func:`rotary_embedding_reference`.  ``force_scalar`` is a
    regression-test hook for the C++ typed fallback when SIMD is unavailable.
    """
    contract = _extract_integer_contract(op)
    if contract is None:
        return None

    x, cos_cache, sin_cache, B, H, L, D_half = _validate_and_normalize(op, values)
    orig_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    x_cfg, cos_cfg, sin_cfg, y_cfg = contract["configs"]
    x_exp, cos_exp, sin_exp, y_exp = contract["exponents"]
    x_scale, cos_scale, sin_scale, y_scale = contract["scales"]
    x_bits, cos_bits, sin_bits, y_bits = contract["bits"]
    op_platform = getattr(op, "platform", None)
    rounding = getattr(y_cfg, "rounding", None)
    if rounding not in (RoundingPolicy.ROUND_HALF_EVEN, RoundingPolicy.ROUND_HALF_UP):
        rounding = (
            RoundingPolicy.ROUND_HALF_UP if op_platform in _ROPE_HALF_UP_PLATFORMS else RoundingPolicy.ROUND_HALF_EVEN
        )

    # ``QuantableOperation`` receives fake-quantized values. Recover exactly
    # what ESP-DL stores, not merely a rounded copy of the ideal fp32 input.
    qx = _to_integer(x.float(), x_cfg).reshape(B, H, L, D_half * 2)
    qc = _to_integer(cos_cache.float(), cos_cfg)
    qs = _to_integer(sin_cache.float(), sin_cfg)
    if qc.shape[0] == 1 and B != 1:
        qc = qc.expand(B, -1, -1)
        qs = qs.expand(B, -1, -1)
    qc = qc.unsqueeze(1)
    qs = qs.unsqueeze(1)

    qx_first, qx_second = qx[..., :D_half], qx[..., D_half:]
    is_p4_platform = op_platform in _ROPE_P4_PLATFORMS or op_platform is None
    simd_shift = (
        None
        if force_scalar or not is_p4_platform
        else _simd_eligible(
            (x_bits, cos_bits, sin_bits, y_bits),
            (x_exp, cos_exp, sin_exp, y_exp),
            D_half,
        )
    )

    typed_integer_combo = (
        x_bits == cos_bits == sin_bits == y_bits == 8
        or x_bits == cos_bits == sin_bits == y_bits == 16
        or (x_bits == y_bits == 8 and cos_bits == sin_bits == 16)
    )
    if typed_integer_combo and cos_exp == sin_exp:
        mac_shift = y_exp - x_exp - cos_exp

        # Pure-width assembly uses a saturating VSUB. Production dispatch
        # scans x2 and sends any row containing INT_MIN to the exact scalar
        # integer path; rows without INT_MIN are already mathematically exact.
        pure_width = x_bits == cos_bits == sin_bits == y_bits
        if simd_shift is not None and pure_width:
            neg_min = -(1 << (x_bits - 1))
            neg_max = (1 << (x_bits - 1)) - 1
            simd_neg_second = (-qx_second).clamp(min=neg_min, max=neg_max)
            row_uses_scalar = (qx_second == neg_min).any(dim=-1, keepdim=True)
            neg_second = torch.where(row_uses_scalar, -qx_second, simd_neg_second)
        else:
            neg_second = -qx_second
        acc_real = qx_first * qc + neg_second * qs
        acc_imag = qx_first * qs + qx_second * qc
        q_real = _rescale_integer(acc_real, mac_shift, rounding=rounding)
        q_imag = _rescale_integer(acc_imag, mac_shift, rounding=rounding)
    else:
        # Unsupported typed combinations or independently-scaled caches use
        # the generic fp32 fallback. Explicit fp32 preserves device rounding
        # rather than accidentally using Python fp64.
        kc = torch.tensor(x_scale * cos_scale / y_scale, dtype=torch.float32, device=x.device)
        ks = torch.tensor(x_scale * sin_scale / y_scale, dtype=torch.float32, device=x.device)
        if cos_exp == sin_exp:
            real = (qx_first * qc - qx_second * qs).to(torch.float32) * kc
            imag = (qx_first * qs + qx_second * qc).to(torch.float32) * kc
        else:
            real = (qx_first * qc).to(torch.float32) * kc - (qx_second * qs).to(torch.float32) * ks
            imag = (qx_first * qs).to(torch.float32) * ks + (qx_second * qc).to(torch.float32) * kc
        q_real = ppq_tensor_round(real, rounding).to(torch.int64)
        q_imag = ppq_tensor_round(imag, rounding).to(torch.int64)

    qy = torch.cat((_saturate(q_real, y_bits), _saturate(q_imag, y_bits)), dim=-1)
    return (qy.to(dtype=orig_dtype) * float(y_scale)).contiguous()


def espdl_rotary_embedding_forward(
    op: Operation,
    values: List[torch.Tensor],
    ctx=None,
    **kwargs,
) -> torch.Tensor:
    """RoPE executor entry point: P4 integer model when eligible, fp32 otherwise."""
    if _use_device_emulator(op):
        emulated = simulate_espdl_rotary_embedding(op, values)
        if emulated is not None:
            return emulated
    return rotary_embedding_reference(op, values, ctx=ctx, **kwargs)


__all__ = [
    "espdl_rotary_embedding_forward",
    "rotary_embedding_reference",
    "set_espdl_rotary_embedding_simulation_enabled",
    "simulate_espdl_rotary_embedding",
]

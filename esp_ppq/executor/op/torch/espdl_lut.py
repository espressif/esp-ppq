"""Bit-exact ESP-DL INT16 LUT activation emulation.

This module makes esp-ppq's *simulation* of activation functions match, bit for
bit, what the ESP-DL runtime computes on device when the activation is deployed
as an INT16 Look-Up-Table (LUT).

Background
----------
On ESP32-P4 / ESP32-S3 an INT16 activation (Swish/SiLU, Sigmoid, Tanh, ...) is
deployed as a step-sub-sampled LUT (see ``dl/module/include/dl_module_lut.hpp``
in esp-dl). For a quantized input value ``q`` the hardware does::

    idx = q + 32768                      // shift signed -> unsigned
    if   step == 1:            out = table[idx]
    elif step is power-of-two: out = table[round_half_even(idx / step)]   // PIE8 SIMD
    else:                      x = table[idx / step]; y = table[idx/step + 1]
                              out = x + (idx % step) * (y - x) / step     // interp

The stock esp-ppq executor instead evaluates the *ideal* floating point
activation (``torch.sigmoid`` etc.) during quantization / validation. Because
the on-device LUT only stores ``65536 / step + 1`` pivots and reconstructs every
other point by nearest-neighbour / interpolation, the Python simulation and the
device output diverge - the model "cannot be aligned" with hardware.

This module closes that gap: activation ops that are rewritten to the ``LUT``
type (see :class:`esp_ppq.quantization.optim.EspdlLUTFusionPass`) are executed
through :class:`HardwareEmulator`, a pure-integer digital twin of the device
kernel, while table generation during export transparently switches back to the
ideal math so the exported table itself stays accurate.
"""

from enum import Enum
from typing import Callable, List

import torch

from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
from esp_ppq.utils.round import ppq_tensor_round

__all__ = [
    "SimulationMode",
    "GlobalMode",
    "set_simulation_mode",
    "HardwareEmulator",
    "lut_forward_provider",
    "register_lut_op_handler",
    "resolve_ideal_math_fn",
]


class SimulationMode(Enum):
    """Execution mode of the LUT emulator."""

    #: Bit-exact hardware emulation (default). Use for PTQ / QAT / validation.
    SIMULATION = 1
    #: Ideal floating point math. Used internally while the exporter builds the
    #: LUT table so the stored pivots are exact.
    IDEAL_MATH = 2


class GlobalMode:
    """Process-wide switch selecting the current :class:`SimulationMode`."""

    _current_mode = SimulationMode.SIMULATION

    @classmethod
    def set(cls, mode: SimulationMode):
        cls._current_mode = mode

    @classmethod
    def get(cls) -> SimulationMode:
        return cls._current_mode


def set_simulation_mode(mode: SimulationMode):
    """Convenience helper to set the global emulator mode."""
    GlobalMode.set(mode)


def resolve_ideal_math_fn(original_type: str) -> Callable:
    """Return a type-agnostic forward function for ``original_type``.

    The LUT op has ``op.type == "LUT"`` but its underlying math is described by
    ``original_type`` (e.g. ``"Sigmoid"``). Some default forwards
    (``UnaryEltwise_forward``) dispatch on ``op.type``, so we temporarily
    restore the original type around the call.
    """
    if original_type not in DEFAULT_BACKEND_TABLE:
        raise KeyError(
            f"Mathematical ground truth for '{original_type}' is missing from "
            "DEFAULT_BACKEND_TABLE. Register a forward function for it first."
        )
    real_fn = DEFAULT_BACKEND_TABLE[original_type]
    if real_fn is None:
        raise KeyError(f"Ideal math function for '{original_type}' is None.")

    def ideal_math_fn(op, values, **kwargs):
        saved_type = op.type
        op.type = original_type
        try:
            return real_fn(op, values, **kwargs)
        finally:
            op.type = saved_type

    return ideal_math_fn


def _as_scalar(scale) -> float:
    if isinstance(scale, torch.Tensor):
        return float(scale.flatten()[0].item())
    return float(scale)


def _broadcast_scale(scale, input_tensor):
    if isinstance(scale, torch.Tensor) and scale.ndim > 0:
        if input_tensor.ndim == 4:
            return scale.view(1, -1, 1, 1)
    return scale


class HardwareEmulator(torch.autograd.Function):
    """Pure-integer digital twin of the ESP-DL INT16 LUT kernel.

    ``forward`` mirrors ``dl::module::LUT::forward`` for
    ``QUANT_TYPE_SYMM_16BIT``. Depending on ``step`` it reproduces the direct
    lookup (``step == 1``), the PIE8 nearest-neighbour SIMD path (power-of-two
    ``step``) or the scalar linear-interpolation fallback.

    ``backward`` uses the smooth ideal activation gradient so the op stays
    trainable under QAT (straight-through-style estimator).
    """

    # Cache: maps id(op) -> int16 table tensor (host/CPU int32).
    _table_cache = {}

    @staticmethod
    def _build_table(math_fn, op_context, in_scale, out_scale, step, rounding):
        """Build the INT16 LUT exactly as the esp-ppq exporter does.

        Mirrors ``AddLUTPattern.calculate_lut``::

            input = arange(min, max + step, step) * in_scale
            output = forward(op, [input])
            lut = round(output / out_scale).clamp(-32768, 32767).to(int16)
        """
        n_entries = 65536 // step + 1

        table_input_int = torch.arange(0, n_entries, dtype=torch.float32) * step - 32768
        table_input_float = table_input_int * _as_scalar(in_scale)

        table_output_float = math_fn(op_context, [table_input_float])

        table_int16 = ppq_tensor_round(table_output_float / _as_scalar(out_scale), rounding)
        table_int16 = torch.clamp(table_int16, -32768, 32767).to(torch.int32)
        return table_int16.flatten()

    @staticmethod
    def forward(ctx, input_tensor, math_fn, op_context, in_scale, out_scale, step, rounding):
        ctx.math_fn = math_fn
        ctx.op_context = op_context
        ctx.save_for_backward(input_tensor)

        step = int(step)

        # --- Step 1: quantize input to INT16 ---
        in_scale_bc = _broadcast_scale(in_scale, input_tensor)
        # float64 division avoids float32 rounding amplification for tiny scales.
        if isinstance(in_scale_bc, torch.Tensor):
            input_int = ppq_tensor_round(input_tensor.double() / in_scale_bc.double(), rounding)
        else:
            input_int = ppq_tensor_round(input_tensor.double() / float(in_scale_bc), rounding)
        input_int = torch.clamp(input_int, -32768, 32767).to(torch.int32)

        # --- Step 2: build / retrieve the LUT table ---
        cache_key = id(op_context)
        if cache_key not in HardwareEmulator._table_cache:
            HardwareEmulator._table_cache[cache_key] = HardwareEmulator._build_table(
                math_fn, op_context, in_scale, out_scale, step, rounding
            )
        table = HardwareEmulator._table_cache[cache_key].to(input_int.device)

        # --- Step 3: table lookup (mirrors dl_module_lut.hpp) ---
        idx = input_int + 32768  # signed -> unsigned, matches XOR 0x8000
        orig_shape = idx.shape

        is_power_of_two = step > 1 and (step & (step - 1)) == 0

        if step == 1:
            lookup = torch.clamp(idx, 0, table.shape[0] - 1)
            output_int = table[lookup.flatten().long()].view(orig_shape).to(torch.int32)
        elif is_power_of_two:
            # PIE8 SIMD: nearest-neighbour with HALF_EVEN rounding. step is a
            # power of two so idx/step is exact in float32 (no rounding error).
            nn_idx = torch.round(idx.float() / step).to(torch.int32)
            nn_idx = torch.clamp(nn_idx, 0, table.shape[0] - 1)
            output_int = table[nn_idx.flatten().long()].view(orig_shape).to(torch.int32)
        else:
            # Scalar linear-interpolation fallback.
            base_idx = idx // step  # C truncation == floor for idx >= 0
            base_idx = torch.clamp(base_idx, 0, table.shape[0] - 2)
            remainder = (idx % step).to(torch.int32)

            base_flat = base_idx.flatten().long()
            x = table[base_flat].to(torch.int32)
            y = table[base_flat + 1].to(torch.int32)

            numerator = remainder.flatten() * (y - x)
            # C integer division truncates toward zero.
            interp = torch.where(numerator >= 0, numerator // step, -((-numerator) // step))
            output_int = (x + interp).view(orig_shape)

        output_int = torch.clamp(output_int, -32768, 32767)

        # --- Step 4: dequantize back to float for the pipeline ---
        out_scale_bc = _broadcast_scale(out_scale, input_tensor)
        return output_int.float() * out_scale_bc

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        math_fn = ctx.math_fn
        op_context = ctx.op_context

        with torch.enable_grad():
            x = input_tensor.detach().requires_grad_(True)
            y = math_fn(op_context, [x])
            grad = torch.autograd.grad(y.sum(), x)[0]

        return grad_output * grad, None, None, None, None, None, None

    @classmethod
    def clear_cache(cls):
        cls._table_cache.clear()


def lut_forward_provider(op, values, ctx=None, **kwargs) -> torch.Tensor:
    """Executor forward for ``LUT`` ops.

    * :attr:`SimulationMode.IDEAL_MATH` (used during export table generation)
      returns the exact float activation.
    * :attr:`SimulationMode.SIMULATION` (default) returns the bit-exact
      hardware LUT emulation.
    """
    input_tensor = values[0]

    for attr in ("original_op_type", "int16_lut_step"):
        if attr not in op.attributes:
            raise AttributeError(
                f"LUT op '{op.name}' is missing mandatory attribute '{attr}'. "
                "It must be produced by EspdlLUTFusionPass."
            )

    original_type = op.attributes["original_op_type"]
    step = op.attributes["int16_lut_step"]

    ideal_math_fn = resolve_ideal_math_fn(original_type)

    if GlobalMode.get() == SimulationMode.IDEAL_MATH:
        return ideal_math_fn(op, values)

    in_scale = op.input_quant_config[0].scale
    out_scale = op.output_quant_config[0].scale
    rounding = op.input_quant_config[0].rounding

    return HardwareEmulator.apply(input_tensor, ideal_math_fn, op, in_scale, out_scale, step, rounding)


def register_lut_op_handler(verbose: bool = False):
    """Register the ``LUT`` op globally so esp-ppq can execute/export it.

    Idempotent - safe to call multiple times.
    """
    # Imported lazily to keep this module free of a hard dependency on the
    # parser package (avoids import cycles at package initialization time).
    from esp_ppq.parser.espdl.espdl_typedef import ACTIVATION_OP_SET, PASSIVE_LAYOUT_OP_SET

    ACTIVATION_OP_SET.add("LUT")
    PASSIVE_LAYOUT_OP_SET.add("LUT")

    for platform in OPERATION_FORWARD_TABLE:
        OPERATION_FORWARD_TABLE[platform]["LUT"] = lut_forward_provider

    if verbose:
        print("[ESP-PPQ][LUT] Bit-exact LUT operation handler registered globally.")

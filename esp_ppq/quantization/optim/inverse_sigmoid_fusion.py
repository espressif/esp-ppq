"""ESP-DL InverseSigmoid fusion and export validation."""

from __future__ import annotations

import math
import struct
from typing import List, Optional, Tuple

from esp_ppq.core import (
    DataType,
    QuantizationProperty,
    QuantizationStates,
    TargetPlatform,
    convert_any_to_torch_tensor,
)
from esp_ppq.IR import BaseGraph, Operation, QuantableOperation, Variable

from .base import QuantizationOptimizationPass

INVERSE_SIGMOID_OP_TYPE = "InverseSigmoid"
DEFAULT_INVERSE_SIGMOID_EPS = 1e-5


def _as_float32(value) -> Optional[float]:
    try:
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error, TypeError, ValueError):
        return None


def _positive_static_shape(var: Variable, role: str) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    shape = getattr(var, "shape", None)
    if shape is None:
        return None, None
    dims: List[int] = []
    for dim in shape:
        if isinstance(dim, bool) or dim is None:
            return None, f"{role} shape must contain only positive static dimensions, got {list(shape)}"
        try:
            value = int(dim)
        except (TypeError, ValueError):
            return None, f"{role} shape must contain only positive static dimensions, got {list(shape)}"
        if value <= 0 or value != dim:
            return None, f"{role} shape must contain only positive static dimensions, got {list(shape)}"
        dims.append(value)
    return tuple(dims), None


def _inverse_sigmoid_runtime_dtype_from_config(config) -> Tuple[Optional[DataType], Optional[str]]:
    if config is None:
        return None, "missing quantization configuration"
    if getattr(config, "state", None) == QuantizationStates.FP32:
        return DataType.FP32, None

    root = getattr(config, "dominated_by", config)
    state = getattr(config, "state", None)
    if state not in {
        QuantizationStates.ACTIVATED,
        QuantizationStates.PASSIVE,
        QuantizationStates.OVERLAPPED,
        QuantizationStates.BAKED,
        QuantizationStates.PASSIVE_BAKED,
    }:
        return None, f"state {getattr(state, 'name', state)} is not export-ready"
    policy = getattr(config, "policy", None)
    if policy is None or not (
        policy.has_property(QuantizationProperty.LINEAR)
        and policy.has_property(QuantizationProperty.SYMMETRICAL)
        and policy.has_property(QuantizationProperty.PER_TENSOR)
        and policy.has_property(QuantizationProperty.POWER_OF_2)
    ):
        return None, "requires symmetric per-tensor linear POWER_OF_2 quantization"

    bits = int(getattr(config, "num_of_bits", 0))
    exponent_bits = int(getattr(config, "exponent_bits", 0))
    if bits not in (8, 16) or exponent_bits != 0:
        return None, f"{bits}-bit quantization (exponent_bits={exponent_bits}) is unsupported (only INT8/INT16)"
    scale = getattr(root, "scale", None)
    offset = getattr(root, "offset", None)
    if scale is None:
        return None, "requires one calibrated scalar scale"
    try:
        scale_tensor = convert_any_to_torch_tensor(scale, accept_none=False, device="cpu")
    except (RuntimeError, TypeError, ValueError):
        return None, "requires one calibrated scalar scale"
    if scale_tensor.numel() != 1:
        return None, "requires one calibrated scalar scale"
    if offset is None:
        return None, "requires zero symmetric offset"
    try:
        offset_tensor = convert_any_to_torch_tensor(offset, accept_none=False, device="cpu")
    except (RuntimeError, TypeError, ValueError):
        return None, "requires zero symmetric offset"
    if offset_tensor.numel() != 1 or float(offset_tensor.item()) != 0.0:
        return None, "requires zero symmetric offset"
    value = float(scale_tensor.item())
    if not math.isfinite(value) or value <= 0.0:
        return None, f"invalid scale {value!r}"
    mantissa, _ = math.frexp(value)
    if mantissa != 0.5:
        return None, f"scale {value:.9g} is not an exact power of two"
    return (DataType.INT8 if bits == 8 else DataType.INT16), None


def inverse_sigmoid_unsupported_reason(op: Operation) -> Optional[str]:
    if op.type != INVERSE_SIGMOID_OP_TYPE:
        return f"expected {INVERSE_SIGMOID_OP_TYPE}, got {op.type}"
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        return f"expected 1 input and 1 output, got {len(op.inputs)} and {len(op.outputs)}"
    for role, var in (("input", op.inputs[0]), ("output", op.outputs[0])):
        if getattr(var, "dtype", None) != DataType.FP32:
            dtype = getattr(getattr(var, "dtype", None), "name", getattr(var, "dtype", None))
            return f"{role} dtype must be FP32, got {dtype}"
    in_shape, in_shape_reason = _positive_static_shape(op.inputs[0], "input")
    if in_shape_reason is not None:
        return in_shape_reason
    out_shape, out_shape_reason = _positive_static_shape(op.outputs[0], "output")
    if out_shape_reason is not None:
        return out_shape_reason
    if in_shape is not None and out_shape is not None and in_shape != out_shape:
        return f"output shape {list(out_shape)} must match input shape {list(in_shape)}"
    eps = _as_float32(op.attributes.get("eps", DEFAULT_INVERSE_SIGMOID_EPS))
    if eps is None or not math.isfinite(eps) or not 0.0 < eps < 1.0:
        return f"eps must be a finite float32 in (0, 1), got {op.attributes.get('eps')!r}"
    if isinstance(op, QuantableOperation):
        in_cfg = list(getattr(op, "input_quant_config", ()) or ())
        out_cfg = list(getattr(op, "output_quant_config", ()) or ())
        if len(in_cfg) != 1 or len(out_cfg) != 1:
            return (
                f"quantized InverseSigmoid needs 1 input config and 1 output config, got {len(in_cfg)}/{len(out_cfg)}"
            )
        in_dtype, in_reason = _inverse_sigmoid_runtime_dtype_from_config(in_cfg[0])
        if in_reason is not None:
            return f"input quantization is not exportable: {in_reason}"
        out_dtype, out_reason = _inverse_sigmoid_runtime_dtype_from_config(out_cfg[0])
        if out_reason is not None:
            return f"output quantization is not exportable: {out_reason}"
        if in_dtype != out_dtype:
            return (
                f"quantization would export input={in_dtype.name} and output={out_dtype.name}; "
                "InverseSigmoid requires matching input/output storage dtypes"
            )
    return None


def finalize_espdl_inverse_sigmoid_for_export(graph: BaseGraph) -> None:
    for op in graph.operations.values():
        if op.type != INVERSE_SIGMOID_OP_TYPE:
            continue
        reason = inverse_sigmoid_unsupported_reason(op)
        if reason is not None:
            raise ValueError(f"InverseSigmoid {op.name}: {reason}")


def _all_io_fp32(*ops: Operation) -> bool:
    return all(getattr(var, "dtype", None) == DataType.FP32 for op in ops for var in (*op.inputs, *op.outputs))


def _param_value(var: Optional[Variable]):
    """Return the underlying value of a parameter Variable, else None."""
    if var is None or not getattr(var, "is_parameter", False):
        return None
    return getattr(var, "value", None)


def _param_tensor(var: Optional[Variable]):
    value = _param_value(var)
    if value is None:
        return None
    try:
        return convert_any_to_torch_tensor(value, accept_none=False, device="cpu")
    except (RuntimeError, TypeError, ValueError):
        return None


def _param_scalar(var: Optional[Variable]) -> Optional[float]:
    """Return a scalar parameter value without discarding tensor rank."""
    tensor = _param_tensor(var)
    if tensor is None or tensor.ndim != 0 or tensor.numel() != 1:
        return None
    try:
        return float(tensor.item())
    except (RuntimeError, TypeError, ValueError):
        return None


def _clip_input_is_omitted(op: Operation, input_index: int) -> bool:
    return input_index >= len(op.inputs) or not op.inputs[input_index].name


def _decode_clip_bound(
    op: Operation,
    attribute_name: str,
    input_index: int,
) -> Tuple[bool, Optional[float]]:
    """Return ``(resolved, value)`` for one Clip bound.

    ``value is None`` means the bound is proven to be omitted. ``resolved`` is
    false when a bound exists but is dynamic, malformed, or otherwise not a
    compile-time scalar.
    """
    if attribute_name in op.attributes:
        try:
            return True, float(op.attributes[attribute_name])
        except (TypeError, ValueError):
            return False, None

    if _clip_input_is_omitted(op, input_index):
        return True, None

    value = _param_scalar(op.inputs[input_index])
    if value is None:
        return False, None
    return True, value


def _decode_clip(op: Optional[Operation]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Return ``(min, max)`` for a Clip op, or None if op is not Clip.

    Supports both attribute-encoded (opset < 11) and input-encoded
    (opset >= 11) forms. A missing side is represented by ``None`` inside the
    tuple; a supplied but unresolved bound rejects the whole decode.
    """
    if op is None or op.type != "Clip":
        return None
    lo_resolved, lo = _decode_clip_bound(op, "min", 1)
    hi_resolved, hi = _decode_clip_bound(op, "max", 2)
    if not lo_resolved or not hi_resolved:
        return None
    return (lo, hi)


def _static_shape(var: Variable) -> Optional[Tuple[int, ...]]:
    shape = getattr(var, "shape", None)
    if shape is None:
        source = getattr(var, "source_op", None)
        if source is not None and source.type == "Clip" and source.inputs:
            shape = getattr(source.inputs[0], "shape", None)
    if shape is None:
        return None
    static = []
    for dim in shape:
        if dim is None:
            return None
        try:
            value = int(dim)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        static.append(value)
    return tuple(static)


def _one_preserves_x_shape(one_shape: Tuple[int, ...], x_shape: Tuple[int, ...]) -> bool:
    if len(one_shape) > len(x_shape):
        return False
    padded_one = (1,) * (len(x_shape) - len(one_shape)) + one_shape
    return all(one_dim == 1 or one_dim == x_dim for one_dim, x_dim in zip(padded_one, x_shape))


def _is_compatible_one(var: Optional[Variable], x_var: Variable) -> bool:
    tensor = _param_tensor(var)
    if tensor is None or tensor.numel() == 0:
        return False
    try:
        if not bool((tensor == 1.0).all().item()):
            return False
    except (RuntimeError, TypeError, ValueError):
        return False
    if tensor.ndim == 0:
        return True
    x_shape = _static_shape(x_var)
    if x_shape is None:
        return False
    return _one_preserves_x_shape(tuple(int(dim) for dim in tensor.shape), x_shape)


def _inverse_sigmoid_prequant_needs_trace(graph: BaseGraph, do_fusion: bool = True) -> bool:
    if not do_fusion:
        return False
    for sub_op in graph.operations.values():
        if sub_op.type != "Sub" or len(sub_op.inputs) != 2 or len(sub_op.outputs) != 1:
            continue
        one = _param_tensor(sub_op.inputs[0])
        if one is None or one.ndim == 0 or one.numel() == 0 or _static_shape(sub_op.inputs[1]) is not None:
            continue
        try:
            if not bool((one == 1.0).all().item()):
                continue
        except (RuntimeError, TypeError, ValueError):
            continue
        sub_consumers = sub_op.outputs[0].dest_ops
        if len(sub_consumers) != 1 or sub_consumers[0].type != "Clip":
            continue
        clip_out = sub_consumers[0].outputs
        if len(clip_out) != 1 or len(clip_out[0].dest_ops) != 1 or clip_out[0].dest_ops[0].type != "Div":
            continue
        div_out = clip_out[0].dest_ops[0].outputs
        if len(div_out) == 1 and any(consumer.type == "Log" for consumer in div_out[0].dest_ops):
            return True
    return False


def _clip_max_is_noop(hi: Optional[float]) -> bool:
    """True if Clip max cannot change a value already in ``[0, 1]``.

    The fused kernel only floors the numerator / denominator at ``eps``.
    A restrictive inner max (e.g. ``Clip(eps, 0.5)``) is a different op
    and must not be rewritten as ``InverseSigmoid``. ``None`` reaches this
    function only after the decoder has proved the max slot is omitted.
    """
    if hi is None:
        return True
    try:
        h = float(hi)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(h):
        return h > 0.0
    return h >= 1.0


def _exclusive_use(graph: BaseGraph, var: Optional[Variable], consumer: Operation) -> bool:
    """True if ``var`` is consumed only by ``consumer`` and is not a graph output.

    Required before deleting ``consumer``'s producer: a fan-out or a graph
    output on the intermediate would be orphaned by ``_emit_fused``.
    """
    if var is None:
        return False
    if set(getattr(var, "dest_ops", []) or []) != {consumer}:
        return False
    outputs = getattr(graph, "outputs", None) or {}
    if var.name in outputs:
        return False
    return True


def _validate_two_phase_removal(
    graph: BaseGraph,
    operations: Tuple[Operation, ...],
    preserved_outputs: Tuple[Variable, ...] = (),
) -> None:
    if len({op.name for op in operations}) != len(operations):
        raise ValueError("duplicate operation in two-phase removal")
    doomed = set(operations)
    preserved = set(preserved_outputs)

    for op in operations:
        if graph.operations.get(op.name) is not op:
            raise ValueError(f"operation {op.name} is not attached to the graph")
        for output in op.outputs:
            if output in preserved:
                continue
            if output.name in graph.outputs:
                raise ValueError(f"cannot remove graph output {output.name}")
            if any(consumer not in doomed for consumer in output.dest_ops):
                raise ValueError(f"cannot remove {output.name}: it has an external consumer")


def _remove_operations_two_phase(
    graph: BaseGraph,
    operations: Tuple[Operation, ...],
    preserved_outputs: Tuple[Variable, ...] = (),
) -> None:
    _validate_two_phase_removal(graph, operations, preserved_outputs)
    preserved = set(preserved_outputs)
    candidates = set()
    for op in operations:
        candidates.update(op.inputs)
        candidates.update(op.outputs)

    for op in operations:
        for input_var in op.inputs:
            if op in input_var.dest_ops:
                input_var.dest_ops.remove(op)
        for output_var in op.outputs:
            if output_var not in preserved and output_var.source_op is op:
                output_var.source_op = None
        op.inputs.clear()
        op.outputs.clear()

    for op in operations:
        graph.remove_operation(op, keep_coherence=False, remove_unlinked_variable=False)

    for var in candidates:
        if (
            var.name in graph.variables
            and var.source_op is None
            and not var.dest_ops
            and var.name not in graph.inputs
            and var.name not in graph.outputs
        ):
            graph.remove_variable(var)


class InverseSigmoidFusionPass(QuantizationOptimizationPass):
    """Fuse the DETR inverse-sigmoid pattern into ``InverseSigmoid``."""

    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="ESPDL InverseSigmoid Fusion Pass")
        self.verbose = verbose
        self.fused_count = 0

    def _resolve_fused_platform(
        self,
        doomed: Tuple[Operation, ...],
        log_op: Operation,
    ) -> Optional[TargetPlatform]:
        if all(op.platform == log_op.platform for op in doomed):
            return log_op.platform
        if log_op.platform != TargetPlatform.FP32:
            return None
        non_log_ops = tuple(op for op in doomed if op is not log_op)
        if not non_log_ops:
            return None
        platform = non_log_ops[0].platform
        if any(op.platform != platform for op in non_log_ops):
            return None
        return platform

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        # Snapshot: we mutate graph.operations during iteration.
        for op in list(graph.operations.values()):
            if op.name not in graph.operations:
                continue
            if op.type != "Log" or len(op.inputs) != 1:
                continue
            self._try_fuse(graph, op)
        if self.verbose:
            print(f"[esp-inverse-sigmoid-fusion] fused {self.fused_count} pattern(s)")

    # ------------------------------------------------------------------
    def _try_fuse(self, graph: BaseGraph, log_op: Operation) -> None:
        div_op = log_op.inputs[0].source_op
        if div_op is None or div_op.type != "Div" or len(div_op.inputs) != 2:
            return

        num_clip = div_op.inputs[0].source_op
        denom_clip = div_op.inputs[1].source_op
        if num_clip is None or denom_clip is None:
            return
        if num_clip.type != "Clip" or denom_clip.type != "Clip":
            return

        if len(log_op.outputs) != 1:
            return

        num_range = _decode_clip(num_clip)
        denom_range = _decode_clip(denom_clip)
        if num_range is None or denom_range is None:
            return
        num_lo, num_hi = num_range
        denom_lo, denom_hi = denom_range
        if num_lo is None or denom_lo is None:
            return
        if num_lo != denom_lo:
            return
        eps = float(num_lo)
        if not 0.0 < eps < 1.0:
            return
        # Inner clips must be floor-only. A finite max below 1 would change
        # values the fused kernel leaves untouched.
        if not _clip_max_is_noop(num_hi) or not _clip_max_is_noop(denom_hi):
            return

        # Numerator branch: Clip(min=eps)(xc)
        if not num_clip.inputs:
            return
        xc_var = num_clip.inputs[0]

        # Denominator branch: Clip(min=eps)(Sub(one_const, xc))
        if not denom_clip.inputs:
            return
        sub_op = denom_clip.inputs[0].source_op
        if sub_op is None or sub_op.type != "Sub" or len(sub_op.inputs) != 2:
            return
        if not _is_compatible_one(sub_op.inputs[0], xc_var):
            return
        if sub_op.inputs[1] is not xc_var:
            return

        # Deleting Div / inner Clips / Sub requires each intermediate edge to
        # be exclusive to this pattern and not exposed as a graph output.
        if not _exclusive_use(graph, log_op.inputs[0], log_op):
            return
        if not _exclusive_use(graph, div_op.inputs[0], div_op):
            return
        if not _exclusive_use(graph, div_op.inputs[1], div_op):
            return
        if not _exclusive_use(graph, denom_clip.inputs[0], denom_clip):
            return

        # The outer Clip(0, 1) is required because InverseSigmoid clamps its
        # input to [0, 1] internally.
        outer_candidate = xc_var.source_op
        if outer_candidate is None or outer_candidate.type != "Clip":
            return
        outer_range = _decode_clip(outer_candidate)
        if outer_range is None or outer_range[0] is None or outer_range[1] is None:
            return
        if outer_range != (0.0, 1.0):
            return
        if not _all_io_fp32(log_op, div_op, num_clip, denom_clip, sub_op, outer_candidate):
            return

        # Fold the outer clip only when the two inner branches are its sole consumers.
        outer_clip: Optional[Operation] = None
        x_var = xc_var
        xc_is_graph_output = xc_var.name in (getattr(graph, "outputs", None) or {})
        if set(xc_var.dest_ops) == {num_clip, sub_op} and not xc_is_graph_output:
            outer_clip = outer_candidate
            x_var = outer_candidate.inputs[0]

        self._emit_fused(
            graph,
            log_op=log_op,
            div_op=div_op,
            num_clip=num_clip,
            denom_clip=denom_clip,
            sub_op=sub_op,
            outer_clip=outer_clip,
            x_var=x_var,
            eps=eps,
        )

    # ------------------------------------------------------------------
    def _emit_fused(
        self,
        graph: BaseGraph,
        log_op: Operation,
        div_op: Operation,
        num_clip: Operation,
        denom_clip: Operation,
        sub_op: Operation,
        outer_clip: Optional[Operation],
        x_var: Variable,
        eps: float,
    ) -> None:
        y_var = log_op.outputs[0]
        to_delete: List[Operation] = [log_op, div_op, num_clip, denom_clip, sub_op]
        if outer_clip is not None:
            to_delete.append(outer_clip)
        doomed = tuple(to_delete)
        _validate_two_phase_removal(graph, doomed, (y_var,))
        fused_platform = self._resolve_fused_platform(doomed, log_op)
        if fused_platform is None:
            return

        fused_name = f"InverseSigmoid_{log_op.name}"
        if fused_name in graph.operations:
            return
        fused = graph.create_operation(
            op_type=INVERSE_SIGMOID_OP_TYPE,
            name=fused_name,
            attributes={"eps": float(eps)},
            inputs=[x_var],
            outputs=[y_var],
            platform=fused_platform,
        )
        # Re-parent the surviving output variable to the new op.
        y_var.source_op = fused

        _remove_operations_two_phase(
            graph,
            operations=doomed,
            preserved_outputs=(y_var,),
        )

        self.fused_count += 1
        if self.verbose:
            print(f"[esp-inverse-sigmoid-fusion] fused @ {log_op.name} -> {fused.name} (x={x_var.name}, eps={eps})")

"""ESP-DL RotaryEmbedding fusion, legalization, and export validation."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from esp_ppq.core import (
    DataType,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
)
from esp_ppq.IR import BaseGraph, Operation, Variable
from esp_ppq.IR.quantize import QuantableOperation

from .base import QuantizationOptimizationPass

ROTARY_EMBEDDING = "RotaryEmbedding"
# The esp-dl RotaryEmbedding module has typed load/store paths only for these
# dtypes. Keep this as the single host-side contract; the C++ module validates
# the same set before dispatching its scalar or SIMD implementation.
ROPE_SUPPORTED_DTYPES = frozenset({DataType.FP32, DataType.INT8, DataType.INT16})
ROPE_P4_PLATFORMS = frozenset(
    {
        TargetPlatform.ESPDL_INT8,
        TargetPlatform.ESPDL_INT16,
        TargetPlatform.ESPDL_H_PRE_INT16,
        TargetPlatform.ESPDL_W8A16,
    }
)
ROPE_HALF_UP_PLATFORMS = frozenset(
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


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _static_shape(var: Optional[Variable]) -> Optional[List[int]]:
    """Return a concrete positive shape, or ``None`` when metadata is absent/dynamic.

    RotaryEmbedding admission is shape-dependent: rank decides the ONNX layout
    and X/cache dimensions decide whether the device can index the tensor.  A
    symbolic or absent dimension is therefore *not* safe to treat as an
    unknown-but-supported value.  Parameter values are a fallback because
    parser versions do not always copy their metadata into ``Variable.shape``.
    """
    if var is None:
        return None
    shape = getattr(var, "shape", None)
    if shape is None:
        value = _param_value(var)
        shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        raw_dims = list(shape)
    except TypeError:
        return None
    dims: List[int] = []
    for dim in raw_dims:
        # ``bool`` is an int subclass but never a valid tensor dimension.
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
            return None
        dim = int(dim)
        if dim <= 0:
            return None
        dims.append(dim)
    return dims


def _var_dtype(var: Optional[Variable]) -> Optional[DataType]:
    """Return PPQ dtype without allowing missing/unsupported values to escape."""
    if var is None:
        return None
    try:
        dtype = getattr(var, "dtype", None)
    except (TypeError, ValueError, OverflowError):
        return None
    return dtype if isinstance(dtype, DataType) else None


def _dtype_name(dtype: Optional[DataType]) -> str:
    return dtype.name if isinstance(dtype, DataType) else "unknown"


def _supported_runtime_dtype(dtype: Optional[DataType]) -> bool:
    return dtype in ROPE_SUPPORTED_DTYPES


def _param_value(var: Optional[Variable]):
    """Return the underlying value of a parameter or Constant-backed Variable, else None.

    ``Variable.value`` may be a torch tensor or numpy array depending on
    parser; both share ``.tolist()`` for scalar/small-tensor extraction.
    Also falls back to ``source_op.attributes['value']`` when ``var`` is the
    output of an unfolded ``Constant`` node.
    """
    if var is None:
        return None
    if getattr(var, "is_parameter", False) and getattr(var, "value", None) is not None:
        return var.value
    src = getattr(var, "source_op", None)
    if src is not None and src.type == "Constant":
        return src.attributes.get("value")
    return None


def _to_int_list(val) -> Optional[List[int]]:
    if val is None:
        return None
    try:
        if hasattr(val, "tolist"):
            v = val.tolist()
        else:
            v = list(val)
        if isinstance(v, (int, float)):
            return [int(v)]
        return [int(x) for x in v]
    except Exception:
        return None


def _decode_slice(op: Operation) -> Optional[dict]:
    """Return ``{'starts', 'ends', 'axes', 'steps'}`` for a Slice op.

    Supports both attribute-encoded (opset < 10) and input-encoded (opset >= 10)
    forms. Values not present in the op default to the ONNX-standard defaults
    (axes = list(range(len(starts))), steps = 1s).

    Returns None if any required piece is missing or non-constant.
    """
    if op.type != "Slice":
        return None
    starts_v = op.attributes.get("starts")
    ends_v = op.attributes.get("ends")
    axes_v = op.attributes.get("axes")
    steps_v = op.attributes.get("steps")

    if starts_v is None and len(op.inputs) >= 3:
        starts_v = _to_int_list(_param_value(op.inputs[1]))
        ends_v = _to_int_list(_param_value(op.inputs[2]))
        if len(op.inputs) >= 4:
            axes_v = _to_int_list(_param_value(op.inputs[3]))
        if len(op.inputs) >= 5:
            steps_v = _to_int_list(_param_value(op.inputs[4]))

    starts_v = _to_int_list(starts_v)
    ends_v = _to_int_list(ends_v)
    if starts_v is None or ends_v is None or len(starts_v) != len(ends_v):
        return None

    if axes_v is None:
        axes_v = list(range(len(starts_v)))
    else:
        axes_v = _to_int_list(axes_v)
        if axes_v is None or len(axes_v) != len(starts_v):
            return None
    if steps_v is None:
        steps_v = [1] * len(starts_v)
    else:
        steps_v = _to_int_list(steps_v)
        if steps_v is None or len(steps_v) != len(starts_v):
            return None

    return {"starts": starts_v, "ends": ends_v, "axes": axes_v, "steps": steps_v}


def _slice_range_on_axis(sl: dict, axis: int, rank: int) -> Optional[Tuple[int, int, int]]:
    """Return ``(start, end, step)`` on the given axis, else None.

    RoPE matching deliberately accepts only a one-axis, unit-step Slice.
    A Slice that also touches batch/head/token axes may happen to have the
    expected last-dimension bounds for one sample, but deleting it would
    change the graph. Negative axes are normalized against the traced rank;
    never infer ``-1`` from an arbitrary positive axis.
    """
    if rank < 1 or len(sl["axes"]) != 1 or len(sl["starts"]) != 1 or len(sl["ends"]) != 1:
        return None
    if len(sl["steps"]) != 1 or int(sl["steps"][0]) != 1:
        return None
    target = axis + rank if axis < 0 else axis
    sliced = int(sl["axes"][0])
    sliced = sliced + rank if sliced < 0 else sliced
    if not (0 <= target < rank) or sliced != target:
        return None
    return int(sl["starts"][0]), int(sl["ends"][0]), 1


def _split_half_bounds_on_last_axis(
    split_op: Operation,
    x_shape: List[int],
) -> Optional[Tuple[int, int]]:
    """Return ``(split, rot_end)`` when ``split_op`` halves the last axis of ``X``."""
    if split_op.type != "Split" or len(split_op.outputs) != 2 or not (1 <= len(split_op.inputs) <= 2):
        return None
    x_rank = len(x_shape)
    if x_rank < 1:
        return None
    try:
        axis = int(split_op.attributes.get("axis", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    normalized_axis = axis + x_rank if axis < 0 else axis
    if normalized_axis != x_rank - 1:
        return None

    num_outputs = split_op.attributes.get("num_outputs")
    if num_outputs is not None:
        try:
            if int(num_outputs) != 2:
                return None
        except (TypeError, ValueError, OverflowError):
            return None

    dim = int(x_shape[-1])
    if dim < 2 or dim % 2 != 0:
        return None
    half = dim // 2

    split_sizes = split_op.attributes.get("split")
    if split_sizes is not None:
        sizes = _to_int_list(split_sizes)
        if sizes != [half, half]:
            return None
    if len(split_op.inputs) == 2:
        sizes = _to_int_list(_param_value(split_op.inputs[1]))
        if sizes != [half, half]:
            return None

    return half, dim


def halve_rope_cache(
    value,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """Fold a full-width ``cat((f, f), -1)`` RoPE table into a half-width ONNX cache.

    Accepts rank-2 ``[L, D]``, rank-3 ``[1, L, D]``, and rank-4 ``[B|1, 1, L, D]``
    tables whose two halves along the last axis are bit-identical.

    Returns:
        Tuple[Optional[Union[np.ndarray, torch.Tensor]], Optional[str]]:
            ``(cache, None)`` with shape ``[B|1, L, D/2]`` on success, or
            ``(None, reason)`` on failure.
    """
    if value is None:
        return None, "table has no value"
    was_numpy = isinstance(value, np.ndarray)
    try:
        table = torch.from_numpy(value) if was_numpy else value
        if not isinstance(table, torch.Tensor):
            return None, f"unsupported table container {type(value).__name__}"
    except Exception as exc:
        return None, f"cannot read table ({exc})"

    if table.dim() not in (2, 3, 4):
        return None, f"table rank {table.dim()} is not one of the broadcastable ranks 2, 3, or 4"
    width = int(table.shape[-1])
    if width < 2 or width % 2 != 0:
        return None, f"table width {width} is not a positive even number"
    half = width // 2

    cache_batch = 1
    if table.dim() == 3:
        # [1, L, D] aligns to X's head axis under ONNX broadcasting.
        if int(table.shape[0]) != 1:
            return None, (
                f"rank-3 table leading dim {int(table.shape[0])} is a per-head axis, "
                "which cannot be expressed as an ONNX cache"
            )
    elif table.dim() == 4:
        table_batch, table_heads = (int(table.shape[0]), int(table.shape[1]))
        if table_heads != 1:
            return None, (
                f"rank-4 table head axis is {table_heads}; per-head caches cannot be expressed as an ONNX cache"
            )
        if table_batch != 1:
            if batch_size is None:
                return None, (f"rank-4 table batch axis is {table_batch}, but X batch is unknown")
            if table_batch != int(batch_size):
                return None, (f"rank-4 table batch axis {table_batch} does not match X batch {batch_size}")
        cache_batch = table_batch

    folded = table[..., :half]
    if not torch.equal(folded, table[..., half:]):
        return None, "table halves are not bit-identical, so a half-width cache would change results"

    seq_t = int(table.shape[-2])
    cache = folded.reshape(cache_batch, seq_t, half)
    if seq_t == 1:
        if seq_len is None or seq_len < 1:
            return None, "table covers a single token and X's sequence length is unknown"
        if seq_len != 1:
            cache = cache.expand(cache_batch, seq_len, half)
    cache = cache.contiguous()
    return (cache.numpy() if was_numpy else cache), None


def _identity_rope_legalization_spec(op: Operation) -> Optional[Tuple[Variable, List[Variable]]]:
    """Return rewrite inputs when a native RoPE is provably an identity.

    Matches non-quantized 3-input 4D RoPE ops with valid output/cache/attribute
    contracts where every ``cos`` entry is 1 and every ``sin`` entry is 0.
    """
    if isinstance(op, QuantableOperation) or len(op.inputs) != 3 or len(op.outputs) != 1:
        return None
    x_var, cos_var, sin_var = op.inputs
    y_var = op.outputs[0]
    # Keep graph cleanup local and deterministic: initializers are safe to
    # detach, while an unfolded Constant may still need its producing node
    # pruned as a separate graph rewrite.
    if cos_var.source_op is not None or sin_var.source_op is not None:
        return None
    x_shape = _static_shape(x_var)
    y_shape = _static_shape(y_var)
    cos_shape = _static_shape(cos_var)
    sin_shape = _static_shape(sin_var)
    if x_shape is None or len(x_shape) != 4 or y_shape != x_shape or cos_shape is None or cos_shape != sin_shape:
        return None
    x_dtype = _var_dtype(x_var)
    cos_dtype = _var_dtype(cos_var)
    sin_dtype = _var_dtype(sin_var)
    y_dtype = _var_dtype(y_var)
    if any(not _supported_runtime_dtype(dtype) for dtype in (x_dtype, cos_dtype, sin_dtype, y_dtype)):
        return None
    if x_dtype != y_dtype or cos_dtype != sin_dtype:
        return None
    B, H, L, D = x_shape
    if D % 2 != 0:
        return None
    try:
        num_heads = int(op.attributes.get("num_heads", 0) or 0)
        rdim = int(op.attributes.get("rotary_embedding_dim", 0) or 0)
        interleaved = int(op.attributes.get("interleaved", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if interleaved not in (0, 1):
        return None
    if num_heads and num_heads != H:
        return None
    if rdim < 0 or rdim % 2 != 0 or rdim > D:
        return None
    rotated = rdim or D
    expected_half = rotated // 2
    if len(cos_shape) == 2:
        cache_valid = cos_shape == [L, expected_half]
    elif len(cos_shape) == 3:
        cache_valid = cos_shape[0] in (1, B) and cos_shape[1:] == [L, expected_half]
    else:
        cache_valid = False
    if not cache_valid:
        return None

    cos_value, sin_value = _param_value(cos_var), _param_value(sin_var)
    if cos_value is None or sin_value is None:
        return None
    try:
        cos = cos_value if isinstance(cos_value, torch.Tensor) else torch.from_numpy(np.asarray(cos_value))
        sin = sin_value if isinstance(sin_value, torch.Tensor) else torch.from_numpy(np.asarray(sin_value))
    except Exception:
        return None
    if tuple(cos.shape) != tuple(cos_shape) or tuple(sin.shape) != tuple(sin_shape):
        return None
    if not bool(torch.all(cos == 1).item()) or not bool(torch.all(sin == 0).item()):
        return None
    return x_var, [cos_var, sin_var]


def legalize_espdl_rotary_embedding_graph(graph: BaseGraph, verbose: bool = False) -> int:
    """Rewrite statically provable identity native RoPE nodes into ``Identity``.

    Unsupported dynamic layouts remain explicit validation errors. Silently
    decomposing position-id gathers or M-RoPE would risk a seemingly successful
    export rotating the wrong token. The identity reduction is exact,
    target-independent and removes the cache parameters, so it is a safe
    graceful downgrade for direct native ONNX exports.
    """
    legalized = 0
    for op in list(graph.operations.values()):
        if op.type != ROTARY_EMBEDDING or op.name not in graph.operations:
            continue
        spec = _identity_rope_legalization_spec(op)
        if spec is None:
            continue
        x_var, cache_vars = spec
        for cache_var in cache_vars:
            if op in cache_var.dest_ops:
                cache_var.dest_ops.remove(op)
        op.inputs[:] = [x_var]
        op.type = "Identity"
        op.attributes.clear()
        for cache_var in cache_vars:
            if (
                cache_var.name in graph.variables
                and cache_var.source_op is None
                and not cache_var.dest_ops
                and cache_var.name not in graph.inputs
                and cache_var.name not in graph.outputs
            ):
                graph.remove_variable(cache_var)
        legalized += 1
        if verbose:
            print(f"[rotary-embedding-legalize] {op.name}: identity cos/sin -> Identity")
    return legalized


def _negated_table(cache):
    """Return a negated copy of the cache in its original container type.

    Used when the graph uses ``Concat([x2, -x1], -1)`` instead of the canonical
    ``Concat([-x2, x1], -1)``.
    """
    if isinstance(cache, np.ndarray):
        return np.negative(cache).copy()
    return torch.neg(cache).contiguous()


class RotaryEmbeddingFusionPass(QuantizationOptimizationPass):
    """Fuse ``X * cos + rotate_half(X) * sin`` into ``RotaryEmbedding``.

    Args:
        verbose (bool): Whether to print per-node skip reasons and fusion summary.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(name="ESPDL RotaryEmbedding Fusion Pass")
        self.verbose = verbose
        self.fused_count = 0
        self.fused_swapped_count = 0

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        if _rope_fusion_needs_trace(graph):
            _ensure_traced_shapes(graph, **kwargs)

        for add_op in list(graph.operations.values()):
            if add_op.name not in graph.operations:
                continue
            if add_op.type != "Add" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            self._try_fuse(graph, add_op)
        if self.verbose:
            print(
                f"[rotary-embedding-fusion] fused {self.fused_count} RoPE pattern(s) "
                f"(swapped={self.fused_swapped_count})"
            )

    def _skip(self, add_op: Operation, reason: str) -> None:
        if self.verbose:
            print(f"[rotary-embedding-fusion] skip {add_op.name}: {reason}")

    # ------------------------------------------------------------------
    def _try_fuse(self, graph: BaseGraph, add_op: Operation) -> None:
        for cos_side, sin_side in ((0, 1), (1, 0)):
            mul_cos = add_op.inputs[cos_side].source_op
            mul_sin = add_op.inputs[sin_side].source_op
            if mul_cos is None or mul_sin is None:
                continue
            if mul_cos.type != "Mul" or mul_sin.type != "Mul":
                continue

            cos_pair = self._parse_mul_with_param(mul_cos)
            sin_pair = self._parse_mul_with_param(mul_sin)
            if cos_pair is None or sin_pair is None:
                continue
            x_direct, cos_var = cos_pair
            rotated_var, sin_var = sin_pair

            # rotate_half matcher (canonical + negated-sin). The GPT-J
            # interleaved layout is out of scope for the vision subset; such
            # a graph simply stays unfused and check_rotary_embedding will
            # reject any hand-authored interleaved=1 node.
            parsed = self._parse_rotate_half(rotated_var.source_op, add_op=add_op)
            if parsed.get("ok") and parsed["x"] is x_direct:
                # _fuse_non_interleaved logs the reason when it declines.
                self._fuse_non_interleaved(graph, add_op, mul_cos, mul_sin, x_direct, cos_var, sin_var, parsed)
                return

    # ------------------------------------------------------------------
    def _parse_mul_with_param(self, mul_op: Operation):
        """Return ``(activation_var, parameter_var)`` for ``Mul`` iff exactly
        one operand has a static parameter/Constant value. Returns None otherwise."""
        if mul_op.type != "Mul" or len(mul_op.inputs) != 2:
            return None
        a, b = mul_op.inputs[0], mul_op.inputs[1]
        a_is_param = _param_value(a) is not None
        b_is_param = _param_value(b) is not None
        if a_is_param and not b_is_param:
            return b, a
        if b_is_param and not a_is_param:
            return a, b
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_x_seq_len(x_var: Variable, x_rank: Optional[int]) -> Optional[int]:
        """Best-effort read of X's sequence-length dimension.

        This helper is intentionally conservative. Layouts follow the ONNX
        RotaryEmbedding spec:
            * 4D X = [B, H, L, D]  -> sequence axis 2
            * 3D X = [B, L, H*D]   -> sequence axis 1
        """
        shape = _static_shape(x_var)
        if shape is None:
            return None
        if x_rank is None:
            x_rank = len(shape)
        if x_rank == 4 and len(shape) >= 3:
            candidate = shape[2]
        elif x_rank == 3 and len(shape) >= 2:
            candidate = shape[1]
        else:
            return None
        try:
            v = int(candidate)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    # ------------------------------------------------------------------
    def _parse_rotate_half(
        self,
        concat_op: Optional[Operation],
        add_op: Optional[Operation] = None,
    ) -> dict:
        """Match the non-interleaved rotate_half pattern.

        ``rotate_half`` maps ``(x1, x2) -> (∓x2, ±x1)``: concat slot 0 always
        carries the UPPER slice ``[split, rot_end)`` and slot 1 the LOWER
        slice ``[0, split)``. Only which slot carries the ``Neg`` varies, and
        that decides the sin sign::

            Concat([-x2,  x1], -1)  ->  RoPE with +sin  (canonical HF-LLaMA)
            Concat([ x2, -x1], -1)  ->  RoPE with -sin  (negate the cache)

        The mirrored layouts ``Concat([x1, -x2])`` / ``Concat([-x1, x2])``
        keep each half aligned with its own cache entry, producing
        ``(x1*(c±s), x2*(c∓s))`` — an elementwise scaling, not a rotation.
        Those must NOT match: no sin-sign flip turns them into RoPE, so
        fusing them would silently change the model's numerics.

        Whether ``split`` is the head midpoint or a partial-RoPE boundary is
        left to the caller; this parser only enforces adjacency.

        Returns dict with keys ``ok``, ``x``, ``x_rank``, ``split``,
        ``rot_end``, ``swapped``. On failure only ``ok=False``.
        """
        fail = {"ok": False}
        if concat_op is None or concat_op.type != "Concat":
            return fail
        if len(concat_op.inputs) != 2:
            return fail
        axis = concat_op.attributes.get("axis", None)
        if axis is None:
            return fail
        try:
            axis = int(axis)
        except (TypeError, ValueError, OverflowError):
            return fail

        src0 = concat_op.inputs[0].source_op
        src1 = concat_op.inputs[1].source_op
        if src0 is None or src1 is None:
            return fail

        # Exactly one slot must go through a Neg.
        neg0 = src0.type == "Neg"
        neg1 = src1.type == "Neg"
        if neg0 == neg1:
            return fail
        swapped = neg1

        def _leaf_var_and_op(var: Variable) -> Tuple[Optional[Variable], Optional[Operation]]:
            """Resolve a concat slot variable and its producer, stepping over a Neg."""
            op = var.source_op
            if op is not None and op.type == "Neg":
                if not op.inputs:
                    return None, None
                var = op.inputs[0]
                op = var.source_op
            if op is None or op.type not in ("Slice", "Split") or not op.inputs:
                return None, None
            return var, op

        var_hi, op_hi = _leaf_var_and_op(concat_op.inputs[0])
        var_lo, op_lo = _leaf_var_and_op(concat_op.inputs[1])
        if op_hi is None or op_lo is None:
            return fail
        if op_hi.inputs[0] is not op_lo.inputs[0]:
            return fail
        x_var = op_lo.inputs[0]
        x_shape = _static_shape(x_var)
        if x_shape is None:
            return fail
        x_rank = len(x_shape)
        normalized_axis = axis + x_rank if axis < 0 else axis
        # The ESP-DL node rotates the last dimension. Resolve the axis from
        # traced rank; never equate axis=-1 with an arbitrary positive axis.
        if normalized_axis != x_rank - 1:
            return fail

        if op_hi.type == "Split" or op_lo.type == "Split":
            if op_hi is not op_lo:
                return fail
            if len(op_hi.outputs) != 2 or var_lo is not op_hi.outputs[0] or var_hi is not op_hi.outputs[1]:
                return fail
            bounds = _split_half_bounds_on_last_axis(op_hi, x_shape)
            if bounds is None:
                return fail
            split, rot_end = bounds
        else:
            info_hi = _decode_slice(op_hi)
            info_lo = _decode_slice(op_lo)
            if info_hi is None or info_lo is None:
                return fail
            r_hi = _slice_range_on_axis(info_hi, normalized_axis, x_rank)
            r_lo = _slice_range_on_axis(info_lo, normalized_axis, x_rank)
            if r_hi is None or r_lo is None:
                return fail

            # Slot 1 is the lower half [0, split); slot 0 the upper [split, end).
            if r_lo[0] != 0:
                return fail
            split = r_lo[1]
            if r_hi[0] != split:
                return fail
            rot_end = r_hi[1]

        return {
            "ok": True,
            "x": x_var,
            "x_rank": x_rank,
            "split": split,
            "rot_end": rot_end,
            "swapped": swapped,
        }

    # ------------------------------------------------------------------
    def _fuse_non_interleaved(
        self,
        graph: BaseGraph,
        add_op: Operation,
        mul_cos: Operation,
        mul_sin: Operation,
        x_direct: Variable,
        cos_var: Variable,
        sin_var: Variable,
        parsed: dict,
    ) -> bool:
        x_rank = parsed["x_rank"]
        split = parsed["split"]
        raw_rot_end = parsed["rot_end"]
        swapped = parsed["swapped"]

        # Never infer the layout from Slice axes alone. A negative ``axis``
        # carries no rank information, and accepting it without the traced X
        # metadata can turn a 3D or partial graph into a device node that only
        # fails at first inference. The normal quantize / FP32-export paths
        # trace before this pass; direct callers without a sample simply keep
        # the primitive graph.
        x_shape = _static_shape(x_direct)
        if x_shape is None:
            self._skip(
                add_op,
                "X shape is missing or dynamic; trace with executor + inputs before RoPE fusion",
            )
            return False
        if len(x_shape) != 4:
            self._skip(
                add_op,
                f"X is {len(x_shape)}D; the ESP-DL vision subset only supports 4D [B,H,L,D]",
            )
            return False

        # ONNX Slice clips a positive ``end`` to the selected dimension.
        # PyTorch exports ``x[..., split:]`` with INT64_MAX rather than the
        # concrete head width, so normalize the decoded upper bound only
        # after tracing has made that width authoritative.  This also accepts
        # any other explicit overrun with the same ONNX semantics, while
        # preserving ordinary partial ranges below ``head_dim D``.
        head_dim = x_shape[3]
        rot_end = min(raw_rot_end, head_dim) if raw_rot_end >= 0 else raw_rot_end
        y_shape = _static_shape(add_op.outputs[0])
        if y_shape is not None and y_shape != x_shape:
            self._skip(
                add_op,
                f"Add output shape {tuple(y_shape)} != X shape {tuple(x_shape)}; "
                "fusion cannot remove cache broadcasting axes",
            )
            return False

        # Both halves of the rotated span [0, split) and [split, rot_end) must have equal width.
        if rot_end != 2 * split:
            self._skip(
                add_op,
                f"rotate_half slices are unbalanced: [0,{split}) and [{split},{rot_end}); "
                "widths differ, not a valid rotate_half",
            )
            return False

        # Compute the cache half — must match the rotated half.
        seq_len_hint = self._infer_x_seq_len(x_direct, x_rank)
        batch_size = x_shape[0]
        cos_cache, why = halve_rope_cache(
            _param_value(cos_var),
            seq_len=seq_len_hint,
            batch_size=batch_size,
        )
        if cos_cache is None:
            self._skip(add_op, f"cos table: {why}")
            return False
        sin_cache, why = halve_rope_cache(
            _param_value(sin_var),
            seq_len=seq_len_hint,
            batch_size=batch_size,
        )
        if sin_cache is None:
            self._skip(add_op, f"sin table: {why}")
            return False
        if cos_cache.shape != sin_cache.shape:
            self._skip(add_op, f"cos cache {tuple(cos_cache.shape)} != sin cache {tuple(sin_cache.shape)}")
            return False

        cache_half = int(cos_cache.shape[-1])
        if cache_half != split:
            self._skip(
                add_op,
                f"cache half {cache_half} does not match rotated half {split}; "
                "cos/sin tables and the graph split disagree on the rotated width",
            )
            return False

        # Swap variant: build a negated sin cache, do not touch the source.
        if swapped:
            sin_cache = _negated_table(sin_cache)

        # Everything below is out of the vision subset's scope. Decline here
        # rather than emit a node that check_rotary_embedding would then turn
        # into a hard error: an unfused primitive subgraph is still correct,
        # a rejected fused node aborts the whole quantization run.
        if rot_end != head_dim:
            self._skip(
                add_op,
                f"only [0,{rot_end}) of head_dim D={head_dim} is rotated; partial "
                "rotation is not supported by the ESP-DL vision subset",
            )
            return False
        L = x_shape[2]
        cache_L = int(cos_cache.shape[-2])
        if cache_L != L:
            self._skip(
                add_op,
                f"cache sequence length L={cache_L} != X's L={L}",
            )
            return False

        # Every intermediate op in the 7-op chain must have no external consumers
        # and must not be a graph output; detaching a shared intermediate would
        # break the sibling consumer.
        concat_op = None
        for var in mul_sin.inputs:
            if var.source_op is not None and var.source_op.type == "Concat":
                concat_op = var.source_op
                break
        if concat_op is None:
            return False

        interior_ops: List[Operation] = [mul_cos, mul_sin, concat_op]
        for var in concat_op.inputs:
            src = var.source_op
            if src is None:
                return False
            if src.type == "Neg":
                if src not in interior_ops:
                    interior_ops.append(src)
                if not src.inputs or src.inputs[0].source_op is None:
                    return False
                leaf = src.inputs[0].source_op
                if leaf.type not in ("Slice", "Split"):
                    return False
                if leaf not in interior_ops:
                    interior_ops.append(leaf)
            elif src.type in ("Slice", "Split"):
                if src not in interior_ops:
                    interior_ops.append(src)
            else:
                return False

        chain_set = set(interior_ops) | {add_op}
        for iop in interior_ops:
            for ov in iop.outputs:
                if ov.name in graph.outputs:
                    self._skip(add_op, f"intermediate tensor {ov.name} is a graph output")
                    return False
                if any(dest not in chain_set for dest in ov.dest_ops):
                    self._skip(add_op, f"intermediate op {iop.name} has external consumer(s)")
                    return False

        if any(op.platform != add_op.platform for op in chain_set):
            self._skip(add_op, "matched runtime operators use different platforms")
            return False

        fused_name = f"RotaryEmbedding_{add_op.name}"
        if (
            fused_name in graph.operations
            or f"{fused_name}_cos_cache" in graph.variables
            or f"{fused_name}_sin_cache" in graph.variables
        ):
            self._skip(add_op, f"target fused name {fused_name} is already in use")
            return False

        self._emit_fused(
            graph,
            add_op=add_op,
            mul_cos=mul_cos,
            mul_sin=mul_sin,
            x_var=x_direct,
            y_var=add_op.outputs[0],
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            attrs={},
        )
        self.fused_count += 1
        if swapped:
            self.fused_swapped_count += 1
        return True

    # ------------------------------------------------------------------
    def _emit_fused(
        self,
        graph: BaseGraph,
        *,
        add_op: Operation,
        mul_cos: Operation,
        mul_sin: Operation,
        x_var: Variable,
        y_var: Variable,
        cos_cache,
        sin_cache,
        attrs: dict,
    ) -> None:
        # Collect the rest of the 7-op chain (Concat + Neg + the two Slices)
        # so it can be detached along with the Add / Mul pair.
        legacy_ops: List[Operation] = []
        concat_op = None
        for var in mul_sin.inputs:
            if var.source_op is not None and var.source_op.type == "Concat":
                concat_op = var.source_op
                break
        if concat_op is not None:
            legacy_ops.append(concat_op)
            for var in concat_op.inputs:
                src = var.source_op
                if src is None:
                    continue
                if src.type == "Neg":
                    if src not in legacy_ops:
                        legacy_ops.append(src)
                    if src.inputs and src.inputs[0].source_op is not None:
                        leaf = src.inputs[0].source_op
                        if leaf not in legacy_ops:
                            legacy_ops.append(leaf)
                elif src.type in ("Slice", "Split"):
                    if src not in legacy_ops:
                        legacy_ops.append(src)

        # Fresh parameter Variables avoid ``BaseGraph.remove_operation`` clearing
        # the original Mul parameter inputs.
        fused_name = f"RotaryEmbedding_{add_op.name}"
        fused = graph.create_operation(
            op_type=ROTARY_EMBEDDING,
            name=fused_name,
            attributes=dict(attrs),
            inputs=[x_var],
            outputs=[y_var],
            platform=add_op.platform,
        )
        y_var.source_op = fused
        x_shape = _static_shape(x_var)
        if x_shape is not None:
            y_var.shape = x_shape

        graph.create_variable(name=f"{fused_name}_cos_cache", value=cos_cache, is_parameter=True, dest_ops=[fused])
        graph.create_variable(name=f"{fused_name}_sin_cache", value=sin_cache, is_parameter=True, dest_ops=[fused])

        # Detach old ops. add_op keeps its output variable — that variable is
        # now the fused node's output; everything else can drop its outputs.
        for old_op in (add_op, mul_cos, mul_sin, *legacy_ops):
            if old_op is None:
                continue
            self._detach_op(graph, old_op, keep_output=(old_op is add_op))

        # Prune any Constant nodes that became dead after detaching the 7-op chain.
        for op in list(graph.operations.values()):
            if op.type == "Constant" and all(not ov.dest_ops and ov.name not in graph.outputs for ov in op.outputs):
                self._detach_op(graph, op, keep_output=False)

        if self.verbose:
            print(
                f"[rotary-embedding-fusion] fused RoPE @ {add_op.name} -> {fused.name} (x={x_var.name}, attrs={attrs})"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _detach_op(graph: BaseGraph, op: Operation, keep_output: bool = False) -> None:
        """Unlink ``op`` from graph and remove it. If ``keep_output`` is True, the
        op's output variables are kept (they've been rewired externally)."""
        for iv in list(op.inputs):
            try:
                if op in iv.dest_ops:
                    iv.dest_ops.remove(op)
            except Exception:
                pass
            # BaseGraph.remove_operation unconditionally wipes and deletes every
            # parameter input still attached to ``op.inputs``. If a parameter
            # (e.g. a shared cos/sin table or Slice constant) is still used by
            # another op in the graph or exposed on the graph boundary, detach
            # it from ``op.inputs`` first so ``remove_operation`` leaves it intact.
            if getattr(iv, "is_parameter", False) and (
                iv.dest_ops or iv.name in graph.inputs or iv.name in graph.outputs
            ):
                op.inputs.remove(iv)

        for ov in list(op.outputs):
            if keep_output:
                # Already rewired to the fused node. Drop it from this op's
                # output list: ``BaseGraph.remove_operation`` resets
                # ``source_op = None`` on every output it still owns, which
                # would otherwise orphan the graph output right after we
                # pointed it at the fused node (executor then yields None).
                op.outputs.remove(ov)
                continue
            if ov.source_op is op:
                ov.source_op = None
            if ov.source_op is None and not ov.dest_ops and ov.name in graph.variables:
                try:
                    graph.remove_variable(ov)
                except Exception:
                    pass

        if op.name in graph.operations:
            try:
                graph.remove_operation(op, keep_coherence=False, remove_unlinked_variable=False)
            except Exception:
                graph.operations.pop(op.name, None)


class EspdlRotaryEmbeddingLegalizePass(QuantizationOptimizationPass):
    """Legalize native RoPE nodes, then enforce the ESP-DL static contract."""

    def __init__(
        self,
        platform: Optional[TargetPlatform] = None,
        strict: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(name="ESPDL RotaryEmbedding Legalize Pass")
        self.platform = platform
        self.strict = strict
        self.verbose = verbose
        self.legalized_count = 0
        self.errors: List[str] = []
        self.simd_hints: List[str] = []

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        if _rope_native_needs_trace(graph):
            _ensure_traced_shapes(graph, **kwargs)
        self.legalized_count += legalize_espdl_rotary_embedding_graph(graph, verbose=self.verbose)
        self.errors, self.simd_hints = check_rotary_embedding(
            graph,
            verbose=self.verbose,
            strict=self.strict,
            platform=self.platform,
        )


def _record_error(op: Operation, reason: str, errors: List[str]) -> None:
    errors.append(f"{op.name}: {reason}")


def _record_hint(op: Operation, reason: str, hints: List[str]) -> None:
    hints.append(f"{op.name}: {reason}")


def check_rotary_embedding(
    graph: BaseGraph,
    verbose: bool = False,
    strict: bool = True,
    platform: Optional[TargetPlatform] = None,
) -> Tuple[List[str], List[str]]:
    """Validate every ``RotaryEmbedding`` node in ``graph`` against the ESP-DL contract.

    Args:
        graph (BaseGraph): Graph to validate.
        verbose (bool): Whether to print a summary when all ops are supported.
        strict (bool): Whether to raise ``ValueError`` on hard errors.
        platform (Optional[TargetPlatform]): Target platform for P4 SIMD hints.

    Returns:
        Tuple[List[str], List[str]]: Hard errors and scalar-fallback hints.
    """
    errors: List[str] = []
    hints: List[str] = []
    for op in graph.operations.values():
        if op.type != ROTARY_EMBEDDING:
            continue
        _check_op(op, errors)
        op_platform = platform if platform is not None else getattr(op, "platform", None)
        if platform is None or op_platform in ROPE_P4_PLATFORMS:
            _collect_simd_hints(op, hints)

    if errors:
        summary = f"RotaryEmbedding support check found {len(errors)} unsupported config(s):\n  - " + (
            "\n  - ".join(errors)
        )
        if strict:
            raise ValueError(summary)
        print(f"[rotary-embedding-support] WARNING: {summary}")
    elif verbose:
        print("[rotary-embedding-support] all RotaryEmbedding ops are supported")

    if hints:
        print(
            f"[rotary-embedding-support] note: {len(hints)} RotaryEmbedding op(s) "
            "will fall back to the scalar kernel on ESP32-P4:\n  - " + ("\n  - ".join(hints))
        )

    return errors, hints


def _collect_simd_hints(
    op: Operation,
    hints: List[str],
    *,
    include_static: bool = True,
    include_quantized: bool = True,
) -> None:
    """Record soft hints when an op will fall back from P4 SIMD to scalar execution.

    ESP32-P4 SIMD kernels support three quantized combinations:
      * int8×3 (X / cos / sin / Y all int8): ``half % 16 == 0``, ``cos_exp == sin_exp``,
        and ``0 < mac_shift <= 31``.
      * int16×3 (X / cos / sin / Y all int16): ``half % 8 == 0``, ``cos_exp == sin_exp``,
        and ``0 < mac_shift <= 63``.
      * mixed s8/s16c (X / Y int8, cos / sin int16): 16-pair or 8-byte-aligned 8-pair
        kernel, ``cos_exp == sin_exp``, and ``0 < mac_shift <= 31``.
    """
    try:
        rdim = int(op.attributes.get("rotary_embedding_dim", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return
    half = None
    cache_dtype = None
    if len(op.inputs) >= 2:
        v = _param_value(op.inputs[1])
        if v is not None:
            t = v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v))
            if t.dim() in (2, 3):
                half = int(t.shape[-1])
            cache_dtype = t.dtype

    if rdim and half and rdim // 2 != half:
        return

    if len(op.inputs) >= 3:
        v_sin = _param_value(op.inputs[2])
        if v_sin is not None and cache_dtype is not None:
            t_sin = v_sin if isinstance(v_sin, torch.Tensor) else torch.from_numpy(np.asarray(v_sin))
            if t_sin.dtype != cache_dtype:
                if include_static:
                    _record_hint(
                        op,
                        f"cos ({cache_dtype}) and sin ({t_sin.dtype}) cache dtypes disagree; "
                        "the P4 SIMD kernels require both caches to share dtype",
                        hints,
                    )
                return

    if include_static:
        if cache_dtype is None or cache_dtype in (torch.float32, torch.float16):
            if half is not None and half % 8 != 0:
                _record_hint(
                    op,
                    f"half={half} is not a multiple of 8 (mixed s8/s16c SIMD gate); "
                    "the default int8 lowering will fall to scalar",
                    hints,
                )
        elif cache_dtype == torch.int8:
            if half is not None and half % 16 != 0:
                _record_hint(
                    op, f"half={half} is not a multiple of 16 (int8×3 SIMD gate); will fall to scalar unroll", hints
                )
        elif cache_dtype == torch.int16:
            if half is not None and half % 8 != 0:
                _record_hint(
                    op,
                    f"half={half} is not a multiple of 8 (int16×3/mixed SIMD gate); will fall to scalar unroll",
                    hints,
                )
        else:
            _record_hint(op, f"cos cache dtype {cache_dtype} has no P4 SIMD kernel", hints)

    # Post-calibration exponent checks when quantization configs are populated.
    in_cfg = getattr(op, "input_quant_config", None) or []
    out_cfg = getattr(op, "output_quant_config", None) or []
    if include_quantized and len(in_cfg) >= 3 and len(out_cfg) >= 1:
        x_exp, x_bits = _extract_tqc_exponent_and_bits(in_cfg[0])
        c_exp, c_bits = _extract_tqc_exponent_and_bits(in_cfg[1])
        s_exp, s_bits = _extract_tqc_exponent_and_bits(in_cfg[2])
        y_exp, y_bits = _extract_tqc_exponent_and_bits(out_cfg[0])
        if c_exp is not None and s_exp is not None and c_exp != s_exp:
            _record_hint(
                op,
                f"cos exponent ({c_exp}) != sin exponent ({s_exp}); P4 SIMD requires "
                "same-k (cos_exp == sin_exp) and will fall back to scalar",
                hints,
            )
        elif None not in (x_exp, c_exp, s_exp, y_exp):
            mac_shift = y_exp - x_exp - c_exp
            max_shift = 63 if (x_bits == 16 and c_bits == 16 and y_bits == 16) else 31
            if not (0 < mac_shift <= max_shift):
                _record_hint(
                    op,
                    f"mac_shift={mac_shift} (y_exp={y_exp} - x_exp={x_exp} - cos_exp={c_exp}) "
                    f"is outside (0, {max_shift}]; P4 SIMD will fall back to scalar",
                    hints,
                )


def _extract_tqc_exponent_and_bits(config) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(exponent, num_of_bits)`` for an activated symmetric per-tensor TQC, else ``(None, None)``."""
    if config is None:
        return None, None
    if getattr(config, "dominated_by", None) is not None:
        config = config.dominated_by
    ready = {
        QuantizationStates.ACTIVATED,
        QuantizationStates.PASSIVE,
        QuantizationStates.OVERLAPPED,
        QuantizationStates.BAKED,
        QuantizationStates.PASSIVE_BAKED,
    }
    if getattr(config, "state", None) not in ready:
        return None, None
    policy = getattr(config, "policy", None)
    if policy is None:
        return None, None
    if not (
        policy.has_property(QuantizationProperty.LINEAR)
        and policy.has_property(QuantizationProperty.PER_TENSOR)
        and policy.has_property(QuantizationProperty.SYMMETRICAL)
        and policy.has_property(QuantizationProperty.POWER_OF_2)
    ):
        return None, None
    bits = int(getattr(config, "num_of_bits", 0))
    if bits not in (8, 16):
        return None, None
    scale = getattr(config, "scale", None)
    if scale is None:
        return None, None
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            return None, None
        val = float(scale.item())
    else:
        val = float(scale)
    if not np.isfinite(val) or val <= 0.0:
        return None, None
    mantissa, exponent = math.frexp(val)
    if mantissa != 0.5:
        return None, None
    return exponent - 1, bits


def _rope_runtime_dtype_from_config(config) -> Tuple[Optional[DataType], Optional[str]]:
    """Resolve the storage dtype an ESP-DL exporter will materialize for one TQC.

    ``FP32`` is a valid deliberate fallback. Any other state must be a finished
    symmetric, per-tensor, linear power-of-two configuration because the device
    model serializes its scale as a single integer exponent.
    """
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
    if bits not in (8, 16):
        return None, f"{bits}-bit quantization is unsupported (only INT8/INT16)"
    scale = getattr(root, "scale", None)
    offset = getattr(root, "offset", None)
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
        return None, "requires one calibrated scalar scale"
    if not isinstance(offset, torch.Tensor) or offset.numel() != 1 or float(offset.item()) != 0.0:
        return None, "requires zero symmetric offset"
    value = float(scale.item())
    if not math.isfinite(value) or value <= 0.0:
        return None, f"invalid scale {value!r}"
    mantissa, _ = math.frexp(value)
    if mantissa != 0.5:
        return None, f"scale {value:.9g} is not an exact power of two"
    return (DataType.INT8 if bits == 8 else DataType.INT16), None


def _rope_expected_rounding(platform: Optional[TargetPlatform]) -> Optional[RoundingPolicy]:
    if platform in ROPE_P4_PLATFORMS:
        return RoundingPolicy.ROUND_HALF_EVEN
    if platform in ROPE_HALF_UP_PLATFORMS:
        return RoundingPolicy.ROUND_HALF_UP
    return None


def _collect_espdl_rotary_embedding_quantization_contract(
    graph: BaseGraph,
    *,
    platform: Optional[TargetPlatform] = None,
) -> Tuple[List[str], List[str]]:
    """Collect calibration-dependent RoPE export errors and SIMD hints."""
    errors: List[str] = []
    hints: List[str] = []
    for op in graph.operations.values():
        if op.type != ROTARY_EMBEDDING:
            continue
        op_platform = platform if platform is not None else getattr(op, "platform", None)
        if platform is None or op_platform in ROPE_P4_PLATFORMS:
            _collect_simd_hints(op, hints, include_static=False)

        if not isinstance(op, QuantableOperation):
            continue
        in_cfg = list(getattr(op, "input_quant_config", ()) or ())
        out_cfg = list(getattr(op, "output_quant_config", ()) or ())
        if len(in_cfg) != 3 or len(out_cfg) != 1:
            _record_error(
                op,
                f"quantized RotaryEmbedding needs 3 input configs and 1 output config, got "
                f"{len(in_cfg)}/{len(out_cfg)}",
                errors,
            )
            continue

        labels = ("X", "cos cache", "sin cache", "Y")
        runtime_dtypes: List[Optional[DataType]] = []
        for label, cfg in zip(labels, (*in_cfg, out_cfg[0])):
            dtype, reason = _rope_runtime_dtype_from_config(cfg)
            runtime_dtypes.append(dtype)
            if reason is not None:
                _record_error(op, f"{label} quantization is not exportable: {reason}", errors)
        if any(dtype is None for dtype in runtime_dtypes):
            continue

        x_dtype, cos_dtype, sin_dtype, y_dtype = runtime_dtypes
        if x_dtype != y_dtype:
            _record_error(
                op,
                f"quantization would export X={_dtype_name(x_dtype)} and Y={_dtype_name(y_dtype)}; "
                "the device requires X/Y to have the same storage dtype",
                errors,
            )
        if cos_dtype != sin_dtype:
            _record_error(
                op,
                f"quantization would export cos={_dtype_name(cos_dtype)} and sin={_dtype_name(sin_dtype)}; "
                "the device requires equal cache storage dtypes",
                errors,
            )
        expected_rounding = _rope_expected_rounding(op_platform)
        if y_dtype != DataType.FP32 and expected_rounding is not None:
            actual_rounding = getattr(out_cfg[0], "rounding", None)
            if actual_rounding != expected_rounding:
                _record_error(
                    op,
                    f"Y rounding {getattr(actual_rounding, 'name', actual_rounding)} does not match "
                    f"target policy {expected_rounding.name}",
                    errors,
                )
    return errors, hints


def _report_espdl_rotary_embedding_export_validation(
    errors: List[str],
    hints: List[str],
    *,
    strict: bool,
    verbose: bool,
) -> Tuple[List[str], List[str]]:
    if errors:
        summary = (
            f"RotaryEmbedding export finalizer found {len(errors)} invalid configuration(s):\n  - "
            + "\n  - ".join(errors)
        )
        if strict:
            raise ValueError(summary)
        print(f"[rotary-embedding-finalizer] WARNING: {summary}")
    elif verbose:
        print("[rotary-embedding-finalizer] all RotaryEmbedding configurations are exportable")

    if hints:
        print(
            f"[rotary-embedding-finalizer] note: {len(hints)} RotaryEmbedding op(s) "
            "will use the scalar path on ESP32-P4:\n  - " + "\n  - ".join(hints)
        )
    return errors, hints


def _finalize_espdl_rotary_embedding_quantization_contract(
    graph: BaseGraph,
    *,
    platform: Optional[TargetPlatform] = None,
    strict: bool = True,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """Validate only RoPE facts introduced by calibration."""
    errors, hints = _collect_espdl_rotary_embedding_quantization_contract(graph, platform=platform)
    return _report_espdl_rotary_embedding_export_validation(errors, hints, strict=strict, verbose=verbose)


def finalize_espdl_rotary_embedding_for_export(
    graph: BaseGraph,
    *,
    platform: Optional[TargetPlatform] = None,
    strict: bool = True,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """Validate the complete RoPE contract before direct ESP-DL export."""
    errors: List[str] = []
    hints: List[str] = []
    for op in graph.operations.values():
        if op.type != ROTARY_EMBEDDING:
            continue
        _check_op(op, errors)
        op_platform = platform if platform is not None else getattr(op, "platform", None)
        if platform is None or op_platform in ROPE_P4_PLATFORMS:
            _collect_simd_hints(op, hints, include_quantized=False)

    quant_errors, quant_hints = _collect_espdl_rotary_embedding_quantization_contract(graph, platform=platform)
    errors.extend(quant_errors)
    hints.extend(quant_hints)
    return _report_espdl_rotary_embedding_export_validation(errors, hints, strict=strict, verbose=verbose)


def _rope_native_needs_trace(graph: BaseGraph) -> bool:
    """Return whether native RotaryEmbedding legalization needs metadata."""
    for op in graph.operations.values():
        if op.type != ROTARY_EMBEDDING:
            continue
        # Malformed arity is a hard contract error, not a metadata-trace
        # problem. Otherwise trace every tensor that _check_op needs,
        # including Y: parsers commonly leave intermediate output metadata
        # unset even when X/cache initializers are fully described.
        if len(op.inputs) < 3 or len(op.outputs) != 1:
            continue
        checked_vars = [*op.inputs[:3], op.outputs[0]]
        if any(_static_shape(var) is None for var in checked_vars):
            return True
        if any(_var_dtype(var) is None for var in checked_vars):
            return True
    return False


def _primitive_rope_candidate_io(
    add_op: Operation,
) -> Optional[Tuple[Variable, Variable]]:
    """Return the prospective fused ``(X, Y)`` without consulting metadata.

    This intentionally checks topology only. Fusion needs X's static shape to
    prove layout, slice bounds, and cache broadcasting; intermediate Mul /
    Concat metadata is not part of that proof.
    """
    if add_op.type != "Add" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
        return None
    mul_ops = [variable.source_op for variable in add_op.inputs]
    if any(op is None or op.type != "Mul" or len(op.inputs) != 2 for op in mul_ops):
        return None

    activations: List[Variable] = []
    for mul_op in mul_ops:
        first, second = mul_op.inputs
        first_is_param = _param_value(first) is not None
        second_is_param = _param_value(second) is not None
        if first_is_param == second_is_param:
            return None
        activations.append(second if first_is_param else first)

    def leaf_source(var: Variable) -> Tuple[Optional[Variable], Optional[Operation]]:
        op = var.source_op
        if op is not None and op.type == "Neg":
            if not op.inputs:
                return None, None
            var = op.inputs[0]
            op = var.source_op
        if op is not None and op.type in ("Slice", "Split") and op.inputs:
            return var, op
        return None, None

    for direct_index, rotated_index in ((0, 1), (1, 0)):
        x_var = activations[direct_index]
        concat = activations[rotated_index].source_op
        if concat is None or concat.type != "Concat" or len(concat.inputs) != 2:
            continue
        sources = [variable.source_op for variable in concat.inputs]
        if sum(source is not None and source.type == "Neg" for source in sources) != 1:
            continue
        var_hi, op_hi = leaf_source(concat.inputs[0])
        var_lo, op_lo = leaf_source(concat.inputs[1])
        if op_hi is None or op_lo is None:
            continue
        if op_hi.type == "Split" or op_lo.type == "Split":
            if op_hi is not op_lo or len(op_hi.outputs) != 2:
                continue
            if var_lo is not op_hi.outputs[0] or var_hi is not op_hi.outputs[1]:
                continue
        if op_hi.inputs[0] is x_var and op_lo.inputs[0] is x_var:
            return x_var, add_op.outputs[0]
    return None


def _rope_fusion_needs_trace(graph: BaseGraph) -> bool:
    """Return whether fusion lacks X shape needed to prove equivalence."""
    for op in graph.operations.values():
        candidate = _primitive_rope_candidate_io(op)
        if candidate is None:
            continue
        x_var, _ = candidate
        if _static_shape(x_var) is None:
            return True
    return False


def _rope_fusion_result_needs_legalize_trace(graph: BaseGraph) -> bool:
    """Return whether a prospective fused node lacks Legalize dtype metadata."""
    for op in graph.operations.values():
        candidate = _primitive_rope_candidate_io(op)
        if candidate is None:
            continue
        x_var, y_var = candidate
        if _var_dtype(x_var) is None or _var_dtype(y_var) is None:
            return True
    return False


def _rope_prequant_needs_trace(graph: BaseGraph, do_fusion: bool = True) -> bool:
    """Return True when the configured prequant RoPE passes need metadata."""
    return _rope_native_needs_trace(graph) or (
        do_fusion and (_rope_fusion_needs_trace(graph) or _rope_fusion_result_needs_legalize_trace(graph))
    )


def _ensure_traced_shapes(graph: BaseGraph, **kwargs) -> None:
    """Populate variable metadata via ``executor.tracing_operation_meta`` if provided."""
    executor = kwargs.get("executor")
    inputs = kwargs.get("inputs")
    if executor is None or inputs is None:
        return
    if not hasattr(executor, "tracing_operation_meta"):
        return
    executor.tracing_operation_meta(inputs=inputs)


def _check_op(op: Operation, errors: List[str]) -> None:
    # The vision subset takes exactly 3 inputs. The spec's optional 4th
    # input (position_ids, for LLM decode-step cache gathering) is rejected
    # here rather than at inference time.
    if len(op.inputs) != 3:
        _record_error(
            op,
            f"expected exactly 3 inputs (X, cos, sin), got {len(op.inputs)}; position_ids "
            "is not supported by the ESP-DL vision subset",
            errors,
        )
    if len(op.inputs) < 3:
        return
    if len(op.outputs) != 1:
        _record_error(op, f"expected exactly 1 output Y, got {len(op.outputs)}", errors)
    x_var = op.inputs[0]
    cos_var = op.inputs[1]
    sin_var = op.inputs[2]
    y_var = op.outputs[0] if op.outputs else None

    # ---- X shape/rank must be concrete before a device node is admitted. ----
    x_shape = _static_shape(x_var)
    B = None
    L = None
    D = None
    try:
        num_heads = int(op.attributes.get("num_heads", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        num_heads = -1
        _record_error(op, f"num_heads={op.attributes.get('num_heads')!r} is not an integer", errors)
    if x_shape is None:
        _record_error(
            op,
            "X shape is missing or dynamic; trace with executor + inputs before validating RotaryEmbedding",
            errors,
        )
    else:
        x_rank = len(x_shape)
        if x_rank == 4:
            B, H, L, D = x_shape
            if num_heads and num_heads != H:
                _record_error(op, f"num_heads={num_heads} contradicts X's H axis ({H})", errors)
        else:
            _record_error(
                op,
                f"X must be 4D [B,H,L,D], got rank {x_rank}; 3D [B,L,H*D] is not supported by the ESP-DL vision subset",
                errors,
            )

    if D is not None and D % 2 != 0:
        _record_error(op, f"head_dim D={D} must be even", errors)

    # Full rotation only: 0 and head_dim D both mean "rotate the whole head".
    try:
        rdim = int(op.attributes.get("rotary_embedding_dim", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        rdim = None
        _record_error(
            op,
            f"rotary_embedding_dim={op.attributes.get('rotary_embedding_dim')!r} is not an integer",
            errors,
        )
    if rdim is not None:
        if rdim < 0 or rdim % 2 != 0:
            _record_error(op, f"rotary_embedding_dim {rdim} must be a non-negative even integer", errors)
        elif rdim and D is not None and rdim != D:
            _record_error(
                op,
                f"rotary_embedding_dim {rdim} != head_dim D={D}; partial rotation is not "
                "supported by the ESP-DL vision subset",
                errors,
            )

    try:
        interleaved = int(op.attributes.get("interleaved", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        interleaved = None
        _record_error(
            op,
            f"interleaved={op.attributes.get('interleaved')!r} is not an integer",
            errors,
        )
    if interleaved is not None and interleaved != 0:
        _record_error(
            op,
            f"interleaved={interleaved} (GPT-J layout) is not supported by the ESP-DL "
            "vision subset, which implements interleaved=0 (half-split) only",
            errors,
        )

    # ---- Runtime dtype contract. ----
    x_dtype = _var_dtype(x_var)
    cos_dtype = _var_dtype(cos_var)
    sin_dtype = _var_dtype(sin_var)
    y_dtype = _var_dtype(y_var)
    for label, dtype in (
        ("X", x_dtype),
        ("cos cache", cos_dtype),
        ("sin cache", sin_dtype),
        ("Y", y_dtype),
    ):
        if dtype is None:
            _record_error(
                op,
                f"{label} dtype is missing; trace with executor + inputs before validating RotaryEmbedding",
                errors,
            )
        elif not _supported_runtime_dtype(dtype):
            _record_error(
                op,
                f"{label} dtype {_dtype_name(dtype)} is unsupported; ESP-DL RotaryEmbedding accepts "
                "FP32, INT8, or INT16 only",
                errors,
            )
    if cos_dtype is not None and sin_dtype is not None and cos_dtype != sin_dtype:
        _record_error(
            op,
            f"cos cache dtype {_dtype_name(cos_dtype)} != sin cache dtype {_dtype_name(sin_dtype)}",
            errors,
        )
    if x_dtype is not None and y_dtype is not None and x_dtype != y_dtype:
        _record_error(
            op,
            f"X dtype {_dtype_name(x_dtype)} != Y dtype {_dtype_name(y_dtype)}; "
            "ESP-DL RotaryEmbedding requires matching X/Y storage types",
            errors,
        )

    y_shape = _static_shape(y_var)
    if y_shape is None:
        _record_error(
            op,
            "Y shape is missing or dynamic; trace with executor + inputs before validating RotaryEmbedding",
            errors,
        )
    elif x_shape is not None and y_shape != x_shape:
        _record_error(
            op,
            f"Y shape {tuple(y_shape)} != X shape {tuple(x_shape)}; RotaryEmbedding preserves X shape",
            errors,
        )

    # ---- Cache geometry must be concrete, equal and device-indexable. ----
    cos_shape = _static_shape(cos_var)
    sin_shape = _static_shape(sin_var)
    if cos_shape is None:
        _record_error(
            op,
            "cos cache shape is missing or dynamic; trace with executor + inputs before validating RotaryEmbedding",
            errors,
        )
    if sin_shape is None:
        _record_error(
            op,
            "sin cache shape is missing or dynamic; trace with executor + inputs before validating RotaryEmbedding",
            errors,
        )
    if cos_shape is None or sin_shape is None:
        return
    if cos_shape != sin_shape:
        _record_error(op, f"cos cache {tuple(cos_shape)} != sin cache {tuple(sin_shape)}", errors)
        return

    c_rank = len(cos_shape)
    if c_rank not in (2, 3):
        _record_error(op, f"cache must be 2D or 3D, got rank {c_rank}", errors)
        return

    cache_half = cos_shape[-1]
    expected = rdim // 2 if rdim else (D // 2 if D is not None else None)
    if expected is not None and cache_half != expected:
        _record_error(
            op,
            f"cache last dim {cache_half} != D/2 ({expected}); "
            "the ONNX cache holds one entry per rotated pair, not per element",
            errors,
        )
    cache_L = cos_shape[-2]
    if L is not None and cache_L != L:
        _record_error(
            op,
            f"cache sequence length L={cache_L} != X's L={L}",
            errors,
        )
    if c_rank == 3 and B is not None:
        cache_B = cos_shape[0]
        if cache_B != B and cache_B != 1:
            _record_error(
                op,
                f"cache batch B={cache_B} matches neither X's B={B} nor 1",
                errors,
            )

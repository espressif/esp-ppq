"""Fuse MatMul/Softmax attention subgraphs, then keep or expand the result."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from esp_ppq.core import DataType, TargetPlatform
from esp_ppq.executor.op.torch.espdl_attention import (
    _decode_layout,
    _layout_head_seq_axes,
    espdl_attention_attributes_ineligible_reason,
    espdl_attention_eligible,
    espdl_attention_head_attrs_ineligible_reason,
    espdl_attention_ineligible_reason,
    espdl_attention_ineligible_reason_from_parts,
    espdl_attention_mask_ineligible_reason,
    espdl_attention_op_logit_scale_ineligible_reason,
)
from esp_ppq.IR import BaseGraph, Operation, Variable
from esp_ppq.IR.search import SearchableGraph
from esp_ppq.log import NaiveLogger
from esp_ppq.quantization.optim.base import QuantizationOptimizationPass

logger = NaiveLogger.get_logger('PPQ')

ATTENTION = "Attention"

# Every esp-dl chip can run a fused Attention node. PIE v2 (P4 / S31) uses the
# QACC kernels; S3 and plain C use the scalar fallback in dl_base_attention.cpp.
# S3 assembly is intentionally not required here.
ESPDL_ATTENTION_PLATFORMS = frozenset(
    {
        TargetPlatform.ESPDL_INT8,
        TargetPlatform.ESPDL_INT16,
        TargetPlatform.ESPDL_H_PRE_INT16,
        TargetPlatform.ESPDL_W8A16,
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


def espdl_attention_platform_unsupported_reason(platform: Optional[TargetPlatform]) -> Optional[str]:
    """None if ``platform`` can run a fused Attention node (``None`` = unspecified)."""
    if platform is None or platform in ESPDL_ATTENTION_PLATFORMS:
        return None
    return f"platform {platform.name} has no esp-dl Attention kernel (requires an ESP-DL target platform)"


def _static_dims(dims) -> bool:
    return bool(dims) and all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in dims)


def _shape(var: Optional[Variable]) -> Optional[List[int]]:
    """Shape of ``var`` as a plain list, or None when metadata is missing."""
    if var is None:
        return None
    shape = getattr(var, "shape", None)
    if shape is not None:
        return list(shape)
    value = getattr(var, "value", None)
    if value is not None and hasattr(value, "shape"):
        try:
            dims = [int(dim) for dim in value.shape]
        except (TypeError, ValueError):
            dims = []
        if _static_dims(dims):
            return dims
    return None


_REPORT_KEY = "espdl_attention_report"
_REPORT_MAX_NAMES = 3
_FOLDED_WHERE_MASK_ADD_KEY = "espdl_attention_folded_where_mask_add"


@dataclass
class _EspdlAttentionReport:
    """What fusion and legalize decided, logged once per export by legalize."""

    fused: int = 0
    kept: int = 0
    compacted: int = 0
    unfolded_where: int = 0
    left: Dict[str, List[str]] = field(default_factory=dict)
    expanded: Dict[str, List[str]] = field(default_factory=dict)
    actionable: Set[str] = field(default_factory=set)

    def record_left(self, name: str, reason: str, actionable: bool) -> None:
        self.left.setdefault(reason, []).append(name)
        if actionable:
            self.actionable.add(reason)

    def record_expanded(self, name: str, reason: str, actionable: bool) -> None:
        self.expanded.setdefault(reason, []).append(name)
        if actionable:
            self.actionable.add(reason)

    def empty(self) -> bool:
        return not (self.kept or self.left or self.expanded or self.unfolded_where)


def _report(graph: BaseGraph) -> _EspdlAttentionReport:
    attribs = getattr(graph, "extension_attrib", None)
    if attribs is None:
        return _EspdlAttentionReport()
    report = attribs.get(_REPORT_KEY)
    if report is None:
        report = _EspdlAttentionReport()
        graph.set_extension_attrib(_REPORT_KEY, report)
    return report


def _names(names: List[str]) -> str:
    shown = ", ".join(names[:_REPORT_MAX_NAMES])
    if len(names) > _REPORT_MAX_NAMES:
        shown += f", ... (+{len(names) - _REPORT_MAX_NAMES})"
    return shown


def _log_attention_report(graph: BaseGraph) -> None:
    """One summary line, then one line per distinct reason.

    Reasons the user can act on (a dimension above the configured ceiling on a
    platform that has the kernel) are WARNINGs; expected outcomes stay INFO.
    """
    attribs = getattr(graph, "extension_attrib", None)
    report = attribs.pop(_REPORT_KEY, None) if attribs is not None else None
    if report is None or report.empty():
        return
    kept = f"{report.kept} Attention kept for the device kernel"
    if report.fused:
        kept += f" ({report.fused} fused from MatMul/Softmax)"
    parts = [kept]
    if report.expanded:
        parts.append(f"{sum(map(len, report.expanded.values()))} expanded to MatMul/Softmax")
    if report.left:
        parts.append(f"{sum(map(len, report.left.values()))} MatMul/Softmax pattern(s) left unfused")
    if report.compacted:
        parts.append(f"{report.compacted} constant mask(s) stored as bool")
    if report.unfolded_where:
        parts.append(f"{report.unfolded_where} Where mask(s) not constant-foldable")
    logger.info(f'[espdl-attention] {", ".join(parts)}')
    for verdict, groups in (("left as MatMul/Softmax", report.left), ("expanded to MatMul/Softmax", report.expanded)):
        for reason, names in groups.items():
            log_fn = logger.warning if reason in report.actionable else logger.info
            log_fn(f'[espdl-attention] {verdict} ({len(names)}: {_names(names)}): {reason}')
    if report.unfolded_where:
        logger.info(
            '[espdl-attention] a Where(mask) before Softmax folds into the Attention only when its '
            '``cond`` and ``-inf`` side are static constants (esp-dl has no Where kernel); re-export '
            'with ``attn_mask`` as an additive float tensor to enable fusion'
        )


def _connected_mask_unshaped(op: Operation) -> bool:
    """True when the optional ``attn_mask`` socket (4 inputs total) has no shape.

    With five or more inputs the fourth socket is past_key (KV cache), not mask.
    """
    return op.num_of_input == 4 and op.inputs[3] is not None and _shape(op.inputs[3]) is None


def _attention_op_shapes_incomplete(op: Operation) -> bool:
    """True when a native ``Attention`` op still lacks Q/K/V or mask shape."""
    return op.type == ATTENTION and (any(_shape(var) is None for var in op.inputs[:3]) or _connected_mask_unshaped(op))


def _is_candidate_scale_factor(var: Optional[Variable], seen: Optional[set] = None) -> bool:
    """True when ``var`` is a scalar factor or a ``Shape -> Slice`` scale chain."""
    if var is None:
        return False
    if _eval_scale_factor(var) is not None:
        return True
    seen = set() if seen is None else seen
    if id(var) in seen:
        return False
    seen.add(id(var))
    src = getattr(var, "source_op", None)
    if src is None or not src.inputs:
        return False
    if src.type in ("Cast", "Identity", "Sqrt"):
        return _is_candidate_scale_factor(src.inputs[0], seen)
    if src.type == "Div" and len(src.inputs) == 2:
        return _is_candidate_scale_factor(src.inputs[0], seen) and _is_candidate_scale_factor(src.inputs[1], seen)
    if src.type == "Slice" and len(src.inputs) >= 3:
        shape_src = getattr(src.inputs[0], "source_op", None)
        return shape_src is not None and shape_src.type == "Shape"
    return False


def _candidate_qk_matmul(scores_var: Optional[Variable]) -> Optional[Operation]:
    """Return ``qk_matmul`` if ``scores_var`` is ``MatMul`` or ``MatMul -> Mul/Div``."""
    if scores_var is None:
        return None
    src = getattr(scores_var, "source_op", None)
    if src is None:
        return None
    if src.type == "MatMul" and src.num_of_parameter == 0 and len(src.inputs) == 2:
        return src
    if src.type in ("Mul", "Div") and len(src.inputs) == 2:
        for idx, inp in enumerate(src.inputs):
            if src.type == "Div" and idx != 0:
                continue
            inp_src = getattr(inp, "source_op", None)
            if (
                inp_src is not None
                and inp_src.type == "MatMul"
                and inp_src.num_of_parameter == 0
                and len(inp_src.inputs) == 2
                and len(inp.dest_ops) == 1
                and inp.dest_ops[0] is src
                and _is_candidate_scale_factor(src.inputs[1 - idx])
            ):
                return inp_src
    return None


def _candidate_prescale_data_vars(var: Variable, consumer: Operation) -> List[Variable]:
    """Candidate data inputs before an optional prescale ``Mul``."""
    src = getattr(var, "source_op", None)
    if src is None or src.type != "Mul" or len(src.inputs) != 2:
        return [var]
    if not _has_only_expected_consumer(var, consumer):
        return []
    # Before shapes are traced, a Shape->Slice->Sqrt prescale factor cannot be
    # folded yet; consider any non-parameter input of the Mul as a candidate.
    candidates = [inp for inp in src.inputs if not getattr(inp, "is_parameter", False)]
    return candidates or list(src.inputs)


def _softmax_in_attention_candidate(graph: BaseGraph, softmax: Operation) -> bool:
    """True when ``softmax`` sits in a structural ``MatMul -> ... -> Softmax -> MatMul``
    attention candidate whose variables still lack traced shapes.

    Avoids running an extra whole-graph forward trace before prequant on
    classification models whose ``Softmax`` is just a classifier head.
    """
    if softmax.type != "Softmax" or softmax.num_of_input < 1 or softmax.num_of_output != 1:
        return False
    probs_var = softmax.outputs[0]
    if _var_is_graph_output(graph, probs_var) or len(probs_var.dest_ops) != 1:
        return False
    pv_matmul = probs_var.dest_ops[0]
    if (
        pv_matmul.type != "MatMul"
        or pv_matmul.num_of_parameter != 0
        or len(pv_matmul.inputs) != 2
        or pv_matmul.inputs[0] is not probs_var
    ):
        return False

    sm_in = softmax.inputs[0]
    if sm_in is None or _var_is_graph_output(graph, sm_in) or not _has_only_expected_consumer(sm_in, softmax):
        return False
    pre_op = getattr(sm_in, "source_op", None)
    if pre_op is None:
        return False

    qk_matmul: Optional[Operation] = None
    mask_var: Optional[Variable] = None
    if pre_op.type == "Add" and len(pre_op.inputs) == 2:
        for idx, inp in enumerate(pre_op.inputs):
            cand = _candidate_qk_matmul(inp)
            if cand is not None:
                qk_matmul = cand
                mask_var = pre_op.inputs[1 - idx]
                break
    elif pre_op.type == "Where" and len(pre_op.inputs) == 3:
        for inp in pre_op.inputs[1:]:
            cand = _candidate_qk_matmul(inp)
            if cand is not None and _try_fold_where_mask(pre_op, inp) is not None:
                qk_matmul = cand
                break
    else:
        qk_matmul = _candidate_qk_matmul(sm_in)

    if qk_matmul is None:
        return False

    q_candidates = _candidate_prescale_data_vars(qk_matmul.inputs[0], qk_matmul)
    k_candidates = _candidate_prescale_data_vars(qk_matmul.inputs[1], qk_matmul)
    if not q_candidates or not k_candidates:
        return False

    matched_k_vars: List[Variable] = []
    for k_data in k_candidates:
        head_merge = _match_head_merge(graph, k_data)
        if head_merge is not None:
            matched_k_vars.extend([head_merge[1], k_data])
        else:
            k_transpose, _ = _match_k_transpose(k_data)
            if k_transpose is None:
                # Could be a 4-D kt_4d candidate once shapes are traced.
                matched_k_vars.append(k_data)
            elif len(k_transpose.attributes.get("perm", []) or []) in (3, 4):
                matched_k_vars.extend([k_transpose.inputs[0], k_data])
            # A 2-D Transpose (perm=[1, 0]) is never fused and is skipped here.
    if not matched_k_vars:
        return False

    check_vars: List[Optional[Variable]] = [
        *q_candidates,
        *matched_k_vars,
        pv_matmul.inputs[1],
        qk_matmul.outputs[0],
        sm_in,
        probs_var,
        pv_matmul.outputs[0],
        mask_var,
    ]
    return any(var is not None and _shape(var) is None for var in check_vars)


def _attention_shapes_incomplete(graph: BaseGraph) -> bool:
    """True when an Attention node, or a Softmax inside a structural primitive
    attention candidate, still has an unshaped variable."""
    for op in graph.operations.values():
        if _attention_op_shapes_incomplete(op):
            return True
        if op.type == "Softmax" and _softmax_in_attention_candidate(graph, op):
            return True
    return False


def _attention_prequant_needs_trace(graph: BaseGraph, platform: Optional[TargetPlatform], fusion: bool = True) -> bool:
    """Whether quantize should trace before prequant.

    Fusion and legalize only read shapes. The sample has to be the same
    ``inputs`` the base quantizer traces with after prequant, so this trace
    lives on the quantizer. Skip it when shapes are already present, when
    ``setting.espdl_attention_fusion`` is off and the graph has no Attention
    node to legalize, or when the platform has no Attention kernel.
    """
    if any(_attention_op_shapes_incomplete(op) for op in graph.operations.values()):
        return True
    if not fusion or espdl_attention_platform_unsupported_reason(platform) is not None:
        return False
    return any(op.type == "Softmax" and _softmax_in_attention_candidate(graph, op) for op in graph.operations.values())


def _ensure_traced_shapes(graph: BaseGraph, **kwargs) -> None:
    """Trace only when the caller passed ``inputs``.

    Quantize traces before prequant, and FP32 export traces before fusion, so
    both paths already have shapes when they reach here. A direct call that
    still has an executor and an explicit sample is traced once. A calibration
    batch is not a substitute: it can disagree with the sample quantization
    traces afterwards. No executor or no ``inputs`` leaves the graph untouched.
    A trace that raises propagates.
    """
    if not _attention_shapes_incomplete(graph):
        return
    executor = kwargs.get("executor")
    batch = kwargs.get("inputs")
    if executor is None or batch is None:
        return
    executor.tracing_operation_meta(inputs=batch)


def _get_const_scalar(var: Variable) -> Optional[float]:
    """Python float if ``var`` is a scalar / 1-element parameter, else None."""
    if var is None or not getattr(var, "is_parameter", False):
        return None
    val = getattr(var, "value", None)
    if val is None:
        return None
    try:
        if hasattr(val, "numel"):
            if int(val.numel()) != 1:
                return None
            return float(val.reshape(-1)[0].item())
        arr = val
        if hasattr(arr, "reshape"):
            arr = arr.reshape(-1)
            if len(arr) != 1:
                return None
            return float(arr[0])
    except (RuntimeError, ValueError, TypeError):
        return None
    return None


def _mul_div_pre_softmax(qk_matmul: Operation) -> Optional[Operation]:
    """Return the scalar ``Mul``/``Div`` consuming qk_matmul's output, if any.

    ``Div`` must divide the scores (``scores / c``); ``c / scores`` is not a scale.
    """
    dst = qk_matmul.outputs[0].dest_ops
    if len(dst) != 1:
        return None
    nxt = dst[0]
    if nxt.type not in ("Mul", "Div"):
        return None
    if len(nxt.inputs) != 2:
        return None
    if nxt.type == "Div" and nxt.inputs[0].source_op is not qk_matmul:
        return None
    other_idx = 1 if nxt.inputs[0].source_op is qk_matmul else 0
    if _eval_scale_factor(nxt.inputs[other_idx]) is None:
        return None
    return nxt


def _match_k_transpose(k_operand: Variable) -> Tuple[Optional[Operation], Optional[str]]:
    """Unwrap the Transpose that turns K into K^T.

    Returns ``(op, mode)``: ``"drop"`` (last-two-axis swap) or
    ``"rewrite_to_bhnd"`` (BNHD -> [B,H,D,N] or LND -> [N,D,L] rewritten to
    keep the sequence and feature axes in attention order).
    """
    src = getattr(k_operand, "source_op", None)
    if src is None or src.type != "Transpose":
        return None, None
    perm = list(src.attributes.get("perm", []))
    n = len(perm)
    if n < 2:
        return None, None
    swap_last = list(range(n))
    swap_last[-1], swap_last[-2] = swap_last[-2], swap_last[-1]
    if perm == swap_last:
        return src, "drop"
    if n == 4 and perm == [0, 2, 3, 1]:
        return src, "rewrite_to_bhnd"
    if n == 3 and perm == [1, 2, 0]:
        return src, "rewrite_to_bhnd"
    return None, None


def _rewrite_k_transpose_for_bhnd(k_transpose: Operation) -> Optional[Variable]:
    """Rewrite ``k_transpose`` in place from BNHD/LND -> K^T to BHND/NLD."""
    if len(k_transpose.outputs[0].dest_ops) != 1:
        return None
    perm = list(k_transpose.attributes.get("perm", []) or [])
    in_shape = _shape(k_transpose.inputs[0])
    if len(perm) == 3:
        k_transpose._attributes["perm"] = [1, 0, 2]
        if in_shape is not None and len(in_shape) == 3:
            ln, h, d = in_shape[0], in_shape[1], in_shape[2]
            k_transpose.outputs[0].shape = [h, ln, d]
    else:
        k_transpose._attributes["perm"] = [0, 2, 1, 3]
        if in_shape is not None and len(in_shape) == 4:
            b, ln, h, d = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
            k_transpose.outputs[0].shape = [b, h, ln, d]
    return k_transpose.outputs[0]


def _var_is_graph_output(graph: BaseGraph, var: Variable) -> bool:
    return var.name in graph.outputs


def _has_only_expected_consumer(var: Variable, expected: Operation) -> bool:
    """True when ``var`` has exactly one consumer, and it is ``expected``."""
    consumers = list(var.dest_ops)
    return len(consumers) == 1 and consumers[0] is expected


def _where_as_mask_pre_softmax(qk_matmul: Operation) -> Optional[Operation]:
    """Detect (but do not fuse) ``MatMul -> [scale] -> Where -> Softmax``."""
    dst = qk_matmul.outputs[0].dest_ops
    if len(dst) != 1 or dst[0].type != "Where":
        return None
    where = dst[0]
    if len(where.outputs[0].dest_ops) != 1 or where.outputs[0].dest_ops[0].type != "Softmax":
        return None
    return where


def _var_static_value(var: Variable) -> Optional[torch.Tensor]:
    """Return ``var``'s materialized value as a torch tensor, or None.

    Covers two cases:
    - ``var`` is a parameter (``is_parameter=True``) with a value.
    - ``var`` is produced by a ``Constant`` op whose ``value`` attribute is a
      tensor-like scalar/array.
    """
    if var is None:
        return None
    if getattr(var, "is_parameter", False):
        val = getattr(var, "value", None)
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val
        try:
            return torch.as_tensor(val)
        except (TypeError, ValueError):
            return None
    src = getattr(var, "source_op", None)
    if src is None or src.type != "Constant":
        return None
    val = src.attributes.get("value")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val
    try:
        return torch.as_tensor(val)
    except (TypeError, ValueError):
        return None


def _is_neg_inf(val: Optional[torch.Tensor]) -> bool:
    """True iff ``val`` is a real tensor whose every element is -inf."""
    if val is None or val.numel() == 0:
        return False
    try:
        flat = val.reshape(-1).to(torch.float64)
    except (RuntimeError, TypeError):
        return False
    return bool(torch.isneginf(flat).all().item())


def _try_fold_where_mask(where: Operation, scores_output: Variable) -> Optional[torch.Tensor]:
    """Constant-fold ``Where(cond, X, Y)`` where one branch is scores (``scores_output``)
    and the other is a ``-inf`` broadcastable constant, into an additive fp32
    mask tensor.

    Returns the folded mask as a fp32 tensor, or None if either
    (a) both branches are constants (no scores side),
    (b) neither branch matches ``scores_output``,
    (c) the ``-inf`` side is not statically -inf, or
    (d) ``cond`` is not a static constant.

    ``cond`` must be static because esp-dl has no runtime ``Where`` kernel;
    runtime bool masks stay decomposed to ``MatMul/Softmax`` upstream.
    """
    if len(where.inputs) != 3:
        return None
    cond_var, x_var, y_var = where.inputs
    if x_var is scores_output and y_var is not scores_output:
        mask_true_is_scores = True
    elif y_var is scores_output and x_var is not scores_output:
        mask_true_is_scores = False
    else:
        return None
    neg_inf_var = y_var if mask_true_is_scores else x_var
    neg_inf_val = _var_static_value(neg_inf_var)
    if not _is_neg_inf(neg_inf_val):
        return None
    cond_val = _var_static_value(cond_var)
    if cond_val is None:
        return None
    cond_bool = cond_val.to(torch.bool)
    # Additive mask: 0 where cond selects scores, -inf where it selects -inf.
    # PyTorch/NumPy ``where(cond, X, Y)`` picks X when cond is True.
    if mask_true_is_scores:
        # Where(cond, scores, -inf): cond True keeps scores, False masks.
        mask = torch.where(
            cond_bool,
            torch.zeros_like(cond_bool, dtype=torch.float32),
            torch.full_like(cond_bool, float("-inf"), dtype=torch.float32),
        )
    else:
        # Where(cond, -inf, scores): cond True masks, False keeps scores.
        mask = torch.where(
            cond_bool,
            torch.full_like(cond_bool, float("-inf"), dtype=torch.float32),
            torch.zeros_like(cond_bool, dtype=torch.float32),
        )
    return mask


def _add_pre_softmax(scores_output: Variable, next_op: Operation) -> Optional[Operation]:
    """``next_op`` if it is a valid mask-add feeding softmax, else None."""
    if next_op is None or next_op.type != "Add":
        return None
    if next_op.inputs[0].source_op is None and next_op.inputs[1].source_op is None:
        return next_op
    if scores_output not in next_op.inputs:
        return None
    return next_op


@dataclass
class _AttentionPrimitiveMatch:
    """A ``MatMul -> [Mul/Div]? -> [Add]? -> Softmax -> MatMul`` that is attention.

    Built without touching the graph, so a match the device cannot run is left
    exactly as the user exported it.
    """

    qk_matmul: Operation
    scale_op: Optional[Operation]
    add: Optional[Operation]
    softmax: Operation
    pv_matmul: Operation
    k_transpose: Operation
    k_transpose_mode: str
    scale: Optional[float]
    mask: Optional[Variable]
    q_var: Variable
    k_var: Variable
    v_var: Variable
    k_data: Optional[Variable] = None
    prescale_ops: Tuple[Operation, ...] = ()
    head_merge_ops: Tuple[Operation, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.qk_matmul.name}/{self.softmax.name}/{self.pv_matmul.name}"

    @property
    def attributes(self) -> dict:
        # always emit scale: absent means 1/sqrt(D), which would double-apply a folded Mul/Div
        return {"scale": float(self.scale) if self.scale is not None else 1.0}

    def qkv_shapes(self) -> List[Optional[List[int]]]:
        k_shape = _shape(self.k_var)
        if self.k_transpose_mode == "rewrite_to_bhnd" and k_shape is not None:
            if len(k_shape) == 4:
                k_shape = [k_shape[0], k_shape[2], k_shape[1], k_shape[3]]
            elif len(k_shape) == 3:
                k_shape = [k_shape[1], k_shape[0], k_shape[2]]
        elif self.k_transpose_mode == "kt_4d" and k_shape is not None and len(k_shape) == 4:
            # ``k_var`` is already K^T, laid out ``[B, H, D, L]``.
            k_shape = [k_shape[0], k_shape[1], k_shape[3], k_shape[2]]
        return [
            _shape(self.q_var),
            k_shape,
            _shape(self.v_var),
        ]

    def ineligible_reason(self) -> str:
        raw_shapes = self.qkv_shapes()
        mask_shape = None if self.mask is None else _shape(self.mask)
        rank = _pattern_rank(self)
        shapes = raw_shapes
        if rank == 3 and all(s is not None and len(s) == 3 for s in raw_shapes):
            if mask_shape is not None and len(mask_shape) > 3:
                return f"attn_mask rank {len(mask_shape)} exceeds 3-D attention rank 3"
            shapes = [[1, *s] for s in raw_shapes]
        base_reason = espdl_attention_ineligible_reason_from_parts(
            self.attributes,
            3 if self.mask is None else 4,
            shapes,
            mask_shape=mask_shape,
            mask_dtype=None if self.mask is None else self.mask.dtype,
        )
        if base_reason:
            return base_reason

        # Primitive MatMul->Softmax->MatMul has no GQA repeat_interleave node,
        # so the batch/head axis of Q, K (in BHND/NLD), and V must match
        # exactly. Rejecting Hq != Hkv here prevents BNHD subgraphs with
        # Lq % Lk == 0 or axis-1 broadcasts from fusing as BHND GQA.
        q_s, k_s, v_s = shapes[0], shapes[1], shapes[2]
        if q_s[1] != k_s[1] or q_s[1] != v_s[1]:
            return (
                f"primitive MatMul attention requires equal head counts (Hq == Hkv == Hv); "
                f"got Q={q_s[1]} K={k_s[1]} V={v_s[1]}"
            )

        if rank == 4:
            bnhd_check_vars = [self.q_var, self.v_var]
            if self.k_transpose_mode in ("drop", "head_merge"):
                bnhd_check_vars.append(self.k_var)
            if any(_is_untransposed_bnhd_reshape(var) for var in bnhd_check_vars):
                return (
                    "Q/K/V comes directly from a 3-D [B, N, H*D] -> 4-D [B, N, H, D] "
                    "BNHD Reshape without transposing to BHND"
                )
            b, h, lq, d = q_s
            lk = k_s[2]
            if self.k_data is not None:
                kd_s = _shape(self.k_data)
                if kd_s is not None and kd_s != [b, h, d, lk]:
                    return f"K^T shape {kd_s} does not match expected [B={b}, H={h}, D={d}, Lk={lk}]"
            if self.k_transpose_mode == "head_merge" and len(self.head_merge_ops) == 3:
                inner, transpose, outer = self.head_merge_ops
                in_out = _shape(inner.outputs[0])
                if in_out is not None and in_out != [b * h, lk, d]:
                    return f"head_merge inner Reshape shape {in_out} does not match [B*H={b * h}, Lk={lk}, D={d}]"
                tr_out = _shape(transpose.outputs[0])
                if tr_out is not None and tr_out != [b * h, d, lk]:
                    return f"head_merge Transpose shape {tr_out} does not match [B*H={b * h}, D={d}, Lk={lk}]"
                out_out = _shape(outer.outputs[0])
                if out_out is not None and out_out != [b, h, d, lk]:
                    return f"head_merge outer Reshape shape {out_out} does not match [B={b}, H={h}, D={d}, Lk={lk}]"
            qk_s = _shape(self.qk_matmul.outputs[0])
            if qk_s is not None and qk_s != [b, h, lq, lk]:
                return f"QK MatMul output shape {qk_s} does not match expected [B={b}, H={h}, Lq={lq}, Lk={lk}]"
            pv_s = _shape(self.pv_matmul.outputs[0])
            if pv_s is not None and pv_s != [b, h, lq, v_s[3]]:
                return f"PV MatMul output shape {pv_s} does not match expected [B={b}, H={h}, Lq={lq}, D={v_s[3]}]"
        elif rank == 3:
            n_heads, lq, d = q_s[1], q_s[2], q_s[3]
            lk = k_s[2]
            if self.k_data is not None:
                kd_s = _shape(self.k_data)
                if kd_s is not None and kd_s != [n_heads, d, lk]:
                    return f"K^T shape {kd_s} does not match expected [N={n_heads}, D={d}, Lk={lk}]"
            qk_s = _shape(self.qk_matmul.outputs[0])
            if qk_s is not None and qk_s != [n_heads, lq, lk]:
                return f"QK MatMul output shape {qk_s} does not match expected [N={n_heads}, Lq={lq}, Lk={lk}]"
            pv_s = _shape(self.pv_matmul.outputs[0])
            if pv_s is not None and pv_s != [n_heads, lq, v_s[3]]:
                return f"PV MatMul output shape {pv_s} does not match expected [N={n_heads}, Lq={lq}, D={v_s[3]}]"
        return ""


def _rewrite_where_masks_to_add(
    graph: BaseGraph,
    dispatching_overrides: Optional[Set[str]] = None,
    allow_automatic_fp32_where: bool = False,
) -> int:
    """Fold ``Where(cond, scores, -inf)`` into ``Add(scores, fp32_mask)``.

    Detects every ``MatMul → [Mul/Div]? → Where → Softmax`` where one
    Where branch is scores and the other is a ``-inf`` constant, then
    constant-folds a matching additive mask (``0`` where cond keeps
    scores, ``-inf`` where cond masks) and rewires the graph.

    Rewrite (before → after)::

        qk_matmul → [scale] → Where(cond, scores, -inf) → Softmax → pv_matmul
                         │
                         cond (must be static constant)

        qk_matmul → [scale] ──────────────────────────→ Add → Softmax → pv_matmul
                                                ↑
                                                fp32 mask parameter

    Only static ``cond`` is supported: esp-dl has no ``Where`` module,
    so a runtime cond would produce an additive mask that cannot be
    materialised on device. Runtime bool masks stay a
    ``_report_unfused_where_masks`` INFO after this rewrite runs.

    Covered by Swin Transformer (windowed-attention constant masks),
    causal masks precomputed with ``torch.triu``, and any decoder whose
    padding mask is baked in at export.
    """
    rewired = 0
    for op in list(graph.operations.values()):
        if op.type != "MatMul":
            continue
        scale_end = _mul_div_pre_softmax(op) or op
        where = _where_as_mask_pre_softmax(scale_end)
        if where is None:
            continue
        scores_output = scale_end.outputs[0]
        mask_tensor = _try_fold_where_mask(where, scores_output)
        if mask_tensor is None:
            # Fall through to _report_unfused_where_masks.
            continue
        softmax = where.outputs[0].dest_ops[0]
        fusion_platform = where.platform
        if (
            allow_automatic_fp32_where
            and dispatching_overrides is not None
            and where.name not in dispatching_overrides
            and where.platform == TargetPlatform.FP32
        ):
            fusion_platform = None
        if not _rewire_where_to_add(
            graph,
            where,
            softmax,
            scores_output,
            mask_tensor,
            fusion_platform,
        ):
            continue
        rewired += 1
    return rewired


def _rewire_where_to_add(
    graph: BaseGraph,
    where: Operation,
    softmax: Operation,
    scores_output: Variable,
    mask_tensor: torch.Tensor,
    fusion_platform: Optional[TargetPlatform],
) -> bool:
    """Replace ``scores_op → Where → softmax`` with ``scores_op → Add(mask) → softmax``.

    - Creates a ``Constant`` op whose output holds the folded fp32 mask.
      Uses Constant (rather than a raw parameter variable) because the
      downstream ``_detach_op`` → ``remove_operation`` pipeline treats
      ``is_parameter=True`` inputs as exclusively-owned by the op being
      removed and zeroes their ``.value`` / clears ``.dest_ops``. A
      Constant's output is an activation-shaped variable, which
      survives Add's later removal and stays consumable by the fused
      Attention as its 4th input.
    - Creates an ``Add`` op consuming ``scores_output`` and the constant's
      output, stamped ``TargetPlatform.FP32`` so the quantizer does not
      try to int8 it (mask has -inf and won't survive int8 rounding).
    - Reuses ``where.outputs[0]`` in place as the ``Add`` output so a
      declared graph output on the masked logits is never dropped.
    - Detaches ``where`` from the graph.
    """
    if softmax.inputs[0] is not where.outputs[0]:
        return False

    mask_tensor_final = mask_tensor.detach().clone().contiguous().to(torch.float32)
    mask_const_out = graph.create_variable(
        name=f"attention_mask_from_{where.name}",
        value=mask_tensor_final,
        is_parameter=False,
    )
    mask_const_out.shape = list(mask_tensor_final.shape)
    mask_const_out.dtype = DataType.FP32
    mask_const_op = graph.create_operation(
        op_type="Constant",
        name=f"attention_mask_const_{where.name}",
        attributes={"value": mask_tensor_final},
        platform=TargetPlatform.FP32,
        inputs=[],
        outputs=[mask_const_out],
    )
    mask_const_out.source_op = mask_const_op

    # Unlink where_out from Where before removing Where so remove_operation
    # does not clear where_out.source_op, and reuse where_out as the Add
    # output so any graph.outputs entry on where_out stays intact.
    where_out = where.outputs[0]
    where.outputs.remove(where_out)
    where_out.source_op = None

    for iv in list(where.inputs):
        if where in iv.dest_ops:
            iv.dest_ops.remove(where)
        if iv is not scores_output and not iv.dest_ops and not _var_is_graph_output(graph, iv):
            src = iv.source_op
            if src is not None and src.type == "Constant":
                _detach_op(graph, src)
            elif iv.is_parameter and iv.name in graph.variables and iv.name not in graph.inputs:
                graph.remove_variable(iv)
    where.inputs.clear()
    if where.name in graph.operations:
        try:
            graph.remove_operation(where, keep_coherence=False, remove_unlinked_variable=False)
        except Exception:  # noqa: BLE001 - remove_operation is defensive
            graph.operations.pop(where.name, None)

    add_op = graph.create_operation(
        op_type="Add",
        attributes={},
        platform=TargetPlatform.FP32,
        inputs=[scores_output, mask_const_out],
        outputs=[where_out],
    )
    add_op.set_extension_attrib(_FOLDED_WHERE_MASK_ADD_KEY, fusion_platform)
    where_out.source_op = add_op
    if softmax not in where_out.dest_ops:
        where_out.dest_ops.append(softmax)
    return True


def _report_unfused_where_masks(graph: BaseGraph) -> None:
    """Record Where-masks that survived the constant-fold pass.

    These require a runtime ``Where`` kernel on device, which esp-dl
    does not yet ship. Users hitting this should re-export with the
    additive fp32 mask baked into the graph, or with an explicit
    ``attn_mask`` argument.
    """
    misses = 0
    for op in graph.operations.values():
        if op.type != "MatMul":
            continue
        scale_end = _mul_div_pre_softmax(op) or op
        where = _where_as_mask_pre_softmax(scale_end)
        if where is None:
            continue
        misses += 1
    if misses:
        _report(graph).unfolded_where += misses
        logger.debug(
            f'[espdl-attention-fusion] detected {misses} MatMul->[scale?]->Where(mask)'
            f'->Softmax->MatMul pattern(s) that could not be constant-folded'
        )


# pattern_matching walks exact edges; Mul/Div scale needs its own pattern.
_SCALE_SPELLINGS = (None, "Mul", "Div")


def _fuse_primitive_pattern(graph: BaseGraph) -> int:
    """Fuse ``MatMul -> [scale?] -> Add(mask) -> Softmax -> MatMul``."""
    return _run_fuse(graph, has_mask=True)


def _fuse_no_mask_primitive_pattern(graph: BaseGraph) -> int:
    """Fuse ``MatMul -> [scale?] -> Softmax -> MatMul`` (no mask)."""
    return _run_fuse(graph, has_mask=False)


def _run_fuse(graph: BaseGraph, has_mask: bool) -> int:
    search = SearchableGraph(graph=graph)
    seen = set()
    fused = 0
    mask_pattern = ["Add"] if has_mask else []
    for scale_type in _SCALE_SPELLINGS:
        patterns = [
            "MatMul",
            *([scale_type] if scale_type else []),
            *mask_pattern,
            "Softmax",
            "MatMul",
        ]
        edges = [[i, i + 1] for i in range(len(patterns) - 1)]
        for match in search.pattern_matching(patterns=patterns, edges=edges, exclusive=False):
            if any(graph.operations.get(op.name) is not op for op in match):
                continue
            qk_matmul = match[0]
            pv_matmul = match[-1]
            softmax = match[-2]
            add = match[-3] if has_mask else None
            key = (qk_matmul.name, softmax.name, pv_matmul.name)
            if key in seen:
                continue
            seen.add(key)
            fused += _try_fuse(graph, qk_matmul, add, softmax, pv_matmul)
    return fused


def _pattern_rank(match: _AttentionPrimitiveMatch) -> int:
    """Rank this pattern claims from its K-transpose topology.

    A last-two-axis swap's perm length is that rank, so length 2 stays
    MatMul/Softmax while length 3 and 4 are candidates once shapes are known.
    Dynamo SDPA head-merge (``Reshape -> Transpose([0, 2, 1]) -> Reshape``)
    and dynamo MultiheadAttention's already-transposed K^T ``[B, H, D, L]``
    are 4-D layouts; ``rewrite_to_bhnd`` carries perm length 3 or 4.
    """
    if match.k_transpose_mode in ("head_merge", "kt_4d"):
        return 4
    perm = list(match.k_transpose.attributes.get("perm", []) or [])
    return len(perm)


def _primitive_doomed_ops(
    graph: BaseGraph,
    match: _AttentionPrimitiveMatch,
) -> Tuple[Operation, ...]:
    drop_k_transpose = match.k_transpose_mode == "drop" and len(match.k_transpose.outputs[0].dest_ops) == 1
    drop_k_transpose = drop_k_transpose and not _var_is_graph_output(graph, match.k_transpose.outputs[0])

    doomed = match.head_merge_ops + match.prescale_ops + (match.qk_matmul,)
    if match.scale_op is not None:
        doomed = doomed + (match.scale_op,)
    if match.add is not None:
        doomed = doomed + (match.add,)
    doomed = doomed + (match.softmax, match.pv_matmul)
    if drop_k_transpose:
        doomed = (match.k_transpose,) + doomed
    return doomed


def _primitive_platform_mismatch_reason(
    match: _AttentionPrimitiveMatch,
    doomed: Tuple[Operation, ...],
) -> str:
    mismatched = []
    for op in doomed:
        platform = op.platform
        if op is match.add and _FOLDED_WHERE_MASK_ADD_KEY in op.extension_attrib:
            platform = op.extension_attrib[_FOLDED_WHERE_MASK_ADD_KEY]
        if platform is not None and platform != match.qk_matmul.platform:
            mismatched.append((op, platform))
    if not mismatched:
        return ""
    assignments = ", ".join(
        [f"{match.qk_matmul.type}={match.qk_matmul.platform.name}"]
        + [f"{op.type}={platform.name}" for op, platform in mismatched]
    )
    return f"matched runtime operators use different platforms ({assignments})"


def _try_fuse(
    graph: BaseGraph,
    qk_matmul: Operation,
    add: Optional[Operation],
    softmax: Operation,
    pv_matmul: Operation,
) -> int:
    """Fuse one match. Returns 1 when an Attention node is written, else 0."""
    match = _match_primitive(graph, qk_matmul, add, softmax, pv_matmul)
    if match is None:
        return 0
    doomed = _primitive_doomed_ops(graph, match)
    # Perm length rejects rank 2. 3-D and 4-D layouts fuse only after Q, K, V,
    # and a connected mask all have shapes; a missing shape stays MatMul/Softmax.
    rank = _pattern_rank(match)
    shapes = match.qkv_shapes()
    mask_shape = None if match.mask is None else _shape(match.mask)
    known_bad_rank = any(shape is not None and len(shape) != rank for shape in shapes)
    shapes_ready = all(shape is not None for shape in shapes) and (match.mask is None or mask_shape is not None)
    if rank not in (3, 4):
        reason = f"matched attention rank {rank} is not 3-D or 4-D"
        actionable = False
    elif not shapes_ready:
        reason = "Q/K/V or attn_mask shape is missing"
        actionable = False
    elif known_bad_rank or (mask_shape is not None and len(mask_shape) > rank):
        reason = match.ineligible_reason() or f"Q/K/V or attn_mask rank does not match {rank}-D pattern"
        actionable = _is_ceiling_reason(reason)
    else:
        reason = match.ineligible_reason()
        actionable = _is_ceiling_reason(reason)
    if not reason:
        reason = _primitive_platform_mismatch_reason(match, doomed)
        actionable = False
    if reason:
        _report(graph).record_left(match.name, reason, actionable)
        logger.debug(f'[espdl-attention-fusion] {match.name}: left as MatMul/Softmax/MatMul — {reason}')
        return 0
    fused = _rewrite_primitive(graph, match, doomed)
    _report(graph).fused += 1
    logger.debug(
        f'[espdl-attention-fusion] fused MatMul->'
        f"{'{Mul/Div}->' if match.scale_op else ''}"
        f"{'Add->' if match.add is not None else ''}"
        f'Softmax->MatMul into {fused.name} (scale={match.scale})'
    )
    return 1


def _is_untransposed_bnhd_reshape(var: Optional[Variable]) -> bool:
    """True when ``var`` is produced directly by a 3-D ``[B, N, H*D]`` -> 4-D
    ``[B, N, H, D]`` Reshape (splitting the hidden axis into ``(H, D)``) without
    a subsequent ``Transpose(perm=[0, 2, 1, 3])`` into BHND.
    """
    cur = var
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        src = getattr(cur, "source_op", None)
        if src is None or not getattr(src, "inputs", None):
            return False
        if src.type == "Identity":
            cur = src.inputs[0]
            continue
        if src.type == "Reshape":
            in_shape = _shape(src.inputs[0])
            out_shape = _shape(cur)
            if (
                in_shape is not None
                and out_shape is not None
                and len(in_shape) == 3
                and len(out_shape) == 4
                and out_shape[1] > 1
                and out_shape[2] > 1
                and in_shape[0] == out_shape[0]
                and in_shape[1] == out_shape[1]
                and in_shape[2] == out_shape[2] * out_shape[3]
            ):
                return True
        return False
    return False


def _const_int_vector(var: Variable) -> Optional[List[int]]:
    """Integer vector stored on ``var``, or None."""
    static = _var_static_value(var)
    if static is None:
        return None
    flat = static.detach().reshape(-1)
    if flat.numel() == 0:
        return None
    out: List[int] = []
    for item in flat.tolist():
        if isinstance(item, float) and not float(item).is_integer():
            return None
        out.append(int(item))
    return out


def _eval_scale_factor(var: Variable, seen: Optional[set] = None) -> Optional[float]:
    """Fold a scale factor that is a constant, or Shape/Sqrt/Div/Cast of one.

    PyTorch's SDPA lowering multiplies Q and K^T by ``sqrt(scale)`` before
    the score MatMul. The legacy exporter builds that factor from ``Shape``;
    the dynamo exporter bakes a scalar constant. Both have to become the
    single ``scale`` attribute.
    """
    if var is None:
        return None
    seen = set() if seen is None else seen
    if id(var) in seen:
        return None
    seen.add(id(var))

    direct = _get_const_scalar(var)
    if direct is not None:
        return direct
    static = _var_static_value(var)
    if static is not None and int(static.numel()) == 1:
        try:
            return float(static.reshape(-1)[0].item())
        except (RuntimeError, ValueError, TypeError):
            return None

    src = getattr(var, "source_op", None)
    if src is None or not src.inputs:
        return None
    if src.type in ("Cast", "Identity"):
        return _eval_scale_factor(src.inputs[0], seen)
    if src.type == "Sqrt":
        val = _eval_scale_factor(src.inputs[0], seen)
        if val is None or val <= 0 or not math.isfinite(val):
            return None
        return math.sqrt(val)
    if src.type == "Div" and len(src.inputs) == 2:
        left = _eval_scale_factor(src.inputs[0], seen)
        right = _eval_scale_factor(src.inputs[1], seen)
        if left is None or right is None or right == 0:
            return None
        return left / right
    if src.type == "Slice":
        return _eval_shape_slice(src)
    return None


def _eval_shape_slice(slice_op: Operation) -> Optional[float]:
    """One element of ``Shape(x)`` when ``x``'s shape is already known."""
    if len(slice_op.inputs) < 3:
        return None
    shaped = slice_op.inputs[0]
    shape_src = getattr(shaped, "source_op", None)
    if shape_src is None or shape_src.type != "Shape" or not shape_src.inputs:
        return None
    dims = _shape(shape_src.inputs[0])
    if dims is None:
        return None
    starts = _const_int_vector(slice_op.inputs[1])
    ends = _const_int_vector(slice_op.inputs[2])
    if starts is None or ends is None or len(starts) != 1 or len(ends) != 1:
        return None
    if len(slice_op.inputs) > 3:
        axes = _const_int_vector(slice_op.inputs[3])
        if axes not in (None, [0]):
            return None
    n = len(dims)
    start, end = starts[0], ends[0]
    if start < 0:
        start += n
    if end < 0:
        end += n
    start = min(max(start, 0), n)
    end = min(max(end, 0), n)
    if end - start != 1:
        return None
    return float(dims[start])


def _unwrap_prescale(
    graph: BaseGraph, var: Variable, consumer: Operation
) -> Optional[Tuple[Variable, float, Optional[Operation]]]:
    """Drop a scalar ``Mul`` in front of ``consumer``.

    Returns ``(data, factor, mul)``. ``factor`` is ``1`` and ``mul`` is
    ``None`` when there is no Mul. Returns ``None`` when a Mul is present
    but its factor is not a positive finite scalar, or the Mul feeds
    something else: folding it would change that other user.
    """
    src = getattr(var, "source_op", None)
    if src is None or src.type != "Mul" or len(src.inputs) != 2:
        return var, 1.0, None
    if not _has_only_expected_consumer(var, consumer) or _var_is_graph_output(graph, var):
        return None
    folded = [_eval_scale_factor(inp) for inp in src.inputs]
    if folded[0] is not None and folded[1] is None:
        data, factor = src.inputs[1], folded[0]
    elif folded[1] is not None and folded[0] is None:
        data, factor = src.inputs[0], folded[1]
    else:
        return None
    if not math.isfinite(factor) or factor <= 0:
        return None
    return data, factor, src


def _match_head_merge(graph: BaseGraph, data: Variable) -> Optional[Tuple[Operation, Variable, Tuple[Operation, ...]]]:
    """``Reshape -> Transpose([0, 2, 1]) -> Reshape`` used by dynamo SDPA.

    That 3-D transpose is how the exporter writes a 4-D last-two swap after
    merging the head into the batch. Dynamo refuses to export SDPA unless
    Q, K and V are 4-D, so this chain is a 4-D K^T.
    """
    outer = getattr(data, "source_op", None)
    if outer is None or outer.type != "Reshape" or not outer.inputs:
        return None
    if len(outer.outputs[0].dest_ops) != 1 or _var_is_graph_output(graph, outer.outputs[0]):
        return None
    transpose = getattr(outer.inputs[0], "source_op", None)
    if transpose is None or transpose.type != "Transpose":
        return None
    if list(transpose.attributes.get("perm", []) or []) != [0, 2, 1]:
        return None
    if len(transpose.outputs[0].dest_ops) != 1 or _var_is_graph_output(graph, transpose.outputs[0]):
        return None
    inner = getattr(transpose.inputs[0], "source_op", None)
    if inner is None or inner.type != "Reshape" or not inner.inputs:
        return None
    if len(inner.outputs[0].dest_ops) != 1 or _var_is_graph_output(graph, inner.outputs[0]):
        return None
    if len(inner.inputs) >= 2 and len(outer.inputs) >= 2:
        inner_len = _shape_vector_length(inner.inputs[1])
        outer_len = _shape_vector_length(outer.inputs[1])
        if (inner_len is not None and inner_len != 3) or (outer_len is not None and outer_len != 4):
            return None
        inner_vec = _const_int_vector(inner.inputs[1])
        if inner_vec is not None and len(inner_vec) == 3:
            # Merging (H, L) into axis 1 writes [0, -1, D] or [B, -1, D],
            # whereas head_merge merges (B, H) into axis 0 ([B*H, L, D] or [-1, L, D]).
            if inner_vec[1] == -1 and inner_vec[0] != -1:
                return None
        outer_vec = _const_int_vector(outer.inputs[1])
        if outer_vec is not None and len(outer_vec) == 4:
            if outer_vec[0] == 0 and (outer_vec[2] == -1 or outer_vec[3] == -1):
                return None
    k_in_shape = _shape(inner.inputs[0])
    if k_in_shape is not None:
        if len(k_in_shape) != 4:
            return None
        b, h, lk, d = k_in_shape
        inner_out = _shape(inner.outputs[0])
        if inner_out is not None and inner_out != [b * h, lk, d]:
            return None
        trans_out = _shape(transpose.outputs[0])
        if trans_out is not None and trans_out != [b * h, d, lk]:
            return None
        outer_out = _shape(outer.outputs[0])
        if outer_out is not None and outer_out != [b, h, d, lk]:
            return None
    return transpose, inner.inputs[0], (inner, transpose, outer)


def _match_kt_4d(k_data: Variable, q_var: Variable) -> bool:
    """True when ``k_data`` is K^T already, shaped ``[B, H, D, Lk]`` against Q.

    Dynamo's ``MultiheadAttention`` export reaches the scale Mul with a
    4-D reshape of that form, instead of the SDPA ``Reshape -> Transpose
    -> Reshape`` chain. The transpose back to ``[B, H, Lk, D]`` is inserted
    at rewrite.
    """
    k_shape = _shape(k_data)
    q_shape = _shape(q_var)
    if k_shape is None or q_shape is None or len(k_shape) != 4 or len(q_shape) != 4:
        return False
    b, h, _lq, d = q_shape
    if k_shape[:3] != [b, h, d]:
        return False
    lk = k_shape[3]
    outer = getattr(k_data, "source_op", None)
    if outer is not None and outer.type == "Reshape" and outer.inputs:
        in_shape = _shape(outer.inputs[0])
        if in_shape is not None and len(in_shape) == 3 and in_shape != [b * h, d, lk]:
            return False
    return True


def _shape_vector_length(var: Variable) -> Optional[int]:
    """How many dims a Reshape shape-input writes, when that count is static."""
    static = _var_static_value(var)
    if static is not None:
        return int(static.numel())
    src = getattr(var, "source_op", None)
    if src is not None and src.type == "Concat":
        total = 0
        for inp in src.inputs:
            length = _shape_vector_length(inp)
            if length is None:
                return None
            total += length
        return total
    return None


def _softmax_default_axis(softmax: Operation) -> int:
    """ONNX Softmax default: ``-1`` from opset 13, otherwise ``1``."""
    if softmax.opset.onnx_opset_version() >= 13:
        return -1
    return 1


def _match_primitive(
    graph: BaseGraph,
    qk_matmul: Operation,
    add: Optional[Operation],
    softmax: Operation,
    pv_matmul: Operation,
) -> Optional[_AttentionPrimitiveMatch]:
    """The attention reading of this chain, or None if it is something else.

    Structural checks only; device limits are left to the caller. Does not
    modify the graph.
    """
    if qk_matmul.num_of_parameter != 0 or pv_matmul.num_of_parameter != 0:
        return None
    if len(qk_matmul.inputs) != 2 or len(pv_matmul.inputs) != 2 or not softmax.inputs:
        return None
    if pv_matmul.inputs[0].source_op is not softmax:
        return None
    # Opset < 13 defaults axis to 1 and softmaxes every dim from there on.
    # Attention only normalizes the last dim, so that default is not a match
    # unless the rank is known and the last dim is axis 1.
    axis = softmax.attributes.get("axis", _softmax_default_axis(softmax))
    score_shape = _shape(qk_matmul.outputs[0]) or _shape(qk_matmul.inputs[0])
    last_axis = len(score_shape or []) - 1
    if axis not in (-1, last_axis):
        return None

    scale: Optional[float] = None
    scale_op = _mul_div_pre_softmax(qk_matmul)
    if scale_op is not None:
        other_idx = 1 if scale_op.inputs[0].source_op is qk_matmul else 0
        const_val = _eval_scale_factor(scale_op.inputs[other_idx])
        if const_val is None:
            return None
        if scale_op.type == "Mul":
            scale = const_val
        else:
            scale = float("inf") if const_val == 0 else (1.0 / const_val)
    scores_op: Operation = scale_op if scale_op is not None else qk_matmul

    if add is not None:
        if softmax.inputs[0].source_op is not add:
            return None
        if scores_op.outputs[0] not in add.inputs:
            return None
    else:
        if softmax.inputs[0].source_op is not scores_op:
            return None

    mask_var: Optional[Variable] = None
    if add is not None:
        for var in add.inputs:
            if var.source_op is scores_op:
                continue
            mask_var = var
            break
        if mask_var is None:
            return None

    # Q and K may each be multiplied by sqrt(scale) before the score MatMul.
    # The product of those factors is the Attention scale.
    left = _unwrap_prescale(graph, qk_matmul.inputs[0], qk_matmul)
    right = _unwrap_prescale(graph, qk_matmul.inputs[1], qk_matmul)
    if left is None or right is None:
        return None
    q_var, left_factor, left_mul = left
    k_data, right_factor, right_mul = right
    prescale_ops = tuple(op for op in (left_mul, right_mul) if op is not None)
    if prescale_ops:
        prescale = left_factor * right_factor
        scale = prescale if scale is None else scale * prescale

    # fused op computes K^T itself. Input 1 is K^T; input 0 is Q.
    head_merge = _match_head_merge(graph, k_data)
    if head_merge is not None:
        k_transpose, k_var, head_merge_ops = head_merge
        k_transpose_mode = "head_merge"
    else:
        k_transpose, k_transpose_mode = _match_k_transpose(k_data)
        if k_transpose is not None:
            k_var = k_transpose.inputs[0]
            head_merge_ops = ()
        elif _match_kt_4d(k_data, q_var):
            # No Transpose node: K^T is the tensor itself. ``k_transpose`` stays unused.
            k_transpose = qk_matmul
            k_var = k_data
            k_transpose_mode = "kt_4d"
            head_merge_ops = ()
        else:
            return None
    v_var = pv_matmul.inputs[1]
    if _match_k_transpose(q_var)[0] is not None or _match_head_merge(graph, q_var) is not None:
        return None
    if _match_k_transpose(v_var)[0] is not None or _match_head_merge(graph, v_var) is not None:
        return None
    if k_transpose_mode == "rewrite_to_bhnd" and (
        len(k_transpose.outputs[0].dest_ops) != 1 or _var_is_graph_output(graph, k_transpose.outputs[0])
    ):
        return None
    if not _interior_ok(graph, qk_matmul, scores_op, add, softmax):
        return None
    return _AttentionPrimitiveMatch(
        qk_matmul=qk_matmul,
        scale_op=scale_op,
        add=add,
        softmax=softmax,
        pv_matmul=pv_matmul,
        k_transpose=k_transpose,
        k_transpose_mode=k_transpose_mode,
        scale=scale,
        mask=mask_var,
        q_var=q_var,
        k_var=k_var,
        v_var=v_var,
        k_data=k_data,
        prescale_ops=prescale_ops,
        head_merge_ops=head_merge_ops,
    )


def _rewrite_primitive(
    graph: BaseGraph,
    match: _AttentionPrimitiveMatch,
    doomed: Tuple[Operation, ...],
) -> Operation:
    """Replace an accepted match with one Attention op."""
    k_transpose = match.k_transpose
    if match.k_transpose_mode == "rewrite_to_bhnd":
        k_var = _rewrite_k_transpose_for_bhnd(k_transpose)
    elif match.k_transpose_mode == "head_merge":
        k_var = match.k_var
    elif match.k_transpose_mode == "kt_4d":
        kt_shape = _shape(match.k_var)
        out_shape = None if kt_shape is None else [kt_shape[0], kt_shape[1], kt_shape[3], kt_shape[2]]
        k_var = _emit_op(
            graph,
            "Transpose",
            [match.k_var],
            {"perm": [0, 1, 3, 2]},
            out_shape=out_shape,
            platform=match.qk_matmul.platform,
        )
    else:  # k_transpose_mode == "drop"
        k_var = match.k_var

    out_var = match.pv_matmul.outputs[0]
    q_shape, k_shape, v_shape = match.qkv_shapes()
    assert q_shape is not None and k_shape is not None and v_shape is not None
    is_3d = len(q_shape) == 3 and len(k_shape) == 3 and len(v_shape) == 3
    q_in = _reshape(graph, match.q_var, [1, *q_shape], platform=match.qk_matmul.platform) if is_3d else match.q_var
    k_in = _reshape(graph, k_var, [1, *k_shape], platform=match.qk_matmul.platform) if is_3d else k_var
    v_in = _reshape(graph, match.v_var, [1, *v_shape], platform=match.qk_matmul.platform) if is_3d else match.v_var
    if not is_3d:
        if q_in.shape is None:
            q_in.shape = list(q_shape)
        if k_in.shape is None:
            k_in.shape = list(k_shape)
        if v_in.shape is None:
            v_in.shape = list(v_shape)
    if match.mask is not None and match.mask.shape is None:
        mask_shape = _shape(match.mask)
        if mask_shape is not None:
            match.mask.shape = list(mask_shape)
    if out_var.shape is None:
        out_var.shape = (
            [q_shape[0], q_shape[1], v_shape[2]] if is_3d else [q_shape[0], q_shape[1], q_shape[2], v_shape[3]]
        )

    attn_out = graph.create_variable() if is_3d else out_var
    if is_3d:
        attn_out.shape = [1, q_shape[0], q_shape[1], v_shape[2]]
        attn_out.dtype = out_var.dtype
    inputs = [q_in, k_in, v_in]
    if match.mask is not None:
        inputs.append(match.mask)
    fused = graph.create_operation(
        op_type=ATTENTION,
        attributes=match.attributes,
        inputs=inputs,
        outputs=[attn_out],
        platform=match.qk_matmul.platform,
    )
    attn_out.source_op = fused
    if is_3d:
        _reshape(
            graph,
            attn_out,
            [q_shape[0], q_shape[1], v_shape[2]],
            platform=match.qk_matmul.platform,
            output=out_var,
        )
    for old_op in doomed:
        _detach_op(graph, old_op, keep_output=(old_op is match.pv_matmul))
    return fused


def _interior_ok(
    graph: BaseGraph,
    qk_matmul: Operation,
    scores_op: Operation,
    add: Optional[Operation],
    softmax: Operation,
) -> bool:
    """No interior value may have an outside consumer or be a graph output."""
    if not _has_only_expected_consumer(
        qk_matmul.outputs[0], scores_op if scores_op is not qk_matmul else (add or softmax)
    ):
        return False
    if _var_is_graph_output(graph, qk_matmul.outputs[0]):
        return False

    if scores_op is not qk_matmul:
        next_consumer = add if add is not None else softmax
        if not _has_only_expected_consumer(scores_op.outputs[0], next_consumer):
            return False
        if _var_is_graph_output(graph, scores_op.outputs[0]):
            return False

    if add is not None:
        if not _has_only_expected_consumer(add.outputs[0], softmax):
            return False
        if _var_is_graph_output(graph, add.outputs[0]):
            return False

    if not all(cons.type == "MatMul" for cons in softmax.outputs[0].dest_ops):
        return False
    if len(softmax.outputs[0].dest_ops) != 1:
        return False
    if _var_is_graph_output(graph, softmax.outputs[0]):
        return False
    return True


def _detach_op(graph: BaseGraph, op: Operation, keep_output: bool = False) -> None:
    """Detach ``op`` from graph, unlinking its I/O variables."""
    for iv in list(op.inputs):
        try:
            if op in iv.dest_ops:
                iv.dest_ops.remove(op)
        except Exception:
            pass
        # remove_operation clears and drops every parameter input, even one
        # the fused Attention now consumes (e.g. a constant attn_mask) or one
        # still exposed on the graph boundary.
        if iv.is_parameter and (iv.dest_ops or iv.name in graph.inputs or _var_is_graph_output(graph, iv)):
            op.inputs.remove(iv)
        elif (
            not iv.is_parameter
            and iv.source_op is None
            and not iv.dest_ops
            and iv.name in graph.variables
            and iv.name not in graph.inputs
            and iv.name not in graph.outputs
        ):
            # Interior edge whose producer was detached first (e.g. Mul -> Add).
            graph.remove_variable(iv)

    for ov in list(op.outputs):
        if keep_output:
            op.outputs.remove(ov)
            continue
        if ov.source_op is op:
            ov.source_op = None
        if (
            ov.source_op is None
            and not ov.dest_ops
            and ov.name in graph.variables
            and not _var_is_graph_output(graph, ov)
        ):
            try:
                graph.remove_variable(ov)
            except Exception:
                pass

    if op.name in graph.operations:
        try:
            graph.remove_operation(op, keep_coherence=False, remove_unlinked_variable=False)
        except Exception:
            graph.operations.pop(op.name, None)


# Attribute names carried only by opset-23 / com.microsoft nodes; a node built
# by primitive fusion stamps just ``scale``. Used to pick the remediation.
_NON_FUSION_MARKER_ATTRS = frozenset({"qkv_hidden_sizes", "num_heads", "q_num_heads", "kv_num_heads"})

# Layouts we can restore with a Transpose; anything else stays a hard error.
_DECOMPOSABLE_LAYOUTS = frozenset({"BHND", "BNHD"})
_DECOMPOSABLE_MASK_DTYPES = frozenset({DataType.BOOL, DataType.FP16, DataType.FP32, DataType.FP64})


def _ineligible_attention_message(op: Operation, reason: str, qkv_shapes: List[Optional[List[int]]]) -> str:
    """Diagnostic naming the node, the rule it breaks, and a way out."""
    if "qkv_hidden_sizes" in (op.attributes or {}):
        remediation = (
            "This is com.microsoft.Attention (packed QKV projection). "
            "esp-dl has no kernel for that operator; re-export as "
            "ai.onnx::Attention or as a MatMul / Softmax / MatMul subgraph."
        )
    elif _NON_FUSION_MARKER_ATTRS & set(op.attributes or {}):
        remediation = (
            "This native ai.onnx::Attention node could not be expanded to "
            "MatMul / Softmax (KV-cache, softcap, runtime bool mask, "
            "q_num_heads/kv_num_heads disagreeing with the 4-D head axis, or "
            "unranked Q/K/V). Disabling primitive attention fusion has no bearing on it. "
            "Drop the unsupported feature or export with opset < 23 so "
            "PyTorch emits the primitive decomposition."
        )
    else:
        remediation = (
            "The node carries only ``scale``. Primitive fusion refuses "
            "ineligible matches, so this one was constructed directly or "
            "rewritten by an earlier pass rather than produced by fusion. "
            "Reshape the block, or emit MatMul / Softmax / MatMul yourself."
        )
    return (
        f"Attention {op.name} cannot run on esp-dl: {reason}. "
        f"Q/K/V shapes are {qkv_shapes}. esp-dl runs a subset of "
        f"ai.onnx::Attention. {remediation}"
    )


def _is_microsoft_attention(op: Operation) -> bool:
    """True for packed-QKV ``com.microsoft.Attention``, which we do not expand."""
    attrs = op.attributes or {}
    if "qkv_hidden_sizes" in attrs:
        return True
    if "q_num_heads" in attrs or "kv_num_heads" in attrs:
        return False
    if op.num_of_input < 2:
        return False
    weight = op.inputs[1]
    if not getattr(weight, "is_parameter", False):
        return False
    shape = _shape(weight)
    return shape is not None and len(shape) == 2


def _static_int_shape(shape: Optional[Sequence]) -> Optional[List[int]]:
    if shape is None or len(shape) not in (3, 4):
        return None
    out: List[int] = []
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            return None
        out.append(int(dim))
    return out


def _ensure_in_graph(graph: BaseGraph, var: Variable) -> None:
    if var.name not in graph.variables:
        graph.append_variable(var)


def _emit_op(
    graph: BaseGraph,
    op_type: str,
    inputs: List[Variable],
    attributes: Optional[dict] = None,
    out_shape: Optional[List[int]] = None,
    platform=None,
    output: Optional[Variable] = None,
) -> Variable:
    """Create ``op_type``; return its output variable."""
    for var in inputs:
        _ensure_in_graph(graph, var)
    out = output if output is not None else graph.create_variable()
    _ensure_in_graph(graph, out)
    if out_shape is not None:
        out.shape = list(out_shape)
    if inputs and getattr(inputs[0], "dtype", None) is not None and output is None:
        out.dtype = inputs[0].dtype
    kwargs = {"op_type": op_type, "attributes": dict(attributes or {}), "inputs": list(inputs), "outputs": [out]}
    if platform is not None:
        kwargs["platform"] = platform
    graph.create_operation(**kwargs)
    return out


def _reshape(
    graph: BaseGraph,
    data: Variable,
    shape: List[int],
    platform=None,
    output: Optional[Variable] = None,
) -> Variable:
    shape_var = graph.create_variable(value=torch.tensor(shape, dtype=torch.int64), is_parameter=True)
    shape_var.shape = [len(shape)]
    shape_var.dtype = DataType.INT64
    return _emit_op(
        graph, "Reshape", [data, shape_var], {"allowzero": 0}, out_shape=shape, platform=platform, output=output
    )


def _transpose(graph: BaseGraph, data: Variable, perm: List[int], out_shape: List[int], platform=None) -> Variable:
    return _emit_op(graph, "Transpose", [data], {"perm": list(perm)}, out_shape=out_shape, platform=platform)


def _concat_repeat(
    graph: BaseGraph, data: Variable, axis: int, times: int, out_shape: List[int], platform=None
) -> Variable:
    if times == 1:
        return data
    return _emit_op(graph, "Concat", [data] * times, {"axis": int(axis)}, out_shape=out_shape, platform=platform)


def _repeat_kv_heads(
    graph: BaseGraph,
    data: Variable,
    shape_bhnd: List[int],
    group: int,
    platform=None,
) -> Variable:
    """``repeat_interleave(group, dim=1)`` via Reshape / Concat / Reshape.

    Concat on the head axis would tile (``[h0,h1,h0,h1,...]``); GQA needs
    ``[h0,h0,...,h1,h1,...]``. Flatten heads into batch first to stay 4-D.
    """
    if group == 1:
        return data
    batch, h_kv, length, dim = shape_bhnd
    flat = _reshape(graph, data, [batch * h_kv, 1, length, dim], platform)
    tiled = _concat_repeat(graph, flat, 1, group, [batch * h_kv, group, length, dim], platform)
    return _reshape(graph, tiled, [batch, h_kv * group, length, dim], platform)


def _to_bhnd(
    graph: BaseGraph,
    var: Variable,
    shape: List[int],
    layout: str,
    num_heads: int,
    platform=None,
) -> Tuple[Variable, List[int]]:
    """Lift ``var`` to BHND ``[B, H, L, D]``."""
    if len(shape) == 4:
        if layout == "BNHD":
            b, length, heads, dim = shape
            bhnd = [b, heads, length, dim]
            return _transpose(graph, var, [0, 2, 1, 3], bhnd, platform), bhnd
        return var, list(shape)
    if len(shape) != 3:
        raise ValueError(f"Attention decompose: expected rank 3 or 4, got {shape}")
    if num_heads <= 0:
        raise ValueError("Attention decompose: 3-D Q/K/V needs q_num_heads / kv_num_heads")
    batch, length, hidden = shape
    if hidden % num_heads != 0:
        raise ValueError(f"Attention decompose: hidden={hidden} not divisible by num_heads={num_heads}")
    dim = hidden // num_heads
    bnhd = [batch, length, num_heads, dim]
    lifted = _reshape(graph, var, bnhd, platform)
    bhnd = [batch, num_heads, length, dim]
    return _transpose(graph, lifted, [0, 2, 1, 3], bhnd, platform), bhnd


def _from_bhnd(
    graph: BaseGraph,
    var: Variable,
    bhnd: List[int],
    orig_rank: int,
    orig_layout: str,
    platform=None,
    output: Optional[Variable] = None,
) -> Variable:
    """Restore the original Q layout/rank from BHND ``var``."""
    b, heads, length, dim = bhnd
    bnhd = [b, length, heads, dim]
    if orig_rank == 4:
        return _emit_op(
            graph,
            "Transpose",
            [var],
            {"perm": [0, 2, 1, 3]},
            out_shape=bnhd,
            platform=platform,
            output=output,
        )
    as_bnhd = _transpose(graph, var, [0, 2, 1, 3], bnhd, platform)
    return _reshape(graph, as_bnhd, [b, length, heads * dim], platform=platform, output=output)


def _mask_is_runtime_bool(mask: Variable) -> bool:
    if _var_static_value(mask) is not None:
        return False
    return mask.dtype == DataType.BOOL


def _additive_mask(graph: BaseGraph, mask: Variable, platform=None) -> Variable:
    """Return a float additive mask, converting a constant bool mask."""
    value = _var_static_value(mask)
    if value is not None and value.dtype == torch.bool:
        additive = torch.zeros(value.shape, dtype=torch.float32)
        additive.masked_fill_(~value, float("-inf"))
        src = mask.source_op
        if (
            src is not None
            and src.type == "Constant"
            and len(mask.dest_ops) <= 1
            and not _var_is_graph_output(graph, mask)
        ):
            _detach_op(graph, src)
        out = graph.create_variable(value=additive, is_parameter=True)
        out.shape = list(additive.shape)
        out.dtype = DataType.FP32
        return out
    return mask


def _decompose_blocker(op: Operation, qkv_shapes: List[Optional[List[int]]]) -> Optional[str]:
    """None if ``op`` can be expanded to MatMul/Softmax; otherwise why not."""
    if _is_microsoft_attention(op):
        return "com.microsoft.Attention packed QKV is not expanded"
    if op.num_of_output > 1:
        return "Attention extra outputs (qk_matmul_output) are not expanded"
    if op.num_of_input > 4:
        return "KV cache (past_key / past_value) is not expanded"
    attrs = op.attributes or {}
    attr_reason = espdl_attention_attributes_ineligible_reason(attrs)
    if attr_reason:
        return attr_reason
    layout = _decode_layout(attrs.get("layout", "BHND")) or "BHND"
    if layout not in _DECOMPOSABLE_LAYOUTS:
        return f"layout={layout!r} is not expanded"
    static = [_static_int_shape(s) for s in qkv_shapes[:3]]
    if any(s is None for s in static):
        return "Q/K/V shapes are not static integers"
    q_shape, k_shape, v_shape = static
    if any(len(s) != len(q_shape) for s in (k_shape, v_shape)) or len(q_shape) not in (3, 4):
        return f"Q/K/V must all be 3-D or all 4-D, got {[len(s) for s in static]}"
    if len(q_shape) == 3:
        head_reason = espdl_attention_head_attrs_ineligible_reason(attrs, static, layout)
        if head_reason:
            return head_reason
        q_heads = int(attrs.get("q_num_heads", 0) or 0)
        kv_heads = int(attrs.get("kv_num_heads", 0) or 0)
        if q_heads <= 0:
            return "3-D Q needs q_num_heads"
        if kv_heads <= 0:
            return "3-D K/V need kv_num_heads"
        if q_shape[2] <= 0 or q_shape[2] % q_heads != 0:
            return f"3-D Q hidden={q_shape[2]} is not divisible by q_num_heads={q_heads}"
        if k_shape[2] <= 0 or k_shape[2] % kv_heads != 0:
            return f"3-D K hidden={k_shape[2]} is not divisible by kv_num_heads={kv_heads}"
        if v_shape[2] <= 0 or v_shape[2] % kv_heads != 0:
            return f"3-D V hidden={v_shape[2]} is not divisible by kv_num_heads={kv_heads}"
        b_q, h_q, l_q, d_q = q_shape[0], q_heads, q_shape[1], q_shape[2] // q_heads
        b_k, h_k, l_k, d_k = k_shape[0], kv_heads, k_shape[1], k_shape[2] // kv_heads
        b_v, h_v, l_v = v_shape[0], kv_heads, v_shape[1]
    else:
        # A contradictory head attribute must not be expanded away. Expansion
        # follows the tensor shape and would drop the attribute, which is the
        # same silent ignore the device export used to do.
        head_reason = espdl_attention_head_attrs_ineligible_reason(attrs, static, layout)
        if head_reason:
            return head_reason
        head_axis, seq_axis = _layout_head_seq_axes(layout)
        b_q, h_q, l_q, d_q = q_shape[0], q_shape[head_axis], q_shape[seq_axis], q_shape[3]
        b_k, h_k, l_k, d_k = k_shape[0], k_shape[head_axis], k_shape[seq_axis], k_shape[3]
        b_v, h_v, l_v = v_shape[0], v_shape[head_axis], v_shape[seq_axis]
    if op.num_of_input >= 4 and op.inputs[3] is not None:
        mask = op.inputs[3]
        if _mask_is_runtime_bool(mask):
            return "runtime bool attn_mask is not expanded (no Where kernel)"
        if mask.dtype is not None and mask.dtype not in _DECOMPOSABLE_MASK_DTYPES:
            dtype_name = getattr(mask.dtype, "name", str(mask.dtype))
            return f"{dtype_name} attn_mask is neither bool nor float (ONNX Attention has no integer mask)"
        # A mask that cannot broadcast onto [B, Hq, Lq, Lk] is not a
        # fused-kernel limitation, it breaks ai.onnx::Attention itself. Do not
        # expand it either: the replacement Add would numpy-broadcast the
        # scores to a higher rank and the graph would fail somewhere further
        # downstream, where the mask is no longer in the picture.
        mask_reason = espdl_attention_mask_ineligible_reason(_shape(mask), b_q, h_q, l_q, l_k)
        if mask_reason:
            return mask_reason
    if b_q != b_k or b_q != b_v:
        return "batch mismatch"
    if h_k != h_v or l_k != l_v:
        return "K/V disagree on (Hkv, Lk)"
    if d_q != d_k:
        return "Q/K head_dim mismatch"
    if h_k < 1 or h_q % h_k != 0:
        return "Hq is not a multiple of Hkv"
    return None


def _nonfinite_mask(mask: Variable) -> bool:
    """True when an additive mask is bool or contains ``-inf`` and cannot be quantized.

    A Where fold stores the mask on a Constant output (``is_parameter`` is false).
    That tensor still cannot be quantized, so it is read the same way as a parameter.
    """
    value = _var_static_value(mask)
    if value is None:
        return False
    if value.dtype == torch.bool:
        return True
    return bool(value.is_floating_point() and torch.isinf(value).any())


def _decompose_attention(graph: BaseGraph, op: Operation, *, quantized: bool = False) -> None:
    """Replace ``op`` with ``MatMul(Q, K^T) -> [Mul] -> [Add] -> Softmax -> MatMul``.

    Compute ops keep ``op.platform`` so a later quantizer can quantize them.
    A ``-inf`` / bool mask Add stays FP32 when ``quantized`` is set: that constant
    has no integer range, and the MatMul / Softmax beside it still do.
    """
    attrs = op.attributes or {}
    platform = op.platform
    layout = _decode_layout(attrs.get("layout", "BHND")) or "BHND"
    q_var, k_var, v_var = op.inputs[0], op.inputs[1], op.inputs[2]
    q_shape = _static_int_shape(_shape(q_var))
    k_shape = _static_int_shape(_shape(k_var))
    v_shape = _static_int_shape(_shape(v_var))
    assert q_shape is not None and k_shape is not None and v_shape is not None

    q_heads = int(attrs.get("q_num_heads", 0) or 0)
    kv_heads = int(attrs.get("kv_num_heads", 0) or 0)
    orig_rank = len(q_shape)

    q_bhnd, qs = _to_bhnd(graph, q_var, q_shape, layout, q_heads, platform)
    k_bhnd, ks = _to_bhnd(graph, k_var, k_shape, layout, kv_heads, platform)
    v_bhnd, vs = _to_bhnd(graph, v_var, v_shape, layout, kv_heads, platform)

    batch, h_q, l_q, d_q = qs
    _, h_kv, l_k, d_k = ks
    d_v = vs[3]
    if d_q != d_k:
        raise ValueError(f"Attention {op.name}: Q/K head_dim mismatch {d_q} vs {d_k}")
    if h_kv < 1 or h_q % h_kv != 0:
        raise ValueError(f"Attention {op.name}: Hq={h_q} is not a multiple of Hkv={h_kv}")
    if h_q != h_kv:
        group = h_q // h_kv
        k_bhnd = _repeat_kv_heads(graph, k_bhnd, [batch, h_kv, l_k, d_k], group, platform)
        v_bhnd = _repeat_kv_heads(graph, v_bhnd, [batch, h_kv, l_k, d_v], group, platform)

    k_t = _transpose(graph, k_bhnd, [0, 1, 3, 2], [batch, h_q, d_k, l_k], platform)
    scores = _emit_op(graph, "MatMul", [q_bhnd, k_t], out_shape=[batch, h_q, l_q, l_k], platform=platform)

    scale = attrs.get("scale")
    scale_val = 1.0 / math.sqrt(d_q) if scale is None else float(scale)
    if scale_val != 1.0:
        scale_var = graph.create_variable(value=torch.tensor(scale_val, dtype=torch.float32), is_parameter=True)
        scale_var.shape = []
        scale_var.dtype = DataType.FP32
        scores = _emit_op(graph, "Mul", [scores, scale_var], out_shape=[batch, h_q, l_q, l_k], platform=platform)

    if op.num_of_input >= 4 and op.inputs[3] is not None:
        raw_mask = op.inputs[3]
        mask_is_nonfinite = _nonfinite_mask(raw_mask)
        mask = _additive_mask(graph, raw_mask, platform)
        add_platform = TargetPlatform.FP32 if quantized and (mask_is_nonfinite or _nonfinite_mask(mask)) else platform
        scores = _emit_op(graph, "Add", [scores, mask], out_shape=[batch, h_q, l_q, l_k], platform=add_platform)

    if int(attrs.get("is_causal", 0) or 0) != 0:
        keep = torch.ones(l_q, l_k, dtype=torch.bool).tril()
        causal = torch.zeros(1, 1, l_q, l_k, dtype=torch.float32)
        causal.masked_fill_(~keep, float("-inf"))
        causal_var = graph.create_variable(value=causal, is_parameter=True)
        causal_var.shape = list(causal.shape)
        causal_var.dtype = DataType.FP32
        add_platform = TargetPlatform.FP32 if quantized else platform
        scores = _emit_op(graph, "Add", [scores, causal_var], out_shape=[batch, h_q, l_q, l_k], platform=add_platform)

    probs = _emit_op(graph, "Softmax", [scores], {"axis": -1}, out_shape=[batch, h_q, l_q, l_k], platform=platform)

    out_var = op.outputs[0]
    op.outputs.remove(out_var)
    out_var.source_op = None

    bhnd_shape = [batch, h_q, l_q, d_v]
    if orig_rank == 4 and layout == "BHND":
        _emit_op(graph, "MatMul", [probs, v_bhnd], out_shape=bhnd_shape, platform=platform, output=out_var)
    else:
        bhnd_out = _emit_op(graph, "MatMul", [probs, v_bhnd], out_shape=bhnd_shape, platform=platform)
        _from_bhnd(graph, bhnd_out, bhnd_shape, orig_rank, layout, platform, output=out_var)

    for iv in list(op.inputs):
        if op in iv.dest_ops:
            iv.dest_ops.remove(op)
        if (
            iv.source_op is None
            and not iv.dest_ops
            and iv.name in graph.variables
            and not _var_is_graph_output(graph, iv)
            and iv.name not in graph.inputs
        ):
            graph.remove_variable(iv)
    op.inputs.clear()
    if op.name in graph.operations:
        graph.remove_operation(op, keep_coherence=False, remove_unlinked_variable=False)


def _is_ceiling_reason(reason: str) -> bool:
    """True iff ``reason`` is a head_dim / Lk ceiling the user can raise.

    Distinguishes "the user could keep fusion by raising both knobs" from "the
    kernel just doesn't support this variant" (GQA grouping, layout, mask
    dtype or shape). Judged on the reason actually reported, not on the shapes:
    a node over the head_dim ceiling that is first rejected for its mask dtype
    gains nothing from raising the knob. The ceiling reasons of
    espdl_attention_shapes_ineligible_reason are the only ones naming
    the device compile definitions.
    """
    return "ESP_DL_ATTENTION_MAX_" in reason


def _unshaped_mask_message(op: Operation) -> str:
    """Diagnostic for a connected attn_mask whose shape never materialised."""
    return (
        f"Attention {op.name} cannot be verified against the esp-dl kernels: "
        f"attn_mask is connected but its shape is still unknown after tracing. "
        f"A missing shape is not an absent mask. The device checks the live tensor "
        f"on the first forward and zeroes the Attention output when the broadcast, "
        f"dtype, or data is invalid. Export the ONNX with a static attn_mask shape, "
        f"or pass ``inputs`` / ``input_shape`` that let shape tracing fill it in."
    )


def _unverifiable_attention_message(op: Operation, qkv_shapes: List[Optional[List[int]]]) -> str:
    """Diagnostic for an Attention node whose Q/K/V shapes never materialised."""
    return (
        f"Attention {op.name} cannot be verified against the esp-dl kernels: "
        f"Q/K/V shapes are still {qkv_shapes} after tracing. Eligibility "
        f"(head_dim, Lk, rank, GQA grouping) is a function of the shapes, so an "
        f"unshaped node is being exported on the assumption that it fits. It may "
        f"not: ``Attention::deserialize`` can only re-check when the .espdl "
        f"carries shapes, and without them the first forward logs an error and "
        f"leaves the output tensor zeroed — a wrong answer rather than a failed "
        f"load. Export the ONNX with static Q/K/V dims (no symbolic axes), or "
        f"pass ``inputs`` / ``input_shape`` that let shape tracing run, so this "
        f"node can be either accepted or decomposed before quantization."
    )


def _compact_constant_mask(graph: BaseGraph, op: Operation) -> bool:
    """Store a constant fp32 ``attn_mask`` holding only ``0`` / ``-inf`` as bool.

    The device reads a bool mask as ``True -> 0``, ``False -> -inf``, so the
    rewrite is exact and cuts the mask to one byte per element. Masks with any
    other value, runtime masks, and masks shared with other consumers or graph
    outputs are left untouched.
    """
    if op.num_of_input < 4 or op.inputs[3] is None:
        return False
    mask = op.inputs[3]
    if len(mask.dest_ops) != 1 or _var_is_graph_output(graph, mask):
        return False
    value = _var_static_value(mask)
    if value is None or not value.is_floating_point() or value.numel() == 0:
        return False
    keep = value == 0
    if not bool((keep | torch.isneginf(value)).all().item()):
        return False
    keep = keep.contiguous()
    if not mask.is_parameter:
        src = mask.source_op
        if src is None or src.type != "Constant" or any(_var_is_graph_output(graph, ov) for ov in src.outputs):
            return False
        src.attributes["value"] = keep
    mask.value = keep
    mask.dtype = DataType.BOOL
    return True


def legalize_espdl_attention_graph(
    graph: BaseGraph, platform: Optional[TargetPlatform] = None, quantized: bool = False, **kwargs
) -> None:
    """Keep a device-runnable Attention node, expand it, or raise.

    Q/K/V and a connected attn_mask must already have shapes. A platform
    without the kernel expands every node. With ``quantized``, only a
    ``-inf`` or bool mask Add stays on ``TargetPlatform.FP32``. A kept
    node's constant ``0`` / ``-inf`` mask is rewritten to bool.
    """
    _ensure_traced_shapes(graph, **kwargs)
    unsupported = espdl_attention_platform_unsupported_reason(platform)
    report = _report(graph)
    for op in list(graph.operations.values()):
        if op.type != ATTENTION:
            continue
        qkv_shapes = [_shape(op.inputs[i]) for i in range(min(3, op.num_of_input))]
        if len(qkv_shapes) < 3 or any(shape is None for shape in qkv_shapes):
            raise ValueError(_unverifiable_attention_message(op, qkv_shapes))
        if _connected_mask_unshaped(op):
            raise ValueError(_unshaped_mask_message(op))
        if unsupported is None:
            if espdl_attention_eligible(op, qkv_shapes):
                report.kept += 1
                if _compact_constant_mask(graph, op):
                    report.compacted += 1
                    logger.debug(f'[espdl-attention-check] {op.name}: constant 0/-inf attn_mask stored as bool')
                continue
        reason = unsupported or espdl_attention_ineligible_reason(op, qkv_shapes)
        blocker = _decompose_blocker(op, qkv_shapes)
        if blocker is not None:
            raise ValueError(_ineligible_attention_message(op, f"{reason}; {blocker}", qkv_shapes))
        _decompose_attention(graph, op, quantized=quantized)
        # A head_dim / Lk above the Python bound gives up the fused kernel even
        # though the device may already be built with a bigger
        # ``ESP_DL_ATTENTION_MAX_*`` ceiling, so the summary warns with both
        # knobs. Other reasons (no kernel on this platform, GQA grouping,
        # layout, mask shape) are not knob-fixable.
        actionable = unsupported is None and _is_ceiling_reason(reason)
        report.record_expanded(op.name, reason or "ineligible for the fused kernel", actionable)
        logger.debug(f'[espdl-attention-check] {op.name}: expanded to MatMul/Softmax ({reason})')
    if quantized:
        for op in graph.operations.values():
            if op.type == "Add" and any(_nonfinite_mask(inp) for inp in op.inputs if inp is not None):
                op.platform = TargetPlatform.FP32
    _log_attention_report(graph)


def finalize_espdl_attention_for_export(graph: BaseGraph) -> None:
    """Check ``logit_scale`` once Q/K scales exist. Does not rewrite the graph.

    Explicit ``FP32`` Attention nodes are accepted when their float ``scale``
    contract holds. Otherwise a missing or unusable Q/K/V/O scale is an error,
    and ``logit_scale`` must be a finite, positive, normal float32. Shape,
    mask, and platform checks stay in ``legalize_espdl_attention_graph``.
    """
    for op in graph.operations.values():
        if op.type != ATTENTION:
            continue
        qkv_shapes = [_shape(op.inputs[i]) for i in range(min(3, op.num_of_input))]
        q_shape = qkv_shapes[0] if qkv_shapes else None
        head_dim = q_shape[3] if q_shape is not None and len(q_shape) > 3 else 0
        logit_reason = espdl_attention_op_logit_scale_ineligible_reason(op, head_dim, require_scales=True)
        if logit_reason:
            raise ValueError(_ineligible_attention_message(op, logit_reason, qkv_shapes))


def _prune_dead_ops(graph: BaseGraph) -> None:
    """Iteratively detach ops whose outputs have no consumers and are not graph outputs."""
    if not graph.outputs:
        return
    changed = True
    while changed:
        changed = False
        for op in list(graph.operations.values()):
            if op.type == ATTENTION or not op.outputs:
                continue
            if all(not v.dest_ops and not _var_is_graph_output(graph, v) for v in op.outputs):
                _detach_op(graph, op)
                changed = True


def fuse_espdl_attention_graph(
    graph: BaseGraph,
    *,
    dispatching_overrides: Optional[Set[str]] = None,
    allow_automatic_fp32_where: bool = False,
    **kwargs,
) -> None:
    """Fuse ``MatMul -> [Mul/Div]? -> [Add]? -> Softmax -> MatMul`` into Attention."""

    _ensure_traced_shapes(graph, **kwargs)
    rewired = _rewrite_where_masks_to_add(
        graph,
        dispatching_overrides=dispatching_overrides,
        allow_automatic_fp32_where=allow_automatic_fp32_where,
    )
    fused_primitive = _fuse_primitive_pattern(graph) + _fuse_no_mask_primitive_pattern(graph)
    if rewired or fused_primitive:
        _prune_dead_ops(graph)
    _report_unfused_where_masks(graph)
    if fused_primitive:
        logger.debug(f'[espdl-attention-fusion] fused {fused_primitive} primitive MatMul->Softmax->MatMul pattern(s)')


class EspdlAttentionFusionPass(QuantizationOptimizationPass):
    """Fuse MatMul->Softmax->MatMul patterns into Attention nodes."""

    def __init__(
        self,
        dispatching_overrides: Optional[Set[str]] = None,
        allow_automatic_fp32_where: bool = False,
    ) -> None:
        super().__init__(name="ESPDL Attention Fusion Pass")
        self.dispatching_overrides = dispatching_overrides
        self.allow_automatic_fp32_where = allow_automatic_fp32_where

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        fuse_espdl_attention_graph(
            graph,
            dispatching_overrides=self.dispatching_overrides,
            allow_automatic_fp32_where=self.allow_automatic_fp32_where,
            **kwargs,
        )


class EspdlAttentionLegalizePass(QuantizationOptimizationPass):
    """Keep, expand, or reject Attention nodes after fusion."""

    def __init__(self, platform: Optional[TargetPlatform] = None, quantized: bool = True) -> None:
        super().__init__(name="ESPDL Attention Legalize Pass")
        self.platform = platform
        self.quantized = quantized

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        legalize_espdl_attention_graph(graph, platform=self.platform, quantized=self.quantized, **kwargs)

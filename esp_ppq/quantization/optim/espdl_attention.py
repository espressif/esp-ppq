"""Prepare native Attention nodes and fuse device-eligible primitive subgraphs."""

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
    lower_espdl_attention_attributes,
)
from esp_ppq.IR import BaseGraph, Operation, Variable
from esp_ppq.IR.search import SearchableGraph
from esp_ppq.log import NaiveLogger

logger = NaiveLogger.get_logger('PPQ')

ATTENTION = "Attention"

# ESP-DL platforms supporting fused Attention.
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
    return f"platform {platform.name} has no Attention kernel in the current implementation"


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
        if dims and all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in dims):
            return dims
    return None


_REPORT_KEY = "espdl_attention_report"
_NONFINITE_MASK_ADD_KEY = "espdl_attention_nonfinite_mask_add"
_DECOMPOSED_ATTENTION_KEY = "espdl_attention_decomposed_softmax"
_PREPARED_ATTENTION_KEY = "espdl_attention_prepared_node"


@dataclass
class _EspdlAttentionReport:
    """What Attention preparation and fusion decided."""

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


def _clear_attention_report(graph: BaseGraph) -> None:
    """Drop the in-graph fusion report. Old PPQ operators do not print a pass summary."""
    attribs = getattr(graph, "extension_attrib", None)
    if attribs is not None:
        attribs.pop(_REPORT_KEY, None)


def _attention_op_shapes_incomplete(op: Operation) -> bool:
    """True when a native ``Attention`` op still lacks Q/K/V or mask shape."""
    if op.type != ATTENTION:
        return False
    mask_unshaped = op.num_of_input == 4 and op.inputs[3] is not None and _shape(op.inputs[3]) is None
    return any(_shape(var) is None for var in op.inputs[:3]) or mask_unshaped


def _attention_shapes_incomplete(graph: BaseGraph) -> bool:
    """True when an Attention node, or a Softmax inside a structural primitive
    attention candidate, still has an unshaped variable."""

    def _is_candidate_scale_factor(var: Optional[Variable], seen: Optional[set] = None) -> bool:
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
                    and inp.dest_ops == [src]
                    and _is_candidate_scale_factor(src.inputs[1 - idx])
                ):
                    return inp_src
        return None

    def _prescale_data_vars(var: Variable, consumer: Operation) -> List[Variable]:
        src = getattr(var, "source_op", None)
        if src is None or src.type != "Mul" or len(src.inputs) != 2:
            return [var]
        if var.dest_ops != [consumer]:
            return []
        candidates = [inp for inp in src.inputs if not getattr(inp, "is_parameter", False)]
        return candidates or list(src.inputs)

    def _softmax_in_candidate(softmax: Operation) -> bool:
        if softmax.type != "Softmax" or softmax.num_of_input < 1 or softmax.num_of_output != 1:
            return False
        if softmax.extension_attrib.get(_DECOMPOSED_ATTENTION_KEY, False):
            return False
        probs_var = softmax.outputs[0]
        if probs_var.name in graph.outputs or len(probs_var.dest_ops) != 1:
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
        if sm_in is None or sm_in.name in graph.outputs or sm_in.dest_ops != [softmax]:
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

        q_candidates = _prescale_data_vars(qk_matmul.inputs[0], qk_matmul)
        k_candidates = _prescale_data_vars(qk_matmul.inputs[1], qk_matmul)
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
                    matched_k_vars.append(k_data)
                elif len(k_transpose.attributes.get("perm", []) or []) in (3, 4):
                    matched_k_vars.extend([k_transpose.inputs[0], k_data])
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

    for op in graph.operations.values():
        if _attention_op_shapes_incomplete(op):
            return True
        if op.type == "Softmax" and _softmax_in_candidate(op):
            return True
    return False


def _ensure_traced_shapes(graph: BaseGraph, **kwargs) -> None:
    """Trace only when the caller passed ``inputs`` and shapes are still missing."""
    if not _attention_shapes_incomplete(graph):
        return
    executor = kwargs.get("executor")
    batch = kwargs.get("inputs")
    if executor is None or batch is None:
        return
    executor.tracing_operation_meta(inputs=batch)


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


def _var_static_value(var: Variable) -> Optional[torch.Tensor]:
    """Return ``var``'s materialized parameter value as a torch tensor, or None."""
    if var is None or not getattr(var, "is_parameter", False):
        return None
    val = getattr(var, "value", None)
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val
    try:
        return torch.as_tensor(val)
    except (TypeError, ValueError):
        return None


def _try_fold_where_mask(where: Operation, scores_output: Variable) -> Optional[torch.Tensor]:
    """Fold ``Where(cond, scores, -inf)`` with static ``cond`` into an additive fp32 mask tensor."""
    if len(where.inputs) != 3:
        return None
    cond_var, x_var, y_var = where.inputs
    if x_var is scores_output and y_var is not scores_output:
        mask_true_is_scores = True
    elif y_var is scores_output and x_var is not scores_output:
        mask_true_is_scores = False
    else:
        return None
    neg_inf_val = _var_static_value(y_var if mask_true_is_scores else x_var)
    if neg_inf_val is None or neg_inf_val.numel() == 0:
        return None
    try:
        if not bool(torch.isneginf(neg_inf_val.reshape(-1).to(torch.float64)).all().item()):
            return None
    except (RuntimeError, TypeError):
        return None
    cond_val = _var_static_value(cond_var)
    if cond_val is None:
        return None
    cond_bool = cond_val.to(torch.bool)
    if mask_true_is_scores:
        return torch.where(
            cond_bool,
            torch.zeros_like(cond_bool, dtype=torch.float32),
            torch.full_like(cond_bool, float("-inf"), dtype=torch.float32),
        )
    return torch.where(
        cond_bool,
        torch.full_like(cond_bool, float("-inf"), dtype=torch.float32),
        torch.zeros_like(cond_bool, dtype=torch.float32),
    )


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

    @property
    def rank(self) -> int:
        if self.k_transpose_mode in ("head_merge", "kt_4d"):
            return 4
        return len(self.k_transpose.attributes.get("perm", []) or [])

    def doomed_ops(self, graph: BaseGraph) -> Tuple[Operation, ...]:
        drop_k_transpose = (
            self.k_transpose_mode == "drop"
            and len(self.k_transpose.outputs[0].dest_ops) == 1
            and self.k_transpose.outputs[0].name not in graph.outputs
        )
        doomed = self.head_merge_ops + self.prescale_ops + (self.qk_matmul,)
        if self.scale_op is not None:
            doomed = doomed + (self.scale_op,)
        if self.add is not None:
            doomed = doomed + (self.add,)
        doomed = doomed + (self.softmax, self.pv_matmul)
        if drop_k_transpose:
            doomed = (self.k_transpose,) + doomed
        return doomed

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
        def _is_untransposed_bnhd_reshape(var: Optional[Variable]) -> bool:
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

        raw_shapes = self.qkv_shapes()
        mask_shape = None if self.mask is None else _shape(self.mask)
        rank = self.rank
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

        # Primitive MatMul attention has no GQA repeat_interleave node, so Q/K/V head counts must match.
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


def _rewrite_where_masks_to_add(graph: BaseGraph) -> int:
    """Fold static ``Where(cond, scores, -inf)`` before ``Softmax`` into ``Add(scores, fp32_mask)``."""
    rewired = 0
    misses = 0
    for op in list(graph.operations.values()):
        if op.type != "MatMul":
            continue
        scale_end = _mul_div_pre_softmax(op) or op
        dst = scale_end.outputs[0].dest_ops
        if len(dst) != 1 or dst[0].type != "Where":
            continue
        where = dst[0]
        if len(where.outputs[0].dest_ops) != 1 or where.outputs[0].dest_ops[0].type != "Softmax":
            continue
        scores_output = scale_end.outputs[0]
        mask_tensor = _try_fold_where_mask(where, scores_output)
        if mask_tensor is None:
            misses += 1
            continue
        softmax = where.outputs[0].dest_ops[0]
        if softmax.inputs[0] is not where.outputs[0]:
            misses += 1
            continue

        mask_tensor_final = mask_tensor.detach().clone().contiguous().to(torch.float32)
        mask_var = graph.create_variable(
            name=f"attention_mask_from_{where.name}",
            value=mask_tensor_final,
            is_parameter=True,
        )
        mask_var.shape = list(mask_tensor_final.shape)
        mask_var.dtype = DataType.FP32

        # Reuse where_out as the Add output so any graph.outputs entry stays intact.
        where_out = where.outputs[0]
        where.outputs.remove(where_out)
        where_out.source_op = None

        for iv in list(where.inputs):
            if where in iv.dest_ops:
                iv.dest_ops.remove(where)
            if iv is not scores_output and not iv.dest_ops and iv.name not in graph.outputs:
                if iv.is_parameter and iv.name in graph.variables and iv.name not in graph.inputs:
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
            platform=where.platform,
            inputs=[scores_output, mask_var],
            outputs=[where_out],
        )
        add_op.set_extension_attrib(_NONFINITE_MASK_ADD_KEY, True)
        where_out.source_op = add_op
        if softmax not in where_out.dest_ops:
            where_out.dest_ops.append(softmax)
        rewired += 1

    if misses:
        _report(graph).unfolded_where += misses
        logger.debug(
            f'[espdl-attention-fusion] detected {misses} MatMul->[scale?]->Where(mask)'
            f'->Softmax->MatMul pattern(s) that could not be constant-folded'
        )
    return rewired


# pattern_matching walks exact edges; Mul/Div scale needs its own pattern.
_SCALE_SPELLINGS = (None, "Mul", "Div")


def _fuse_primitive_patterns(graph: BaseGraph) -> int:
    """Fuse ``MatMul -> [scale?] -> [Add(mask)]? -> Softmax -> MatMul``."""

    def _try_fuse(
        qk_matmul: Operation,
        add: Optional[Operation],
        softmax: Operation,
        pv_matmul: Operation,
    ) -> int:
        match = _match_primitive(graph, qk_matmul, add, softmax, pv_matmul)
        if match is None:
            return 0
        doomed = match.doomed_ops(graph)
        rank = match.rank
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
            actionable = "ESP_DL_ATTENTION_MAX_" in reason
        else:
            reason = match.ineligible_reason()
            actionable = "ESP_DL_ATTENTION_MAX_" in reason
        if reason:
            # Keep unfused non-finite mask Adds in FP32 so calibration does not quantize -inf.
            if match.add is not None and any(_nonfinite_mask(inp) for inp in match.add.inputs if inp is not None):
                match.add.platform = TargetPlatform.FP32
                match.add.set_extension_attrib(_NONFINITE_MASK_ADD_KEY, True)
            _report(graph).record_left(match.name, reason, actionable)
            logger.debug(f'[espdl-attention-fusion] {match.name}: left as MatMul/Softmax/MatMul — {reason}')
            return 0

        k_transpose = match.k_transpose
        if match.k_transpose_mode == "rewrite_to_bhnd":
            if len(k_transpose.outputs[0].dest_ops) != 1:
                k_var = None
            else:
                perm = list(k_transpose.attributes.get("perm", []) or [])
                in_shape = _shape(k_transpose.inputs[0])
                if len(perm) == 3:
                    k_transpose._attributes["perm"] = [1, 0, 2]
                    if in_shape is not None and len(in_shape) == 3:
                        k_transpose.outputs[0].shape = [in_shape[1], in_shape[0], in_shape[2]]
                else:
                    k_transpose._attributes["perm"] = [0, 2, 1, 3]
                    if in_shape is not None and len(in_shape) == 4:
                        k_transpose.outputs[0].shape = [in_shape[0], in_shape[2], in_shape[1], in_shape[3]]
                k_var = k_transpose.outputs[0]
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
        else:
            k_var = match.k_var

        out_var = match.pv_matmul.outputs[0]
        q_shape, k_shape, v_shape = shapes
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
        if match.mask is not None and match.mask.shape is None and mask_shape is not None:
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

        fused.set_extension_attrib(_PREPARED_ATTENTION_KEY, True)
        report = _report(graph)
        report.fused += 1
        report.kept += 1
        if _compact_constant_mask(graph, fused):
            report.compacted += 1
            logger.debug(f'[espdl-attention-fusion] {fused.name}: constant 0/-inf attn_mask stored as bool')
        logger.debug(
            f'[espdl-attention-fusion] fused MatMul->'
            f"{'{Mul/Div}->' if match.scale_op else ''}"
            f"{'Add->' if match.add is not None else ''}"
            f'Softmax->MatMul into {fused.name} (scale={match.scale})'
        )
        return 1

    search = SearchableGraph(graph=graph)
    fused_total = 0
    for has_mask in (True, False):
        seen = set()
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
                fused_total += _try_fuse(qk_matmul, add, softmax, pv_matmul)
    return fused_total


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
    """Evaluate a constant or ``Shape -> Slice -> Sqrt -> Div`` attention scale factor."""
    if var is None:
        return None
    seen = set() if seen is None else seen
    if id(var) in seen:
        return None
    seen.add(id(var))

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
    if src.type == "Slice" and len(src.inputs) >= 3:
        shape_src = getattr(src.inputs[0], "source_op", None)
        if shape_src is None or shape_src.type != "Shape" or not shape_src.inputs:
            return None
        dims = _shape(shape_src.inputs[0])
        if dims is None:
            return None
        starts = _const_int_vector(src.inputs[1])
        ends = _const_int_vector(src.inputs[2])
        if starts is None or ends is None or len(starts) != 1 or len(ends) != 1:
            return None
        if len(src.inputs) > 3 and _const_int_vector(src.inputs[3]) not in (None, [0]):
            return None
        n = len(dims)
        start = min(max(starts[0] + n if starts[0] < 0 else starts[0], 0), n)
        end = min(max(ends[0] + n if ends[0] < 0 else ends[0], 0), n)
        return float(dims[start]) if end - start == 1 else None
    return None


def _match_head_merge(graph: BaseGraph, data: Variable) -> Optional[Tuple[Operation, Variable, Tuple[Operation, ...]]]:
    """Match dynamo SDPA's ``Reshape -> Transpose([0, 2, 1]) -> Reshape`` 4-D K^T chain."""

    def _shape_vector_length(var: Variable) -> Optional[int]:
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

    outer = getattr(data, "source_op", None)
    if outer is None or outer.type != "Reshape" or not outer.inputs:
        return None
    if len(outer.outputs[0].dest_ops) != 1 or outer.outputs[0].name in graph.outputs:
        return None
    transpose = getattr(outer.inputs[0], "source_op", None)
    if transpose is None or transpose.type != "Transpose":
        return None
    if list(transpose.attributes.get("perm", []) or []) != [0, 2, 1]:
        return None
    if len(transpose.outputs[0].dest_ops) != 1 or transpose.outputs[0].name in graph.outputs:
        return None
    inner = getattr(transpose.inputs[0], "source_op", None)
    if inner is None or inner.type != "Reshape" or not inner.inputs:
        return None
    if len(inner.outputs[0].dest_ops) != 1 or inner.outputs[0].name in graph.outputs:
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


def _match_primitive(
    graph: BaseGraph,
    qk_matmul: Operation,
    add: Optional[Operation],
    softmax: Operation,
    pv_matmul: Operation,
) -> Optional[_AttentionPrimitiveMatch]:
    """Match a structural ``MatMul -> [Mul/Div]? -> [Add]? -> Softmax -> MatMul`` subgraph."""

    def _unwrap_prescale(var: Variable, consumer: Operation) -> Optional[Tuple[Variable, float, Optional[Operation]]]:
        src = getattr(var, "source_op", None)
        if src is None or src.type != "Mul" or len(src.inputs) != 2:
            return var, 1.0, None
        if var.dest_ops != [consumer] or var.name in graph.outputs:
            return None
        folded = [_eval_scale_factor(inp) for inp in src.inputs]
        if folded[0] is not None and folded[1] is None:
            data, factor = src.inputs[1], folded[0]
        elif folded[1] is not None and folded[0] is None:
            data, factor = src.inputs[0], folded[1]
        else:
            return None
        return (data, factor, src) if math.isfinite(factor) and factor > 0 else None

    def _is_kt_4d(k_data: Variable, q_var: Variable) -> bool:
        k_shape, q_shape = _shape(k_data), _shape(q_var)
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

    if qk_matmul.num_of_parameter != 0 or pv_matmul.num_of_parameter != 0:
        return None
    if softmax.extension_attrib.get(_DECOMPOSED_ATTENTION_KEY, False):
        return None
    if len(qk_matmul.inputs) != 2 or len(pv_matmul.inputs) != 2 or not softmax.inputs:
        return None
    if pv_matmul.inputs[0].source_op is not softmax:
        return None
    default_axis = -1 if softmax.opset.onnx_opset_version() >= 13 else 1
    axis = softmax.attributes.get("axis", default_axis)
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
        if softmax.inputs[0].source_op is not add or scores_op.outputs[0] not in add.inputs:
            return None
        mask_var = next((var for var in add.inputs if var.source_op is not scores_op), None)
        if mask_var is None:
            return None
    else:
        if softmax.inputs[0].source_op is not scores_op:
            return None
        mask_var = None

    left = _unwrap_prescale(qk_matmul.inputs[0], qk_matmul)
    right = _unwrap_prescale(qk_matmul.inputs[1], qk_matmul)
    if left is None or right is None:
        return None
    q_var, left_factor, left_mul = left
    k_data, right_factor, right_mul = right
    prescale_ops = tuple(op for op in (left_mul, right_mul) if op is not None)
    if prescale_ops:
        prescale = left_factor * right_factor
        scale = prescale if scale is None else scale * prescale

    head_merge = _match_head_merge(graph, k_data)
    if head_merge is not None:
        k_transpose, k_var, head_merge_ops = head_merge
        k_transpose_mode = "head_merge"
    else:
        k_transpose, k_transpose_mode = _match_k_transpose(k_data)
        if k_transpose is not None:
            k_var = k_transpose.inputs[0]
            head_merge_ops = ()
        elif _is_kt_4d(k_data, q_var):
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
        len(k_transpose.outputs[0].dest_ops) != 1 or k_transpose.outputs[0].name in graph.outputs
    ):
        return None

    # Ensure interior edges have no outside consumers or graph outputs.
    expected_after_qk = scores_op if scores_op is not qk_matmul else (add or softmax)
    if qk_matmul.outputs[0].dest_ops != [expected_after_qk] or qk_matmul.outputs[0].name in graph.outputs:
        return None
    if scores_op is not qk_matmul:
        next_consumer = add if add is not None else softmax
        if scores_op.outputs[0].dest_ops != [next_consumer] or scores_op.outputs[0].name in graph.outputs:
            return None
    if add is not None and (add.outputs[0].dest_ops != [softmax] or add.outputs[0].name in graph.outputs):
        return None
    if len(softmax.outputs[0].dest_ops) != 1 or softmax.outputs[0].dest_ops[0].type != "MatMul":
        return None
    if softmax.outputs[0].name in graph.outputs:
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


def _detach_op(graph: BaseGraph, op: Operation, keep_output: bool = False) -> None:
    """Detach ``op`` from graph, unlinking its I/O variables."""
    for iv in list(op.inputs):
        try:
            if op in iv.dest_ops:
                iv.dest_ops.remove(op)
        except Exception:
            pass
        # Keep parameter inputs that are still referenced by the fused op or graph boundary.
        if iv.is_parameter and (iv.dest_ops or iv.name in graph.inputs or iv.name in graph.outputs):
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
        if ov.source_op is None and not ov.dest_ops and ov.name in graph.variables and ov.name not in graph.outputs:
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
        if var.name not in graph.variables:
            graph.append_variable(var)
    out = output if output is not None else graph.create_variable()
    if out.name not in graph.variables:
        graph.append_variable(out)
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


def _decompose_blocker(op: Operation, qkv_shapes: List[Optional[List[int]]]) -> Optional[str]:
    """None if ``op`` can be expanded to MatMul/Softmax; otherwise why not."""
    attrs = op.attributes or {}
    is_msft = "qkv_hidden_sizes" in attrs or (
        "q_num_heads" not in attrs
        and "kv_num_heads" not in attrs
        and op.num_of_input >= 2
        and getattr(op.inputs[1], "is_parameter", False)
        and len(_shape(op.inputs[1]) or []) == 2
    )
    if is_msft:
        return "com.microsoft.Attention packed QKV is not expanded"
    if op.num_of_output > 1:
        return "Attention extra outputs (qk_matmul_output) are not expanded"
    if op.num_of_input > 4:
        return "KV cache (past_key / past_value) is not expanded"
    attr_reason = espdl_attention_attributes_ineligible_reason(attrs)
    if attr_reason:
        return attr_reason
    layout = _decode_layout(attrs.get("layout", "BHND")) or "BHND"
    if layout not in _DECOMPOSABLE_LAYOUTS:
        return f"layout={layout!r} is not expanded"
    static = [
        [int(d) for d in s]
        if s is not None
        and len(s) in (3, 4)
        and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in s)
        else None
        for s in qkv_shapes[:3]
    ]
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
        head_reason = espdl_attention_head_attrs_ineligible_reason(attrs, static, layout)
        if head_reason:
            return head_reason
        head_axis, seq_axis = _layout_head_seq_axes(layout)
        b_q, h_q, l_q, d_q = q_shape[0], q_shape[head_axis], q_shape[seq_axis], q_shape[3]
        b_k, h_k, l_k, d_k = k_shape[0], k_shape[head_axis], k_shape[seq_axis], k_shape[3]
        b_v, h_v, l_v = v_shape[0], v_shape[head_axis], v_shape[seq_axis]
    if op.num_of_input >= 4 and op.inputs[3] is not None:
        mask = op.inputs[3]
        if _var_static_value(mask) is None and mask.dtype == DataType.BOOL:
            return "runtime bool attn_mask is not expanded (no Where kernel)"
        if mask.dtype is not None and mask.dtype not in _DECOMPOSABLE_MASK_DTYPES:
            dtype_name = getattr(mask.dtype, "name", str(mask.dtype))
            return f"{dtype_name} attn_mask is neither bool nor float (ONNX Attention has no integer mask)"
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
    """True when an additive mask is bool or contains ``-inf`` and cannot be quantized."""
    value = _var_static_value(mask)
    if value is None:
        return False
    if value.dtype == torch.bool:
        return True
    return bool(value.is_floating_point() and torch.isinf(value).any())


def _decompose_attention(graph: BaseGraph, op: Operation, *, quantized: bool = False) -> None:
    """Decompose ``op`` into ``MatMul(Q, K^T) -> [Mul] -> [Add] -> Softmax -> MatMul``."""
    attrs = op.attributes or {}
    platform = op.platform
    layout = _decode_layout(attrs.get("layout", "BHND")) or "BHND"

    def _to_bhnd(var: Variable, shape: List[int], num_heads: int) -> Tuple[Variable, List[int]]:
        if len(shape) == 4:
            if layout == "BNHD":
                bhnd = [shape[0], shape[2], shape[1], shape[3]]
                return _emit_op(
                    graph, "Transpose", [var], {"perm": [0, 2, 1, 3]}, out_shape=bhnd, platform=platform
                ), bhnd
            return var, list(shape)
        if len(shape) != 3:
            raise ValueError(f"Attention decompose: expected rank 3 or 4, got {shape}")
        if num_heads <= 0:
            raise ValueError("Attention decompose: 3-D Q/K/V needs q_num_heads / kv_num_heads")
        batch_s, length_s, hidden_s = shape
        if hidden_s % num_heads != 0:
            raise ValueError(f"Attention decompose: hidden={hidden_s} not divisible by num_heads={num_heads}")
        dim_s = hidden_s // num_heads
        lifted = _reshape(graph, var, [batch_s, length_s, num_heads, dim_s], platform)
        bhnd = [batch_s, num_heads, length_s, dim_s]
        return _emit_op(graph, "Transpose", [lifted], {"perm": [0, 2, 1, 3]}, out_shape=bhnd, platform=platform), bhnd

    def _repeat_kv_heads(data: Variable, shape_bhnd: List[int], group: int) -> Variable:
        if group == 1:
            return data
        b_s, h_s, l_s, d_s = shape_bhnd
        flat = _reshape(graph, data, [b_s * h_s, 1, l_s, d_s], platform)
        tiled = _emit_op(
            graph, "Concat", [flat] * group, {"axis": 1}, out_shape=[b_s * h_s, group, l_s, d_s], platform=platform
        )
        return _reshape(graph, tiled, [b_s, h_s * group, l_s, d_s], platform)

    q_var, k_var, v_var = op.inputs[0], op.inputs[1], op.inputs[2]
    q_shape = [int(d) for d in (_shape(q_var) or ())]
    k_shape = [int(d) for d in (_shape(k_var) or ())]
    v_shape = [int(d) for d in (_shape(v_var) or ())]

    q_heads = int(attrs.get("q_num_heads", 0) or 0)
    kv_heads = int(attrs.get("kv_num_heads", 0) or 0)
    orig_rank = len(q_shape)

    q_bhnd, qs = _to_bhnd(q_var, q_shape, q_heads)
    k_bhnd, ks = _to_bhnd(k_var, k_shape, kv_heads)
    v_bhnd, vs = _to_bhnd(v_var, v_shape, kv_heads)

    batch, h_q, l_q, d_q = qs
    _, h_kv, l_k, d_k = ks
    d_v = vs[3]
    if d_q != d_k:
        raise ValueError(f"Attention {op.name}: Q/K head_dim mismatch {d_q} vs {d_k}")
    if h_kv < 1 or h_q % h_kv != 0:
        raise ValueError(f"Attention {op.name}: Hq={h_q} is not a multiple of Hkv={h_kv}")
    if h_q != h_kv:
        group = h_q // h_kv
        k_bhnd = _repeat_kv_heads(k_bhnd, [batch, h_kv, l_k, d_k], group)
        v_bhnd = _repeat_kv_heads(v_bhnd, [batch, h_kv, l_k, d_v], group)

    k_t = _emit_op(
        graph, "Transpose", [k_bhnd], {"perm": [0, 1, 3, 2]}, out_shape=[batch, h_q, d_k, l_k], platform=platform
    )
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
        mask_val = _var_static_value(raw_mask)
        if mask_val is not None and mask_val.dtype == torch.bool:
            additive = torch.zeros(mask_val.shape, dtype=torch.float32).masked_fill_(~mask_val, float("-inf"))
            mask = graph.create_variable(value=additive, is_parameter=True)
            mask.shape = list(additive.shape)
            mask.dtype = DataType.FP32
        else:
            mask = raw_mask
        requires_fp32_add = mask_is_nonfinite or _nonfinite_mask(mask)
        add_platform = TargetPlatform.FP32 if quantized and requires_fp32_add else platform
        scores = _emit_op(graph, "Add", [scores, mask], out_shape=[batch, h_q, l_q, l_k], platform=add_platform)
        if requires_fp32_add and scores.source_op is not None:
            scores.source_op.set_extension_attrib(_NONFINITE_MASK_ADD_KEY, True)

    if int(attrs.get("is_causal", 0) or 0) != 0:
        keep = torch.ones(l_q, l_k, dtype=torch.bool).tril()
        causal = torch.zeros(1, 1, l_q, l_k, dtype=torch.float32)
        causal.masked_fill_(~keep, float("-inf"))
        causal_var = graph.create_variable(value=causal, is_parameter=True)
        causal_var.shape = list(causal.shape)
        causal_var.dtype = DataType.FP32
        add_platform = TargetPlatform.FP32 if quantized else platform
        scores = _emit_op(graph, "Add", [scores, causal_var], out_shape=[batch, h_q, l_q, l_k], platform=add_platform)
        if scores.source_op is not None:
            scores.source_op.set_extension_attrib(_NONFINITE_MASK_ADD_KEY, True)

    probs = _emit_op(graph, "Softmax", [scores], {"axis": -1}, out_shape=[batch, h_q, l_q, l_k], platform=platform)
    if probs.source_op is not None:
        probs.source_op.set_extension_attrib(_DECOMPOSED_ATTENTION_KEY, True)

    out_var = op.outputs[0]
    op.outputs.remove(out_var)
    out_var.source_op = None

    bhnd_shape = [batch, h_q, l_q, d_v]
    if orig_rank == 4 and layout == "BHND":
        _emit_op(graph, "MatMul", [probs, v_bhnd], out_shape=bhnd_shape, platform=platform, output=out_var)
    else:
        bhnd_out = _emit_op(graph, "MatMul", [probs, v_bhnd], out_shape=bhnd_shape, platform=platform)
        bnhd = [batch, l_q, h_q, d_v]
        if orig_rank == 4:
            _emit_op(
                graph,
                "Transpose",
                [bhnd_out],
                {"perm": [0, 2, 1, 3]},
                out_shape=bnhd,
                platform=platform,
                output=out_var,
            )
        else:
            as_bnhd = _emit_op(
                graph, "Transpose", [bhnd_out], {"perm": [0, 2, 1, 3]}, out_shape=bnhd, platform=platform
            )
            _reshape(graph, as_bnhd, [batch, l_q, h_q * d_v], platform=platform, output=out_var)

    for iv in list(op.inputs):
        if op in iv.dest_ops:
            iv.dest_ops.remove(op)
        if (
            iv.source_op is None
            and not iv.dest_ops
            and iv.name in graph.variables
            and iv.name not in graph.outputs
            and iv.name not in graph.inputs
        ):
            graph.remove_variable(iv)
    op.inputs.clear()
    if op.name in graph.operations:
        graph.remove_operation(op, keep_coherence=False, remove_unlinked_variable=False)


def _compact_constant_mask(graph: BaseGraph, op: Operation) -> bool:
    """Store a constant fp32 ``attn_mask`` holding only ``0`` / ``-inf`` as bool."""
    if op.num_of_input < 4 or op.inputs[3] is None:
        return False
    mask = op.inputs[3]
    if not mask.is_parameter or len(mask.dest_ops) != 1 or mask.name in graph.outputs:
        return False
    value = _var_static_value(mask)
    if value is None or not value.is_floating_point() or value.numel() == 0:
        return False
    keep = value == 0
    if not bool((keep | torch.isneginf(value)).all().item()):
        return False
    mask.value = keep.contiguous()
    mask.dtype = DataType.BOOL
    return True


def _prepare_native_espdl_attention_ops(
    graph: BaseGraph,
    attention_ops: Sequence[Operation],
    *,
    platform: Optional[TargetPlatform] = None,
    quantized: bool = False,
    log_report: bool = True,
    **kwargs,
) -> None:
    """Validate and lower eligible native Attention ops, or decompose ineligible ones."""
    _ensure_traced_shapes(graph, **kwargs)
    unsupported = espdl_attention_platform_unsupported_reason(platform)
    report = _report(graph)
    for op in list(attention_ops):
        if op.type != ATTENTION or op.name not in graph.operations or graph.operations[op.name] is not op:
            continue
        qkv_shapes = [_shape(op.inputs[i]) for i in range(min(3, op.num_of_input))]
        if len(qkv_shapes) < 3 or any(shape is None for shape in qkv_shapes):
            raise ValueError(
                f"Attention {op.name} cannot be verified by the current implementation: "
                f"Q/K/V shapes are still {qkv_shapes} after tracing. Export the ONNX "
                f"with static Q/K/V dims, or pass ``inputs`` / ``input_shape`` that let "
                f"shape tracing run."
            )
        if op.num_of_input == 4 and op.inputs[3] is not None and _shape(op.inputs[3]) is None:
            raise ValueError(
                f"Attention {op.name} cannot be verified by the current implementation: "
                f"attn_mask is connected but its shape is still unknown after tracing. "
                f"Export the ONNX with a static attn_mask shape, or pass ``inputs`` / "
                f"``input_shape`` that let shape tracing fill it in."
            )
        if unsupported is None:
            if op.extension_attrib.get(_PREPARED_ATTENTION_KEY, False) and espdl_attention_eligible(op, qkv_shapes):
                continue
            if espdl_attention_eligible(op, qkv_shapes):
                report.kept += 1
                if _compact_constant_mask(graph, op):
                    report.compacted += 1
                    logger.debug(f'[espdl-attention-prepare] {op.name}: constant 0/-inf attn_mask stored as bool')
                lowered = lower_espdl_attention_attributes(op.attributes)
                op.attributes.clear()
                op.attributes.update(lowered)
                if op.outputs and op.outputs[0].shape is None and qkv_shapes and qkv_shapes[0] is not None:
                    op.outputs[0].shape = list(qkv_shapes[0])
                op.set_extension_attrib(_PREPARED_ATTENTION_KEY, True)
                continue
        reason = unsupported or espdl_attention_ineligible_reason(op, qkv_shapes)
        blocker = _decompose_blocker(op, qkv_shapes)
        if blocker is not None:
            attrs = op.attributes or {}
            if "qkv_hidden_sizes" in attrs:
                remediation = (
                    "This is com.microsoft.Attention (packed QKV projection). "
                    "The current implementation has no kernel for that operator; re-export as "
                    "ai.onnx::Attention or as a MatMul / Softmax / MatMul subgraph."
                )
            elif _NON_FUSION_MARKER_ATTRS & set(attrs):
                remediation = (
                    "This native ai.onnx::Attention node could not be expanded to "
                    "MatMul / Softmax. Drop the unsupported feature or export with "
                    "opset < 23 so PyTorch emits the primitive decomposition."
                )
            else:
                remediation = "Reshape the block, or emit MatMul / Softmax / MatMul yourself."
            raise ValueError(
                f"Attention {op.name} is unsupported by the current implementation: {reason}; {blocker}. "
                f"Q/K/V shapes are {qkv_shapes}. {remediation}"
            )
        _decompose_attention(graph, op, quantized=quantized)
        actionable = unsupported is None and "ESP_DL_ATTENTION_MAX_" in reason
        report.record_expanded(op.name, reason or "ineligible for the fused kernel", actionable)
        logger.debug(f'[espdl-attention-prepare] {op.name}: expanded to MatMul/Softmax ({reason})')
    if log_report:
        _clear_attention_report(graph)


def fuse_espdl_attention_graph(
    graph: BaseGraph,
    **kwargs,
) -> None:
    """Fuse ``MatMul -> [Mul/Div]? -> [Add]? -> Softmax -> MatMul`` into Attention."""

    _ensure_traced_shapes(graph, **kwargs)
    rewired = _rewrite_where_masks_to_add(graph)
    fused_primitive = _fuse_primitive_patterns(graph)
    if (rewired or fused_primitive) and graph.outputs:
        changed = True
        while changed:
            changed = False
            for op in list(graph.operations.values()):
                if op.type == ATTENTION or not op.outputs:
                    continue
                if all(not v.dest_ops and v.name not in graph.outputs for v in op.outputs):
                    _detach_op(graph, op)
                    changed = True
    if fused_primitive:
        logger.debug(f'[espdl-attention-fusion] fused {fused_primitive} primitive MatMul->Softmax->MatMul pattern(s)')


def format_espdl_attention_graph(graph: BaseGraph) -> None:
    """Fuse primitive attention subgraphs and prepare/decompose shaped native Attention."""
    native_attention = [
        op
        for op in graph.operations.values()
        if op.type == ATTENTION and not op.extension_attrib.get(_PREPARED_ATTENTION_KEY, False)
    ]
    fuse_espdl_attention_graph(graph)
    shaped_native = [op for op in native_attention if not _attention_op_shapes_incomplete(op)]
    if shaped_native:
        _prepare_native_espdl_attention_ops(
            graph,
            shaped_native,
            log_report=False,
        )
    _clear_attention_report(graph)

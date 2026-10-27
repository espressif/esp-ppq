from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from esp_ppq.api.setting import QuantizationSetting
from esp_ppq.core import (
    OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE,
    PASSIVE_OPERATIONS,
    DataType,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
    convert_any_to_torch_tensor,
    ppq_warning,
)
from esp_ppq.IR import BaseGraph, Operation, Variable
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.quantization.optim import (
    EspdlAttentionFusionPass,
    EspdlAttentionLegalizePass,
    EspdlLUTFusionPass,
    EspdlRotaryEmbeddingLegalizePass,
    InverseSigmoidFusionPass,
    QuantizationOptimizationPipeline,
    RNNArityNormalizationPass,
    RotaryEmbeddingFusionPass,
    espdl_attention_platform_unsupported_reason,
    finalize_espdl_attention_for_export,
    finalize_espdl_inverse_sigmoid_for_export,
)
from esp_ppq.quantization.optim.espdl_attention import _attention_prequant_needs_trace
from esp_ppq.quantization.optim.inverse_sigmoid_fusion import _inverse_sigmoid_prequant_needs_trace
from esp_ppq.quantization.optim.rotary_embedding_fusion import (
    _finalize_espdl_rotary_embedding_quantization_contract,
    _rope_prequant_needs_trace,
)

from .base import BaseQuantizer


def _quant_range(bits: int) -> tuple[int, int]:
    """Return (quant_min, quant_max) for signed symmetric quantization."""
    half = 1 << (bits - 1)
    return -half, half - 1


def _kl_hist_bins(bits: int) -> int:
    """Return KL histogram bin count for given bit width."""
    return 32 * (1 << (bits - 1))


def _apply_bit_width(config, bits: int, state=QuantizationStates.PASSIVE_INIT, observer: str = "minmax") -> None:
    """Set bit width, quant range, state and observer on a TensorQuantizationConfig."""
    config.num_of_bits = bits
    config.quant_min, config.quant_max = _quant_range(bits)
    # Sync KL histogram bins with the updated bit width.
    config.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = _kl_hist_bins(bits)
    config.state = state
    config.observer_algorithm = observer


_GRU_BIAS_INDICES = (3,)  # bias
_LSTM_BIAS_INDICES = (3, 6)  # bias, initial_c
_ATTENTION_MASK_INDEX = 3


@dataclass(frozen=True)
class _PlatformQuantConfig:
    num_of_bits: int
    bias_bits: int
    rounding: RoundingPolicy


_PLATFORM_CONFIGS: dict[TargetPlatform, _PlatformQuantConfig] = {
    TargetPlatform.ESPDL_INT8: _PlatformQuantConfig(8, 32, RoundingPolicy.ROUND_HALF_EVEN),
    TargetPlatform.ESPDL_INT16: _PlatformQuantConfig(16, 40, RoundingPolicy.ROUND_HALF_EVEN),
    TargetPlatform.ESPDL_H_PRE_INT16: _PlatformQuantConfig(16, 40, RoundingPolicy.ROUND_HALF_EVEN),
    TargetPlatform.ESPDL_S3_INT8: _PlatformQuantConfig(8, 20, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_S3_INT16: _PlatformQuantConfig(16, 40, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_S3_H_PRE_INT16: _PlatformQuantConfig(16, 40, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_C_INT8: _PlatformQuantConfig(8, 32, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_C_INT16: _PlatformQuantConfig(16, 64, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_C_H_PRE_INT16: _PlatformQuantConfig(16, 64, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_W8A16: _PlatformQuantConfig(16, 40, RoundingPolicy.ROUND_HALF_EVEN),
    TargetPlatform.ESPDL_S3_W8A16: _PlatformQuantConfig(16, 40, RoundingPolicy.ROUND_HALF_UP),
    TargetPlatform.ESPDL_C_W8A16: _PlatformQuantConfig(16, 64, RoundingPolicy.ROUND_HALF_UP),
}


_W8A16_PLATFORMS = {
    TargetPlatform.ESPDL_W8A16,
    TargetPlatform.ESPDL_S3_W8A16,
    TargetPlatform.ESPDL_C_W8A16,
}

# W8A16 is weight-8 / activation-16. A MatMul with two runtime operands has no
# weight, so keep both sides at 16-bit (w16a16) on the matching INT16 platform.
_W8A16_TO_W16A16 = {
    TargetPlatform.ESPDL_W8A16: TargetPlatform.ESPDL_INT16,
    TargetPlatform.ESPDL_S3_W8A16: TargetPlatform.ESPDL_S3_INT16,
    TargetPlatform.ESPDL_C_W8A16: TargetPlatform.ESPDL_C_INT16,
}
_MATMUL_TYPES = {"MatMul", "PPQBiasFusedMatMul"}
_TOPK_OP_TYPE = "TopK"
_TOPK_INT_MAX = (1 << 31) - 1
_TOPK_SUPPORTED_DTYPES = {
    DataType.INT8,
    DataType.INT16,
    DataType.FP32,
    DataType.FP64,
}


def _topk_integer(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _topk_concrete_shape(var: Variable) -> Optional[Tuple[int, ...]]:
    shape = getattr(var, "shape", None)
    if shape is None:
        return None
    result = []
    for dim in shape:
        value = _topk_integer(dim)
        if value is None or value < 0:
            return None
        result.append(value)
    return tuple(result)


def _topk_parameter_tensor(var: Variable) -> Optional[torch.Tensor]:
    if not var.is_parameter or var.value is None:
        return None
    try:
        return convert_any_to_torch_tensor(var.value, accept_none=False, device="cpu")
    except (RuntimeError, TypeError, ValueError):
        return None


def _topk_dtype_name(dtype) -> str:
    return getattr(dtype, "name", str(dtype))


def _topk_static_unsupported_reason(op: Operation) -> Optional[str]:
    if op.type != _TOPK_OP_TYPE:
        return f"expected {_TOPK_OP_TYPE}, got {op.type}"

    version = op.opset.onnx_opset_version()
    if version < 10 or version > 23:
        return f"ONNX opset {version} is unsupported by ESP-DL; use opset 10-23 where K is an input"
    if len(op.inputs) != 2 or len(op.outputs) != 2:
        return f"expected 2 inputs and 2 outputs, got {len(op.inputs)} and {len(op.outputs)}"
    if version == 10 and any(name in op.attributes for name in ("largest", "sorted")):
        return "largest and sorted attributes require ONNX opset 11 or newer"

    axis = _topk_integer(op.attributes.get("axis", -1))
    if axis is None:
        return f"axis must be an integer, got {op.attributes.get('axis')!r}"
    largest = _topk_integer(op.attributes.get("largest", 1))
    if largest not in (0, 1):
        return f"largest must be 0 or 1, got {op.attributes.get('largest')!r}"
    sorted_output = _topk_integer(op.attributes.get("sorted", 1))
    if sorted_output not in (0, 1):
        return f"sorted must be 0 or 1, got {op.attributes.get('sorted')!r}"

    x_var, k_var = op.inputs
    values_var, indices_var = op.outputs
    if x_var.dtype not in _TOPK_SUPPORTED_DTYPES:
        return f"X dtype {_topk_dtype_name(x_var.dtype)} is unsupported"
    if values_var.dtype != x_var.dtype:
        return f"Values dtype {_topk_dtype_name(values_var.dtype)} must match X dtype {_topk_dtype_name(x_var.dtype)}"
    if indices_var.dtype != DataType.INT64:
        return f"Indices dtype must be INT64, got {_topk_dtype_name(indices_var.dtype)}"

    x_shape = _topk_concrete_shape(x_var)
    if x_shape is None:
        return "X shape is missing or dynamic"
    if not x_shape:
        return "X rank must be at least 1"
    if any(dim <= 0 for dim in x_shape):
        return f"X shape must contain only positive dimensions, got {list(x_shape)}"
    if axis < -len(x_shape) or axis >= len(x_shape):
        return f"axis {axis} is out of range for rank {len(x_shape)}"
    normalized_axis = axis % len(x_shape)

    if not k_var.is_parameter:
        return "K must be a constant parameter, not a runtime input"
    if k_var.dtype != DataType.INT64:
        return f"K dtype must be INT64, got {_topk_dtype_name(k_var.dtype)}"
    k_tensor = _topk_parameter_tensor(k_var)
    if k_tensor is None:
        return "K must contain a constant value"
    if k_tensor.dtype != torch.int64:
        return f"K storage dtype must be torch.int64, got {k_tensor.dtype}"
    if tuple(k_tensor.shape) != (1,):
        return f"K shape must be [1], got {list(k_tensor.shape)}"
    k = int(k_tensor.item())
    axis_size = x_shape[normalized_axis]
    if k <= 0 or k > axis_size:
        return f"K={k} is out of range (0, axis_size={axis_size}]"
    if k > _TOPK_INT_MAX:
        return f"K={k} exceeds ESP-DL's INT_MAX limit"

    expected_shape = list(x_shape)
    expected_shape[normalized_axis] = k
    expected_shape = tuple(expected_shape)
    for role, var in (("Values", values_var), ("Indices", indices_var)):
        shape = _topk_concrete_shape(var)
        if shape is None:
            return f"{role} shape is missing or dynamic"
        if shape != expected_shape:
            return f"{role} shape must be {list(expected_shape)}, got {list(shape)}"

    if math.prod(x_shape) > _TOPK_INT_MAX or math.prod(expected_shape) > _TOPK_INT_MAX:
        return "tensor element count exceeds ESP-DL's INT_MAX indexing limit"
    return None


def _topk_quantization_unsupported_reason(op: QuantableOperation) -> Optional[str]:
    config = op.config
    if len(config.input_quantization_config) != 2 or len(config.output_quantization_config) != 2:
        return "quantization config must contain 2 inputs and 2 outputs"

    k_config = config.input_quantization_config[1]
    if k_config.state != QuantizationStates.FP32:
        return f"K quantization state must be FP32, got {k_config.state.name}"
    if k_config.can_export(export_overlapped=True):
        return "K quantization config must not be exportable"

    indices_config = config.output_quantization_config[1]
    if indices_config.state != QuantizationStates.FP32:
        return f"Indices quantization state must be FP32, got {indices_config.state.name}"
    if indices_config.can_export(export_overlapped=True):
        return "Indices quantization config must not be exportable"

    input_config = config.input_quantization_config[0]
    values_config = config.output_quantization_config[0]
    if values_config.dominated_by is not input_config.dominated_by:
        return "Values must share X's quantization configuration"

    def _storage_signature(cfg):
        return (
            cfg.state == QuantizationStates.FP32,
            cfg.policy._policy,
            cfg.num_of_bits,
            cfg.exponent_bits,
            cfg.quant_min,
            cfg.quant_max,
        )

    if _storage_signature(values_config) != _storage_signature(input_config):
        return "Values and X must use the same storage quantization"
    if input_config.state != QuantizationStates.FP32:
        policy = input_config.policy
        if not (
            policy.has_property(QuantizationProperty.LINEAR)
            and policy.has_property(QuantizationProperty.SYMMETRICAL)
            and policy.has_property(QuantizationProperty.PER_TENSOR)
            and policy.has_property(QuantizationProperty.POWER_OF_2)
        ):
            return "X and Values quantization must be symmetric per-tensor linear POWER_OF_2"
        if input_config.num_of_bits not in (8, 16) or input_config.exponent_bits != 0:
            return "X and Values quantization must be 8-bit or 16-bit integer"
    return None


def _validate_espdl_topk_static_contract(graph: BaseGraph) -> None:
    for op in graph.operations.values():
        if op.type != _TOPK_OP_TYPE:
            continue
        reason = _topk_static_unsupported_reason(op)
        if reason is not None:
            raise ValueError(f"TopK {op.name}: {reason}")


def _validate_espdl_topk_quantization_contract(graph: BaseGraph) -> None:
    for op in graph.operations.values():
        if not isinstance(op, QuantableOperation) or op.type != _TOPK_OP_TYPE:
            continue
        reason = _topk_quantization_unsupported_reason(op)
        if reason is not None:
            raise ValueError(f"TopK {op.name}: {reason}")


def finalize_espdl_topk_for_export(graph: BaseGraph) -> None:
    """Validate static and calibrated ESP-DL TopK deployment contracts."""
    _validate_espdl_topk_static_contract(graph)
    _validate_espdl_topk_quantization_contract(graph)


def _topk_prequant_needs_trace(graph: BaseGraph) -> bool:
    for op in graph.operations.values():
        if op.type != _TOPK_OP_TYPE:
            continue
        version = op.opset.onnx_opset_version()
        if version < 10 or version > 23 or len(op.inputs) != 2 or len(op.outputs) != 2:
            continue
        k_var = op.inputs[1]
        k_tensor = _topk_parameter_tensor(k_var)
        if (
            not k_var.is_parameter
            or k_var.dtype != DataType.INT64
            or k_tensor is None
            or k_tensor.dtype != torch.int64
            or tuple(k_tensor.shape) != (1,)
        ):
            continue
        if any(_topk_concrete_shape(var) is None for var in (op.inputs[0], *op.outputs)):
            return True
    return False


def _fallback_dynamic_matmul_w8a16_to_w16a16(operation: Operation) -> None:
    if operation.type not in _MATMUL_TYPES or operation.platform not in _W8A16_TO_W16A16:
        return
    if operation.num_of_input < 2:
        return
    if operation.inputs[0].is_parameter or operation.inputs[1].is_parameter:
        return

    w16a16_platform = _W8A16_TO_W16A16[operation.platform]
    ppq_warning(
        f"MatMul '{operation.name}' has no weight operand; "
        f"fallback {operation.platform.name} -> {w16a16_platform.name} (w16a16)."
    )
    operation.platform = w16a16_platform


def _var_carries_nonfinite_mask(var) -> bool:
    if var is None:
        return False
    from esp_ppq.quantization.optim.espdl_attention import _nonfinite_mask

    if _nonfinite_mask(var):
        return True
    source = var.source_op
    if source is None:
        return False
    if source.type == "Where":
        return any(_var_carries_nonfinite_mask(inp) for inp in source.inputs[1:])
    if source.type == "Add":
        return any(_var_carries_nonfinite_mask(inp) for inp in source.inputs)
    return False


_P4_PLATFORMS = {
    TargetPlatform.ESPDL_INT8,
    TargetPlatform.ESPDL_INT16,
    TargetPlatform.ESPDL_H_PRE_INT16,
    TargetPlatform.ESPDL_W8A16,
}

_PER_CHANNEL_POLICY = QuantizationPolicy(
    QuantizationProperty.SYMMETRICAL
    + QuantizationProperty.LINEAR
    + QuantizationProperty.PER_CHANNEL
    + QuantizationProperty.POWER_OF_2
)


class BaseEspdlQuantizer(BaseQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = -128
        self._quant_max = +127
        self._custom_tqc = None

    def create_espdl_quant_config(
        self,
        operation: Operation,
        num_of_bits: int,
        quant_min: int,
        quant_max: int,
        bias_bits: int,
    ) -> OperationQuantizationConfig:
        # int16 uses minmax by default, int8 uses percentile.
        default_observer = "minmax" if num_of_bits == 16 else "percentile"
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy,
            rounding=self.rounding_policy,
            op=operation,
            num_of_bits=num_of_bits,
            exponent_bits=0,
            quant_max=quant_max,
            quant_min=quant_min,
            observer_algorithm=default_observer,
        )

        kl_bins = _kl_hist_bins(num_of_bits)
        for index in range(operation.num_of_input):
            if not operation.inputs[index].is_parameter:
                base_quant_config.input_quantization_config[index].detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = (
                    kl_bins
                )

        for index in range(operation.num_of_output):
            base_quant_config.output_quantization_config[index].detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = kl_bins

        if operation.type in {"Conv", "ConvTranspose", "Gemm"}:
            assert operation.num_of_input > 0, "Seems you got a Conv layer with no parameters."

            # Per-channel weight quantization for ESP32-P4 platforms
            if operation.platform in _P4_PLATFORMS and operation.num_of_input > 1:
                weight_config = base_quant_config.input_quantization_config[1]
                weight_config.policy = _PER_CHANNEL_POLICY
                if operation.type == "ConvTranspose":
                    weight_config.channel_axis = 1
                elif operation.type == "Gemm":
                    if operation.attributes.get('transB', 0) == 1:
                        weight_config.channel_axis = 0
                    else:
                        weight_config.channel_axis = 1
                else:
                    weight_config.channel_axis = 0
                weight_config.observer_algorithm = 'minmax'

            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                if operation.platform in _P4_PLATFORMS:
                    bias_config.policy = _PER_CHANNEL_POLICY
                    bias_config.channel_axis = 0
                _apply_bit_width(bias_config, bias_bits)
        elif operation.type in {"GRU", "LSTM"}:
            bias_indices = _LSTM_BIAS_INDICES if operation.type == "LSTM" else _GRU_BIAS_INDICES
            for index in bias_indices:
                if operation.num_of_input > index:
                    _apply_bit_width(base_quant_config.input_quantization_config[index], 16)

            if operation.num_of_output == 3:
                _apply_bit_width(base_quant_config.output_quantization_config[2], 16)
        elif operation.type == "RMSNormalization":
            if operation.num_of_input > 1:
                base_quant_config.input_quantization_config[1].state = QuantizationStates.FP32
        elif operation.type in {"Softmax", "LogSoftmax"}:
            base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
        elif operation.type == "TopK":
            if operation.num_of_input > 1:
                base_quant_config.input_quantization_config[1].state = QuantizationStates.FP32
            if operation.num_of_output > 1:
                base_quant_config.output_quantization_config[1].state = QuantizationStates.FP32
            if operation.num_of_input > 0 and operation.num_of_output > 0:
                base_quant_config.output_quantization_config[
                    0
                ].dominated_by = base_quant_config.input_quantization_config[0]
        elif operation.type == "Attention":
            if operation.num_of_input > _ATTENTION_MASK_INDEX:
                base_quant_config.input_quantization_config[_ATTENTION_MASK_INDEX].state = QuantizationStates.FP32
        elif operation.type == "RotaryEmbedding":
            # Promote cos/sin caches to int16 when activations are int8.
            if operation.num_of_input >= 3 and num_of_bits == 8:
                for index in (1, 2):
                    cache_config = base_quant_config.input_quantization_config[index]
                    _apply_bit_width(
                        cache_config,
                        16,
                        state=cache_config.state,
                        observer="minmax",
                    )

        for index, input_var in enumerate(operation.inputs):
            if not input_var.name:
                # Keep omitted interior optional slots in FP32 so they do not participate in calibration.
                base_quant_config.input_quantization_config[index].state = QuantizationStates.FP32

        if (
            operation.platform in _W8A16_PLATFORMS
            and operation.type in {"Conv", "Gemm", "MatMul"}
            and operation.num_of_input > 1
            # Depthwise convolution stores its filter in a dedicated layout that ESP-DL has no
            # W8A16 kernel for, so those weights stay at the activation bit width.
            and operation.attributes.get("group", 1) == 1
        ):
            weight_var = operation.inputs[1]
            if weight_var.is_parameter:
                weight_config = base_quant_config.input_quantization_config[1]
                weight_observer = "minmax" if operation.platform == TargetPlatform.ESPDL_W8A16 else "percentile"
                _apply_bit_width(weight_config, 8, state=weight_config.state, observer=weight_observer)

        if operation.type in PASSIVE_OPERATIONS:
            base_quant_config.is_active_quant_op = False
            # QuantizeFusionPass skips graph-input ops; align the scale here.
            if operation.num_of_input > 0 and operation.inputs[0].source_op is None:
                input_config = base_quant_config.input_quantization_config[0]
                if input_config.state != QuantizationStates.FP32:
                    for output_config in base_quant_config.output_quantization_config:
                        if output_config.state == QuantizationStates.FP32:
                            continue
                        output_config.dominated_by = input_config

        self._apply_custom_tqc(operation, base_quant_config)

        if operation.type == "RotaryEmbedding" and operation.num_of_input >= 3:
            cos_config = base_quant_config.input_quantization_config[1]
            sin_config = base_quant_config.input_quantization_config[2]
            # Share one scale across cos/sin caches when their quantization schemes match.
            if cos_config.is_same_scheme(sin_config) and sin_config.dominated_by == sin_config:
                sin_config.dominated_by = cos_config

        return base_quant_config

    def _apply_custom_tqc(self, operation: Operation, config: OperationQuantizationConfig) -> None:
        if not self._custom_tqc or not self._custom_tqc.get(operation.name):
            return

        op_configs = self._custom_tqc[operation.name]
        for tqc_name, tqc_value in op_configs.items():
            bits = tqc_value.get("bit_width")
            if not bits:
                continue

            tqc_index = int(re.findall(r"\d+", tqc_name)[0])
            if "input" in tqc_name:
                tqc_list, limit = config.input_quantization_config, operation.num_of_input
                direction = "input"
            elif "output" in tqc_name:
                tqc_list, limit = config.output_quantization_config, operation.num_of_output
                direction = "output"
            else:
                continue

            if tqc_index >= limit:
                ppq_warning(f"Your {direction} tqc index has exceeds num_of_{direction}({limit})!")
                continue

            tqc = tqc_list[tqc_index]
            tqc.num_of_bits = bits
            tqc.quant_min, tqc.quant_max = _quant_range(bits)
            tqc.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = _kl_hist_bins(bits)

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        """Build the standard quant pipeline and, on ESP-DL INT16 platforms,
        append the bit-exact LUT activation fusion pass by default.

        The pass is controlled by ``setting.lut_optimization`` (default on) and
        ``setting.lut_optimization_setting`` (exposes ``step`` / ``target_ops``).
        On non-INT16-LUT platforms it is simply not added.
        """
        pipeline = super().build_quant_pipeline(setting)

        if self._num_of_bits == 16 and setting.lut_optimization:
            lut_setting = setting.lut_optimization_setting
            pipeline.append_optimization_to_pipeline(
                EspdlLUTFusionPass(target_ops=lut_setting.target_ops, lut_step=lut_setting.step)
            )

        return pipeline

    def build_prequant_pipeline(self, setting: QuantizationSetting, executor) -> QuantizationOptimizationPipeline:
        """Prepend ESP-DL Attention, RoPE, and InverseSigmoid prequant passes."""
        pipeline = super().build_prequant_pipeline(setting, executor)

        rope_setting = setting.rotary_embedding_fusion_setting
        do_rope_fusion = bool(setting.rotary_embedding_fusion)
        verbose = bool(rope_setting.verbose)
        strict = bool(rope_setting.strict)
        dispatchings = setting.dispatching_table.dispatchings
        explicit_overrides = set(dispatchings).intersection(self._graph.operations)
        uses_builtin_dispatcher = isinstance(setting.dispatcher, str)

        if setting.inverse_sigmoid_fusion:
            pipeline.append_optimization_to_pipeline(
                InverseSigmoidFusionPass(),
                at_front=True,
            )

        # ``at_front=True`` reverses insertion order, resulting in:
        # Attention Fusion -> Attention Legalize -> RoPE Fusion -> RoPE Legalize -> InverseSigmoid Fusion.
        pipeline.append_optimization_to_pipeline(
            EspdlRotaryEmbeddingLegalizePass(
                platform=self.target_platform,
                strict=strict,
                verbose=verbose,
            ),
            at_front=True,
        )
        if do_rope_fusion:
            pipeline.append_optimization_to_pipeline(
                RotaryEmbeddingFusionPass(verbose=verbose),
                at_front=True,
            )

        pipeline.append_optimization_to_pipeline(
            EspdlAttentionLegalizePass(platform=self.target_platform, quantized=True),
            at_front=True,
        )
        if setting.espdl_attention_fusion and (
            espdl_attention_platform_unsupported_reason(self.target_platform) is None
        ):
            pipeline.append_optimization_to_pipeline(
                EspdlAttentionFusionPass(
                    dispatching_overrides=explicit_overrides,
                    allow_automatic_fp32_where=uses_builtin_dispatcher,
                ),
                at_front=True,
            )

        # Run before any op-specific fusion/legalization so every downstream pass
        # sees GRU/LSTM with canonical ONNX arity.
        pipeline.append_optimization_to_pipeline(RNNArityNormalizationPass(), at_front=True)

        return pipeline

    def quantize(
        self,
        inputs: Union[torch.Tensor, list, dict],
        calib_dataloader,
        executor,
        setting: QuantizationSetting,
        **kwargs,
    ) -> None:
        do_rope_fusion = bool(setting.rotary_embedding_fusion)
        do_attention_fusion = bool(setting.espdl_attention_fusion)
        do_inverse_sigmoid_fusion = bool(setting.inverse_sigmoid_fusion)
        needs_attention_trace = _attention_prequant_needs_trace(self._graph, self.target_platform, do_attention_fusion)
        needs_rope_trace = inputs is not None and _rope_prequant_needs_trace(self._graph, do_fusion=do_rope_fusion)
        needs_inverse_sigmoid_trace = inputs is not None and _inverse_sigmoid_prequant_needs_trace(
            self._graph,
            do_fusion=do_inverse_sigmoid_fusion,
        )
        needs_topk_trace = inputs is not None and _topk_prequant_needs_trace(self._graph)
        if needs_attention_trace or needs_rope_trace or needs_inverse_sigmoid_trace or needs_topk_trace:
            executor.tracing_operation_meta(inputs=inputs)
        _validate_espdl_topk_static_contract(self._graph)

        super().quantize(
            inputs=inputs,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            **kwargs,
        )

        finalize_espdl_inverse_sigmoid_for_export(self._graph)
        _validate_espdl_topk_quantization_contract(self._graph)

        if any(op.type == "Attention" for op in self._graph.operations.values()):
            finalize_espdl_attention_for_export(self._graph)

        rope_setting = setting.rotary_embedding_fusion_setting
        verbose = bool(rope_setting.verbose)
        strict = bool(rope_setting.strict)
        _finalize_espdl_rotary_embedding_quantization_contract(
            self._graph,
            platform=self.target_platform,
            verbose=verbose,
            strict=strict,
        )

    def quantize_operation(self, op_name: str, platform: TargetPlatform = None):
        converting_operation = self._graph.operations[op_name]
        if platform is not None:
            converting_operation.platform = platform
        _fallback_dynamic_matmul_w8a16_to_w16a16(converting_operation)
        return super().quantize_operation(op_name, platform=converting_operation.platform)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        _fallback_dynamic_matmul_w8a16_to_w16a16(operation)
        if operation.platform == self.target_platform:
            num_of_bits = self._num_of_bits
            quant_min = self._quant_min
            quant_max = self._quant_max
            bias_bits = _PLATFORM_CONFIGS[self.target_platform].bias_bits
        elif operation.platform in _PLATFORM_CONFIGS:
            pcfg = _PLATFORM_CONFIGS[operation.platform]
            num_of_bits = pcfg.num_of_bits
            quant_min, quant_max = _quant_range(num_of_bits)
            bias_bits = pcfg.bias_bits
        else:
            raise KeyError(f"EspdlQuantizer do not support operation platform : {operation.platform}.")

        config = self.create_espdl_quant_config(operation, num_of_bits, quant_min, quant_max, bias_bits)
        # Where or an upstream -inf mask Add can introduce -inf. Quantizing that
        # tensor drops the mask to a finite value and the histogram observer
        # rejects a non-finite range. Leave the consumer input in fp32.
        for index, var in enumerate(operation.inputs):
            if not _var_carries_nonfinite_mask(var):
                continue
            if index < len(config.input_quantization_config):
                config.input_quantization_config[index].state = QuantizationStates.FP32
        return config

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_INT8

    @property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @property
    def quant_operation_types(self) -> set:
        return {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "GRU",
            "LSTM",
            "Relu",
            "PRelu",
            "Clip",
            "Pad",
            "Resize",
            "MaxPool",
            "AveragePool",
            "GlobalMaxPool",
            "GlobalAveragePool",
            "Softmax",
            "LogSoftmax",
            "LpNormalization",
            "Mul",
            "Add",
            "Max",
            "Sub",
            "Div",
            "Neg",
            "Reshape",
            "LeakyRelu",
            "Concat",
            "Sigmoid",
            "Interp",
            "ReduceL1",
            "ReduceL2",
            "ReduceMean",
            "ReduceMin",
            "ReduceMax",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "Transpose",
            "Slice",
            "Flatten",
            "HardSwish",
            "HardSigmoid",
            "MatMul",
            "Attention",
            "LayerNormalization",
            "RMSNormalization",
            "Gelu",
            "PPQBiasFusedMatMul",
            "Split",
            "Gather",
            "TopK",
            "ScatterND",
            "Tanh",
            "Elu",
            "Greater",
            "Less",
            "Equal",
            "GreaterOrEqual",
            "LessOrEqual",
            "ReverseSequence",
            "Identity",
            "Swish",
            'Squeeze',
            'Unsqueeze',
            'Exp',
            'DepthToSpace',
            'SpaceToDepth',
            'InsertZeros',
            'Mod',
            'InverseSigmoid',
            'RotaryEmbedding',
        }

    @property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL
            + QuantizationProperty.LINEAR
            + QuantizationProperty.PER_TENSOR
            + QuantizationProperty.POWER_OF_2
        )

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @property
    def activation_fusion_types(self) -> set:
        """
        我不知道这个对不对, 这个是遵循 Xilinx FPGA 的修改，
        如果你的网络中有特殊的激活函数，我建议你手动调整这个融合选项

        Returns:
            set: _description_
        """
        return {"Relu", "Clip"}

    @property
    def custom_tqc(self) -> dict | None:
        return self._custom_tqc

    # The custom_op_tqc format is as follows:
    # {
    #     'op_name': {
    #         'input_0': {
    #             'bit_width': 8
    #             ......
    #         }
    #         ......
    #         'output_0': {
    #             'bit_width': 8
    #             ......
    #         }
    #     }
    # }
    @custom_tqc.setter
    def custom_tqc(self, custom_op_tqc: dict):
        self._custom_tqc = custom_op_tqc


class EspdlQuantizer(BaseEspdlQuantizer):
    pass


class EspdlInt16Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 16
        self._quant_min, self._quant_max = _quant_range(self._num_of_bits)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_INT16


class EspdlHPreInt16Quantizer(EspdlInt16Quantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_H_PRE_INT16


class EspdlW8A16Quantizer(EspdlInt16Quantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_W8A16


class EspdlS3Quantizer(BaseEspdlQuantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_INT8

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_UP


class EspdlS3Int16Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 16
        self._quant_min, self._quant_max = _quant_range(self._num_of_bits)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_INT16

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_UP


class EspdlS3HPreInt16Quantizer(EspdlS3Int16Quantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_H_PRE_INT16


class EspdlS3W8A16Quantizer(EspdlS3Int16Quantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_W8A16


class EspdlCQuantizer(BaseEspdlQuantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_INT8

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_UP


class EspdlCInt16Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 16
        self._quant_min, self._quant_max = _quant_range(self._num_of_bits)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_INT16

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_UP


class EspdlCHPreInt16Quantizer(EspdlCInt16Quantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_H_PRE_INT16


class EspdlCW8A16Quantizer(EspdlCInt16Quantizer):
    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_W8A16

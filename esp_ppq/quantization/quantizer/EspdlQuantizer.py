from __future__ import annotations

import re
from dataclasses import dataclass

from esp_ppq.api.setting import QuantizationSetting
from esp_ppq.core import (
    OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE,
    PASSIVE_OPERATIONS,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
    ppq_warning,
)
from esp_ppq.IR import BaseGraph, Operation
from esp_ppq.quantization.optim import QuantizationOptimizationPipeline

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
    config.state = state
    config.observer_algorithm = observer


_GRU_BIAS_INDICES = (3,)  # bias
_LSTM_BIAS_INDICES = (3, 6)  # bias, initial_c


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
}


_P4_PLATFORMS = {
    TargetPlatform.ESPDL_INT8,
    TargetPlatform.ESPDL_INT16,
    TargetPlatform.ESPDL_H_PRE_INT16,
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
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy,
            rounding=self.rounding_policy,
            op=operation,
            num_of_bits=num_of_bits,
            exponent_bits=0,
            quant_max=quant_max,
            quant_min=quant_min,
            observer_algorithm="percentile",
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
            for index in range(len(operation.inputs)):
                if not operation.inputs[index].name:
                    base_quant_config.input_quantization_config[index].state = QuantizationStates.FP32
        elif operation.type in {"Softmax", "LogSoftmax"}:
            base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            base_quant_config.is_active_quant_op = False

        self._apply_custom_tqc(operation, base_quant_config)

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

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
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

        return self.create_espdl_quant_config(operation, num_of_bits, quant_min, quant_max, bias_bits)

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
            "Gelu",
            "PPQBiasFusedMatMul",
            "Split",
            "Gather",
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

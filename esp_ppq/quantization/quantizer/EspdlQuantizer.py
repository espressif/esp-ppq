import re
from typing import Union

import torch

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


class BaseEspdlQuantizer(BaseQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = -128
        self._quant_max = +127
        self._custom_tqc = None

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        pipeline = super().build_quant_pipeline(setting)
        return pipeline

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

        for index in range(operation.num_of_input):
            if not operation.inputs[index].is_parameter:
                base_quant_config.input_quantization_config[index].detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = (
                    32 * int(pow(2, num_of_bits - 1))
                )

        for index in range(operation.num_of_output):
            base_quant_config.output_quantization_config[index].detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = 32 * int(
                pow(2, num_of_bits - 1)
            )

        if operation.type in {"Conv", "ConvTranspose", "Gemm"}:
            # reset num_of_bits of bias to 32 bits
            assert operation.num_of_input > 0, "Seems you got a Conv layer with no parameters."

            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.num_of_bits = bias_bits
                bias_config.quant_max = int(pow(2, bias_config.num_of_bits - 1)) - 1
                bias_config.quant_min = -int(pow(2, bias_config.num_of_bits - 1))
                bias_config.state = QuantizationStates.PASSIVE_INIT
                bias_config.observer_algorithm = "minmax"
        elif operation.type in {"LSTM"}:
            for index in range(len(operation.inputs)):
                if operation.inputs[index].name is None or len(operation.inputs[index].name) == 0:
                    base_quant_config.input_quantization_config[index].state = QuantizationStates.FP32
        elif operation.type in {"Softmax"}:
            # reset output to float32
            base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False

        # Use custom TQC to override configured TQC.
        if self._custom_tqc and self._custom_tqc.get(operation.name):
            configs = self._custom_tqc.get(operation.name)
            for tqc_name in configs.keys():
                if not configs[tqc_name].get("bit_width"):
                    continue

                tqc_index = int(re.findall(r"\d+", tqc_name)[0])
                if "input" in tqc_name:
                    if tqc_index >= operation.num_of_input:
                        ppq_warning(f"Your input tqc index has exceeds num_of_input({operation.num_of_input})!")
                        continue

                    base_quant_config.input_quantization_config[tqc_index].num_of_bits = configs[tqc_name]["bit_width"]
                    base_quant_config.input_quantization_config[tqc_index].quant_max = (
                        +int(pow(2, configs[tqc_name]["bit_width"] - 1)) - 1
                    )
                    base_quant_config.input_quantization_config[tqc_index].quant_min = -int(
                        pow(2, configs[tqc_name]["bit_width"] - 1)
                    )
                    base_quant_config.input_quantization_config[tqc_index].detail[
                        OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE
                    ] = 32 * int(pow(2, configs[tqc_name]["bit_width"] - 1))
                elif "output" in tqc_name:
                    if tqc_index >= operation.num_of_output:
                        ppq_warning(f"Your output tqc index has exceeds num_of_output({operation.num_of_output})!")
                        continue

                    base_quant_config.output_quantization_config[tqc_index].num_of_bits = configs[tqc_name]["bit_width"]
                    base_quant_config.output_quantization_config[tqc_index].quant_max = (
                        +int(pow(2, configs[tqc_name]["bit_width"] - 1)) - 1
                    )
                    base_quant_config.output_quantization_config[tqc_index].quant_min = -int(
                        pow(2, configs[tqc_name]["bit_width"] - 1)
                    )
                    base_quant_config.output_quantization_config[tqc_index].detail[
                        OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE
                    ] = 32 * int(pow(2, configs[tqc_name]["bit_width"] - 1))

        return base_quant_config

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        if operation.platform == self.target_platform:
            num_of_bits = self._num_of_bits
            quant_min = self._quant_min
            quant_max = self._quant_max
        elif operation.platform in [
            TargetPlatform.ESPDL_INT8,
            TargetPlatform.ESPDL_S3_INT8,
            TargetPlatform.ESPDL_C_INT8,
        ]:
            num_of_bits = 8
            quant_min = -128
            quant_max = 127
        elif operation.platform in [
            TargetPlatform.ESPDL_INT16,
            TargetPlatform.ESPDL_H_PRE_INT16,
            TargetPlatform.ESPDL_S3_INT16,
            TargetPlatform.ESPDL_S3_H_PRE_INT16,
            TargetPlatform.ESPDL_C_INT16,
            TargetPlatform.ESPDL_C_H_PRE_INT16,
        ]:
            num_of_bits = 16
            quant_min = -32768
            quant_max = 32767
        else:
            raise KeyError(f"EspdlQuantizer do not support operation platform : {operation.platform}.")

        bias_bits = 32
        if operation.platform == TargetPlatform.ESPDL_S3_INT8:
            bias_bits = 20
        elif operation.platform in [
            TargetPlatform.ESPDL_INT16,
            TargetPlatform.ESPDL_H_PRE_INT16,
            TargetPlatform.ESPDL_S3_INT16,
            TargetPlatform.ESPDL_S3_H_PRE_INT16,
        ]:
            bias_bits = 40
        elif operation.platform in [TargetPlatform.ESPDL_C_INT16, TargetPlatform.ESPDL_C_H_PRE_INT16]:
            bias_bits = 64

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
            "Mul",
            "Add",
            "Max",
            "Sub",
            "Div",
            "Reshape",
            "LeakyRelu",
            "Concat",
            "Sigmoid",
            "Interp",
            "ReduceMean",
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
    def rounding_policy(self):
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
    def custom_tqc(self) -> dict:
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
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)


class EspdlInt16Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 16
        self._quant_min = -32768
        self._quant_max = +32767
        self._custom_tqc = None

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_INT16


class EspdlHPreInt16Quantizer(EspdlInt16Quantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_H_PRE_INT16


class EspdlS3Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_INT8

    @property
    def rounding_policy(self):
        return RoundingPolicy.ROUND_HALF_UP


class EspdlS3Int16Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 16
        self._quant_min = -32768
        self._quant_max = +32767
        self._custom_tqc = None

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_INT16

    @property
    def rounding_policy(self):
        return RoundingPolicy.ROUND_HALF_UP


class EspdlS3HPreInt16Quantizer(EspdlS3Int16Quantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_S3_H_PRE_INT16


class EspdlCQuantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_INT8

    @property
    def rounding_policy(self):
        return RoundingPolicy.ROUND_HALF_UP


class EspdlCInt16Quantizer(BaseEspdlQuantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)
        self._num_of_bits = 16
        self._quant_min = -32768
        self._quant_max = +32767
        self._custom_tqc = None

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_INT16

    @property
    def rounding_policy(self):
        return RoundingPolicy.ROUND_HALF_UP


class EspdlCHPreInt16Quantizer(EspdlCInt16Quantizer):
    def __init__(self, graph: BaseGraph) -> None:
        super().__init__(graph=graph)

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPDL_C_H_PRE_INT16

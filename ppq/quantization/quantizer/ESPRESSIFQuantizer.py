from typing import Union

import re
import torch
from ppq.api.setting import QuantizationSetting
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform,
                      ppq_warning, OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE)
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim import QuantizationOptimizationPipeline

from .base import BaseQuantizer


class ESPRESSIFQuantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph,
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - 128
        self._quant_max = + 127
        self._custom_tqc = None

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        pipeline = super().build_quant_pipeline(setting)
        return pipeline

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # # if operation has bias
            # if operation.num_of_input > 2:
            #     bias_config = base_quant_config.input_quantization_config[-1]
            #     bias_config.policy = QuantizationPolicy(
            #         QuantizationProperty.SYMMETRICAL +
            #         QuantizationProperty.LINEAR +
            #         QuantizationProperty.PER_TENSOR +
            #         QuantizationProperty.POWER_OF_2)

            #     # Xilinx FPGA bias 并不是 32 位的！
            #     bias_config.num_of_bits = 30
            #     bias_config.quant_max = + int(pow(2, 29))
            #     bias_config.quant_min = - int(pow(2, 29))
            #     bias_config.state = QuantizationStates.PASSIVE_INIT
            #     base_quant_config.input_quantization_config[-1].observer_algorithm = 'Minmax'

            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False

        # Use custom TQC to override configured TQC.
        if self._custom_tqc and self._custom_tqc.get(operation.name):
            configs = self._custom_tqc.get(operation.name)
            for tqc_name in configs.keys():
                if not configs[tqc_name].get('bit_width'):
                    continue

                tqc_index = int(re.findall(r"\d+", tqc_name)[0])
                if 'input' in tqc_name:
                    if tqc_index >= operation.num_of_input:
                        ppq_warning(f'Your input tqc index has exceeds num_of_input({operation.num_of_input})!')
                        continue

                    base_quant_config.input_quantization_config[tqc_index].num_of_bits = configs[tqc_name]['bit_width']
                    base_quant_config.input_quantization_config[tqc_index].quant_max = + int(pow(2, configs[tqc_name]['bit_width'] - 1)) - 1
                    base_quant_config.input_quantization_config[tqc_index].quant_min = - int(pow(2, configs[tqc_name]['bit_width'] - 1))
                    base_quant_config.input_quantization_config[tqc_index].detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = 32 * int(pow(2, configs[tqc_name]['bit_width'] - 1))
                elif 'output' in tqc_name:
                    if tqc_index >= operation.num_of_output:
                        ppq_warning(f'Your output tqc index has exceeds num_of_output({operation.num_of_output})!')
                        continue

                    base_quant_config.output_quantization_config[tqc_index].num_of_bits = configs[tqc_name]['bit_width']
                    base_quant_config.output_quantization_config[tqc_index].quant_max = + int(pow(2, configs[tqc_name]['bit_width'] - 1)) - 1
                    base_quant_config.output_quantization_config[tqc_index].quant_min = - int(pow(2, configs[tqc_name]['bit_width'] - 1))
                    base_quant_config.output_quantization_config[tqc_index].detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] = 32 * int(pow(2, configs[tqc_name]['bit_width'] - 1))

        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ESPRESSIF_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'Max', 'Sub', 'Div',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Slice'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR +
            QuantizationProperty.POWER_OF_2
        )

    @ property
    def rounding_policy(self):
        return RoundingPolicy.ROUND_HALF_UP

    @ property
    def activation_fusion_types(self) -> set:
        """
        我不知道这个对不对, 这个是遵循 Xilinx FPGA 的修改，
        如果你的网络中有特殊的激活函数，我建议你手动调整这个融合选项

        Returns:
            set: _description_
        """
        return {'Relu', 'Clip'}

    @ property
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
    @ custom_tqc.setter
    def custom_tqc(self, custom_op_tqc: dict):
        self._custom_tqc = custom_op_tqc

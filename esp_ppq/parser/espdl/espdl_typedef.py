from typing import List, Union

import torch

from esp_ppq.core import OperationQuantizationConfig, SingletonMeta

ACTIVATION_OP_SET = {
    "Relu",
    "PRelu",
    "Sigmoid",
    "Tanh",
    "HardSwish",
    "HardSigmoid",
    "Elu",
    "Erf",
    "Gelu",
    "Clip",
    "Cast",
    "LeakyRelu",
    "Softplus",
    "Identity",
    "Swish",
}

MATH_OP_SET = {
    "Abs",
    "Pow",
    "Exp",
    "Log",
    "Sqrt",
    "Cos",
    "Sin",
    "Tan",
    "Not",
    "Floor",
    "Round",
    "Ceil",
    "Neg",
}

QUANT_OP_SET = {
    "RequantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "QuantizeFloating",
    "DequantizeFloating",
}
PASSIVE_LAYOUT_OP_SET = ACTIVATION_OP_SET | QUANT_OP_SET | MATH_OP_SET
CONV_LAYOUT_OP_SET = {"Conv", "GlobalAveragePool", "AveragePool", "MaxPool", "ConvTranspose", "GlobalMaxPool"}
ADD_LIKE_OP_SET = {"Add", "Sub", "Mul", "Div", "And", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"}
OTHER_OP_SET = {
    "MatMul",
    "Gemm",
    "Flatten",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Slice",
    "Pad",
    "Concat",
    "Constant",
    "Gather",
    "GatherElements",
    "GatherND",
    "Shape",
    "ConstantOfShape",
    "ReverseSequence",
    "Expand",
    "ReduceMean",
    "ReduceL2",
    "ReduceMax",
    "ReduceSum",
    "LayerNorm",
    "LayerNormalization",
    "Max",
    "Min",
    "ScatterElements",
    "ScatterND",
    "TopK",
    "Where",
    "GRU",
    "LSTM",
    "Sum",
}

SOFTMAX_LIKE_OP_SET = {"Softmax", "LogSoftmax", "Split"}
# QUANT_EXCLUDE_OP_SET refers to operators that do not participate
# in the operations of quantize, dequantize, or requantize.
QUANT_EXCLUDE_OP_SET = {"Shape"}


class EspQuantType:
    F32 = "F32"
    S16 = "S16"
    S8 = "S8"


class LayoutAnnotation:
    NCHW = "NCHW"
    NHWC = "NHWC"
    HWCN = "HWCN"
    N16HWC16 = "(N/16)HWC16"
    N8HWC8 = "(N/8)HWC8"
    N16HWC16_UNALIGNED = "(N/16)HWC16_UNALIGNED"
    N8HWC8_UNALIGNED = "(N/8)HWC8_UNALIGNED"
    NCW = "NCW"
    NWC = "NWC"
    WCN = "WCN"
    N16WC16 = "(N/16)WC16"
    N8WC8 = "(N/8)WC8"
    N16WC16_UNALIGNED = "(N/16)WC16_UNALIGNED"
    N8WC8_UNALIGNED = "(N/8)WC8_UNALIGNED"
    UNKNOWN = "UNK"


class ExporterPatternInfo(metaclass=SingletonMeta):
    var_exponents = {}
    var_layout = {}
    var_permute = {}
    var_config = {}
    luts = {}

    def reset(self):
        self.var_exponents = {}
        self.var_layout = {}
        self.var_permute = {}
        self.var_config = {}
        self.luts = {}

    def get_var_exponents(self, var_name: str, default: List[int] = None) -> Union[int, List[int]]:
        return self.var_exponents.get(var_name, default)

    def get_var_layout(self, var_name: str, default: LayoutAnnotation = None) -> LayoutAnnotation:
        return self.var_layout.get(var_name, default)

    def get_var_permute(self, var_name: str, default: List[int] = None) -> List[int]:
        return self.var_permute.get(var_name, default)

    def get_var_config(self, var_name: str, default: OperationQuantizationConfig = None) -> OperationQuantizationConfig:
        return self.var_config.get(var_name, default)

    def get_lut(self, lut_name: str, default: torch.Tensor = None) -> torch.Tensor:
        return self.luts.get(lut_name, default)

    def add_var_exponents(self, var_name: str, exponent: Union[int, List[int]]):
        if isinstance(exponent, int):
            exponent = [exponent]
        self.var_exponents[var_name] = exponent

    def add_var_layout(self, var_name: str, layout: LayoutAnnotation):
        self.var_layout[var_name] = layout

    def add_var_permute(self, var_name: str, perm: List[int]):
        self.var_permute[var_name] = perm

    def add_var_config(self, var_name: str, config: OperationQuantizationConfig):
        self.var_config[var_name] = config

    def add_lut(self, lut_name: str, lut: torch.Tensor, exponent: Union[int, List[int]]):
        self.luts[lut_name] = lut
        self.var_exponents[lut_name] = exponent

    def print(self):
        print(self.var_exponents)
        print(self.var_layout)
        print(self.var_permute)
        print(self.var_config)

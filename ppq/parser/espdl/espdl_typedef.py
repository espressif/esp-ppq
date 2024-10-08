from typing import List, Union

from ppq.core import OperationQuantizationConfig, SingletonMeta

ACTIVATION_OP_SET = {
    "Relu",
    "PRelu",
    "Sigmoid",
    "Tanh",
    "HardSwish",
    "Elu",
    "Gelu",
    "Clip",
    "Cast",
}
QUANT_OP_SET = {
    "RequantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "QuantizeFloating",
    "DequantizeFloating",
}
PASSIVE_LAYOUT_OP_SET = ACTIVATION_OP_SET | QUANT_OP_SET
CONV_LAYOUT_OP_SET = {"Conv", "GlobalAveragePool", "AveragePool", "MaxPool"}
ADD_LIKE_OP_SET = {"Add", "Sub", "Mul", "Div"}
OTHER_OP_SET = {
    "Matmul",
    "Gemm",
    "Flatten",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Slice",
    "Pad",
    "Split",
    "Concat",
    "Constant",
    "Gather",
    "Shape",
    "ConstantOfShape",
    "Expand",
    "ReduceMean",
    "Softmax",
    "LogSoftmax"
}
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
    N16HWC16 = "N16HWC16"
    N8HWC8 = "N8HWC8"
    N16HWC16_UNALIGNED = "N16HWC16_UNALIGNED"
    N8HWC8_UNALIGNED = "N8HWC8_UNALIGNED"
    UNKNOWN = "UNK"


class ExporterPatternInfo(metaclass=SingletonMeta):
    var_exponents = {}
    var_layout = {}
    var_permute = {}
    var_config = {}

    def reset(self):
        self.var_exponents = {}
        self.var_layout = {}
        self.var_permute = {}
        self.var_config = {}

    def get_var_exponents(
        self, var_name: str, default: List[int] = None
    ) -> Union[int, List[int]]:
        return self.var_exponents.get(var_name, default)

    def get_var_layout(
        self, var_name: str, default: LayoutAnnotation = None
    ) -> LayoutAnnotation:
        return self.var_layout.get(var_name, default)

    def get_var_permute(self, var_name: str, default: List[int] = None) -> List[int]:
        return self.var_permute.get(var_name, default)

    def get_var_config(
        self, var_name: str, default: OperationQuantizationConfig = None
    ) -> OperationQuantizationConfig:
        return self.var_config.get(var_name, default)

    def add_var_exponents(self, var_name: str, exponent: Union[int, List[int]]):
        self.var_exponents[var_name] = exponent

    def add_var_layout(self, var_name: str, layout: LayoutAnnotation):
        self.var_layout[var_name] = layout

    def add_var_permute(self, var_name: str, perm: List[int]):
        self.var_permute[var_name] = perm

    def add_var_config(self, var_name: str, config: OperationQuantizationConfig):
        self.var_config[var_name] = config

    def print(self):
        print(self.var_exponents)
        print(self.var_layout)
        print(self.var_permute)
        print(self.var_config)

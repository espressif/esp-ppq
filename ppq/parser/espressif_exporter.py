import json
import os
import sys
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
import torch

from ppq.core import (
    ONNX_VERSION,
    PPQ_CONFIG,
    DataType,
    QuantizationProperty,
    QuantizationStates,
    TargetPlatform,
    TensorQuantizationConfig,
    convert_any_to_numpy,
    convert_any_to_torch_tensor,
)
from ppq.IR import BaseGraph, GraphExporter, Operation, OperationExporter, Variable
from ppq.IR.quantize import QuantableOperation
from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt

from .fbs_construct import helper
from .onnx_exporter import OP_CONVERTERS
from .onnxruntime_exporter import QDQHelper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from .espressif_export_patterns import (
    FUSE_PATTERNS,
    QUANT_OP_SET,
    EspQuantType,
    InsertDequantNodePattern,
    InsertQuantNodePattern,
    InsertRequantNodePattern,
    LayoutAnnotation,
)
from .fbs_construct.FlatBuffers.Dl.Attribute import AttributeT
from .fbs_construct.FlatBuffers.Dl.Model import ModelT
from .fbs_construct.FlatBuffers.Dl.Node import NodeT
from .fbs_construct.FlatBuffers.Dl.Tensor import TensorT
from .fbs_construct.FlatBuffers.Dl.ValueInfo import ValueInfoT


def as_c_array(byte_arr):
    hex_str = ''
    for idx, byte in enumerate(byte_arr):
        hex_str += "0x{:02x}, ".format(byte)
    return hex_str

def convert_value(value: Union[int, float, np.ndarray, torch.Tensor]) -> Any:
    if type(value) in {int, float}:
        return value
    else:
        value = convert_any_to_numpy(value, accept_none=True)
        if value is None:
            return value  # SOI config has Nona as its scale and
        return value.tolist()
    
def quantize_value_info(var: Variable, op: QuantableOperation):
    quant_type = op.attributes.get("quant_type", None)
    if quant_type == EspQuantType.S8:
        var.dtype = DataType.INT8
    elif quant_type == EspQuantType.S16:
        var.dtype = DataType.INT16
    else:
        var.dtype = DataType.FP32
    
    return var

def transpose_tensor(var: Variable, op: QuantableOperation):
    tensor = var.value
    tensor_ret = tensor
    layout = LayoutAnnotation.NCHW

    if op.type == "Conv":
        if len(tensor.shape) == 4: # Conv2d Filter
            quant_type = op.attributes.get("quant_type", None)
            if quant_type == None or quant_type == EspQuantType.F32:
                return tensor_ret, layout

            n,c,h,w = tensor.shape # n denotes output channels, c denotes input channels,
            tensor = tensor.permute(0, 2, 3, 1)   # NCHW -> NHWC

            align = 16 if quant_type == EspQuantType.S8 else 8
            aligned_len = n // align * align
            aligned_tensor = tensor[0:aligned_len, ...]
            aligned_tensor = aligned_tensor.reshape(n // align,  align, h, w, c) # NHWC -> (N/align,align)HWC
            aligned_tensor = aligned_tensor.permute(0, 2, 3, 4, 1)          #(N/align,align)HWC -> (N/align)HWC(align)
            aligned_tensor = aligned_tensor.reshape(aligned_len, h, w, c)   #(N/align)HWC(align) -> (aligned_len)HWC

            group = op.attributes.get("group", None)
            if n % align != 0:
                unaligned_tensor = tensor[aligned_len:n, ...] # NHWC
                if group == 1:
                    aligned_tensor = torch.cat((aligned_tensor, unaligned_tensor), 0)
                else:
                    n_remain = n - aligned_len
                    unaligned_tensor = unaligned_tensor.permute(3, 1, 2, 0) # depthwise unaligned: NHWC -> CHWN
                    unaligned_tensor = unaligned_tensor.reshape(n_remain, h, w, c)
                    aligned_tensor = torch.cat((aligned_tensor, unaligned_tensor), 0)

                if align == 16:
                    layout = LayoutAnnotation.N16HWC16_UNALIGNED
                else:
                    layout = LayoutAnnotation.N8HWC8_UNALIGNED
            else:
                if align == 16:
                    layout = LayoutAnnotation.N16HWC16
                else:
                    layout = LayoutAnnotation.N8HWC8

            # #TODO:: modify the layout of depthwise conv in ESP-DL, keep it same with conv
            # if group == 1:
            #     aligned_tensor = aligned_tensor.reshape(h,w,c,n) # reshape to HWCN
            # else:
            #     aligned_tensor = aligned_tensor.reshape(h,w,n,c) # reshape to HWNC
            tensor_ret = aligned_tensor.reshape(n, c, h, w)

    elif op.type == "Gemm":
        if len(tensor.shape) == 2:  # Gemm Filter
            quant_type = op.attributes.get("quant_type", None)
            if quant_type == None or quant_type == EspQuantType.F32:
                return tensor_ret, layout

            trans_filter = op.attributes.get("transB", 0)
            if trans_filter:
                tensor = tensor.transpose(0, 1)  # [n, c] -> [c, n]

            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
            c, n, h, w = tensor.shape   # n denotes out_features, c denotes in_features
            tensor = tensor.permute(1, 2, 3, 0)   # CNHW -> NHWC

            align = 16 if quant_type == EspQuantType.S8 else 8
            aligned_len = n // align * align
            aligned_tensor = tensor[0:aligned_len, ...]
            aligned_tensor = aligned_tensor.reshape(n // align, align, h, w, c) # NHWC -> (N/align,align)HWC
            aligned_tensor = aligned_tensor.permute(0, 2, 3, 4, 1)          #(N/align,align)HWC -> (N/align)HWC(align)
            aligned_tensor = aligned_tensor.reshape(aligned_len, h, w, c)   #(N/align)HWC(align) -> (aligned_len)HWC

            if n % align != 0:
                unaligned_tensor = tensor[aligned_len:n, ...] # NHWC
                aligned_tensor = torch.cat((aligned_tensor, unaligned_tensor), 0)

                if align == 16:
                    layout = LayoutAnnotation.N16HWC16_UNALIGNED
                else:
                    layout = LayoutAnnotation.N8HWC8_UNALIGNED
            else:
                if align == 16:
                    layout = LayoutAnnotation.N16HWC16
                else:
                    layout = LayoutAnnotation.N8HWC8

            tensor_ret = aligned_tensor.reshape(n, c, h, w)

    return tensor_ret, layout


def calculate_exponent(config: TensorQuantizationConfig):
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
        raise ValueError('Critical Quantization Error! Asymmetrical config detected.')
    
    if not config.scale:
        return None
    
    exponent = None
    if config.policy.has_property(QuantizationProperty.PER_TENSOR) and config.policy.has_property(QuantizationProperty.POWER_OF_2):
        scale = convert_any_to_numpy(config.scale)
        exponent = [int(np.log2(scale))]
    elif config.policy.has_property(QuantizationProperty.PER_CHANNEL) and config.policy.has_property(QuantizationProperty.POWER_OF_2):
        scale = convert_any_to_numpy(config.scale)
        exponent = np.log2(scale).astype(int)
    return exponent

class EspressifExporter(GraphExporter):
    """
    PPQ 可以将 计算图 导出成 Onnx 标准格式，Onnx Exporter 不会导出 QDQ 节点。
    如需导出带有 QDQ 节点的 Onnx 文件，用户需要使用 OnnxRuntime Exporter

    任何导出器的导出逻辑都是原地进行的，它们将对传入的计算图对象进行原地修改，因此在导出之前你需要手动克隆计算图。
    """

    def __init__(self) -> None:
        super().__init__()

    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        """Export Tensor Quantization Config to File(Json)."""

        render_buffer = {"configs": {}, "dispatchings": {}, "values": {}}

        # Render quantization config.
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                op_dict = {
                    var.name: {
                        "bit_width": config.num_of_bits,
                        "policy": config.policy.to_dict(),
                        "state": config.state.name,
                        "quant_min": config.quant_min,
                        "quant_max": config.quant_max,
                        "hash": config.__hash__(),
                        "dominator": config.dominated_by.__hash__(),
                    }
                    for config, var in operation.config_with_variable
                }

                for config, _ in operation.config_with_variable:
                    if config.dominated_by == config:
                        if config.state != QuantizationStates.FP32:
                            render_buffer["values"][config.__hash__()] = {
                                "scale": convert_value(config.scale),
                                "zero_point": convert_value(config.offset),
                            }

                render_buffer["configs"][operation.name] = op_dict
                render_buffer["dispatchings"][operation.name] = operation.platform.name

        with open(file=config_path, mode="w") as file:
            json.dump(render_buffer, file, indent=4)
    
    def insert_quant_type(self, op: Operation) -> Operation:
        """insert quantization type 

        Args:
            op (Operation): Converting op
        """
        # op.attributes['domain'] = 'esp-dl'
        if op.platform == TargetPlatform.ESPRESSIF_INT8:
            op.attributes["quant_type"] = EspQuantType.S8
        elif op.platform == TargetPlatform.ESPRESSIF_INT16:
            op.attributes["quant_type"] = EspQuantType.S16
        else:
            op.attributes["quant_type"] = EspQuantType.F32
        
        return op
    
    def quantize_variable(self, 
                          op: QuantableOperation, 
                          exponents: Dict[str, List[int]], 
                          graph:BaseGraph, 
                          valuesForTest: Dict[str, Dict[str, torch.Tensor]] = None,
                          valuesForTestQ: Dict[str, Dict[str, np.ndarray]] = None
                        ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        """ 
        Args:
            valuesForTest (Dict[str, Dict[str, torch.Tensor]]): the test values used to compare accuracy.
                                                              The input format is as follows:
                {
                    'inputs': {
                        'input_0_name': torch.Tensor
                        ......
                        'input_n_name': torch.Tensor
                    },
                    'outputs': {
                        'output_0_name': torch.Tensor
                        ......
                        'output_n_name': torch.Tensor
                    },
                }

        """

        layouts = {}

        # collect quantable vars, where we need to quantize parameters
        for config, var in [_ for _ in op.config_with_variable]:
            if not var or not config:
                continue

            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                # print("Warning: skip %s because it's not exportable" % (var.name))
                continue

            if var.name not in exponents:
                var_exponent = calculate_exponent(config)

                if var_exponent:
                    exponents[var.name] = var_exponent
                else:
                    print("Skip %s from (op name:%s, type:%s) because it's not quantized" % (var.name, op.name, op.type))
            else:
                continue

            if var.is_parameter:
                assert len(var.dest_ops) == 1, (
                f'Can not export variable {var.name}, cause it has more than 1 destination operations. '
                'PPQ require all parameters to have only 1 destination operation.')

                # override quantization state, so that we can export parameter correctly.
                if config.state == QuantizationStates.BAKED:
                    config.state = QuantizationStates.ACTIVATED
                if config.state == QuantizationStates.PASSIVE_BAKED:
                    config.state = QuantizationStates.PASSIVE

                if config.policy.has_property(QuantizationProperty.LINEAR):
                    var.value = PPQLinearQuant_toInt(tensor=var.value, config=config)
                    var.value, layout = transpose_tensor(var, op)
                    layouts[var.name] = layout
            elif (not var.is_parameter):
                if config.policy.has_property(QuantizationProperty.LINEAR):
                    quant_type = op.attributes.get("quant_type", None)
                    if quant_type == EspQuantType.S8:
                        var.dtype = DataType.INT8
                    elif quant_type == EspQuantType.S16:
                        var.dtype = DataType.INT16
                    else:
                        var.dtype = DataType.FP32

                    if valuesForTest is not None and valuesForTestQ is not None:
                        if 'inputs' not in valuesForTestQ:
                            valuesForTestQ['inputs'] = {}
                        if 'outputs' not in valuesForTestQ:
                            valuesForTestQ['outputs'] = {}

                        if var.name in valuesForTest.get('inputs', {}):
                            valuesForTestQ['inputs'][var.name] = convert_any_to_numpy(
                                PPQLinearQuant_toInt(tensor = valuesForTest['inputs'][var.name], config = config)
                                )
                        elif var.name in valuesForTest.get('outputs', {}):
                            valuesForTestQ['outputs'][var.name] = convert_any_to_numpy(
                                PPQLinearQuant_toInt(tensor = valuesForTest['outputs'][var.name], config = config)
                                )

        return exponents, layouts


    def convert_operation_from_opset11_to_opset13(self, graph:BaseGraph) -> None:
        """Convert your network from opset 11 standard towards opset 13 With
        Onnx definition, per-channel quant operation requires opset 13.

        Args:
            graph (BaseGraph): Processing graph.
        """
        # this func transform representation of certain op from opset 11 to 13
        for op in graph.operations.values():
            if op.type == 'ReduceSum' or op.type == 'Squeeze' or op.type == 'Unsqueeze':
                if 'axes' not in op.attributes: continue # is already v13
                axes = convert_any_to_torch_tensor(op.attributes.pop('axes'), dtype=torch.int64)
                graph.create_variable(name=None, value=axes, is_parameter=True, dest_ops=[op])

            elif op.type == 'Split':
                if 'split' not in op.attributes: continue # split is already v13
                split = convert_any_to_torch_tensor(op.attributes.pop('split'), dtype=torch.int64)
                graph.create_variable(name=None, value=split, is_parameter=True, dest_ops=[op])

    def prepare_graph(self,
                      graph: BaseGraph) -> BaseGraph:
        """Prepare your graph for exporting.

        There are many works to do with your graph:

            1. Insert Quant and Dequant operation within your graph.

            2. Remove all unnecessary activations.

            3. Quantize all parameters of your graph, convert them to int8.

        Args:
            graph (BaseGraph): Processing Graph

        Returns:
            BaseGraph: Processed Graph
        """
        self.convert_operation_from_opset11_to_opset13(graph)

        # Insert Quant, Dequant or Requant operation within your graph.
        print("Espressif Exporter: Insert Quant, Requant, Dequant Operation ...")
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation): continue
            if op.type in QUANT_OP_SET: continue
            
            # Insert Quant op
            InsertQuantNodePattern().export(op=op, graph=graph)

            # Insert Requant op
            InsertRequantNodePattern().export(op=op, graph=graph)

            # Insert Dequant op
            InsertDequantNodePattern().export(op=op, graph=graph)

        # fuse ops
        print("Espressif Exporter: Execute fusion pattern ...")
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation): continue
            # The FUSE_PATTERNS may remove some ops.
            if op.name not in graph.operations: continue
            if op.type in FUSE_PATTERNS:
                exporter = FUSE_PATTERNS[op.type]()
                assert isinstance(
                    exporter, OperationExporter
                ), f"Expected an OpExporter here, however {type(exporter)} was given."
                op = exporter.export(op=op, graph=graph)
        
        # reset layout
        print("Espressif Exporter: Reset layout ...")

        return graph

    def export_graph(
            self,
            graph: BaseGraph,
            modelVersion: int = 0,
            valuesForTest: Dict[str, Dict[str, torch.Tensor]] = None
        ) -> ModelT:
        """
        Convert a PPQ IR to Onnx IR.
        This export will only convert PPQ Op and var to onnx, all quantization configs will be skipped.

        This function will try to keep the opset version of your graph unchanged.
        However if the opset is not given, ppq will convert it to with the global parameter ppq.core.ONNX_EXPORT_OPSET.
        """
        name = graph._name
        if not name:
            name = f"{PPQ_CONFIG.NAME} - v({PPQ_CONFIG.VERSION})"
        

        # Ready to export onnx graph defination.
        _inputs, _outputs, _initilizers, _nodes, _value_info = [], [], [], [], []
        exponents = {}
        layouts = {}

        # before we can export them, we firstly convert all ops to proper format. Copy from onnx exporter
        for op in [_ for _ in graph.topological_sort()]:
            if op.type in OP_CONVERTERS:
                exporter = OP_CONVERTERS[op.type]()
                assert isinstance(
                    exporter, OperationExporter
                ), f"Expected an OpExporter here, however {type(exporter)} was given."
                op = exporter.export(op=op, graph=graph)

        valuesForTestQ = {}
        for op in graph.topological_sort():
            op = self.insert_quant_type(op)
            if isinstance(op, QuantableOperation):
                exponents, layout = self.quantize_variable(op, exponents, graph, valuesForTest, valuesForTestQ)
                layouts.update(layout)
            _nodes.append(self.build_operator_proto(op))

        for variable in graph.variables.values():
            tensor_proto = self.build_variable_proto(variable, exponents, layouts)
            if variable.name in graph.inputs:
                _inputs.append(tensor_proto)
            if variable.name in graph.outputs:
                _outputs.append(tensor_proto)
            if variable.is_parameter:
                _initilizers.append(tensor_proto)
            else:
                _value_info.append(tensor_proto)

        test_inputs_value, test_outputs_value = helper.make_graph_test_value(valuesForTestQ, exponents)

        graph_def = helper.make_graph(
            name=name,
            nodes=_nodes,
            inputs=_inputs,
            outputs=_outputs,
            initializer=_initilizers,
            value_info=_value_info,
            test_inputs_value = test_inputs_value,
            test_outputs_value = test_outputs_value
        )

        onnx_model = helper.make_model(
                            graph = graph_def,
                            producerName = PPQ_CONFIG.NAME,
                            modelVersion = modelVersion
                        )
        onnx_model.ir_version = graph._detail.get("ir_version", ONNX_VERSION)
        return onnx_model

    def build_operator_proto(self, operation: Operation) -> NodeT:
        """
        Convert PPQ Op to Onnx Operation
        An Op consumes zero or more Tensors, and produces zero or more Tensors.
        """
        attributes = operation.attributes
        for key in attributes:
            value = attributes[key]
            if isinstance(value, DataType):
                attributes[key] = value.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    attributes[key] = None
                elif value.numel() == 1:
                    attributes[key] = convert_any_to_numpy(
                        [value.item()]
                    )  # convert to 1d array
                else:
                    attributes[key] = convert_any_to_numpy(value)

        if PPQ_CONFIG.EXPORT_PPQ_INTERNAL_INFO:
            attributes["platform"] = operation.platform.name

        op_proto = helper.make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            **attributes,
        )

        return op_proto

    def build_variable_proto(self, variable: Variable, exponents: Dict[str, List[int]], layouts: Dict[str, str]) -> Union[ValueInfoT, TensorT]:
        """
        Convert PPQ Variable to Onnx TensorProto, There are 2 different types of Tensor in Onnx:
            Variable: Represents a Tensor whose value is not known until inference-time.
            Constant: Represents a Tensor whose value is known.
        """
        # Parameter Varaible in PPQ, Constant Variable in Onnx
        if variable.is_parameter:
            if variable.value is not None:
                var_shape = variable.value.shape
                pytorch_dtype = variable.value.dtype
                onnx_dtype = DataType.convert_from_torch(pytorch_dtype).value

        # Non Parameter
        else:
            var_shape = variable.shape
            onnx_dtype = variable.dtype.value
        
        # if variable not in exponents, set exponent to 0
        var_exponents = exponents.get(variable.name, [0])

        if not variable.is_parameter:
            tensor_proto = helper.make_tensor_value_info(
                name=variable.name, elem_type=onnx_dtype, shape=var_shape, exponents=var_exponents
            )
        else:
            value = variable.value
            is_raw_format = False
            var_layout = layouts.get(variable.name, "")
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    value = []
                elif value.ndim >= 1:
                    value = convert_any_to_numpy(variable.value).flatten()
                    value = value.tobytes()
                    is_raw_format = True
                elif value.ndim == 0:  # Pytorch Scalar Type
                    value = [
                        value.item(),
                    ]  # it is fine for onnx, shape for this value will be []
            else:
                value = value  # value is python primary type.
            tensor_proto = helper.make_tensor(
                name=variable.name,
                data_type=onnx_dtype,
                dims=var_shape,
                vals=value,
                raw=is_raw_format,
                exponents=var_exponents
            )
            tensor_proto.docString = "layout ==> " + var_layout
        return tensor_proto


    def modify_shape_info(
            self,
            graph: BaseGraph,
            fbsModel: ModelT
        ):

        # modify graph input shape info
        for var in graph.inputs.values():
            for op in var.dest_ops:
                if "Conv" == op.type:
                    var_shape = var.shape
                    if len(var_shape) == 4:     # conv2d, NCHW -> NHWC
                        c = var_shape[1]
                        var_shape[1:3] = var_shape[2:]
                        var_shape[3] = c
                    else:                       # conv1d, NCW -> NWC
                        c = var_shape[1]
                        var_shape[1] = var_shape[2]
                        var_shape[2] = c

                    if var_shape[0] == 1:
                        var_shape.pop(0)

                    var_fbs: ValueInfoT = None
                    input_fbs: ValueInfoT = None
                    for i, var_fbs in enumerate(fbsModel.graph.valueInfo):
                        # modify graph valueInfo
                        if var_fbs.name == var.name:
                            fbsModel.graph.valueInfo[i] = helper.make_tensor_value_info(name = var_fbs.name, 
                                                                                    elem_type = var.dtype.value, 
                                                                                    shape = var_shape, 
                                                                                    exponents = var_fbs.exponents
                                                                                    )
                            for j, input_fbs in enumerate(fbsModel.graph.input):
                                # modify graph input
                                if input_fbs.name == var.name:
                                    fbsModel.graph.input[j] = fbsModel.graph.valueInfo[i]
                                    break

                            break

                    break

        # modify filter shape info
        op_fbs: NodeT = None
        for i, op_fbs in enumerate(fbsModel.graph.node):
            if "Conv" == op_fbs.opType:
                op_group: int = 1
                attr_fbs: AttributeT = None
                for attr_fbs in op_fbs.attribute:
                    if attr_fbs.name == "group":
                        op_group = attr_fbs.i.i
                        break

                filter_name_fbs: str = op_fbs.input[1]
                tensor_fbs: TensorT = None
                for tensor_fbs in fbsModel.graph.initializer:
                    if tensor_fbs.name == filter_name_fbs:
                        if len(tensor_fbs.dims) == 4:   # conv2d
                            if op_group == 1:           # shape info: NCHW -> HWCN
                                n, c = tensor_fbs.dims[0 : 2]
                                tensor_fbs.dims[0:2] = tensor_fbs.dims[2:]
                                tensor_fbs.dims[2] = c
                                tensor_fbs.dims[3] = n
                            else:                       # only support depthwise conv2d: NCHW -> HWNC
                                n, c = tensor_fbs.dims[0 : 2]
                                tensor_fbs.dims[0:2] = tensor_fbs.dims[2:]
                                tensor_fbs.dims[2] = n
                                tensor_fbs.dims[3] = c

                        break

            elif "Gemm" == op_fbs.opType:
                filter_name_fbs: str = op_fbs.input[1]
                tensor_fbs: TensorT = None
                for tensor_fbs in fbsModel.graph.initializer:
                    if tensor_fbs.name == filter_name_fbs:
                        if len(tensor_fbs.dims) == 4:
                            # shape info: NCHW -> HWCN
                            n, c = tensor_fbs.dims[0 : 2]
                            tensor_fbs.dims[0:2] = tensor_fbs.dims[2:]
                            tensor_fbs.dims[2] = c
                            tensor_fbs.dims[3] = n

                        break

        return


    def export(
        self,
        file_path: str,
        graph: BaseGraph,
        config_path: str = None,
        modelVersion: int = 0,
        valuesForTest: Dict[str, Dict[str, torch.Tensor]] = None,
        encrypt_data: bool = False,
        **kwargs: Any
    ):
        """ Export model to flatbuffers file

        Args:
            file_path: flatbuffers file
            graph: ppq graph
            config_path: quantization config
            modelVersion (int): model version
            valuesForTest (Dict[str, Dict[str, np.ndarray]]): the test values used to compare accuracy.
                                                            The input format is as follows:
                {
                    'inputs': {
                        'input_0_name': np.ndarray
                        ......
                        'input_n_name': np.ndarray
                    },
                    'outputs': {
                        'output_0_name': np.ndarray
                        ......
                        'output_n_name': np.ndarray
                    },
                }

        """

        # during export we will remove all boundary operations from graph.
        # we do not want to change the structure of original graph,
        # so there have to take a clone of it.
        graph = graph.copy()

        # In prepare stage, run all graph pattern, e.g. fuse Conv and Relu
        graph = self.prepare_graph(graph = graph)

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

        model: ModelT = self.export_graph(graph = graph,
                                          modelVersion = modelVersion,
                                          valuesForTest = valuesForTest)

        # Export the information of quantization model.
        file_base_name, _ = os.path.splitext(file_path)
        file_info_path = file_base_name + ".info"
        with open(file_info_path, 'w') as file_info:
            file_info.write(helper.printable_graph(model.graph, print_initializer_value = True, print_value_info = True, print_test_value = True))

        self.modify_shape_info(graph = graph, fbsModel = model)

        key = helper.save(
            model,
            file_path,
            encrypt_data   # if True, encrypt data for security
        )
        if key:
            print("AES 128-bit secret key: {}".format(as_c_array(key)))

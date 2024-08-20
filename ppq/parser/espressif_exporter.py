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
    OperationQuantizationConfig,
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
from ppq.utils.round import ppq_tensor_round

from .fbs_construct import helper
from .onnx_exporter import OP_CONVERTERS
from .onnxruntime_exporter import QDQHelper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from .fbs_construct.FlatBuffers.Dl.Attribute import AttributeT
from .fbs_construct.FlatBuffers.Dl.Model import ModelT
from .fbs_construct.FlatBuffers.Dl.Node import NodeT
from .fbs_construct.FlatBuffers.Dl.Tensor import TensorT
from .fbs_construct.FlatBuffers.Dl.ValueInfo import ValueInfoT


class EspQuantType:
    F32 = "F32"
    S16 = "S16"
    S8  = "S8"

class LayoutAnnotation:
    NCHW = "NCHW"
    NHWC = "NHWC"
    N16HWC16 = "N16HWC16"
    N8HWC8 = "N8HWC8"
    N16HWC16_UNALIGNED = "N16HWC16_UNALIGNED"
    N8HWC8_UNALIGNED = "N8HWC8_UNALIGNED"


def fuse_downstream_operation(graph: BaseGraph,
                   fusing_downstream_op: Operation,
                   keep_coherence: bool = False,
                   remove_unlinked_variable: bool = False):
    """Remove operation from graph, this function will unlink removing
    operation from current graph, pop it from graph.operations, and remove
    it from all its input and output variables.

    Parameters of this removing operations will be removed from graph by this function, without warning.

    Args:
        fusing_downstream_op (Operation): [description]

        keep_coherence (bool): if keep_coherence = True,
            PPQ will link downstream operations of removing op to the upstream operation.
            if there is more than 1 input and output variable, ppq will link input[0] with output[0]
    """
    if fusing_downstream_op.name not in graph.operations:
        raise KeyError(f'Can not remove operation {fusing_downstream_op.name}, operation not found.')

    # removing all parameters first.
    for parameter in fusing_downstream_op.inputs.copy():
        if keep_coherence and fusing_downstream_op.type in {'Constant', 'Identity'}: break
        if parameter.is_parameter:

            parameter.dest_ops.clear()
            parameter.value = None # clear memory.
            fusing_downstream_op.inputs.remove(parameter)

            graph.variables.pop(parameter.name)

    related_vars = [var for var in fusing_downstream_op.inputs + fusing_downstream_op.outputs]
    input_var, output_var = (
        fusing_downstream_op.inputs[0] if fusing_downstream_op.num_of_input >= 1 else None,
        fusing_downstream_op.outputs[0] if fusing_downstream_op.num_of_output >= 1 else None)

    # remove operation from its output variables
    for _output_var in fusing_downstream_op.outputs:
        _output_var.source_op = None
    fusing_downstream_op.outputs.clear()

    # remove operation from its input variables
    for _input_var in fusing_downstream_op.inputs:
        if fusing_downstream_op in _input_var.dest_ops:
            _input_var.dest_ops.remove(fusing_downstream_op)
    fusing_downstream_op.inputs.clear()

    if (input_var is not None and
        output_var is not None and
        keep_coherence):

        removing_var = input_var
        source_op = removing_var.source_op
        source_op.outputs[source_op.outputs.index(removing_var)] = output_var
        output_var.source_op = source_op
        removing_var.source_op = None
        removing_var.dest_ops.clear()
        graph.remove_variable(removing_var)

    graph.operations.pop(fusing_downstream_op.name)

    if remove_unlinked_variable:
        for var in related_vars:
            if var.source_op is None and len(var.dest_ops) == 0 and var.name in graph.variables:
                graph.remove_variable(var)

    return graph


class FuseReluLikePattern(OperationExporter):
    def export(self, 
               op: QuantableOperation, 
               graph: BaseGraph, 
               **kwargs) -> Operation:
        op.attributes["activation"] = "Linear"
        downstream_op = graph.get_downstream_operations(op)
        if len(downstream_op) == 1:  # the downstream op have only one op and this op is relu
            if downstream_op[0].type == "Relu":
                conv_quant_config = op.config
                relu_quant_config = downstream_op[0].config
                new_config = OperationQuantizationConfig(
                    conv_quant_config.input_quantization_config,
                    relu_quant_config.output_quantization_config
                )

                # graph.remove_operation(downstream_op[0], keep_coherence=True)
                graph = fuse_downstream_operation(graph, downstream_op[0], keep_coherence = True)
                op.config = new_config
                op.attributes["activation"] = "Relu"

        return op

GRAPH_PATTERN = {
    "Conv": FuseReluLikePattern,
    "Gemm": FuseReluLikePattern,
    # "Resize": ResizeCheckPattern,
}

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


    def infer_qtype(self, config: TensorQuantizationConfig):
        offset_dtype, value_dtype = torch.int8, torch.int8
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            offset_dtype = torch.uint8
            value_dtype  = torch.uint8
        if config.num_of_bits > 8:
            offset_dtype = torch.int16
            value_dtype  = torch.int16
        elif config.num_of_bits > 16:
            offset_dtype = torch.int32
            value_dtype  = torch.int32
        return offset_dtype, value_dtype


    def insert_quantize_node(
        self, graph: BaseGraph, 
        var: Variable, config: TensorQuantizationConfig, 
        op: Operation) -> Operation:
        """
        Insert a Quantize Node on given variable, according to given TensorQuantizationConfig.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            # Following code will export Linear Quantization Config
            # That is for FP32 -> INT
            offset_dtype, value_type = self.infer_qtype(config)
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

            created = graph.create_operation(op_type='QuantizeLinear', attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = None

            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')

            graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

            created.outputs[0].dtype = value_type
            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape
            return created

        else:
            raise TypeError(
                f'PPQ Can not export quantization information with variable {var.name}, '
                'Unexpected Quantization property.')


    def insert_dequantize_node(
        self, graph: BaseGraph, 
        var: Variable, config: TensorQuantizationConfig, 
        op: Operation) -> Operation:
        """
        Insert a DeQuantize Node on given variable, according to given TensorQuantizationConfig.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            offset_dtype, value_type = self.infer_qtype(config)
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

            created = graph.create_operation(op_type='DequantizeLinear', attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = None

            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')

            graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

            created.inputs[0].dtype = value_type
            created.inputs[0].shape = var.shape
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = torch.float32
            return created

        else:
            raise TypeError(
                f'PPQ Can not export quantization information with variable {var.name}, '
                'Unexpected Quantization property.')


    def insert_requantize_node(
        self, graph: BaseGraph, 
        var: Variable, 
        upstream_config: TensorQuantizationConfig,
        config: TensorQuantizationConfig, 
        op: Operation) -> Operation:
        """
        Insert a ReQuantize Node on given variable, according to given TensorQuantizationConfig.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            upstream_offset_dtype, upstream_value_type = self.infer_qtype(upstream_config)
            upstream_scale  = convert_any_to_torch_tensor(upstream_config.scale.clone(), dtype=torch.float32)
            upstream_offset = ppq_tensor_round(upstream_config.offset.clone()).type(torch.float)
            offset_dtype, value_type = self.infer_qtype(config)
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(torch.float)

            created = graph.create_operation(op_type='RequantizeLinear', attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = None

            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')

            rescale = scale / upstream_scale
            reoffset = ppq_tensor_round(offset - ppq_tensor_round(upstream_offset / rescale, config.rounding)).type(offset_dtype)

            graph.create_variable(name=None, value=rescale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=reoffset, is_parameter=True, dest_ops=[created])

            created.inputs[0].dtype = upstream_value_type
            created.inputs[0].shape = var.shape
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = value_type
            return created

        else:
            raise TypeError(
                f'PPQ Can not export quantization information with variable {var.name}, '
                'Unexpected Quantization property.')


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


    def convert_operation(self, graph: BaseGraph, op: QuantableOperation):
        """ For the Espressif platform, quantization scale information 
        is placed in the form of [int(np.log2(scale))] in each value_info 
        and initializer. However, there may be cases where the quantization 
        methods of upstream and downstream operators in the model are different. 
        In such cases, it is necessary to insert requantize, quantize, or dequantize 
        operators.

        Args:
            graph (BaseGraph): PPQ IR
            op (Operation): Converting op
        """
        # collect quantable vars, where we need to insert requantize, quant or dequant op
        for config, var in [_ for _ in op.config_with_variable]:
            inserting, inserting_var = op, var
            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var): continue

            if (not var.is_parameter):

                if (var.source_op is not None and var.source_op.type in {'RequantizeLinear', 'QuantizeLinear', 
                                                                         'DequantizeLinear', 'QuantizeFloating', 'DequantizeFloating'}):
                    assert var.source_op.num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                    assert isinstance(var.source_op, Operation)
                    continue
                if (len(var.dest_ops) == 1 and var.dest_ops[0].type in {'RequantizeLinear', 'QuantizeLinear', 
                                                                        'DequantizeLinear', 'QuantizeFloating', 'DequantizeFloating'}):
                    assert var.dest_ops[0].num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                    assert isinstance(var.dest_ops[0], Operation)
                    continue

                if var in op.inputs:
                    if var.source_op is not None and not isinstance(var.source_op, QuantableOperation):
                        self.insert_quantize_node(
                            graph = graph, var = inserting_var, config = config, op=inserting)

                    elif var.source_op is not None and isinstance(var.source_op, QuantableOperation):
                        print(f"xiewei debug, upstream op is QuantableOperation, op name: {op.name}, var name: {var.name}")
                        source_op_output_var_index = var.source_op.outputs.index(var)
                        source_op_output_config = var.source_op.output_quant_config[source_op_output_var_index]
                        scale_diff     = torch.max(torch.abs(source_op_output_config.scale - config.scale)).item()
                        zeropoint_diff = torch.max(torch.abs(source_op_output_config.offset - config.offset)).item()
                        print(f"xiewei debug, scale_diff: {scale_diff}, zeropoint_diff: {zeropoint_diff}")

                        if (source_op_output_config.num_of_bits != config.num_of_bits or 
                            scale_diff >= 1e-4 or zeropoint_diff >= 1e-1):
                            self.insert_requantize_node(
                                graph = graph, 
                                var = inserting_var, 
                                upstream_config = source_op_output_config,
                                config = config, 
                                op = inserting)

                elif var in op.outputs:
                    for dest_op in var.dest_ops:
                        if dest_op is not None and not isinstance(dest_op, QuantableOperation):
                            inserting = dest_op
                            self.insert_dequantize_node(
                                graph = graph, var = inserting_var, 
                                config = config, op = inserting)


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
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'QuantizeLinear', 
                           'DequantizeLinear', 
                           'QuantizeFloating', 
                           'DequantizeFloating',
                           'RequantizeLinear'}: continue

            self.convert_operation(graph=graph, op=op)

        # fuse ops
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation): continue
            # The GRAPH_PATTERN may remove some ops.
            if op.name not in graph.operations: continue
            if op.type in GRAPH_PATTERN:
                exporter = GRAPH_PATTERN[op.type]()
                assert isinstance(
                    exporter, OperationExporter
                ), f"Expected an OpExporter here, however {type(exporter)} was given."
                op = exporter.export(op=op, graph=graph)

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

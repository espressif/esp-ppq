import os
import sys

import torch

from ppq.core import (
    OperationQuantizationConfig,
    QuantizationProperty,
    TensorQuantizationConfig,
    convert_any_to_torch_tensor,
)
from ppq.IR import BaseGraph, Operation, OperationExporter, Variable
from ppq.IR.quantize import QuantableOperation
from ppq.utils.round import ppq_tensor_round

from .onnxruntime_exporter import QDQHelper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


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


QUANT_OP_SET = {'RequantizeLinear', 'QuantizeLinear', 'DequantizeLinear', 'QuantizeFloating', 'DequantizeFloating'}
# QUANT_EXCLUDE_OP_SET refers to operators that do not participate 
# in the operations of quantize, dequantize, or requantize.
QUANT_EXCLUDE_OP_SET = {'Shape'}

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

def infer_qtype(config: TensorQuantizationConfig):
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
    graph: BaseGraph, 
    var: Variable, config: TensorQuantizationConfig, 
    op: Operation) -> Operation:
    """
    Insert a Quantize Node on given variable, according to given TensorQuantizationConfig.
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        # Following code will export Linear Quantization Config
        # That is for FP32 -> INT
        offset_dtype, value_type = infer_qtype(config)
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
    graph: BaseGraph, 
    var: Variable, config: TensorQuantizationConfig, 
    op: Operation) -> Operation:
    """
    Insert a DeQuantize Node on given variable, according to given TensorQuantizationConfig.
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        offset_dtype, value_type = infer_qtype(config)
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
    graph: BaseGraph, 
    var: Variable, 
    upstream_config: TensorQuantizationConfig,
    config: TensorQuantizationConfig, 
    op: Operation) -> Operation:
    """
    Insert a ReQuantize Node on given variable, according to given TensorQuantizationConfig.
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        upstream_offset_dtype, upstream_value_type = infer_qtype(upstream_config)
        upstream_scale  = convert_any_to_torch_tensor(upstream_config.scale.clone(), dtype=torch.float32)
        upstream_offset = ppq_tensor_round(upstream_config.offset.clone()).type(torch.float)
        offset_dtype, value_type = infer_qtype(config)
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

class InsertQuantNodePattern(OperationExporter):
    def export(self, 
               op: QuantableOperation, 
               graph: BaseGraph, 
               **kwargs) -> Operation:
        
        for config, var in [_ for _ in op.config_with_variable]:
            inserting_op, inserting_var = op, var
            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var): continue

            if not var.is_parameter:
                if var.source_op:
                    if var.source_op.type in QUANT_OP_SET:
                        assert var.source_op.num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                        assert isinstance(var.source_op, Operation)
                        continue
                    elif var in op.inputs:
                        if not isinstance(var.source_op, QuantableOperation) and var.source_op.type not in QUANT_EXCLUDE_OP_SET:
                            insert_quantize_node(graph=graph, var=inserting_var, config=config, op=inserting_op)
        return op

class InsertRequantNodePattern(OperationExporter):
    def export(self, 
               op: QuantableOperation, 
               graph: BaseGraph, 
               **kwargs) -> Operation:
        
        for config, var in [_ for _ in op.config_with_variable]:
            inserting_op, inserting_var = op, var
            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var): continue

            if not var.is_parameter:

                if var.source_op:
                    if var.source_op.type in QUANT_OP_SET:
                        assert var.source_op.num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                        assert isinstance(var.source_op, Operation)
                        continue
                    elif var in op.inputs and isinstance(var.source_op, QuantableOperation):
                        source_op_output_var_index = var.source_op.outputs.index(var)
                        source_op_output_config = var.source_op.output_quant_config[source_op_output_var_index]
                        scale_diff     = torch.max(torch.abs(source_op_output_config.scale - config.scale)).item()
                        zeropoint_diff = torch.max(torch.abs(source_op_output_config.offset - config.offset)).item()
            
                        if source_op_output_config.num_of_bits != config.num_of_bits or scale_diff >= 1e-4 or zeropoint_diff >= 1e-1:
                            # if config 
                            insert_requantize_node(
                                graph = graph, 
                                var = inserting_var, 
                                upstream_config = source_op_output_config,
                                config = config, 
                                op = inserting_op)

        return op

class InsertDequantNodePattern(OperationExporter):
    def export(self, 
               op: QuantableOperation, 
               graph: BaseGraph, 
               **kwargs) -> Operation:
        
        for config, var in [_ for _ in op.config_with_variable]:
            inserting_op, inserting_var = op, var
            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var): continue

            if not var.is_parameter:
                if len(var.dest_ops) == 1 and var.dest_ops[0].type in QUANT_OP_SET:
                    assert var.dest_ops[0].num_of_input == 3, 'Quantize Node Format Error, need as least 3 inputs.'
                    assert isinstance(var.dest_ops[0], Operation)
                    continue
                
                if var in op.outputs:
                    for dest_op in var.dest_ops:
                        if dest_op and not isinstance(dest_op, QuantableOperation) and dest_op.type not in QUANT_EXCLUDE_OP_SET:
                            insert_dequantize_node(
                                graph = graph, var = inserting_var, 
                                config = config, op = dest_op)
        return op


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


FUSE_PATTERNS = {
    "Conv": FuseReluLikePattern,
    "Gemm": FuseReluLikePattern,
}



import os
import sys
from typing import List

import numpy as np
import torch

from esp_ppq.core import (
    DataType,
    OperationQuantizationConfig,
    QuantizationProperty,
    QuantizationStates,
    QuantizationVisibility,
    TensorQuantizationConfig,
    convert_any_to_numpy,
    convert_any_to_torch_tensor,
)
from esp_ppq.IR import BaseGraph, Operation, OperationExporter, Variable
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.log import NaiveLogger
from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
from esp_ppq.utils.round import ppq_tensor_round

logger = NaiveLogger.get_logger("ESPDL")


def transpose_shape(input_shape, perm: List[int]) -> List[int]:
    if not perm:
        return input_shape
    return [input_shape[i] for i in perm]


def get_inverse_transpose(perm: List[int]) -> List[int]:
    """
    tensor == inverse_transpose(transpose(tensor))
    """
    # return perm
    return [perm.index(i) for i in range(len(perm))]


def get_default_perm(var: Variable) -> List[int]:
    """
    return the default permute for given variable, [0,1,2,3,...]
    """
    if not var or not var.shape:
        return []

    return [i for i in range(len(var.shape))]


def insert_transpose_node(graph: BaseGraph, var: Variable, op: Operation, perm: List[int]) -> Operation:
    """
    Insert a Transpose Node on given variable, according to given perm.
    """
    created = None
    if op and perm != range(len(perm)):
        logger.debug(f"insert transpose node: op: {op.name}, var:{var.name}, perm:{perm}")
        if var in op.inputs:
            created = graph.create_operation(op_type="Transpose", attributes={"perm": perm})
            var_index = op.inputs.index(var)
            if isinstance(op, QuantableOperation):
                # For transpose op,  input_quantization_config == output_quantization_config
                new_config = OperationQuantizationConfig(
                    [op.input_quant_config[var_index]],
                    [op.input_quant_config[var_index]],
                )
                created = QuantableOperation(created, new_config, op.platform)
                graph.operations[created.name] = created

            graph.insert_op_before(A=created, B=op, input_idx=var_index)
            new_var = created.outputs[0]
            new_var.shape = [var.shape[i] for i in perm]
            new_var.is_parameter = False
            new_var.dtype = var.dtype

            info = ExporterPatternInfo()
            info.add_var_permute(created.outputs[0].name, get_default_perm(created.outputs[0]))

        else:
            raise ValueError(f"Unexpected Error in Exporting Op {op.name}({op.type}).")

    return created


def restore_origin_shape(op: Operation, graph: BaseGraph):
    info = ExporterPatternInfo()
    for var in op.inputs:
        if var.is_parameter:
            continue

        var_perm = info.get_var_permute(var.name)
        if var_perm and var_perm != get_default_perm(var):
            # There is already a permute, but this op need keep origin shape
            # A transpose node needs to be inserted into the word.
            inverse_perm = get_inverse_transpose(var_perm)
            insert_transpose_node(graph, var, op, inverse_perm)
        else:
            info.add_var_permute(var.name, get_default_perm(var))

    for var in op.outputs:
        info.add_var_permute(var.name, get_default_perm(var))
    return op


def insert_concat_node(
    graph: BaseGraph,
    insert_op_var: Variable,
    insert_op: Operation,
    link_vars: List[Variable],
    link_vars_src_op: List[Operation],
    axis: int = 0,
) -> Operation:
    """
    Insert a Concat Node on given insert_op_var, according to given axis. And using link_vars as the other input,
    these function will use the TQC of the first input as the TQC for all the inputs and output.

    """
    created = None
    if insert_op and insert_op_var in insert_op.inputs:
        logger.debug(
            f"insert concat node: insert_op: {insert_op.name}, insert_op_var:{insert_op_var.name}, link_vars number:{len(link_vars)}"
        )
        if axis < 0:
            axis = axis + len(insert_op_var.shape)
        created = graph.create_operation(op_type="Concat", attributes={"axis": axis})
        var_index = insert_op.inputs.index(insert_op_var)
        if isinstance(insert_op, QuantableOperation):
            # For concat insert_op,  input_quantization_config == output_quantization_config
            input_config = insert_op.input_quant_config[var_index].copy()
            input_config.state = QuantizationStates.PASSIVE
            new_config = OperationQuantizationConfig(
                [input_config] * (len(link_vars) + 1),
                [insert_op.input_quant_config[var_index]],
            )
            created = QuantableOperation(created, new_config, insert_op.platform)
            graph.operations[created.name] = created

        output_shape = insert_op_var.shape.copy()
        for var in link_vars:
            assert len(var.shape) == len(insert_op_var.shape), (
                f"The inputs of concat have inconsistent numbers of dimensions {len(insert_op_var.shape)}, {len(var.shape)}."
            )
            output_shape[axis] = output_shape[axis] + var.shape[axis]

        graph.insert_op_before(A=created, B=insert_op, input_idx=var_index)
        new_var = created.outputs[0]
        new_var.shape = output_shape
        new_var.is_parameter = False
        new_var.dtype = insert_op_var.dtype

        for var, src_op in zip(link_vars, link_vars_src_op):
            graph.create_link_with_op(A=src_op, B=created, variable=var)

    else:
        raise ValueError(f"Unexpected Error in Exporting Op {insert_op.name}({insert_op.type}).")

    return created


def insert_slice_node(
    graph: BaseGraph,
    var: Variable,
    op: Operation,
    starts: List[int],
    ends: List[int],
    axes: List[int],
    steps: List[int],
) -> Operation:
    """
    Insert a Slice Node on given variable, according to given starts, ends, axes, steps.
    """
    created = None
    if op and var in op.inputs:
        logger.debug(
            f"insert slice node: op: {op.name}, var: {var.name}, starts: {starts}, ends: {ends}, axes: {axes}, steps: {steps}"
        )
        created = graph.create_operation(op_type="Slice")

        var_index = op.inputs.index(var)
        if isinstance(op, QuantableOperation):
            # For slice op,  input_quantization_config[0] == output_quantization_config
            input0_config = op.input_quant_config[var_index].copy()
            input0_config.state = QuantizationStates.OVERLAPPED

            starts_config = op.input_quant_config[var_index].copy()
            starts_config.state = QuantizationStates.FP32
            starts_config.observer_algorithm = "percentile"
            ends_config = starts_config.copy()
            axes_config = starts_config.copy()
            steps_config = starts_config.copy()

            output0_config = op.input_quant_config[var_index].copy()
            output0_config.state = QuantizationStates.OVERLAPPED
            output0_config.dominated_by = input0_config

            new_config = OperationQuantizationConfig(
                [input0_config, starts_config, ends_config, axes_config, steps_config],
                [output0_config],
            )
            created = QuantableOperation(created, new_config, op.platform)
            graph.operations[created.name] = created

        dim = len(var.shape)
        output_shape = [0] * dim
        for i in range(len(starts)):
            axis = i
            if axes:
                axis = (axes[i] + dim) % dim

            start_i = starts[i] + var.shape[axis] if starts[i] < 0 else starts[i] % (var.shape[axis] + 1)
            end_i = ends[i] + var.shape[axis] if ends[i] < 0 else ends[i] % (var.shape[axis] + 1)
            if start_i >= end_i:
                raise ValueError(f"Unexpected value, start_i: {start_i}, end_i: {end_i}.")
            else:
                if steps:
                    output_shape[axis] = 1 + (end_i - start_i - 1) // steps[i]
                else:
                    output_shape[axis] = end_i - start_i

        graph.insert_op_before(A=created, B=op, input_idx=var_index)
        new_var = created.outputs[0]
        new_var.shape = output_shape
        new_var.is_parameter = var.is_parameter
        new_var.dtype = var.dtype

        starts_param = graph.create_variable(
            value=torch.Tensor(starts).to(torch.int64), is_parameter=True, dest_ops=[created]
        )
        starts_param.dtype = DataType.INT64
        ends_param = graph.create_variable(
            value=torch.Tensor(ends).to(torch.int64), is_parameter=True, dest_ops=[created]
        )
        ends_param.dtype = DataType.INT64
        axes_param = graph.create_variable(
            value=torch.Tensor(axes).to(torch.int64), is_parameter=True, dest_ops=[created]
        )
        axes_param.dtype = DataType.INT64
        steps_param = graph.create_variable(
            value=torch.Tensor(steps).to(torch.int64), is_parameter=True, dest_ops=[created]
        )
        steps_param.dtype = DataType.INT64

    else:
        raise ValueError(f"Unexpected Error in insert slice node, var: {var.name}, Op: {op.name}.")

    return created


def insert_reshape_node(
    graph: BaseGraph, var: Variable, op: Operation, shape: List[int], allowzero: int = 0
) -> Operation:
    """
    Insert a Reshape Node on given variable, according to given shape.
    """
    created = None
    if op and var in op.inputs:
        logger.debug(f"insert reshape node: op: {op.name}, var: {var.name}, shape: {shape}")
        created = graph.create_operation(op_type="Reshape", attributes={"allowzero": allowzero})

        var_index = op.inputs.index(var)
        if isinstance(op, QuantableOperation):
            # For reshape op,  input_quantization_config[0] == output_quantization_config
            shape_config = op.input_quant_config[var_index].copy()
            shape_config.state = QuantizationStates.FP32
            shape_config.observer_algorithm = "percentile"

            new_config = OperationQuantizationConfig(
                [op.input_quant_config[var_index], shape_config],
                [op.input_quant_config[var_index]],
            )
            created = QuantableOperation(created, new_config, op.platform)
            graph.operations[created.name] = created

        graph.insert_op_before(A=created, B=op, input_idx=var_index)
        new_var = created.outputs[0]
        new_var.shape = shape
        new_var.is_parameter = var.is_parameter
        new_var.dtype = var.dtype

        shape_param = graph.create_variable(
            value=torch.Tensor(shape).to(torch.int64), is_parameter=True, dest_ops=[created]
        )
        shape_param.dtype = DataType.INT64

    else:
        raise ValueError(f"Unexpected Error in insert reshape node, var: {var.name}, Op: {op.name}.")

    return created


def fuse_downstream_operation(
    graph: BaseGraph,
    fusing_downstream_op: Operation,
    keep_coherence: bool = False,
    remove_unlinked_variable: bool = False,
):
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
        raise KeyError(f"Can not remove operation {fusing_downstream_op.name}, operation not found.")

    # removing all parameters first.
    for parameter in fusing_downstream_op.inputs.copy():
        if keep_coherence and fusing_downstream_op.type in {"Constant", "Identity"}:
            break
        if parameter.is_parameter:
            parameter.dest_ops.clear()
            parameter.value = None  # clear memory.
            fusing_downstream_op.inputs.remove(parameter)

            graph.variables.pop(parameter.name)

    related_vars = [var for var in fusing_downstream_op.inputs + fusing_downstream_op.outputs]
    input_var, output_var = (
        fusing_downstream_op.inputs[0] if fusing_downstream_op.num_of_input >= 1 else None,
        fusing_downstream_op.outputs[0] if fusing_downstream_op.num_of_output >= 1 else None,
    )

    # remove operation from its output variables
    for _output_var in fusing_downstream_op.outputs:
        _output_var.source_op = None
    fusing_downstream_op.outputs.clear()

    # remove operation from its input variables
    for _input_var in fusing_downstream_op.inputs:
        if fusing_downstream_op in _input_var.dest_ops:
            _input_var.dest_ops.remove(fusing_downstream_op)
    fusing_downstream_op.inputs.clear()

    if input_var is not None and output_var is not None and keep_coherence:
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
        value_dtype = torch.uint8
    if config.num_of_bits > 8:
        offset_dtype = torch.int16
        value_dtype = torch.int16
    elif config.num_of_bits > 16:
        offset_dtype = torch.int32
        value_dtype = torch.int32
    return offset_dtype, value_dtype


def insert_quantize_node(graph: BaseGraph, var: Variable, config: TensorQuantizationConfig, op: Operation) -> Operation:
    """
    Insert a Quantize Node on given variable, according to given TensorQuantizationConfig.
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        # Following code will export Linear Quantization Config
        # That is for FP32 -> INT
        offset_dtype, value_type = infer_qtype(config)
        scale = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
        offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

        created = graph.create_operation(op_type="QuantizeLinear", attributes={})
        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            created.attributes["axis"] = config.channel_axis
        else:
            created.attributes["axis"] = None

        if var in op.inputs:
            graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
        elif var in op.outputs:
            graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
        else:
            raise ValueError(f"Unexpected Error in Exporting Op {op.name}({op.type}).")

        graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
        graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

        created.outputs[0].dtype = value_type
        created.outputs[0].shape = var.shape
        created.inputs[0].shape = var.shape
        return created

    else:
        raise TypeError(
            f"PPQ Can not export quantization information with variable {var.name}, Unexpected Quantization property."
        )


def insert_requantize_node(
    graph: BaseGraph,
    var: Variable,
    upstream_config: TensorQuantizationConfig,
    config: TensorQuantizationConfig,
    op: Operation,
) -> Operation:
    """
    Insert a ReQuantize Node on given variable, according to given TensorQuantizationConfig.
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        upstream_offset_dtype, upstream_value_type = infer_qtype(upstream_config)
        upstream_scale = convert_any_to_torch_tensor(upstream_config.scale.clone(), dtype=torch.float32)
        upstream_offset = ppq_tensor_round(upstream_config.offset.clone()).type(torch.float)
        offset_dtype, value_type = infer_qtype(config)
        scale = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
        offset = ppq_tensor_round(config.offset.clone()).type(torch.float)

        created = graph.create_operation(op_type="RequantizeLinear", attributes={})
        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            created.attributes["axis"] = config.channel_axis
        else:
            created.attributes["axis"] = None

        if var in op.inputs:
            graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
        elif var in op.outputs:
            graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
        else:
            raise ValueError(f"Unexpected Error in Exporting Op {op.name}({op.type}).")

        rescale = scale / upstream_scale
        reoffset = ppq_tensor_round(offset - ppq_tensor_round(upstream_offset / rescale, config.rounding)).type(
            offset_dtype
        )

        graph.create_variable(name=None, value=rescale, is_parameter=True, dest_ops=[created])
        graph.create_variable(name=None, value=reoffset, is_parameter=True, dest_ops=[created])

        created.inputs[0].dtype = upstream_value_type
        created.inputs[0].shape = var.shape
        created.outputs[0].shape = var.shape
        created.outputs[0].dtype = value_type
        return created

    else:
        raise TypeError(
            f"PPQ Can not export quantization information with variable {var.name}, Unexpected Quantization property."
        )


def insert_dequantize_node(
    graph: BaseGraph, var: Variable, config: TensorQuantizationConfig, op: Operation
) -> Operation:
    """
    Insert a DeQuantize Node on given variable, according to given TensorQuantizationConfig.
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        offset_dtype, value_type = infer_qtype(config)
        scale = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
        offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

        created = graph.create_operation(op_type="DequantizeLinear", attributes={})
        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            created.attributes["axis"] = config.channel_axis
        else:
            created.attributes["axis"] = None

        if var in op.inputs:
            graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
        elif var in op.outputs:
            graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
        else:
            raise ValueError(f"Unexpected Error in Exporting Op {op.name}({op.type}).")

        graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
        graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

        created.inputs[0].dtype = value_type
        created.inputs[0].shape = var.shape
        created.outputs[0].shape = var.shape
        created.outputs[0].dtype = torch.float32
        return created

    else:
        raise TypeError(
            f"PPQ Can not export quantization information with variable {var.name}, Unexpected Quantization property."
        )

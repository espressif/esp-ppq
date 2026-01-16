import copy
import json
import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import torch

from esp_ppq.core import (
    DataType,
    OperationQuantizationConfig,
    QuantizationProperty,
    QuantizationStates,
    QuantizationVisibility,
    SingletonMeta,
    TensorQuantizationConfig,
    convert_any_to_numpy,
    convert_any_to_torch_tensor,
)
from esp_ppq.IR import BaseGraph, Operation, OperationExporter, Variable
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.log import NaiveLogger
from esp_ppq.parser.espdl.espdl_graph_utils import (
    fuse_downstream_operation,
    get_default_perm,
    get_inverse_transpose,
    insert_transpose_node,
    restore_origin_shape,
    transpose_shape,
)
from esp_ppq.parser.espdl.espdl_typedef import (
    ACTIVATION_OP_SET,
    ADD_LIKE_OP_SET,
    MATH_OP_SET,
    QUANT_OP_SET,
    ExporterPatternInfo,
)
from esp_ppq.utils.round import ppq_tensor_round

logger = NaiveLogger.get_logger("ESPDL")

CONV_STREAMING_OP_SET = {
    "Conv",
    "AveragePool",
    "MaxPool",
    "ConvTranspose",
}

BYPASS_OTHER_OP_SET = {
    "MatMul",
    "Gemm",
    "Flatten",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Expand",
    "LayerNorm",
    "LayerNormalization",
    "Max",
    "Min",
    "TopK",
    "GRU",
    "LSTM",
    "InsertZeros",
}

STREADMING_BYPASS_OP_SET = ACTIVATION_OP_SET | MATH_OP_SET | QUANT_OP_SET | ADD_LIKE_OP_SET | BYPASS_OTHER_OP_SET


class StreamingTable(metaclass=SingletonMeta):
    _table: Dict[str, Dict[str, Any]] = {}
    _input_frame_axis: int = 1
    _input_frame_num: int = 1

    def reset(self, graph: BaseGraph = None) -> None:
        self._table = {}
        self._graph = graph

    def get_table(self) -> Dict[str, Dict[str, Any]]:
        return self._table

    def append(self, d) -> Dict[str, Dict[str, Any]]:
        if isinstance(d, dict):
            self._table.update(d)
        elif isinstance(d, list):
            for item in d:
                self._table.update(item)
        return self._table

    def add(self, var_name: str, op_name: str, window_size: int, frame_axis: int = None) -> None:
        self._table[var_name] = {
            'op_name': op_name,
            'window_size': window_size,
            'frame_axis': frame_axis if frame_axis is not None else self._input_frame_axis,
        }

    def check(self, graph: BaseGraph) -> bool:
        if graph is None:
            logger.error("Graph is None, check failed.")
            return False
        for var_name in self._table:
            if var_name not in graph.variables:
                logger.error(f"Variable {var_name} not found in graph.")
                return False
        logger.info("Streaming table check passed.")
        return True

    def print(self) -> None:
        print(json.dumps(self._table, indent=4))

    def set_input_frame_axis(self, axis: int) -> None:
        self._input_frame_axis = axis

    def set_input_frame_num(self, num: int) -> None:
        self._input_frame_num = num

    # ---------- add dict interface ----------
    def __contains__(self, var_name: str) -> bool:
        return var_name in self._table

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self._table[key] = value

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self._table[key]

    def __delitem__(self, key: str) -> None:
        del self._table[key]

    def __len__(self) -> int:
        return len(self._table)

    def __iter__(self):
        return iter(self._table)


def insert_streaming_cache_op(graph: BaseGraph, var_name: str, attrs: Dict[str, Any]) -> Operation:
    """
    Insert a Streaming Node on given variable, according to given perm.
    """
    window_size = attrs['window_size']
    op_name = attrs.get('op_name', None)
    frame_axis = attrs.get('frame_axis', 1)
    if window_size <= 1:
        logger.info(f"Bypass inserting StreamingCache on Variable {var_name} due to window_size <= 1.")
        return None
    created = None
    op = None
    var = graph.variables[var_name]
    var_input_index = None
    var_output_index = None
    info = ExporterPatternInfo()

    if op_name:
        op = graph.operations[op_name]
        created_op_name = f"{var_name}_{op_name}_SCache"
        if var in op.inputs:
            var_input_index = op.inputs.index(var)
        elif var in op.outputs:
            var_output_index = op.outputs.index(var)
        logger.info(
            f"Inserting StreamingCache between Variable({var.name}) and Operation({op.name}): window_size {window_size}, frame_axis {frame_axis}"
        )
    elif len(var.dest_ops) > 0:
        op = var.dest_ops[0]
        created_op_name = f"{var_name}_SCache"
        var_input_index = op.inputs.index(var)
        logger.info(
            f"Inserting StreamingCache on Variable {var.name}: window_size {window_size}, frame_axis {frame_axis}"
        )
    elif var.source_op != None:
        op = var.source_op
        created_op_name = f"{var_name}_SCache"
        var_output_index = op.outputs.index(var)
        logger.info(
            f"Inserting StreamingCache on Variable {var.name}: window_size {window_size}, frame_axis {frame_axis}"
        )
    else:
        raise ValueError(f"Variable {var.name} has no connected operations.")

    created = graph.create_operation(
        op_type="StreamingCache",
        attributes={"window_size": window_size, "frame_axis": frame_axis},
        name=created_op_name,
    )
    if isinstance(op, QuantableOperation):
        # For StreamingCache op,  input_quantization_config == output_quantization_config
        if var_input_index != None:
            new_config = OperationQuantizationConfig(
                [op.input_quant_config[var_input_index]],
                [op.input_quant_config[var_input_index]],
            )
        else:
            new_config = OperationQuantizationConfig(
                [op.output_quant_config[var_output_index]],
                [op.output_quant_config[var_output_index]],
            )
        created = QuantableOperation(created, new_config, op.platform)
        created.attributes["quant_type"] = op.attributes["quant_type"]
        graph.operations[created.name] = created

    if op_name:
        if var_input_index != None:
            graph.insert_op_before(A=created, B=op, input_idx=var_input_index)
        elif var_output_index != None:
            graph.insert_op_after(A=created, B=op, output_idx=var_output_index)
    else:
        graph.insert_op_on_var(created, var_name)

    new_var = created.outputs[0]
    new_var.shape = copy.deepcopy(var.shape)
    perm = info.get_var_permute(var_name)
    frame_axis_perm = perm.index(frame_axis)
    new_var.shape[frame_axis_perm] = window_size + var.shape[frame_axis_perm] - 1
    new_var.is_parameter = False
    new_var.dtype = var.dtype
    info.add_var_exponents(new_var.name, info.get_var_exponents(var_name))
    info.add_var_permute(new_var.name, perm)

    return created


def set_conv_new_padding(op: Operation) -> None:
    padding = op.attributes.get('pads', None)

    if padding != None:
        if len(padding) == 2:
            new_padding = [0, 0]
        elif len(padding) == 4:
            new_padding = [0, padding[1], 0, padding[3]]
        op.attributes['pads'] = new_padding


def get_conv_cache_window_size(op: Operation) -> int:
    """
    Get suitable window size for convolution operation.
    """
    if op.type not in CONV_STREAMING_OP_SET:
        raise ValueError(f"Operation {op.name} is not a convolution operation.")

    # Assuming the first input is the feature map and the second input is the kernel
    input_var = op.inputs[0]
    if len(op.inputs) > 1:
        kernel_shape = op.inputs[1].shape
    elif 'kernel_shape' in op.attributes:
        kernel_shape = op.attributes['kernel_shape']
    else:
        raise ValueError(f"Cannot determine kernel shape for operation {op.name}.")

    window_size = 0
    kernel_height = kernel_shape[0]
    padding = op.attributes.get('pads', None)
    stride = op.attributes.get('strides', [1])
    dilation = op.attributes.get('dilations', [1])
    auto_pad = op.attributes.get('auto_pad', 'NOTSET')
    effective_kernel_size = dilation[0] * (kernel_height - 1) + 1

    if auto_pad in ['SAME_UPPER', 'SAME_LOWER'] and effective_kernel_size > 1:
        # Handle auto padding cases if necessary
        logger.error(f"Auto padding '{auto_pad}' is not supported in streaming node.")
        return window_size

    if padding != None:
        if len(padding) == 2:
            pad_bottom = padding[1]
        elif len(padding) == 4:
            pad_bottom = padding[2]
    else:
        pad_bottom = 0

    if pad_bottom > 0:
        logger.error(f"{padding} is not supported in streaming node.")
        return window_size

    window_size = effective_kernel_size

    return window_size


class AutoStreamingPattern(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        streaming_table = StreamingTable()

        if op.type in CONV_STREAMING_OP_SET:
            input_var = op.inputs[0]
            if input_var.name in streaming_table:
                logger.info(f"Bypass streaming node for op {op.name} as it is already in streaming table.")
            else:
                window_size = get_conv_cache_window_size(op)
                if window_size > 1:
                    streaming_table.add(input_var.name, op.name, window_size)
                    logger.info(
                        f"Insert streaming node for op {op.name} with window size {window_size} on variable {input_var.name}."
                    )
                elif window_size == 1:
                    logger.info(f"Bypass streaming node for op {op.name} due to window size == 1.")
                else:
                    logger.error(f"Failed to insert streaming node for op {op.name} due to invalid window size.")
            set_conv_new_padding(op)
        elif op.type in STREADMING_BYPASS_OP_SET:
            logger.info(f"Bypass streaming node for op {op.name} of type {op.type}.")
        else:
            manual_flag = False
            for var in op.inputs:
                if var.name in streaming_table:
                    logger.info(
                        f"Bypass streaming node for op {op.name} as input var {var.name} is already in streaming table."
                    )
                    manual_flag = True
            if not manual_flag:
                logger.error(
                    f"Operation {op.name} of type {op.type} is not supported for AutoStreamingPattern. Please insert manually on variables: {[var.name for var in op.inputs]}."
                )

        return op


def insert_streaming_nodes(graph: BaseGraph):
    """
    Reset layout from NCHW -> NHWC
    """

    streaming_table = StreamingTable()
    streaming_table.check(graph)

    for var_name, attrs in streaming_table.get_table().items():
        insert_streaming_cache_op(graph, var_name, attrs)

from typing import List

import numpy as np
import torch

from esp_ppq.core import (
    DataType,
    OperationQuantizationConfig,
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
    ADD_LIKE_OP_SET,
    CONV_LAYOUT_OP_SET,
    OTHER_OP_SET,
    PASSIVE_LAYOUT_OP_SET,
    SOFTMAX_LIKE_OP_SET,
    ExporterPatternInfo,
)

logger = NaiveLogger.get_logger('ESPDL')
# logger.set_level("DEBUG")


class ResetConvLayoutPattern(OperationExporter):
    """
    Modify Conv inputs and outputs layout from NCHW to NHWC
    And Update all variable's shape
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        info = ExporterPatternInfo()
        if op.type in CONV_LAYOUT_OP_SET:
            for var in op.inputs:
                if var.is_parameter:
                    continue

                var_shape = var.shape
                if len(var_shape) == 4:  # conv2d, NCHW -> NHWC
                    perm = [0, 2, 3, 1]
                elif len(var_shape) == 3:  # conv1d, NCW -> NWC
                    perm = [0, 2, 1]

                var_perm = info.get_var_permute(var.name)

                if var_perm:
                    if perm != var_perm:
                        # There is already a permute, but it does not match the conv layout.
                        # A transpose node needs to be inserted into the graph.
                        inverse_perm = get_inverse_transpose(var_perm)
                        new_perm = transpose_shape(inverse_perm, perm)
                        insert_transpose_node(graph, var, op, new_perm)
                else:
                    info.add_var_permute(var.name, perm)

            for var in op.outputs:
                var_shape = var.shape
                if len(var_shape) == 4:  # conv2d, NCHW -> NHWC
                    perm = [0, 2, 3, 1]
                else:  # conv1d, NCW -> NWC
                    perm = [0, 2, 1]
                info.add_var_permute(var.name, perm)

        return op


class RestoreOriginLayoutPattern(OperationExporter):
    """
    Restore original layout
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in OTHER_OP_SET:
            return restore_origin_shape(op, graph)

        return op


class BypassPassiveLayoutPattern(OperationExporter):
    """
    Passive Node inherit transpose from upstream
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in PASSIVE_LAYOUT_OP_SET:
            info = ExporterPatternInfo()
            assert len(op.outputs) == 1
            input1 = op.inputs[0]
            output = op.outputs[0]
            assert input1.shape == output.shape

            var_perm = info.get_var_permute(input1.name)
            if not var_perm:
                var_perm = get_default_perm(input1)
                info.add_var_permute(input1.name, var_perm)

            info.add_var_permute(op.outputs[0].name, var_perm)
        return op


class BypassAddLikePattern(OperationExporter):
    """
    Add,Mul,Sub,Div:

    two input and one output,
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ADD_LIKE_OP_SET:
            info = ExporterPatternInfo()
            input1 = op.inputs[0]
            input2 = op.inputs[1]
            output = op.outputs[0]
            input1_perm = info.get_var_permute(input1.name)
            input2_perm = info.get_var_permute(input2.name)
            output_perm = None
            if not input1.is_parameter and not input2.is_parameter:
                if input1_perm == input2_perm:
                    if input1_perm:
                        # using upstream's perm
                        output_perm = input1_perm
                    else:
                        # input1_perm is None, add new perm
                        info.add_var_permute(input1.name, get_default_perm(input1))
                        info.add_var_permute(input2.name, get_default_perm(input2))
                        output_perm = get_default_perm(output)
                    # logger.debug(f"{info.get_var_permute(input1.name)}, {info.get_var_permute(input2.name)}, {output_perm}")
                    info.add_var_permute(output.name, output_perm)
                else:
                    # insert transpose node and restore origin shape
                    return restore_origin_shape(op, graph)
            elif input2.is_parameter or input1.is_parameter:
                return restore_origin_shape(op, graph)

        return op


class BypassSoftmaxLayoutPattern(OperationExporter):
    """
    Softmax pattern with one input and one output and axis attribute
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in SOFTMAX_LIKE_OP_SET:
            info = ExporterPatternInfo()
            input = op.inputs[0]

            var_perm = info.get_var_permute(input.name)
            if var_perm and var_perm != get_default_perm(input):
                # There is already a permute, change axis accordingly
                axis = (int(op.attributes["axis"]) + len(var_perm)) % len(var_perm)
                new_axis = var_perm.index(axis)
                op.attributes["axis"] = new_axis
            else:
                var_perm = get_default_perm(input)
                info.add_var_permute(input.name, var_perm)

            # use input perm
            for output in op.outputs:
                info.add_var_permute(output.name, var_perm)

        return op


class ResetConcatPattern(OperationExporter):
    """
    Concat pattern with two input and one output,
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ["Concat"]:
            perm_dict = {}
            info = ExporterPatternInfo()

            for var in op.inputs:
                var_perm = info.get_var_permute(var.name)
                perm_str = str(var_perm)
                if perm_str not in perm_dict:
                    perm_dict[perm_str] = var_perm

            output_var = op.outputs[0]

            if len(perm_dict) == 1:  # all input have same perm, output bypass
                var_perm = list(perm_dict.values())[0]
                if not var_perm:
                    restore_origin_shape(op, graph)
                else:
                    axis = op.attributes["axis"]
                    new_axis = var_perm.index(int(axis))
                    op.attributes["axis"] = new_axis
                    info.add_var_permute(output_var.name, var_perm)
                    logger.debug(f"{op.name} update axes from {axis} to {new_axis}")
            else:
                logger.debug(f"transpose perm {perm_dict}")
                restore_origin_shape(op, graph)
        return op


class ResetResizePattern(OperationExporter):
    """
    Reize Layout Pattern
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ["Resize"]:
            info = ExporterPatternInfo()
            input_var = op.inputs[0]
            input_shape_orig = input_var.shape
            perm = None
            if len(input_shape_orig) == 4:  # 2d, NCHW -> NHWC
                perm = [0, 2, 3, 1]
            elif len(input_shape_orig) == 3:  # 1d, NCW -> NWC
                perm = [0, 2, 1]
            else:
                logger.error(f"Reize: do not support shape for {input_shape_orig}")

            input_perm = info.get_var_permute(input_var.name)
            if input_perm:
                if perm != input_perm:
                    # There is already a permute, but it does not match the Reize layout.
                    # A transpose node needs to be inserted into the graph.
                    inverse_perm = get_inverse_transpose(input_perm)
                    new_perm = transpose_shape(inverse_perm, perm)
                    insert_transpose_node(graph, input_var, op, new_perm)
            else:
                info.add_var_permute(input_var.name, perm)

            # Get roi
            roi_var = op.inputs[1]
            if not roi_var.name or roi_var.value.numel() == 0:
                roi_var = None

            # Get scales
            scales_var = op.inputs[2]
            if not scales_var.name:
                scales_var = None

            # Get sizes
            if len(op.inputs) > 3:
                sizes_var = op.inputs[3]
            else:
                sizes_var = None

            if scales_var is None and sizes_var is None:
                raise ValueError("scales_var is None and sizes_var is None.")

            rank = len(input_shape_orig)
            axes = op.attributes.get("axes", None)

            # Adjust roi, scales, and sizes to the same dimensions as input.
            # But the data arrangement is still in NCHW or NCW format.
            if axes is not None:
                # Adjust roi parameters
                if roi_var is not None:
                    new_roi_value = ([0.0] * rank) + ([1.0] * rank)
                    naxes = len(axes)
                    for i, d in enumerate(axes):
                        new_roi_value[d] = roi_var.value[i]
                        new_roi_value[rank + d] = roi_var.value[naxes + i]
                    roi_var.value = torch.tensor(new_roi_value).type(roi_var.dtype.to_torch())
                    roi_var.shape = roi_var.value.shape

                # Adjust scales parameters
                if scales_var is not None:
                    new_scales_value = [1.0] * rank
                    for i, d in enumerate(axes):
                        new_scales_value[d] = scales_var.value[i]
                    scales_var.value = torch.tensor(new_scales_value).type(scales_var.dtype.to_torch())
                    scales_var.shape = scales_var.value.shape

                # Adjust sizes parameters
                if sizes_var is not None:
                    new_sizes_value = [input_shape_orig[i] for i in range(rank)]
                    for i, d in enumerate(axes):
                        new_sizes_value[d] = sizes_var.value[i]
                    sizes_var.value = torch.tensor(new_sizes_value).type(sizes_var.dtype.to_torch())
                    sizes_var.shape = sizes_var.value.shape

            else:
                axes = list(range(rank))

            # Adjust scales and sizes according to keep_aspect_ratio_policy.
            # This attribute describes how to interpret the sizes input with regard to keeping the original
            # aspect ratio of the input, and it is not applicable when the scales input is used.
            keep_aspect_ratio_policy = op.attributes.get("keep_aspect_ratio_policy", None)
            if sizes_var is not None:
                scale_factors = [sizes_var.value[i] / input_shape_orig[i] for i in range(rank)]
                if keep_aspect_ratio_policy and keep_aspect_ratio_policy != "stretch":
                    if keep_aspect_ratio_policy == "not_larger":
                        scale = np.array(scale_factors)[axes].min()
                    elif keep_aspect_ratio_policy == "not_smaller":
                        scale = np.array(scale_factors)[axes].max()
                    else:
                        raise ValueError(f"Invalid keep_aspect_ratio_policy={keep_aspect_ratio_policy!r}")

                    scale_factors = [scale if i in axes else 1.0 for i in range(rank)]

                    def round_half_up(x: float) -> int:
                        return int(x + 0.5)

                    output_size = [
                        round_half_up(scale * input_shape_orig[i]) if i in axes else input_shape_orig[i]
                        for i in range(rank)
                    ]
                else:
                    output_size = sizes_var.value.tolist()

                if scales_var:
                    scales_var.value = torch.tensor(scale_factors).type(dtype=DataType.to_torch(scales_var.dtype))
            else:
                output_size = (scales_var.value * torch.tensor(input_shape_orig)).type(torch.int64)  # type: ignore[union-attr]

            if sizes_var:
                sizes_var.value = torch.tensor(output_size).type(dtype=DataType.to_torch(sizes_var.dtype))

            # Align the data arrangement of the parameters with that of the input.
            # if perm:
            #     if roi_var is not None:
            #         roi_value_1 = torch.tensor(transpose_shape(roi_var.value[0:rank], perm)).type(dtype=DataType.to_torch(roi_var.dtype))
            #         roi_value_2 = torch.tensor(transpose_shape(roi_var.value[rank:], perm)).type(dtype=DataType.to_torch(roi_var.dtype))
            #         roi_var.value = torch.cat((roi_value_1, roi_value_2))
            #         roi_var.shape = roi_var.value.shape
            #     if scales_var is not None:
            #         scales_var.value = torch.tensor(transpose_shape(scales_var.value, perm)).type(dtype=DataType.to_torch(scales_var.dtype))
            #         scales_var.shape = scales_var.value.shape
            #     if sizes_var is not None:
            #         sizes_var.value = torch.tensor(transpose_shape(sizes_var.value, perm)).type(dtype=DataType.to_torch(sizes_var.dtype))
            #         sizes_var.shape = sizes_var.value.shape

            for var in op.inputs[1:]:
                if var:
                    info.add_var_permute(var.name, get_default_perm(var))
            info.add_var_permute(op.outputs[0].name, perm)

        return op


class FuseTransposePattern(OperationExporter):
    """
    Fuse Transpose Pattern
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # The FUSE_OP_PATTERNS may remove some ops.
        if op.name not in graph.operations:
            return op

        if op.type == "Transpose":
            downstream_transpose_op = None
            perm = op.attributes["perm"]
            while True:
                downstream_op = graph.get_downstream_operations(op)
                # the downstream op have only one op and this op is Transpose
                if len(downstream_op) == 1 and downstream_op[0].type == "Transpose":
                    downstream_transpose_op = downstream_op[0]
                    perm = transpose_shape(perm, downstream_op[0].attributes["perm"])

                    if isinstance(op, QuantableOperation):
                        op_config = op.config
                        downstream_config = downstream_transpose_op.config
                        new_config = OperationQuantizationConfig(
                            op_config.input_quantization_config,
                            downstream_config.output_quantization_config,
                        )
                        op.config = new_config
                    graph = fuse_downstream_operation(graph, downstream_transpose_op, keep_coherence=True)
                    op.attributes["perm"] = perm
                else:
                    break

            perm = op.attributes["perm"]
            if perm == [i for i in range(len(perm))]:
                # Removed redundant transpose
                graph.remove_operation(op, keep_coherence=True)
        return op


def print_vars(op: Operation):
    logger.info(f"Op: {op.name}, {op.type}, {op.attributes}")
    for var in op.inputs:
        print("inputs:", var.name, var.shape)
    for var in op.outputs:
        print("outputs:", var.name, var.shape)


def reset_graph_layout(graph: BaseGraph):
    """
    Reset layout from NCHW -> NHWC
    """

    layout_patterns = [
        [CONV_LAYOUT_OP_SET, ResetConvLayoutPattern],
        [PASSIVE_LAYOUT_OP_SET, BypassPassiveLayoutPattern],
        [ADD_LIKE_OP_SET, BypassAddLikePattern],
        [SOFTMAX_LIKE_OP_SET, BypassSoftmaxLayoutPattern],
        [["Concat"], ResetConcatPattern],
        [["Resize"], ResetResizePattern],
        [OTHER_OP_SET, RestoreOriginLayoutPattern],
    ]

    for op in graph.topological_sort():
        flag = 1
        for pattern in layout_patterns:
            if op.type in pattern[0]:
                pattern[1]().export(op, graph)
                flag = 0
                break
        if flag:
            logger.error(f"Can not reset {op.type}:{op.name} layout")

    # fuse transpose op
    pattern = FuseTransposePattern()
    for op in graph.topological_sort():
        pattern.export(op, graph)

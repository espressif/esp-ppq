import os
import sys

import numpy as np
import torch

from esp_ppq.core import (
    DataType,
    OperationQuantizationConfig,
    QuantizationProperty,
    QuantizationStates,
    QuantizationVisibility,
    TargetPlatform,
    TensorQuantizationConfig,
    convert_any_to_numpy,
)
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.IR import BaseGraph, Operation, OperationExporter, Variable
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.log import NaiveLogger
from esp_ppq.parser.espdl.espdl_graph_utils import (
    fuse_downstream_operation,
    insert_concat_node,
    insert_dequantize_node,
    insert_quantize_node,
    insert_requantize_node,
    insert_reshape_node,
    insert_slice_node,
    insert_transpose_node,
)
from esp_ppq.parser.espdl.espdl_typedef import (
    ACTIVATION_OP_SET,
    MATH_OP_SET,
    QUANT_EXCLUDE_OP_SET,
    QUANT_OP_SET,
    EspQuantType,
    ExporterPatternInfo,
    LayoutAnnotation,
)
from esp_ppq.quantization.qfunction.linear import PPQLinearQuant_toInt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

logger = NaiveLogger.get_logger("ESPDL")


class EspdlQuantHelper:
    """Helper class for processing onnx qdq format"""

    @staticmethod
    def TQC_Exportable_Check(TQC: TensorQuantizationConfig, bounded_var: Variable) -> bool:
        if not TQC.can_export(True):
            logger.info(f"Skip {bounded_var.name} because it's not exportable")
            return False

        if TQC.visibility == QuantizationVisibility.INTERNAL:
            logger.info(f"Skip {bounded_var.name} because TAC visibility is internal")
            return False

        if TQC.num_of_bits == 8 and TQC.policy.has_property(QuantizationProperty.LINEAR):
            if TQC.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                range_check = TQC.quant_max <= 255 and TQC.quant_min >= 0
            else:
                range_check = TQC.quant_max <= 127 and TQC.quant_min >= -128
        else:
            range_check = True

        if not range_check:
            logger.warning(
                f"Is it not safe to export TQC({bounded_var.name}) to Onnx, "
                f"INT8 value range must be [-128, 127] or [0, 255], "
                f"however [{TQC.quant_min, TQC.quant_max}] was given."
            )
            return False
        return True


class InsertQuantTypePattern(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.platform in [TargetPlatform.ESPDL_INT8, TargetPlatform.ESPDL_S3_INT8, TargetPlatform.ESPDL_C_INT8]:
            op.attributes["quant_type"] = EspQuantType.S8
        elif op.platform in [
            TargetPlatform.ESPDL_INT16,
            TargetPlatform.ESPDL_S3_INT16,
            TargetPlatform.ESPDL_H_PRE_INT16,
            TargetPlatform.ESPDL_S3_H_PRE_INT16,
            TargetPlatform.ESPDL_C_INT16,
            TargetPlatform.ESPDL_C_H_PRE_INT16,
        ]:
            op.attributes["quant_type"] = EspQuantType.S16
        else:
            op.attributes["quant_type"] = EspQuantType.F32

        return op


class InsertQuantNodePattern(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in QUANT_OP_SET or not isinstance(op, QuantableOperation):
            return op

        for config, var in [_ for _ in op.config_with_variable]:
            inserting_op, inserting_var = op, var
            if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                continue

            if not var.is_parameter:
                if var.source_op:
                    if var.source_op.type in QUANT_OP_SET:
                        assert var.source_op.num_of_input == 3, "Quantize Node Format Error, need as least 3 inputs."
                        assert isinstance(var.source_op, Operation)
                        continue
                    elif var in op.inputs:
                        if (
                            not isinstance(var.source_op, QuantableOperation)
                            or var.source_op.output_quant_config[var.source_op.outputs.index(var)].state
                            == QuantizationStates.FP32
                        ) and var.source_op.type not in QUANT_EXCLUDE_OP_SET:
                            logger.debug(f"Insert Quantize Node for {op.name}:{var.name}")
                            insert_quantize_node(
                                graph=graph,
                                var=inserting_var,
                                config=config,
                                op=inserting_op,
                            )
        return op


class InsertRequantNodePattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in QUANT_OP_SET or not isinstance(op, QuantableOperation):
            return op

        for config, var in [_ for _ in op.config_with_variable]:
            inserting_op, inserting_var = op, var
            if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                continue

            if not var.is_parameter:
                if var.source_op:
                    if var.source_op.type in QUANT_OP_SET:
                        assert var.source_op.num_of_input == 3, "Quantize Node Format Error, need as least 3 inputs."
                        assert isinstance(var.source_op, Operation)
                        continue
                    elif var in op.inputs and isinstance(var.source_op, QuantableOperation):
                        source_op_output_var_index = var.source_op.outputs.index(var)
                        source_op_output_config = var.source_op.output_quant_config[source_op_output_var_index]
                        scale_diff = torch.max(torch.abs(source_op_output_config.scale - config.scale)).item()
                        zeropoint_diff = torch.max(torch.abs(source_op_output_config.offset - config.offset)).item()

                        if (
                            source_op_output_config.num_of_bits != config.num_of_bits
                            or scale_diff >= 1e-5
                            or zeropoint_diff >= 1e-1
                        ):
                            # if config
                            logger.debug(f"Insert Requantize Node for {op.name}:{var.name}")
                            insert_requantize_node(
                                graph=graph,
                                var=inserting_var,
                                upstream_config=source_op_output_config,
                                config=config,
                                op=inserting_op,
                            )

        return op


class InsertDequantNodePattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in QUANT_OP_SET or not isinstance(op, QuantableOperation):
            return op

        for config, var in [_ for _ in op.config_with_variable]:
            inserting_op, inserting_var = op, var
            if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                continue

            if not var.is_parameter:
                if var in op.outputs:
                    for dest_op in var.dest_ops:
                        if (
                            dest_op
                            and dest_op.type not in QUANT_OP_SET
                            and not isinstance(dest_op, QuantableOperation)
                            and dest_op.type not in QUANT_EXCLUDE_OP_SET
                        ):
                            logger.debug(f"Insert Dequantize Node for {op.name}:{var.name}")
                            insert_dequantize_node(
                                graph=graph,
                                var=inserting_var,
                                config=config,
                                op=dest_op,
                            )
        return op


class InsertPreNodeOfMatMulPattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type != "MatMul" or op.inputs[1].is_parameter or not isinstance(op, QuantableOperation):
            return op

        input0 = op.inputs[0]
        input1 = op.inputs[1]
        input1_config = op.input_quant_config[1]
        input1_num_of_bits = input1_config.num_of_bits
        input1_n_size = input1.shape[-1]
        if input0.shape is None or input1.shape is None:
            logger.error("input shape is None")
            return op

        if input1_num_of_bits != 8 and input1_num_of_bits != 16:
            logger.warning(f"The num_of_bits of input1 {input1_num_of_bits} is not supported.")
            return op

        input0_dims = len(input0.shape)
        input1_dims = len(input1.shape)
        input1_orig_shape = input1.shape
        align = 16 if input1_num_of_bits == 8 else 8

        if op.platform in [
            TargetPlatform.ESPDL_C_INT8,
            TargetPlatform.ESPDL_C_INT16,
            TargetPlatform.ESPDL_C_H_PRE_INT16,
        ]:
            if input1_dims >= 2:
                # *CN -> *NC
                # Because esp-dl's Conv2D implementation in C requires the filter layout to be NHWC.
                insert_transpose_node(
                    graph=graph, var=op.inputs[1], op=op, perm=list(range(len(input1.shape) - 2)) + [-1, -2]
                )
                insert_reshape_node(graph=graph, var=op.inputs[1], op=op, shape=input1_orig_shape)
        else:
            # align
            if input1_n_size % align == 0:
                if input1_dims >= 2:
                    # *CN -> *(N/align)C(align) = *(N/align)HWC(align)
                    c, n = input1.shape[-2:]
                    insert_reshape_node(
                        graph=graph, var=op.inputs[1], op=op, shape=input1.shape[:-2] + [c, n // align, align]
                    )
                    insert_transpose_node(
                        graph=graph, var=op.inputs[1], op=op, perm=list(range(len(input1.shape) - 2)) + [-2, -3, -1]
                    )
                    insert_reshape_node(graph=graph, var=op.inputs[1], op=op, shape=input1_orig_shape)
            # unalign
            else:
                aligned_len = input1_n_size // align * align

                if input1_dims >= 2:
                    c, n = input1.shape[-2:]
                    trans_op = insert_transpose_node(
                        graph=graph, var=op.inputs[1], op=op, perm=list(range(len(input1.shape) - 2)) + [-1, -2]
                    )
                    if aligned_len > 0:
                        insert_slice_node(
                            graph=graph,
                            var=op.inputs[1],
                            op=op,
                            starts=[0] * len(input1.shape),
                            ends=input1.shape[:-2] + [aligned_len, c],
                            axes=list(range(len(input1.shape))),
                            steps=[1] * len(input1.shape),
                        )
                        insert_reshape_node(
                            graph=graph, var=op.inputs[1], op=op, shape=input1.shape[:-2] + [n // align, align, c]
                        )
                        insert_transpose_node(
                            graph=graph, var=op.inputs[1], op=op, perm=list(range(len(input1.shape) - 2)) + [-3, -1, -2]
                        )
                        insert_reshape_node(
                            graph=graph, var=op.inputs[1], op=op, shape=input1.shape[:-2] + [aligned_len, c]
                        )
                        # concat align and unalign
                        concat_op = insert_concat_node(
                            graph=graph,
                            insert_op_var=op.inputs[1],
                            insert_op=op,
                            link_vars=[trans_op.outputs[0]],
                            link_vars_src_op=[trans_op],
                            axis=-2,
                        )
                        # insert unalign
                        insert_slice_node(
                            graph=graph,
                            var=concat_op.inputs[1],
                            op=concat_op,
                            starts=[0] * (len(input1.shape) - 2) + [aligned_len, 0],
                            ends=input1.shape[:-2] + [input1_n_size, c],
                            axes=list(range(len(input1.shape))),
                            steps=[1] * len(input1.shape),
                        )
                        concat_axis = concat_op.attributes["axis"]
                        concat_op.outputs[0].shape[concat_axis] = 0
                        for input in concat_op.inputs:
                            concat_op.outputs[0].shape[concat_axis] = (
                                concat_op.outputs[0].shape[concat_axis] + input.shape[concat_axis]
                            )

                    insert_reshape_node(graph=graph, var=op.inputs[1], op=op, shape=input1_orig_shape)

        return op


class FuseReluLikePattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if not isinstance(op, QuantableOperation):
            return op

        # The FUSE_OP_PATTERNS may remove some ops.
        if op.name not in graph.operations:
            return op

        if op.type in ["Conv", "Gemm", "MatMul"]:
            op.attributes["activation"] = "Linear"
            downstream_op = graph.get_downstream_operations(op)
            if len(downstream_op) == 1:  # the downstream op have only one op and this op is relu
                # if downstream_op[0].type in ["Relu", "Clip"]:
                if downstream_op[0].type in ["Relu"]:
                    logger.debug(f"fuse {op.type}:{op.name} and {downstream_op[0].type}:{downstream_op[0].name}")
                    conv_quant_config = op.config
                    relu_quant_config = downstream_op[0].config
                    new_config = OperationQuantizationConfig(
                        conv_quant_config.input_quantization_config,
                        relu_quant_config.output_quantization_config,
                    )

                    # graph.remove_operation(downstream_op[0], keep_coherence=True)
                    graph = fuse_downstream_operation(graph, downstream_op[0], keep_coherence=True)
                    op.config = new_config
                    op.attributes["activation"] = downstream_op[0].type

        return op


class QuantVariableToIntPattern(OperationExporter):
    @staticmethod
    def calculate_exponent(config: TensorQuantizationConfig):
        if not config.policy.has_property(QuantizationProperty.LINEAR):
            raise ValueError("Critical Quantization Error! Non-linear config detected.")
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            raise ValueError("Critical Quantization Error! Asymmetrical config detected.")

        if not config.scale:
            return None

        exponent = None
        if config.policy.has_property(QuantizationProperty.PER_TENSOR) and config.policy.has_property(
            QuantizationProperty.POWER_OF_2
        ):
            scale = convert_any_to_numpy(config.scale)
            exponent = [int(np.log2(scale))]
        elif config.policy.has_property(QuantizationProperty.PER_CHANNEL) and config.policy.has_property(
            QuantizationProperty.POWER_OF_2
        ):
            scale = convert_any_to_numpy(config.scale)
            exponent = np.log2(scale).astype(int).tolist()
        return exponent

    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if not isinstance(op, QuantableOperation):
            logger.info("skip not QuantableOperation")
            return op

        # collect quantable vars, where we need to quantize parameters
        info = ExporterPatternInfo()

        for config, var in [_ for _ in op.config_with_variable]:
            if not var or not config:
                logger.info("skip not config or var")
                continue

            if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                continue

            if not info.get_var_config(var.name):
                info.add_var_config(var.name, config)

            if not info.get_var_exponents(var.name):
                exponent = QuantVariableToIntPattern.calculate_exponent(config)
                if exponent:
                    info.add_var_exponents(var.name, exponent)
                    info.add_var_config(var.name, config)
                    logger.debug(f"{var.name} exponent: {exponent}")
                else:
                    logger.info(
                        "Skip %s from (op name:%s, type:%s) because it's not quantized" % (var.name, op.name, op.type)
                    )
            else:
                continue

            if var.is_parameter:
                assert len(var.dest_ops) == 1, (
                    f"Can not export variable {var.name}, cause it has more than 1 destination operations. "
                    "PPQ require all parameters to have only 1 destination operation."
                )

                # override quantization state, so that we can export parameter correctly.
                if config.state == QuantizationStates.BAKED:
                    config.state = QuantizationStates.ACTIVATED
                if config.state == QuantizationStates.PASSIVE_BAKED:
                    config.state = QuantizationStates.PASSIVE

                if config.policy.has_property(QuantizationProperty.LINEAR):
                    if (
                        op.platform
                        in [
                            TargetPlatform.ESPDL_H_PRE_INT16,
                            TargetPlatform.ESPDL_S3_H_PRE_INT16,
                            TargetPlatform.ESPDL_C_H_PRE_INT16,
                        ]
                    ) and config.num_of_bits >= 16:
                        var.value = PPQLinearQuant_toInt(tensor=var.value.type(dtype=torch.float64), config=config)
                    else:
                        var.value = PPQLinearQuant_toInt(tensor=var.value, config=config)
            elif not var.is_parameter:
                if config.policy.has_property(QuantizationProperty.LINEAR):
                    quant_type = op.attributes.get("quant_type", None)
                    if quant_type == EspQuantType.S8:
                        var.dtype = DataType.INT8
                    elif quant_type == EspQuantType.S16:
                        var.dtype = DataType.INT16
                    else:
                        var.dtype = DataType.FP32

        return op


class ResetParamLayoutPattern(OperationExporter):
    def reset_conv_filter_layout(self, tensor, quant_type, group=None, platform=TargetPlatform.ESPDL_INT8):
        layout = LayoutAnnotation.NCHW

        if len(tensor.shape) != 3 and len(tensor.shape) != 4:
            logger.error(f"Conv filter don't support {len(tensor.shape)}D tensor.")
            return tensor, layout

        if len(tensor.shape) == 3:
            n, c, w = tensor.shape  # n denotes output channels, c denotes input channels,
            tensor = tensor.permute(0, 2, 1)  # NCW -> NWC

            if platform in [
                TargetPlatform.ESPDL_C_INT8,
                TargetPlatform.ESPDL_C_INT16,
                TargetPlatform.ESPDL_C_H_PRE_INT16,
            ]:
                layout = LayoutAnnotation.NWC
                if group > 1:
                    tensor = tensor.permute(1, 2, 0)  # depthwise: NWC -> WCN
                    layout = LayoutAnnotation.WCN
            else:
                align = 16 if quant_type == EspQuantType.S8 else 8
                aligned_len = n // align * align
                aligned_tensor = tensor[0:aligned_len, ...]
                aligned_tensor = aligned_tensor.reshape(n // align, align, w, c)  # NWC -> (N/align,align)WC
                # (N/align,align)WC -> (N/align)WC(align)
                aligned_tensor = aligned_tensor.permute(0, 2, 3, 1)
                # (N/align)WC(align) -> (aligned_len)WC
                aligned_tensor = aligned_tensor.reshape(aligned_len, w, c)

                if n % align != 0:
                    unaligned_tensor = tensor[aligned_len:n, ...]  # NWC
                    if group == 1 or group == None:
                        aligned_tensor = torch.cat((aligned_tensor, unaligned_tensor), 0)
                    else:
                        n_remain = n - aligned_len
                        unaligned_tensor = unaligned_tensor.permute(2, 1, 0)  # depthwise unaligned: NWC -> CWN
                        unaligned_tensor = unaligned_tensor.reshape(n_remain, w, c)
                        aligned_tensor = torch.cat((aligned_tensor, unaligned_tensor), 0)

                    if align == 16:
                        layout = LayoutAnnotation.N16WC16_UNALIGNED
                    else:
                        layout = LayoutAnnotation.N8WC8_UNALIGNED
                else:
                    if align == 16:
                        layout = LayoutAnnotation.N16WC16
                    else:
                        layout = LayoutAnnotation.N8WC8

                tensor = aligned_tensor

            # TODO:: modify the layout of depthwise conv in ESP-DL, keep it same with conv
            if group == 1 or group == None:
                tensor = tensor.reshape(w, c, n)  # reshape to WCN
            else:
                tensor = tensor.reshape(w, n, c)  # reshape to WNC

        elif len(tensor.shape) == 4:
            n, c, h, w = tensor.shape  # n denotes output channels, c denotes input channels,
            tensor = tensor.permute(0, 2, 3, 1)  # NCHW -> NHWC

            if platform in [
                TargetPlatform.ESPDL_C_INT8,
                TargetPlatform.ESPDL_C_INT16,
                TargetPlatform.ESPDL_C_H_PRE_INT16,
            ]:
                layout = LayoutAnnotation.NHWC
                if group is not None and group > 1:
                    tensor = tensor.permute(1, 2, 3, 0)  # depthwise: NHWC -> HWCN
                    layout = LayoutAnnotation.HWCN
            else:
                align = 16 if quant_type == EspQuantType.S8 else 8
                aligned_len = n // align * align
                aligned_tensor = tensor[0:aligned_len, ...]
                aligned_tensor = aligned_tensor.reshape(n // align, align, h, w, c)  # NHWC -> (N/align,align)HWC
                # (N/align,align)HWC -> (N/align)HWC(align)
                aligned_tensor = aligned_tensor.permute(0, 2, 3, 4, 1)
                # (N/align)HWC(align) -> (aligned_len)HWC
                aligned_tensor = aligned_tensor.reshape(aligned_len, h, w, c)

                if n % align != 0:
                    unaligned_tensor = tensor[aligned_len:n, ...]  # NHWC
                    if group == 1 or group == None:
                        aligned_tensor = torch.cat((aligned_tensor, unaligned_tensor), 0)
                    else:
                        n_remain = n - aligned_len
                        unaligned_tensor = unaligned_tensor.permute(3, 1, 2, 0)  # depthwise unaligned: NHWC -> CHWN
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

                tensor = aligned_tensor

            # TODO:: modify the layout of depthwise conv in ESP-DL, keep it same with conv
            if group == 1 or group == None:
                tensor = tensor.reshape(h, w, c, n)  # reshape to HWCN
            else:
                tensor = tensor.reshape(h, w, n, c)  # reshape to HWNC

        return tensor, layout

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        quant_type = op.attributes.get("quant_type", None)
        if quant_type == None or quant_type == EspQuantType.F32 or not isinstance(op, QuantableOperation):
            return op

        info = ExporterPatternInfo()

        if op.type == "Conv":
            for var in op.inputs:
                if not var.is_parameter:
                    continue

                tensor = var.value
                if len(tensor.shape) == 3 or len(tensor.shape) == 4:  # Conv1d/Conv2d Filter
                    group = op.attributes.get("group", None)
                    aligned_tensor, layout = self.reset_conv_filter_layout(tensor, quant_type, group, op.platform)
                    info.add_var_layout(var.name, layout)
                    var.value = aligned_tensor
                    logger.debug(f"reset {op.type}:{op.name}, shape:{tensor.shape}, layout to {layout}")

        elif op.type == "Gemm":
            for var in op.inputs:
                if not var.is_parameter:
                    continue

                # fix the transB attribute is 0
                tensor = var.value
                alpha = op.attributes.get("alpha", 1.0)
                beta = op.attributes.get("beta", 1.0)
                assert alpha == 1.0 and beta == 1.0, "alpha and beta must be 1.0 and 0.0"

                if len(tensor.shape) == 2:  # Gemm Filter
                    trans_filter = op.attributes.get("transB", 0)
                    if trans_filter != 0:
                        logger.debug("transB is not 0, transpose the filter and reset transB")
                        op.attributes["transB"] = 0  # update 'transB'
                        tensor = tensor.transpose(1, 0)  # [N, C] -> [C, N]
                    tensor = tensor.unsqueeze(-1).unsqueeze(-1)  # CN -> CNHW
                    # CNHW -> NCHW, same with conv2d filter
                    tensor = tensor.permute(1, 0, 2, 3)

                    aligned_tensor, layout = self.reset_conv_filter_layout(tensor, quant_type, None, op.platform)
                    info.add_var_layout(var.name, layout)
                    var.value = aligned_tensor
                    logger.debug(f"reset {op.type}:{op.name}, shape:{var.value.shape}, layout to {layout}")

        elif op.type == "MatMul" and op.inputs[1].is_parameter and len(op.inputs[1].shape) >= 2:
            tensor = op.inputs[1].value
            tensor_orig_shape = tensor.shape
            c, n = tensor_orig_shape[-2:]
            tensor = tensor.reshape(-1, c, n)
            tensor_reset = []
            layout = None

            for tensor_tmp in tensor:
                tensor_tmp = tensor_tmp.unsqueeze(-1).unsqueeze(-1)  # CN -> CNHW
                # CNHW -> NCHW, same with conv2d filter
                tensor_tmp = tensor_tmp.permute(1, 0, 2, 3)

                tensor_tmp, layout = self.reset_conv_filter_layout(tensor_tmp, quant_type, None, op.platform)
                # HWCN -> CN
                tensor_tmp.squeeze(0).squeeze(0)
                tensor_reset.append(tensor_tmp)

            info.add_var_layout(op.inputs[1].name, layout)
            op.inputs[1].value = torch.stack(tensor_reset).reshape(tensor_orig_shape)

        return op


class AddLUTPattern(OperationExporter):
    def __init__(self, int16_step=1) -> None:
        super().__init__()
        self.int16_step = int(int16_step)  # the step of int16 LUT

    def get_scale(self, var: Variable, info: ExporterPatternInfo) -> torch.Tensor:
        exponent = info.get_var_exponents(var.name)
        scale = 1.0
        if exponent:
            if isinstance(exponent, list):
                scale = 2 ** exponent[0]
            else:
                scale = 2**exponent

        return scale

    def calculate_lut(
        self,
        op: QuantableOperation,
        info: ExporterPatternInfo,
        max: int,
        min: int,
        step: int = 1,
    ) -> torch.Tensor:
        # Get forward function
        platform_dispatching_table = OPERATION_FORWARD_TABLE[op.platform]
        if op.type not in platform_dispatching_table:
            raise NotImplementedError(
                f"Graph op: {op.name}({op.type}) "
                f"has no backend implementation on target platform {op.platform}. "
                "Register this op to esp_ppq.executor.base.py and esp_ppq.executor.op first"
            )
        operation_forward_func = platform_dispatching_table[op.type]

        # Calculate output and lut
        input = torch.arange(min, max + 1, step=step, dtype=torch.float)
        input = input * self.get_scale(op.inputs[0], info)
        inputs = [input]

        if len(op.inputs) > 1:
            for op_input in op.inputs[1:]:
                inputs.append(op_input.value * self.get_scale(op_input, info))
        output = operation_forward_func(op, inputs)
        device = op.output_quant_config[0].scale.device
        lut = PPQLinearQuant_toInt(output.to(device), op.output_quant_config[0])

        return lut

    def get_lut_name(self, op: Operation, info: ExporterPatternInfo):
        index = len(info.luts)
        name = f"{op.type}_lut_{index}"
        return name

    def check_op(self, op: Operation):
        """
        True if this op can be implemented by LUT, otherwise False
        """

        if op.type == "PRelu":
            if op.inputs[1].value.numel() == 1:
                return True
            else:
                return False
        elif op.type == "Clip":
            return True
        elif len(op.outputs) > 1 or len(op.inputs) > 1:
            return False
        elif op.type in ACTIVATION_OP_SET or op.type in MATH_OP_SET:
            return True

        return False

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        quant_type = op.attributes.get("quant_type", None)
        if quant_type == None or quant_type == EspQuantType.F32 or not isinstance(op, QuantableOperation):
            return op

        info = ExporterPatternInfo()

        if self.check_op(op):
            lut = None
            if quant_type == EspQuantType.S8:
                lut = self.calculate_lut(op, info, 127, -128, 1)
            elif quant_type == EspQuantType.S16 and self.int16_step > 0:
                lut = self.calculate_lut(op, info, 2**15 - 1, -(2**15), self.int16_step)

            if lut != None:
                lut_name = self.get_lut_name(op, info)
                op.attributes["lut"] = lut_name
                info.add_lut(lut_name, lut, info.get_var_exponents(op.outputs[0].name))

        return op

import os
import sys

import numpy as np
import torch

from ppq.core import (
    DataType,
    OperationQuantizationConfig,
    QuantizationProperty,
    QuantizationStates,
    QuantizationVisibility,
    TargetPlatform,
    TensorQuantizationConfig,
    convert_any_to_numpy,
    convert_any_to_torch_tensor,
)
from ppq.executor.base import OPERATION_FORWARD_TABLE
from ppq.IR import BaseGraph, Operation, OperationExporter, Variable
from ppq.IR.quantize import QuantableOperation
from ppq.log import NaiveLogger
from ppq.parser.espdl.espdl_typedef import (
    ACTIVATION_OP_SET,
    MATH_OP_SET,
    QUANT_EXCLUDE_OP_SET,
    QUANT_OP_SET,
    EspQuantType,
    ExporterPatternInfo,
    LayoutAnnotation,
)
from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt
from ppq.utils.round import ppq_tensor_round

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

logger = NaiveLogger.get_logger("ESPDL")


class EspdlQuantHelper:
    """Helper class for processing onnx qdq format"""

    @staticmethod
    def TQC_Exportable_Check(
        TQC: TensorQuantizationConfig, bounded_var: Variable
    ) -> bool:
        if not TQC.can_export(True):
            logger.warning(
                f"Warning: skip {bounded_var.name} because it's not exportable"
            )
            return False

        if TQC.visibility == QuantizationVisibility.INTERNAL:
            logger.warning(
                f"Warning: skip {bounded_var.name} because TAC visibility is internal"
            )
            return False

        if TQC.num_of_bits == 8 and TQC.policy.has_property(
            QuantizationProperty.LINEAR
        ):
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
        raise KeyError(
            f"Can not remove operation {fusing_downstream_op.name}, operation not found."
        )

    # removing all parameters first.
    for parameter in fusing_downstream_op.inputs.copy():
        if keep_coherence and fusing_downstream_op.type in {"Constant", "Identity"}:
            break
        if parameter.is_parameter:
            parameter.dest_ops.clear()
            parameter.value = None  # clear memory.
            fusing_downstream_op.inputs.remove(parameter)

            graph.variables.pop(parameter.name)

    related_vars = [
        var for var in fusing_downstream_op.inputs + fusing_downstream_op.outputs
    ]
    input_var, output_var = (
        fusing_downstream_op.inputs[0]
        if fusing_downstream_op.num_of_input >= 1
        else None,
        fusing_downstream_op.outputs[0]
        if fusing_downstream_op.num_of_output >= 1
        else None,
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
            if (
                var.source_op is None
                and len(var.dest_ops) == 0
                and var.name in graph.variables
            ):
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


class InsertQuantTypePattern(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if (
            op.platform == TargetPlatform.ESPDL_INT8
            or op.platform == TargetPlatform.ESPDL_S3_INT8
        ):
            op.attributes["quant_type"] = EspQuantType.S8
        elif (
            op.platform == TargetPlatform.ESPDL_INT16
            or op.platform == TargetPlatform.ESPDL_S3_INT16
        ):
            op.attributes["quant_type"] = EspQuantType.S16
        else:
            op.attributes["quant_type"] = EspQuantType.F32

        return op


class InsertQuantNodePattern(OperationExporter):
    @staticmethod
    def insert_quantize_node(
        graph: BaseGraph, var: Variable, config: TensorQuantizationConfig, op: Operation
    ) -> Operation:
        """
        Insert a Quantize Node on given variable, according to given TensorQuantizationConfig.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            # Following code will export Linear Quantization Config
            # That is for FP32 -> INT
            offset_dtype, value_type = infer_qtype(config)
            scale = convert_any_to_torch_tensor(
                config.scale.clone(), dtype=torch.float32
            )
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
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            graph.create_variable(
                name=None, value=scale, is_parameter=True, dest_ops=[created]
            )
            graph.create_variable(
                name=None, value=offset, is_parameter=True, dest_ops=[created]
            )

            created.outputs[0].dtype = value_type
            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape
            return created

        else:
            raise TypeError(
                f"PPQ Can not export quantization information with variable {var.name}, "
                "Unexpected Quantization property."
            )

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
                        assert (
                            var.source_op.num_of_input == 3
                        ), "Quantize Node Format Error, need as least 3 inputs."
                        assert isinstance(var.source_op, Operation)
                        continue
                    elif var in op.inputs:
                        if (
                            not isinstance(var.source_op, QuantableOperation)
                            and var.source_op.type not in QUANT_EXCLUDE_OP_SET
                        ):
                            logger.debug(
                                f"Insert Quantize Node for {op.name}:{var.name}"
                            )
                            InsertQuantNodePattern.insert_quantize_node(
                                graph=graph,
                                var=inserting_var,
                                config=config,
                                op=inserting_op,
                            )
        return op


class InsertRequantNodePattern(OperationExporter):
    @staticmethod
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
            upstream_scale = convert_any_to_torch_tensor(
                upstream_config.scale.clone(), dtype=torch.float32
            )
            upstream_offset = ppq_tensor_round(upstream_config.offset.clone()).type(
                torch.float
            )
            offset_dtype, value_type = infer_qtype(config)
            scale = convert_any_to_torch_tensor(
                config.scale.clone(), dtype=torch.float32
            )
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
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            rescale = scale / upstream_scale
            reoffset = ppq_tensor_round(
                offset - ppq_tensor_round(upstream_offset / rescale, config.rounding)
            ).type(offset_dtype)

            graph.create_variable(
                name=None, value=rescale, is_parameter=True, dest_ops=[created]
            )
            graph.create_variable(
                name=None, value=reoffset, is_parameter=True, dest_ops=[created]
            )

            created.inputs[0].dtype = upstream_value_type
            created.inputs[0].shape = var.shape
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = value_type
            return created

        else:
            raise TypeError(
                f"PPQ Can not export quantization information with variable {var.name}, "
                "Unexpected Quantization property."
            )

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
                        assert (
                            var.source_op.num_of_input == 3
                        ), "Quantize Node Format Error, need as least 3 inputs."
                        assert isinstance(var.source_op, Operation)
                        continue
                    elif var in op.inputs and isinstance(
                        var.source_op, QuantableOperation
                    ):
                        source_op_output_var_index = var.source_op.outputs.index(var)
                        source_op_output_config = var.source_op.output_quant_config[
                            source_op_output_var_index
                        ]
                        scale_diff = torch.max(
                            torch.abs(source_op_output_config.scale - config.scale)
                        ).item()
                        zeropoint_diff = torch.max(
                            torch.abs(source_op_output_config.offset - config.offset)
                        ).item()

                        if (
                            source_op_output_config.num_of_bits != config.num_of_bits
                            or scale_diff >= 1e-4
                            or zeropoint_diff >= 1e-1
                        ):
                            # if config
                            logger.debug(
                                f"Insert Requantize Node for {op.name}:{var.name}"
                            )
                            InsertRequantNodePattern.insert_requantize_node(
                                graph=graph,
                                var=inserting_var,
                                upstream_config=source_op_output_config,
                                config=config,
                                op=inserting_op,
                            )

        return op


class InsertDequantNodePattern(OperationExporter):
    @staticmethod
    def insert_dequantize_node(
        graph: BaseGraph, var: Variable, config: TensorQuantizationConfig, op: Operation
    ) -> Operation:
        """
        Insert a DeQuantize Node on given variable, according to given TensorQuantizationConfig.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            offset_dtype, value_type = infer_qtype(config)
            scale = convert_any_to_torch_tensor(
                config.scale.clone(), dtype=torch.float32
            )
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
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            graph.create_variable(
                name=None, value=scale, is_parameter=True, dest_ops=[created]
            )
            graph.create_variable(
                name=None, value=offset, is_parameter=True, dest_ops=[created]
            )

            created.inputs[0].dtype = value_type
            created.inputs[0].shape = var.shape
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = torch.float32
            return created

        else:
            raise TypeError(
                f"PPQ Can not export quantization information with variable {var.name}, "
                "Unexpected Quantization property."
            )

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
                            logger.debug(
                                f"Insert Dequantize Node for {op.name}:{var.name}"
                            )
                            InsertDequantNodePattern.insert_dequantize_node(
                                graph=graph,
                                var=inserting_var,
                                config=config,
                                op=dest_op,
                            )
        return op


class FuseReluLikePattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if not isinstance(op, QuantableOperation):
            return op

        # The FUSE_OP_PATTERNS may remove some ops.
        if op.name not in graph.operations:
            return op

        if op.type in ["Conv", "Gemm"]:
            op.attributes["activation"] = "Linear"
            downstream_op = graph.get_downstream_operations(op)
            if (
                len(downstream_op) == 1
            ):  # the downstream op have only one op and this op is relu
                # if downstream_op[0].type in ["Relu", "Clip"]:
                if downstream_op[0].type in ["Relu"]:
                    logger.debug(
                        f"fuse {op.type}:{op.name} and {downstream_op[0].type}:{downstream_op[0].name}"
                    )
                    conv_quant_config = op.config
                    relu_quant_config = downstream_op[0].config
                    new_config = OperationQuantizationConfig(
                        conv_quant_config.input_quantization_config,
                        relu_quant_config.output_quantization_config,
                    )

                    # graph.remove_operation(downstream_op[0], keep_coherence=True)
                    graph = fuse_downstream_operation(
                        graph, downstream_op[0], keep_coherence=True
                    )
                    op.config = new_config
                    op.attributes["activation"] = downstream_op[0].type

        return op


class QuantVariableToIntPattern(OperationExporter):
    @staticmethod
    def calculate_exponent(config: TensorQuantizationConfig):
        if not config.policy.has_property(QuantizationProperty.LINEAR):
            raise ValueError("Critical Quantization Error! Non-linear config detected.")
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            raise ValueError(
                "Critical Quantization Error! Asymmetrical config detected."
            )

        if not config.scale:
            return None

        exponent = None
        if config.policy.has_property(
            QuantizationProperty.PER_TENSOR
        ) and config.policy.has_property(QuantizationProperty.POWER_OF_2):
            scale = convert_any_to_numpy(config.scale)
            exponent = [int(np.log2(scale))]
        elif config.policy.has_property(
            QuantizationProperty.PER_CHANNEL
        ) and config.policy.has_property(QuantizationProperty.POWER_OF_2):
            scale = convert_any_to_numpy(config.scale)
            exponent = np.log2(scale).astype(int)
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
                        "Skip %s from (op name:%s, type:%s) because it's not quantized"
                        % (var.name, op.name, op.type)
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
    def reset_conv_filter_layout(self, tensor, quant_type, group=None):
        if len(tensor.shape) != 4:
            logger.error(
                f"Conv filter should be 4D tensor, but got {len(tensor.shape)}D tensor."
            )
            return tensor, LayoutAnnotation.NCHW

        n, c, h, w = (
            tensor.shape
        )  # n denotes output channels, c denotes input channels,
        tensor = tensor.permute(0, 2, 3, 1)  # NCHW -> NHWC

        align = 16 if quant_type == EspQuantType.S8 else 8
        aligned_len = n // align * align
        aligned_tensor = tensor[0:aligned_len, ...]
        aligned_tensor = aligned_tensor.reshape(
            n // align, align, h, w, c
        )  # NHWC -> (N/align,align)HWC
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
                unaligned_tensor = unaligned_tensor.permute(
                    3, 1, 2, 0
                )  # depthwise unaligned: NHWC -> CHWN
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

        # TODO:: modify the layout of depthwise conv in ESP-DL, keep it same with conv
        if group == 1 or group == None:
            aligned_tensor = aligned_tensor.reshape(h, w, c, n)  # reshape to HWCN
        else:
            aligned_tensor = aligned_tensor.reshape(h, w, n, c)  # reshape to HWNC
        return aligned_tensor, layout

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        quant_type = op.attributes.get("quant_type", None)
        if (
            quant_type == None
            or quant_type == EspQuantType.F32
            or not isinstance(op, QuantableOperation)
        ):
            return op

        info = ExporterPatternInfo()

        if op.type == "Conv":
            for var in op.inputs:
                if not var.is_parameter:
                    continue

                tensor = var.value
                if len(tensor.shape) == 4:  # Conv2d Filter
                    group = op.attributes.get("group", None)
                    aligned_tensor, layout = self.reset_conv_filter_layout(
                        tensor, quant_type, group
                    )
                    info.add_var_layout(var.name, layout)
                    var.value = aligned_tensor
                    logger.debug(
                        f"reset {op.type}:{op.name}, shape:{tensor.shape}, layout to {layout}"
                    )

        elif op.type == "Gemm":
            for var in op.inputs:
                if not var.is_parameter:
                    continue

                # fix the transB attribute is 0
                tensor = var.value
                alpha = op.attributes.get("alpha", 1.0)
                beta = op.attributes.get("beta", 1.0)
                assert (
                    alpha == 1.0 and beta == 1.0
                ), "alpha and beta must be 1.0 and 0.0"

                if len(tensor.shape) == 2:  # Gemm Filter
                    trans_filter = op.attributes.get("transB", 0)
                    if trans_filter != 0:
                        logger.debug(
                            "transB is not 0, transpose the filter and reset transB"
                        )
                        op.attributes["transB"] = 0  # update 'transB'
                        tensor = tensor.transpose(1, 0)  # [N, C] -> [C, N]
                    tensor = tensor.unsqueeze(-1).unsqueeze(-1)  # CN -> CNHW
                    # CNHW -> NCHW, same with conv2d filter
                    tensor = tensor.permute(1, 0, 2, 3)

                    aligned_tensor, layout = self.reset_conv_filter_layout(
                        tensor, quant_type, None
                    )
                    info.add_var_layout(var.name, layout)
                    var.value = aligned_tensor
                    logger.debug(
                        f"reset {op.type}:{op.name}, shape:{var.value.shape}, layout to {layout}"
                    )

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
                "Register this op to ppq.executor.base.py and ppq.executor.op first"
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
        lut = PPQLinearQuant_toInt(output, op.output_quant_config[0])

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
        elif len(op.outputs) > 1 or len(op.inputs) > 1:
            return False
        elif op.type in ACTIVATION_OP_SET or op.type in MATH_OP_SET:
            return True

        return False

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        quant_type = op.attributes.get("quant_type", None)
        if (
            quant_type == None
            or quant_type == EspQuantType.F32
            or not isinstance(op, QuantableOperation)
        ):
            return op

        info = ExporterPatternInfo()

        if self.check_op(op):
            lut = None
            if quant_type == EspQuantType.S8:
                lut = self.calculate_lut(op, info, 127, -128, 1)
            elif quant_type == EspQuantType.S16 and self.int16_step > 0:
                lut = self.calculate_lut(op, info, 2**15 - 1, - 2**15, self.int16_step)

            if lut != None:
                lut_name = self.get_lut_name(op, info)
                op.attributes["lut"] = lut_name
                info.add_lut(lut_name, lut, info.get_var_exponents(op.outputs[0].name))

        return op

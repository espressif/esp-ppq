import os
import sys

import numpy as np
import torch

from esp_ppq.core import (
    ESPDL_CONV_LIKE_FUSED_ACTIVATIONS,
    ESPDL_CONV_LIKE_FUSED_OPS,
    ESPDL_QUANT_PLATFORMS,
    ESPDL_S3_PLATFORMS,
    ESPDL_S8_PLATFORMS,
    GRU_QUANT_EXPONENT,
    LSTM_QUANT_EXPONENT,
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
    transpose_shape,
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


def _fusion_edge_is_graph_output(graph: BaseGraph, upstream: Operation, downstream: Operation) -> bool:
    """Whether an edge between two operations is also a public graph output."""
    return any(
        downstream in variable.dest_ops and graph.outputs.get(variable.name) is variable
        for variable in upstream.outputs
    )


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


_W8A16_PLATFORMS = {
    TargetPlatform.ESPDL_W8A16,
    TargetPlatform.ESPDL_S3_W8A16,
    TargetPlatform.ESPDL_C_W8A16,
}
_W8A16_OPS = {"Conv", "Gemm", "MatMul"}
# Output stays FP32 on device, but S8/S16 kernel width follows quantized inputs.
_KERNEL_WIDTH_FROM_INPUT_OPS = {"Softmax", "LogSoftmax"}
# RNN gates/cell states use LSTM_QUANT_BITS; keep INT8/INT16 activations on the main path.
_LSTM_ACTIVATION_WIDTH_EXCEPTION_INPUTS = {3, 6}  # bias B, optional initial_c
_GRU_ACTIVATION_WIDTH_EXCEPTION_INPUTS = {3}  # bias B
_H_PRE_PLATFORMS = {
    TargetPlatform.ESPDL_H_PRE_INT16,
    TargetPlatform.ESPDL_S3_H_PRE_INT16,
    TargetPlatform.ESPDL_C_H_PRE_INT16,
}
_CLIP_BOUND_CODES_BAKED_DETAIL_KEY = "__espdl_clip_bound_codes_baked__"


def _effective_bits(config: TensorQuantizationConfig) -> int:
    return int(config.dominated_by.num_of_bits)


def _share_quant_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Exact grid comparison; ``atol=0`` keeps 2**k / 2**(k+1) apart at any exponent."""
    if left is None or right is None:
        return False
    return torch.allclose(torch.as_tensor(left), torch.as_tensor(right), atol=0.0)


def _quant_dtype_name(dtype: torch.dtype) -> str:
    if dtype in (torch.float32, torch.float64):
        return "FP32"
    if dtype == torch.int8:
        return "S8"
    if dtype == torch.int16:
        return "S16"
    return str(dtype)


def _skipped_cache_tensor_dtype(var: Variable) -> torch.dtype:
    """dtype the exporter leaves on a cache tensor whose own config is skipped.

    ``QuantVariableToIntPattern`` still writes the producer's dtype on a
    non-parameter output, so a runtime cache follows its producer. Anything the
    producer does not explicitly quantize keeps the FP32 default.
    """
    if var.is_parameter:
        value = var.value if isinstance(var.value, torch.Tensor) else torch.as_tensor(var.value)
        return value.dtype
    producer = var.source_op
    if isinstance(producer, QuantableOperation) and var in producer.outputs:
        produced_config = producer.output_quant_config[producer.outputs.index(var)]
        if EspdlQuantHelper.TQC_Exportable_Check(TQC=produced_config, bounded_var=var):
            dom = produced_config.dominated_by
            if dom.policy.has_property(QuantizationProperty.LINEAR):
                bits = int(dom.num_of_bits)
                if bits == 8:
                    return torch.int8
                if bits == 16:
                    return torch.int16
    return torch.float32


def _rope_exported_cache_dtype(op: QuantableOperation, config: TensorQuantizationConfig, var: Variable) -> str:
    """Device dtype of one cos/sin cache after export.

    ESP-DL rejects a model whose two caches do not share one dtype, and the
    loader reads the dtype actually written into the model. A cache config that
    export skips, including one whose state is FP32, leaves the tensor as-is.
    Its dtype then comes from the stored constant or from the producer, not
    from this config's state.
    """
    dom = config.dominated_by
    if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
        dtype = _quant_dtype_name(_skipped_cache_tensor_dtype(var))
        if dtype in {"FP32", "S8", "S16"}:
            return dtype
        raise ValueError(
            f"{op.type} {op.name} on esp-dl requires RotaryEmbedding cos/sin cache "
            f"inputs to use FP32 or linear S8/S16 quantization, got {dtype}."
        )

    if not dom.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError(
            f"{op.type} {op.name} on esp-dl requires RotaryEmbedding cos/sin cache "
            "inputs to use FP32 or linear S8/S16 quantization."
        )
    in_bits = int(dom.num_of_bits)
    if in_bits not in (8, 16):
        raise ValueError(
            f"{op.type} {op.name} on esp-dl requires RotaryEmbedding cos/sin cache "
            f"inputs to use 8- or 16-bit quantization, got {in_bits}."
        )
    return f"S{in_bits}"


def _is_w8a16_mixed_precision_op(op: Operation) -> bool:
    if op.platform not in _W8A16_PLATFORMS or op.type not in _W8A16_OPS:
        return False
    # Depthwise stores the filter in a dedicated layout; ESP-DL has no W8A16 kernel for it.
    if op.type == "Conv" and op.attributes.get("group", 1) != 1:
        return False
    if not isinstance(op, QuantableOperation) or op.num_of_input < 2 or not op.output_quant_config:
        return False
    return (
        _effective_bits(op.input_quant_config[0]) == 16
        and _effective_bits(op.input_quant_config[1]) == 8
        and _effective_bits(op.output_quant_config[0]) == 16
    )


def _kernel_width_from_inputs(op: Operation, output_index: int = 0) -> bool:
    """True when kernel bit width is taken from quantized inputs, not this output.

    ``Logical_Socket``, ``NonZero_Socket``, and ``Topk_Socket`` mark non-activation
    outputs as ``FP32``. ``Softmax`` and ``LogSoftmax`` keep a quantizable socket
    but the device kernel writes float; they are listed in
    ``_KERNEL_WIDTH_FROM_INPUT_OPS``. ``TopK`` values are not in that list: output
    0 stays quantized and carries the kernel width.
    """
    if output_index == 0 and op.type in _KERNEL_WIDTH_FROM_INPUT_OPS:
        return True
    out_plat = op.socket.out_plat or []
    return output_index < len(out_plat) and out_plat[output_index] == TargetPlatform.FP32


def _activation_quant_bits(op: QuantableOperation) -> int:
    """Activation bit width on esp-dl, with consistency checks for integer kernels.

    When ``_kernel_width_from_inputs`` is true for output 0, the S8/S16 width comes from the quantized
    non-parameter inputs. Otherwise output 0 must be an 8- or 16-bit activation,
    and every quantized activation input must use that same width. ``TopK`` takes
    the latter path: values are output 0, indices are kept FP32. ``RotaryEmbedding``
    allows cos/sin cache inputs (indices 1 and 2) to use FP32 or their own 8- or
    16-bit width. The two caches must still use the same dtype, as required by
    the device loader. ``GRU``/``LSTM`` bias and ``LSTM`` ``initial_c`` (and
    ``Y_c`` at export) follow ``LSTM_QUANT_BITS`` and are excluded from this
    same-width rule; ``quant_type`` still follows output 0.
    """

    def _quantized_input_bits() -> list[int]:
        input_bits: list[int] = []
        for cfg, var in zip(op.input_quant_config, op.inputs):
            if var is None or var.name == "" or var.is_parameter:
                continue
            in_cfg = cfg.dominated_by
            if in_cfg.state == QuantizationStates.FP32:
                continue
            input_bits.append(int(in_cfg.num_of_bits))
        return input_bits

    out_cfg = op.output_quant_config[0].dominated_by
    if out_cfg.state == QuantizationStates.FP32:
        if _kernel_width_from_inputs(op, 0):
            input_bits = _quantized_input_bits()
            if not input_bits:
                raise ValueError(
                    f"{op.type} {op.name} on esp-dl keeps FP32 output, but has no quantized activation input "
                    "to pick S8/S16 kernel width."
                )
            if any(bits != input_bits[0] for bits in input_bits[1:]):
                raise ValueError(
                    f"{op.type} {op.name} on esp-dl requires all quantized activation inputs to share one bit width "
                    f"(8 or 16), got {input_bits}."
                )
            if input_bits[0] not in (8, 16):
                raise ValueError(
                    f"{op.type} {op.name} on esp-dl requires 8- or 16-bit activation quantization, "
                    f"got input {input_bits[0]}."
                )
            return input_bits[0]
        raise ValueError(
            f"{op.type} {op.name} on esp-dl requires a quantized activation output (8 or 16-bit), "
            "but output is FP32. Keep this op on FP32 platform, or calibrate and activate output quantization first."
        )
    out_bits = int(out_cfg.num_of_bits)
    if out_bits not in (8, 16):
        raise ValueError(
            f"{op.type} {op.name} on esp-dl requires 8- or 16-bit activation quantization, got {out_bits}."
        )
    cache_dtypes: list[str] = []
    for index, (cfg, var) in enumerate(zip(op.input_quant_config, op.inputs)):
        if var is None or var.name == "":
            continue
        in_cfg = cfg.dominated_by
        if op.type == "RotaryEmbedding" and index in (1, 2):
            cache_dtypes.append(_rope_exported_cache_dtype(op, cfg, var))
            continue
        if op.type == "LSTM" and index in _LSTM_ACTIVATION_WIDTH_EXCEPTION_INPUTS:
            continue
        if op.type == "GRU" and index in _GRU_ACTIVATION_WIDTH_EXCEPTION_INPUTS:
            continue
        if var.is_parameter or in_cfg.state == QuantizationStates.FP32:
            continue
        in_bits = int(in_cfg.num_of_bits)
        if in_bits != out_bits:
            raise ValueError(
                f"{op.type} {op.name} on esp-dl requires input and output activations to share one bit width "
                f"(8 or 16), got input {in_bits} and output {out_bits}."
            )
    if (
        op.type == "RotaryEmbedding"
        and len(cache_dtypes) >= 2
        and any(dtype != cache_dtypes[0] for dtype in cache_dtypes[1:])
    ):
        raise ValueError(
            f"{op.type} {op.name} on esp-dl requires RotaryEmbedding cos/sin cache inputs "
            f"to share one dtype, got {cache_dtypes}."
        )
    return out_bits


def _layernorm_device_quant_type(op: Operation):
    """S8 or S16 when every LayerNorm tensor shares that device kernel.

    None is only for an op that is not LayerNorm. An ESP-DL LayerNorm whose
    X, scale, bias, and Y are not one activated symmetric int8 or int16 width
    cannot be exported: simulation keeps the float reference, while an integer
    ``quant_type`` would run the device kernel. Mixed 8/16 widths raise from
    ``_layernorm_quantization``.
    """
    if op.type not in {"LayerNorm", "LayerNormalization"} or not isinstance(op, QuantableOperation):
        return None
    from esp_ppq.executor.op.torch.default import _layernorm_quantization

    quant = _layernorm_quantization(op, op.num_of_input >= 3)
    if quant is None:
        if getattr(op, "platform", None) in ESPDL_QUANT_PLATFORMS:
            raise ValueError(
                f"LayerNorm {op.name} on esp-dl is not an activated symmetric int8 or int16 kernel. "
                "Simulation uses the float reference, but the device integer kernel reads every tensor as a code. "
                "Calibrate X, scale, bias, and Y to one symmetric 8- or 16-bit width, "
                "or keep this op on an FP32 platform."
            )
        return None
    accum_int32 = quant[2]
    return EspQuantType.S8 if accum_int32 else EspQuantType.S16


class InsertQuantTypePattern(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if (
            op.type in {"Softmax", "LogSoftmax", "Attention"}
            and isinstance(op, QuantableOperation)
            and all(cfg.state == QuantizationStates.FP32 for cfg in op.input_quant_config + op.output_quant_config)
        ):
            op.attributes["quant_type"] = EspQuantType.F32
            return op
        if op.platform in ESPDL_QUANT_PLATFORMS:
            layernorm_type = _layernorm_device_quant_type(op)
            if _is_w8a16_mixed_precision_op(op):
                op.attributes["quant_type"] = EspQuantType.W8A16
            elif layernorm_type is not None:
                op.attributes["quant_type"] = layernorm_type
            elif isinstance(op, QuantableOperation) and op.output_quant_config:
                bits = _activation_quant_bits(op)
                op.attributes["quant_type"] = EspQuantType.S8 if bits == 8 else EspQuantType.S16
            elif op.platform in ESPDL_S8_PLATFORMS:
                op.attributes["quant_type"] = EspQuantType.S8
            else:
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
                            created = insert_quantize_node(
                                graph=graph,
                                var=inserting_var,
                                config=config,
                                op=inserting_op,
                            )
                            if created:
                                info = ExporterPatternInfo()
                                var_perm = info.get_var_permute(inserting_var.name)
                                if var_perm:
                                    info.add_var_permute(created.outputs[0].name, var_perm)
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
                        # Integer Clip compares raw codes directly; even a small power-of-two
                        # scale step must insert RequantizeLinear instead of falling below 1e-5.
                        clip_bound_grid_differs = (
                            op.type == "Clip"
                            and var in list(op.inputs)[1:]
                            and (
                                not _share_quant_tensor(source_op_output_config.scale, config.scale)
                                or not _share_quant_tensor(source_op_output_config.offset, config.offset)
                            )
                        )

                        if (
                            _effective_bits(source_op_output_config) != _effective_bits(config)
                            or scale_diff >= 1e-5
                            or zeropoint_diff >= 1e-1
                            or clip_bound_grid_differs
                        ):
                            # if config
                            logger.debug(f"Insert Requantize Node for {op.name}:{var.name}")
                            created = insert_requantize_node(
                                graph=graph,
                                var=inserting_var,
                                upstream_config=source_op_output_config,
                                config=config,
                                op=inserting_op,
                            )
                            if created:
                                info = ExporterPatternInfo()
                                var_perm = info.get_var_permute(inserting_var.name)
                                if var_perm:
                                    info.add_var_permute(created.outputs[0].name, var_perm)

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
                            created = insert_dequantize_node(
                                graph=graph,
                                var=inserting_var,
                                config=config,
                                op=dest_op,
                            )
                            if created:
                                info = ExporterPatternInfo()
                                var_perm = info.get_var_permute(inserting_var.name)
                                if var_perm:
                                    info.add_var_permute(created.outputs[0].name, var_perm)
        return op


class InsertPreNodeOfMatMulPattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type != "MatMul" or not isinstance(op, QuantableOperation):
            return op

        # Static right-hand operands are packed offline by ResetParamLayoutPattern.
        # Keep this explicit so the runtime never has to infer a physical layout
        # from tensor shapes. Models exported before this attribute was added are
        # treated as packed by ESP-DL for backward compatibility.
        if op.inputs[1].is_parameter:
            op.attributes["input1_layout"] = "packed"
            return op

        # P4/S3 SIMD and the C (ESP32) scalar kernel all accept native KN,
        # including fused ReLU. Dynamic right-hand operands therefore stay in
        # their logical row-major layout and avoid the Reshape/Transpose/
        # Slice/Concat packing graph used by Conv2D.
        if op.platform in [
            TargetPlatform.ESPDL_INT8,
            TargetPlatform.ESPDL_INT16,
            TargetPlatform.ESPDL_W8A16,
            TargetPlatform.ESPDL_H_PRE_INT16,
            TargetPlatform.ESPDL_S3_INT8,
            TargetPlatform.ESPDL_S3_INT16,
            TargetPlatform.ESPDL_S3_W8A16,
            TargetPlatform.ESPDL_S3_H_PRE_INT16,
            TargetPlatform.ESPDL_C_INT8,
            TargetPlatform.ESPDL_C_INT16,
            TargetPlatform.ESPDL_C_W8A16,
            TargetPlatform.ESPDL_C_H_PRE_INT16,
        ] and op.attributes.get("activation", "Linear") in ("Linear", "Relu"):
            op.attributes["input1_layout"] = "native_kn"
            return op

        op.attributes["input1_layout"] = "packed"
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
            TargetPlatform.ESPDL_C_W8A16,
        ]:
            if input1_dims >= 2:
                # *CN -> *NC
                # ESP32 C conv2d (int8 / int16 / w8a16) expects NHWC, not the
                # (N/align)HWC(align) packing used by S3/P4 SIMD kernels.
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

        if op.type in ESPDL_CONV_LIKE_FUSED_OPS:
            op.attributes["activation"] = "Linear"
            downstream_op = graph.get_downstream_operations(op)
            if len(downstream_op) == 1:  # the downstream op have only one op and this op is relu
                # if downstream_op[0].type in ["Relu", "Clip"]:
                if downstream_op[0].type in ESPDL_CONV_LIKE_FUSED_ACTIVATIONS:
                    if _fusion_edge_is_graph_output(graph, op, downstream_op[0]):
                        return op
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


def _clip_integer_kernel_bits(op: Operation):
    """Bit width of the integer Clip kernel, or None when Clip exports a LUT or FP32.

    INT16 Clip always runs on the integer kernel. INT8 Clip runs on the integer
    kernel when at least one bound is a runtime tensor (so ``AddLUTPattern``
    does not replace the op with a step-1 table).
    """
    if op.type != "Clip":
        return None
    quant_type = op.attributes.get("quant_type")
    if quant_type == EspQuantType.S16:
        return 16
    if quant_type == EspQuantType.S8 and any(bound.value is None and bound.name != "" for bound in list(op.inputs)[1:]):
        return 8
    return None


def _clip_bound_on_input_grid(op: Operation, var: Variable) -> bool:
    """True when a Clip constant bound can be baked with the activation grid."""
    if not _clip_bound_requires_integer_codes(op, var):
        return False
    if not isinstance(op, QuantableOperation) or not op.input_quant_config:
        return False
    grid = op.input_quant_config[0].dominated_by
    if grid.scale is None or grid.offset is None:
        return False
    return (
        grid.policy.has_property(QuantizationProperty.LINEAR)
        and grid.policy.has_property(QuantizationProperty.PER_TENSOR)
        and grid.policy.has_property(QuantizationProperty.SYMMETRICAL)
    )


# States for which the executor fake-quants a Clip bound, or the value is already
# the integer code. FP32 and the other inactive states leave the float in place.
_SIMULATED_CLIP_BOUND_STATES = {
    QuantizationStates.ACTIVATED,
    QuantizationStates.PASSIVE,
    QuantizationStates.BAKED,
    QuantizationStates.PASSIVE_BAKED,
}


def _integer_clip_bound_not_simulated(op: Operation, config: TensorQuantizationConfig, var: Variable) -> bool:
    """Integer-kernel Clip bound that the simulator still clamps as a float."""
    if _clip_integer_kernel_bits(op) is None:
        return False
    if var is None or var.name == "" or var not in list(op.inputs)[1:]:
        return False
    return config.state not in _SIMULATED_CLIP_BOUND_STATES


def _clip_bound_requires_integer_codes(op: Operation, var: Variable) -> bool:
    if _clip_integer_kernel_bits(op) is None:
        return False
    if op.type != "Clip":
        return False
    if var is None or var.name == "" or not var.is_parameter:
        return False
    return var in list(op.inputs)[1:] and len(var.dest_ops) == 1


def _clip_code_config(grid: TensorQuantizationConfig, bits: int) -> TensorQuantizationConfig:
    """Quant config that stores codes in the int8 or int16 the Clip kernel reads.

    An activation that already matches ``bits`` keeps its own clamp. Any other
    activation width still stores ``bits``-bit codes so the bound's own bit
    width does not change the storage type.
    """
    if int(grid.num_of_bits) == bits:
        quant_min, quant_max = int(grid.quant_min), int(grid.quant_max)
    else:
        half = 1 << (bits - 1)
        quant_min, quant_max = -half, half - 1
    return TensorQuantizationConfig(
        policy=grid.policy,
        rounding=grid.rounding,
        num_of_bits=bits,
        quant_min=quant_min,
        quant_max=quant_max,
        scale=grid.scale,
        offset=grid.offset,
        state=QuantizationStates.ACTIVATED,
    )


def _write_clip_codes(op: Operation, config: TensorQuantizationConfig, var: Variable) -> None:
    """Replace a Clip bound with integer codes the integer kernel reads."""
    bits = _clip_integer_kernel_bits(op)
    if bits is None or var.value is None:
        return
    if config.detail.get(_CLIP_BOUND_CODES_BAKED_DETAIL_KEY):
        return
    if not _clip_bound_on_input_grid(op, var):
        raise ValueError(
            f"Clip {op.name} bound {var.name} can not be baked onto the Clip input grid: "
            "input activation quantization is not ready as a per-tensor symmetric linear grid."
        )
    grid = op.input_quant_config[0].dominated_by
    code_config = _clip_code_config(grid, bits)
    value = var.value if isinstance(var.value, torch.Tensor) else torch.as_tensor(var.value)
    use_fp64 = op.platform in _H_PRE_PLATFORMS and bits >= 16
    if not value.is_floating_point():
        value = value.to(dtype=torch.float64)
    elif use_fp64:
        value = value.to(dtype=torch.float64)
    # NaN has no code. ±inf is an unbounded side: PPQLinearQuant_toInt clamps it
    # to quant_min or quant_max, the same code the simulator and esp-dl use.
    if torch.isnan(value).any():
        raise ValueError(f"Clip {op.name} bound {var.name} contains NaN and cannot be baked for the integer kernel.")
    var.value = PPQLinearQuant_toInt(tensor=value, config=code_config)
    config.detail[_CLIP_BOUND_CODES_BAKED_DETAIL_KEY] = True


class QuantVariableToIntPattern(OperationExporter):
    @staticmethod
    def calculate_exponent(config: TensorQuantizationConfig):
        if not config.policy.has_property(QuantizationProperty.LINEAR):
            raise ValueError("Critical Quantization Error! Non-linear config detected.")
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            raise ValueError("Critical Quantization Error! Asymmetrical config detected.")

        if config.scale is None:
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

            if _integer_clip_bound_not_simulated(op, config, var):
                raise ValueError(
                    f"Clip {op.name} bound {var.name} is still {config.state.name}. "
                    "The simulator clamps that float value, but esp-dl integer Clip "
                    f"({op.attributes.get('quant_type')}) reads the bound as a code. "
                    "Enable quantize_passive_parameter so PassiveParameterQuantizePass "
                    "aligns the bound to the input grid before export."
                )

            if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                # INTERNAL still withholds scale, exponent, and quant nodes.
                # The integer Clip kernel reads the bound bytes, so bake those bytes only.
                # A bound the simulator never fake-quantized was rejected above.
                if _clip_bound_requires_integer_codes(op, var):
                    _write_clip_codes(op, config, var)
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
                    if _clip_bound_requires_integer_codes(op, var):
                        _write_clip_codes(op, config, var)
                    elif op.platform in _H_PRE_PLATFORMS and config.num_of_bits >= 16:
                        var.value = PPQLinearQuant_toInt(tensor=var.value.type(dtype=torch.float64), config=config)
                    else:
                        var.value = PPQLinearQuant_toInt(tensor=var.value, config=config)
            elif not var.is_parameter:
                dom = config.dominated_by
                if dom.policy.has_property(QuantizationProperty.LINEAR):
                    if dom.num_of_bits == 8 and dom.exponent_bits == 0:
                        var.dtype = DataType.INT8
                    elif dom.num_of_bits == 16 and dom.exponent_bits == 0:
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
                TargetPlatform.ESPDL_C_W8A16,
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

            # Only the "C" (CPU) targets keep the plain NHWC filter layout. ESP32-S3
            # (ESPDL_S3_W8A16) and ESP32-P4 (ESPDL_W8A16) W8A16 convolutions share the
            # same (N/8)HWC8 packed layout below, so both chips export identical layouts.
            if platform in [
                TargetPlatform.ESPDL_C_INT8,
                TargetPlatform.ESPDL_C_INT16,
                TargetPlatform.ESPDL_C_H_PRE_INT16,
                TargetPlatform.ESPDL_C_W8A16,
            ]:
                layout = LayoutAnnotation.NHWC
                if group is not None and group > 1:
                    tensor = tensor.permute(1, 2, 3, 0)  # depthwise: NHWC -> HWCN
                    layout = LayoutAnnotation.HWCN
            else:
                # ESP32-P4/S3 W8A16 accumulate int8 weights in 8-wide QACC lanes,
                # so the packed block size is 8 regardless of the weight width.
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


def _activation_shape_on_device(var: Variable) -> list:
    """Shape esp-dl sees: the IR shape after the export permute.

    Layout passes record that permute without rewriting ``var.shape``. The
    integer PRelu kernel then indexes channel axis 3 of this shape.
    """
    if not var.shape:
        raise ValueError(f"PRelu activation {var.name} has no shape, so the integer kernel's channel axis is unknown.")
    shape = list(var.shape)
    perm = ExporterPatternInfo().get_var_permute(var.name)
    if not perm:
        return shape
    if len(perm) != len(shape):
        raise ValueError(f"PRelu activation {var.name} permute {list(perm)} does not match shape {shape}.")
    return list(transpose_shape(shape, perm))


# Packed ``(N/8)HWC8*`` layouts are for Conv/Gemm weights only, not activations.
_PRELU_ACTIVATION_NHWC_LAYOUTS = frozenset({LayoutAnnotation.NHWC})
_PRELU_DEVICE_NCHW_LAYOUTS = frozenset({LayoutAnnotation.NCHW, LayoutAnnotation.HWCN})


def _prelu_channel_axis3_confirmed(var: Variable, numel: int) -> bool:
    """True when export metadata or conservative IR shape says axis 3 is channel.

    Prefer ``ExporterPatternInfo`` layout when a layout pass marks the *activation*
    as plain ``NHWC`` and ``shape[3]`` matches the per-channel slope count. Packed
    filter layouts (``N8HWC8``, ``N8HWC8_UNALIGNED``, …) describe weight storage,
    not feature-map geometry, and must not be copied onto activations here.
    Annotated NCHW (or any other known non-NHWC layout) refuses this path.
    ``LayoutAnnotation.UNKNOWN`` does not prove axis 3 is channel; use perm
    ``[0, 2, 3, 1]`` or an explicit activation ``NHWC`` layout instead.
    """
    ir_shape = list(var.shape)
    if len(ir_shape) != 4 or int(ir_shape[3]) != numel:
        return False

    layout = ExporterPatternInfo().get_var_layout(var.name, LayoutAnnotation.UNKNOWN)
    if layout in _PRELU_ACTIVATION_NHWC_LAYOUTS:
        return True
    if layout in _PRELU_DEVICE_NCHW_LAYOUTS:
        return False
    return False


def _prelu_activation_is_rank4_on_device(var: Variable) -> bool:
    """True when this activation is rank 4 after export permute.

    A scalar INT16 slope is broadcast by the integer PRelu path, so its routing
    only needs a valid rank-4 activation shape on device. Per-channel slope
    layout checks stay in ``_reshape_prelu_slope``.
    """
    try:
        shape = _activation_shape_on_device(var)
    except ValueError:
        return False
    return len(shape) == 4 and shape[3] is not None


def _var_exponent(var: Variable, config, info: ExporterPatternInfo):
    exp = info.get_var_exponents(var.name)
    if isinstance(exp, list):
        exp = exp[0] if exp else None
    if exp is not None:
        return int(exp)
    if config is not None:
        calculated = QuantVariableToIntPattern.calculate_exponent(config)
        if calculated:
            return int(calculated[0])
    return None


def _s3_prelu_changes_exponent(op: Operation, info: ExporterPatternInfo) -> bool:
    """True on ESP32-S3 when PRelu input and output do not share one exponent.

    ``dl_tie728_s16_prelu_11c`` executes ``EE.VPRELU.S16``. The positive branch
    is copied through with no ``output_shift``, so that branch matches
    simulation only when the exponents are equal. The negative branch is an
    arithmetic right shift on S3, P4, and C (``EE.VPRELU.S16`` and
    ``DL_RIGHT_SHIFT``). Simulation rounds it, so those codes can differ by 1.
    """
    if op.platform not in ESPDL_S3_PLATFORMS or not op.inputs or not op.outputs:
        return False
    in_cfg = op.input_quant_config[0] if isinstance(op, QuantableOperation) and op.input_quant_config else None
    out_cfg = op.output_quant_config[0] if isinstance(op, QuantableOperation) and op.output_quant_config else None
    in_exp = _var_exponent(op.inputs[0], in_cfg, info)
    out_exp = _var_exponent(op.outputs[0], out_cfg, info)
    return in_exp is None or out_exp is None or in_exp != out_exp


def _reshape_prelu_slope(op: Operation) -> None:
    """Store the integer-kernel slope as ``[c, 1, 1]``.

    ``PRelu::deserialize`` asserts that rank. The kernel indexes ``alpha[c]``
    with ``c`` taken from activation axis 3 (NHWC). A slope with any other
    number of elements is rejected: the int8 kernel would read off the end or
    ignore the tail, and the int16 module would copy ``alpha[0]`` onto every
    channel. An elementwise slope is not a supported input. An INT16 slope of
    length 1 is stored as ``[1, 1, 1]``; the int16 module broadcasts that value.
    A longer slope is refused when axis 3 is not the IR channel, including an
    identity permute that leaves the tensor in NCHW. An INT8 scalar stays on
    the step-1 table and never reaches this function. Runs on the export clone,
    after the slope has been turned into integer codes.
    """
    if len(op.inputs) < 2:
        return
    slope = op.inputs[1].value
    if slope is None:
        return
    device_shape = _activation_shape_on_device(op.inputs[0])
    if len(device_shape) < 4 or device_shape[3] is None:
        raise ValueError(
            f"{op.name} exports PRelu on the integer kernel, but the activation "
            f"reaching esp-dl has shape {device_shape}. That kernel indexes channel axis 3."
        )
    channels = int(device_shape[3])
    flat = slope.reshape(-1)
    numel = int(flat.numel())
    if op.attributes.get("quant_type") == EspQuantType.S16 and numel == 1:
        if slope.ndim == 3 and list(slope.shape) == [1, 1, 1]:
            return
        op.inputs[1].value = flat.reshape(1, 1, 1)
        return
    perm = ExporterPatternInfo().get_var_permute(op.inputs[0].name)
    perm_axes = [int(axis) for axis in perm] if perm else None
    nhwc_perm = [0, 2, 3, 1]
    activation = op.inputs[0]

    def _reject_prelu_slope_layout(permute: list | None) -> None:
        shown = "unset" if not permute else list(permute)
        raise ValueError(
            f"{op.name} exports PRelu on the integer kernel, but activation {op.inputs[0].name} "
            f"reaches esp-dl with permute {shown}. That kernel indexes the slope along axis 3, "
            "while simulation applies a per-channel slope along IR axis 1. "
            "Those axes match for an NHWC permute."
        )

    layout_ok = _prelu_channel_axis3_confirmed(activation, numel)
    if perm_axes != nhwc_perm and not layout_ok:
        _reject_prelu_slope_layout(perm_axes)
    if numel != channels:
        raise ValueError(
            f"{op.name} exports PRelu on the integer kernel with {numel} slope elements, "
            f"but the activation has {channels} channels. "
            "esp-dl applies one slope per channel and does not accept an elementwise slope."
        )
    if slope.ndim == 3 and list(slope.shape) == [channels, 1, 1]:
        return
    op.inputs[1].value = flat.reshape(channels, 1, 1)


class AddLUTPattern(OperationExporter):
    def __init__(self, int16_step=1) -> None:
        super().__init__()
        self.int16_step = int(int16_step)  # the step of int16 LUT

    def get_scale(self, var: Variable, info: ExporterPatternInfo) -> float:
        exponent = info.get_var_exponents(var.name)
        if not exponent:
            return 1.0
        if isinstance(exponent, list):
            exponent = exponent[0]
        return 2.0 ** int(exponent)

    def calculate_lut(
        self,
        op: QuantableOperation,
        info: ExporterPatternInfo,
        max: int,
        min: int,
        step: int = 1,
    ) -> torch.Tensor:
        # If this op was rewritten to a "LUT" op during quantization
        # (EspdlLUTFusionPass), recover the original activation type so the
        # table is built from the true math (e.g. "Swish") instead of the
        # placeholder "LUT" type.
        forward_type = op.attributes.get("original_op_type", op.type)

        from esp_ppq.executor.op.torch.espdl_lut import GlobalMode, SimulationMode

        # Get forward function
        platform_dispatching_table = OPERATION_FORWARD_TABLE[op.platform]
        if forward_type not in platform_dispatching_table:
            raise NotImplementedError(
                f"Graph op: {op.name}({forward_type}) "
                f"has no backend implementation on target platform {op.platform}. "
                "Register this op to esp_ppq.executor.base.py and esp_ppq.executor.op first"
            )
        operation_forward_func = platform_dispatching_table[forward_type]

        # Calculate output and lut
        input = torch.arange(min, max + step, step=step, dtype=torch.float)
        input = input * self.get_scale(op.inputs[0], info)
        inputs = [input]

        if len(op.inputs) > 1:
            for op_input in op.inputs[1:]:
                # ONNX optional inputs (e.g. the trailing ``max`` of a Clip
                # produced by ``F.normalize`` -> ``clamp_min(eps)``) are
                # parsed with ``value=None``. Forward through as-is so the
                # op kernel (torch.clamp etc.) applies its missing-input
                # semantics; multiplying ``None * scale`` would TypeError.
                if op_input.value is None:
                    inputs.append(None)
                else:
                    inputs.append(op_input.value.float() * self.get_scale(op_input, info))

        # Temporarily restore the original op type while running the forward, so
        # forwards that dispatch on ``op.type`` (e.g. UnaryEltwise_forward for
        # Sigmoid/Relu) compute the correct activation.
        saved_type = op.type
        saved_mode = GlobalMode.get()
        op.type = forward_type
        GlobalMode.set(SimulationMode.IDEAL_MATH)
        try:
            output = operation_forward_func(op, inputs)
        finally:
            op.type = saved_type
            GlobalMode.set(saved_mode)

        device = op.output_quant_config[0].scale.device
        lut = PPQLinearQuant_toInt(output.to(device), op.output_quant_config[0])

        return lut

    def get_lut_name(self, op: Operation, info: ExporterPatternInfo):
        index = len(info.luts)
        forward_type = op.attributes.get("original_op_type", op.type)
        name = f"{forward_type}_lut_{index}"
        return name

    @staticmethod
    def _is_16bit_unary(op: Operation) -> bool:
        in_cfg = getattr(op, "input_quant_config", None) or ()
        out_cfg = getattr(op, "output_quant_config", None) or ()
        return bool(in_cfg and out_cfg and _effective_bits(in_cfg[0]) == 16 and _effective_bits(out_cfg[0]) == 16)

    def _s16_exports_lut(self, op: Operation, _info: ExporterPatternInfo) -> bool:
        """True when this INT16 node exports a lookup table.

        ``Gelu`` is registered as ``LUT`` on esp-dl and has no other kernel.
        A scalar-slope ``PRelu`` uses the integer kernel when its activation is
        rank 4 on the device and (on ESP32-S3) input and output share one
        exponent; otherwise it falls back to a table. The kernel's negative
        branch truncates on every platform. Simulation rounds it and can differ
        by 1 code.
        """
        if op.type == "LUT":
            return True
        if op.type == "Gelu" and self._is_16bit_unary(op):
            return True
        if op.type == "PRelu" and self.check_op(op):
            return True
        return False

    def check_op(self, op: Operation):
        """
        True if this op can be implemented by LUT, otherwise False
        """

        # INT8 Clip uses a step-1 table unless a bound is a runtime tensor.
        # INT16 Clip returns early above and stays on the integer kernel.
        if op.type == "Clip":
            return all(bound.value is not None or bound.name == "" for bound in op.inputs[1:])
        if op.type == "Not":
            # Bitwise NOT is undefined on the float grid used to sample S8 LUTs.
            return False
        if op.type == "PRelu":
            # INT8 does not broadcast a length-1 slope, so a scalar stays on
            # the step-1 table. INT16 copies that one value onto every channel
            # and uses the integer kernel when the activation is rank 4 on the
            # device and (on ESP32-S3, where EE.VPRELU.S16 passes the positive
            # branch through unchanged) input and output share one exponent.
            # The negative branch is an arithmetic right shift on every
            # platform, so simulation can still differ by 1 code.
            slope = op.inputs[1].value if len(op.inputs) > 1 else None
            if slope is None or int(slope.numel()) != 1:
                return False
            if op.attributes.get("quant_type") == EspQuantType.S8:
                return True
            if not _prelu_activation_is_rank4_on_device(op.inputs[0]):
                return True
            return _s3_prelu_changes_exponent(op, ExporterPatternInfo())
        elif len(op.outputs) > 1 or len(op.inputs) > 1:
            return False
        elif op.type in ACTIVATION_OP_SET or op.type in MATH_OP_SET:
            return True

        return False

    @staticmethod
    def _s8_refuses_int16_lut(op: Operation) -> bool:
        """An INT8 node must not use the INT16 ``LUT`` op."""
        return op.type == "LUT"

    @staticmethod
    def _expected_prelu_slope_bits(quant_type: str):
        if quant_type == EspQuantType.S8:
            return 8
        if quant_type == EspQuantType.S16:
            return 16
        return None

    def _prelu_integer_kernel_has_mismatched_slope_bits(self, op: QuantableOperation, quant_type: str) -> bool:
        # INT8 scalar PRelu keeps its table. Every other PRelu reads the slope.
        if op.type != "PRelu" or self.check_op(op):
            return False
        expected_bits = self._expected_prelu_slope_bits(quant_type)
        if expected_bits is None or len(op.input_quant_config) < 2:
            return False
        slope_cfg = op.input_quant_config[1]
        return _effective_bits(slope_cfg) != expected_bits

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        quant_type = op.attributes.get("quant_type", None)
        if quant_type == None or quant_type == EspQuantType.F32 or not isinstance(op, QuantableOperation):
            return op

        info = ExporterPatternInfo()

        # A PRelu without a lookup table runs on the esp-dl integer kernel.
        # That kernel loads one slope per channel as a rank-3 [c, 1, 1]
        # parameter and can not read a runtime tensor. Reject any other
        # constant slope here, on the export clone, after
        # QuantVariableToIntPattern has turned it into integer codes.
        if op.type == "PRelu" and not self.check_op(op):
            slope = op.inputs[1].value if len(op.inputs) > 1 else None
            if slope is None:
                raise ValueError(
                    f"{op.name} exports PRelu without a lookup table, but it has no constant slope. "
                    "esp-dl loads the PRelu slope as a model parameter, so a runtime slope can not deploy; "
                    "constant-fold the slope, or keep this op in FP32."
                )
            if (
                quant_type == EspQuantType.S16
                and int(slope.numel()) > 1
                and op.platform in ESPDL_S3_PLATFORMS
                and _s3_prelu_changes_exponent(op, info)
            ):
                in_cfg = op.input_quant_config[0] if op.input_quant_config else None
                out_cfg = op.output_quant_config[0] if op.output_quant_config else None
                in_exp = _var_exponent(op.inputs[0], in_cfg, info)
                out_exp = _var_exponent(op.outputs[0], out_cfg, info)
                raise ValueError(
                    f"{op.name} exports INT16 PRelu with per-channel slope on ESP32-S3 without LUT, "
                    f"but input/output exponents are unknown or differ ({in_exp} vs {out_exp}). "
                    "S3 EE.VPRELU.S16 does not requantize the positive branch in this path; "
                    "align output scale with input scale, or use a scalar slope."
                )
            _reshape_prelu_slope(op)

        if self._prelu_integer_kernel_has_mismatched_slope_bits(op, quant_type):
            actual_bits = _effective_bits(op.input_quant_config[1])
            raise ValueError(
                f"{op.name} exports PRelu as {quant_type} without LUT, but slope uses {actual_bits}-bit quantization. "
                "PRelu integer kernels read slope with the activation dtype. "
                "custom_tqc must keep slope bit width aligned with quant_type."
            )

        if quant_type == EspQuantType.S8 and self._s8_refuses_int16_lut(op):
            raise ValueError(f"{op.name} is an INT16 LUT op, but its quant_type is S8.")

        if (
            quant_type == EspQuantType.S16
            and op.type == "Clip"
            and len(op.inputs) < 2
            and any(bound in op.attributes for bound in ("min", "max"))
        ):
            raise ValueError(
                f"Clip {op.name} exports as INT16 with min/max attributes but no bound inputs. "
                "Run format_graph (FORMAT_CLIP) to materialize bounds as inputs, or keep this op on FP32 platform."
            )

        if quant_type == EspQuantType.S16 and op.type in {"LUT", "Gelu"}:
            if op.type == "Gelu" and not self._is_16bit_unary(op):
                raise ValueError(
                    f"Gelu {op.name} exports as INT16, so its input and output must be 16-bit. "
                    "esp-dl registers Gelu as LUT and cannot load it without an INT16 table."
                )
        if quant_type == EspQuantType.S16 and not self._s16_exports_lut(op, info):
            return op
        if quant_type == EspQuantType.S16:
            int16_step = int(op.attributes.get("int16_lut_step", self.int16_step))
            if int16_step <= 0 or int16_step > 65536 or (int16_step & (int16_step - 1)) != 0:
                raise ValueError(
                    f"{op.type} {op.name} requires an INT16 lookup table on esp-dl; "
                    f"int16_lut_step must be a positive power of two (<= 65536), got {int16_step}."
                )

        if self.check_op(op):
            lut = None
            if quant_type == EspQuantType.S8:
                lut = self.calculate_lut(op, info, 127, -128, 1)
            else:
                # Fused ops carry the step EspdlLUTFusionPass stored.
                int16_step = int(op.attributes.get("int16_lut_step", self.int16_step))
                if quant_type == EspQuantType.S16 and int16_step > 0:
                    lut = self.calculate_lut(op, info, 2**15 - 1, -(2**15), int16_step)

            if lut != None:
                lut_name = self.get_lut_name(op, info)
                lut_name = info.add_lut(
                    lut_name,
                    lut,
                    info.get_var_exponents(op.outputs[0].name),
                )
                op.attributes["lut"] = lut_name

        return op


class QuantRNNPattern(OperationExporter):
    def export(self, op: QuantableOperation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ["GRU"] and isinstance(op, QuantableOperation):
            op.attributes["gate_exponent"] = GRU_QUANT_EXPONENT
        elif op.type in ["LSTM"] and isinstance(op, QuantableOperation):
            op.attributes["gate_exponent"] = LSTM_QUANT_EXPONENT

        return op

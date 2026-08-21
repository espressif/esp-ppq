import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from esp_ppq.api.espdl_interface import espdl_quantize_torch, get_target_platform
from esp_ppq.core import NetworkFramework, TargetPlatform
from esp_ppq.IR import BaseGraph, QuantableOperation
from esp_ppq.lib.quant import Quantizer
from esp_ppq.parser.espdl.espdl_typedef import (
    EspQuantType,
    ExporterPatternInfo,
    LayoutAnnotation,
)
from esp_ppq.parser.espdl.export_patterns import InsertQuantTypePattern, ResetParamLayoutPattern

_W8A16_CASES = [
    ("esp32p4", TargetPlatform.ESPDL_W8A16, 40),
    ("esp32s3", TargetPlatform.ESPDL_S3_W8A16, 40),
    ("esp32c3", TargetPlatform.ESPDL_C_W8A16, 64),
]


def _make_operation(
    graph,
    op_type,
    platform,
    constant_weight=True,
    with_bias=False,
    out_channels=17,
):
    if op_type == "Conv":
        activation = graph.create_variable(name="conv_input", shape=[1, 3, 8, 8])
        weight_value = torch.randn(out_channels, 3, 3, 3) if constant_weight else None
        weight = graph.create_variable(
            name="conv_weight",
            value=weight_value,
            is_parameter=constant_weight,
            shape=None if constant_weight else [out_channels, 3, 3, 3],
        )
        output = graph.create_variable(name="conv_output", shape=[1, out_channels, 8, 8])
    else:
        activation = graph.create_variable(name=f"{op_type}_input", shape=[1, 16])
        weight_value = torch.randn(16, out_channels) if constant_weight else None
        weight = graph.create_variable(
            name=f"{op_type}_weight",
            value=weight_value,
            is_parameter=constant_weight,
            shape=None if constant_weight else [16, out_channels],
        )
        output = graph.create_variable(name=f"{op_type}_output", shape=[1, out_channels])

    inputs = [activation, weight]
    if with_bias:
        inputs.append(
            graph.create_variable(
                name=f"{op_type}_bias",
                value=torch.randn(out_channels),
                is_parameter=True,
            )
        )

    return graph.create_operation(
        op_type=op_type,
        name=f"{op_type}_op",
        platform=platform,
        inputs=inputs,
        outputs=[output],
    )


@pytest.mark.parametrize(
    ("args", "kwargs", "platform"),
    [
        (("esp32p4",), {}, TargetPlatform.ESPDL_INT8),
        (("esp32p4", "w8a8"), {}, TargetPlatform.ESPDL_INT8),
        (("esp32p4", "w8a16"), {}, TargetPlatform.ESPDL_W8A16),
        (("esp32p4", "w16a16"), {}, TargetPlatform.ESPDL_INT16),
        (("esp32p4", "w16a16", True), {}, TargetPlatform.ESPDL_H_PRE_INT16),
        (("esp32p4", "none"), {}, TargetPlatform.FP32),
        (("esp32s3", "w8a16"), {}, TargetPlatform.ESPDL_S3_W8A16),
        (("esp32c3", "w8a16"), {}, TargetPlatform.ESPDL_C_W8A16),
        # Legacy: positional / keyword num_of_bits, float, hi_precision
        (("esp32p4", 8), {}, TargetPlatform.ESPDL_INT8),
        (("esp32p4", 16), {}, TargetPlatform.ESPDL_INT16),
        (("esp32p4",), {"num_of_bits": 16}, TargetPlatform.ESPDL_INT16),
        (("esp32p4",), {"num_of_bits": 16, "hi_precision": True}, TargetPlatform.ESPDL_H_PRE_INT16),
        (("esp32p4",), {"float": True}, TargetPlatform.FP32),
    ],
)
def test_get_target_platform_quant_type_and_legacy(args, kwargs, platform):
    assert get_target_platform(*args, **kwargs) == platform


@pytest.mark.parametrize(("target", "platform", "bias_bits"), _W8A16_CASES)
def test_w8a16_quantization_config(target, platform, bias_bits):
    assert get_target_platform(target, quant_type="w8a16") == platform

    for op_type in ("Conv", "Gemm", "MatMul"):
        graph = BaseGraph(name=f"{op_type}_graph", built_from=NetworkFramework.ONNX)
        op = _make_operation(
            graph,
            op_type,
            platform,
            constant_weight=True,
            with_bias=op_type in {"Conv", "Gemm"},
        )
        config = Quantizer(platform, graph).init_quantize_config(op)

        assert config.input_quantization_config[0].num_of_bits == 16
        assert config.input_quantization_config[1].num_of_bits == 8
        assert config.output_quantization_config[0].num_of_bits == 16
        if op_type in {"Conv", "Gemm"}:
            assert config.input_quantization_config[-1].num_of_bits == bias_bits


@pytest.mark.parametrize(("target", "platform", "_"), _W8A16_CASES)
def test_w8a16_keeps_dynamic_operands_and_other_ops_int16(target, platform, _):
    graph = BaseGraph(name="dynamic_graph", built_from=NetworkFramework.ONNX)
    for op_type in ("Gemm", "MatMul"):
        operation = _make_operation(graph, op_type, platform, constant_weight=False)
        operation_config = Quantizer(platform, graph).init_quantize_config(operation)
        assert [config.num_of_bits for config in operation_config.input_quantization_config] == [16, 16]
        assert operation_config.output_quantization_config[0].num_of_bits == 16

    add = graph.create_operation(
        op_type="Add",
        name="Add_op",
        platform=platform,
        inputs=[
            graph.create_variable(name="add_input_0", shape=[1, 17]),
            graph.create_variable(name="add_input_1", shape=[1, 17]),
        ],
        outputs=[graph.create_variable(name="add_output", shape=[1, 17])],
    )
    add_config = Quantizer(platform, graph).init_quantize_config(add)
    assert [config.num_of_bits for config in add_config.input_quantization_config] == [16, 16]
    assert add_config.output_quantization_config[0].num_of_bits == 16


@pytest.mark.parametrize(
    ("target", "platform"),
    [
        ("esp32p4", TargetPlatform.ESPDL_W8A16),
        ("esp32s3", TargetPlatform.ESPDL_S3_W8A16),
    ],
)
@pytest.mark.parametrize(
    ("out_channels", "expected_layout"),
    [
        (16, LayoutAnnotation.N8HWC8),
        (17, LayoutAnnotation.N8HWC8_UNALIGNED),
    ],
)
def test_w8a16_conv_filter_layout_is_n8hwc8_on_s3_and_p4(target, platform, out_channels, expected_layout):
    # ESP32-S3 and ESP32-P4 must export the W8A16 Conv filter in the same
    # (N/8)HWC8 layout; only the "c" targets keep a plain NHWC filter.
    graph = BaseGraph(name="export_graph", built_from=NetworkFramework.ONNX)
    op = _make_operation(
        graph,
        "Conv",
        platform,
        constant_weight=True,
        with_bias=True,
        out_channels=out_channels,
    )
    config = Quantizer(platform, graph).init_quantize_config(op)
    quant_op = QuantableOperation(convert_from=op, quantize_config=config, platform=platform)

    info = ExporterPatternInfo()
    info.reset()
    InsertQuantTypePattern().export(quant_op, graph)
    ResetParamLayoutPattern().export(quant_op, graph)

    # int8 weights are packed with the int16 block size: the ESP32-P4 16-bit MAC accumulates into
    # 8 x 64-bit QACC lanes, so a filter block covers 8 output channels regardless of weight width.
    assert quant_op.attributes["quant_type"] == EspQuantType.W8A16
    assert info.get_var_layout(quant_op.inputs[1].name) == expected_layout

    for op_type in ("Gemm", "MatMul"):
        op_graph = BaseGraph(name=f"{op_type}_export_graph", built_from=NetworkFramework.ONNX)
        mixed_op = _make_operation(op_graph, op_type, platform, constant_weight=True)
        mixed_config = Quantizer(platform, op_graph).init_quantize_config(mixed_op)
        mixed_quant = QuantableOperation(convert_from=mixed_op, quantize_config=mixed_config, platform=platform)
        InsertQuantTypePattern().export(mixed_quant, op_graph)
        assert mixed_quant.attributes["quant_type"] == EspQuantType.W8A16


def test_w8a16_depthwise_and_add_keep_s16():
    graph = BaseGraph(name="other_ops_graph", built_from=NetworkFramework.ONNX)
    platform = TargetPlatform.ESPDL_W8A16
    depthwise = graph.create_operation(
        op_type="Conv",
        name="Depthwise_op",
        platform=platform,
        attributes={"group": 3},
        inputs=[
            graph.create_variable(name="dw_input", shape=[1, 3, 8, 8]),
            graph.create_variable(name="dw_weight", value=torch.randn(3, 1, 3, 3), is_parameter=True),
        ],
        outputs=[graph.create_variable(name="dw_output", shape=[1, 3, 8, 8])],
    )
    add = graph.create_operation(
        op_type="Add",
        name="Add_op",
        platform=platform,
        inputs=[
            graph.create_variable(name="add_input_0", shape=[1, 3, 8, 8]),
            graph.create_variable(name="add_input_1", shape=[1, 3, 8, 8]),
        ],
        outputs=[graph.create_variable(name="add_output", shape=[1, 3, 8, 8])],
    )

    depthwise_quant = QuantableOperation(
        convert_from=depthwise,
        quantize_config=Quantizer(platform, graph).init_quantize_config(depthwise),
        platform=platform,
    )
    add_quant = QuantableOperation(
        convert_from=add,
        quantize_config=Quantizer(platform, graph).init_quantize_config(add),
        platform=platform,
    )
    InsertQuantTypePattern().export(depthwise_quant, graph)
    InsertQuantTypePattern().export(add_quant, graph)

    assert depthwise_quant.attributes["quant_type"] == EspQuantType.S16
    assert add_quant.attributes["quant_type"] == EspQuantType.S16


@pytest.mark.parametrize("target", ["esp32p4", "esp32s3"])
def test_w8a16_end_to_end_export(target):
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 17, 3, padding=1)

        def forward(self, inputs):
            return self.conv(inputs)

    with tempfile.TemporaryDirectory() as directory:
        export_file = os.path.join(directory, "w8a16.espdl")
        calibration_data = DataLoader([torch.randn(3, 8, 8) for _ in range(4)], batch_size=2)
        graph = espdl_quantize_torch(
            model=ConvNet().eval(),
            espdl_export_file=export_file,
            calib_dataloader=calibration_data,
            calib_steps=2,
            input_shape=[1, 3, 8, 8],
            target=target,
            quant_type="w8a16",
            collate_fn=lambda batch: batch,
            device="cpu",
            error_report=False,
            export_config=True,
            verbose=0,
        )

        conv = next(op for op in graph.operations.values() if op.type == "Conv")
        assert conv.input_quant_config[1].num_of_bits == 8
        assert conv.input_quant_config[0].num_of_bits == 16
        assert conv.output_quant_config[0].num_of_bits == 16
        assert os.path.exists(export_file)

        with open(os.path.splitext(export_file)[0] + ".info", encoding="utf-8") as info_file:
            exported_info = info_file.read()
        assert "INT8" in exported_info
        assert "INT16" in exported_info
        # 17 output channels are not a multiple of 8, so the packed filter is
        # (N/8)HWC8_UNALIGNED; ESP32-S3 and ESP32-P4 must agree on this layout.
        assert "layout ==> (N/8)HWC8_UNALIGNED" in exported_info

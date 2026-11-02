"""Relu, Clip, and PRelu export follows the device kernel."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from esp_ppq.api.espdl_interface import espdl_quantize_torch
from esp_ppq.core import NetworkFramework, QuantizationStates, QuantizationVisibility, TargetPlatform
from esp_ppq.core.defs import SingletonMeta
from esp_ppq.executor.op.torch.default import Clip_forward
from esp_ppq.IR import BaseGraph, Operation, QuantableOperation
from esp_ppq.IR.base.opdef import DEFAULT_SOCKET_TABLE
from esp_ppq.parser.espdl.espdl_typedef import EspQuantType, ExporterPatternInfo, LayoutAnnotation
from esp_ppq.parser.espdl.export_patterns import (
    AddLUTPattern,
    InsertQuantTypePattern,
    QuantVariableToIntPattern,
    _kernel_width_from_inputs,
)
from esp_ppq.parser.espdl_exporter import EspdlExporter
from esp_ppq.quantization.quantizer.EspdlQuantizer import (
    EspdlInt16Quantizer,
    EspdlQuantizer,
    EspdlS3Int16Quantizer,
)


def _quantize(op_type, platform, inputs, output_name):
    graph = BaseGraph(name=f"{op_type}_lut_rule", built_from=NetworkFramework.ONNX)
    output = graph.create_variable(name=output_name, shape=[1])
    operation = graph.create_operation(
        op_type=op_type,
        name=f"{op_type}_op",
        platform=platform,
        inputs=inputs(graph),
        outputs=[output],
    )
    is_int16 = "INT16" in platform.name
    if platform == TargetPlatform.ESPDL_S3_INT16:
        quantizer = EspdlS3Int16Quantizer(graph)
    elif is_int16:
        quantizer = EspdlInt16Quantizer(graph)
    else:
        quantizer = EspdlQuantizer(graph)
    config = quantizer.init_quantize_config(operation)
    for tensor_config in config.input_quantization_config + config.output_quantization_config:
        tensor_config.scale = torch.tensor(1.0)
        tensor_config.offset = torch.tensor(0.0)
    quantized = QuantableOperation(convert_from=operation, quantize_config=config, platform=platform)
    bits = 16 if is_int16 else 8
    quantized.attributes["quant_type"] = EspQuantType.S16 if bits == 16 else EspQuantType.S8
    return graph, quantized


def _activation(graph):
    return [graph.create_variable(name="x", shape=[1])]


def _clip_bounds(graph, min_value=0.0, max_value=1.0, omit_max=False):
    lower = graph.create_variable(name="clip_min", value=torch.tensor(min_value), is_parameter=True)
    if omit_max:
        # Trailing empty ONNX name: the slot is omitted, not a runtime tensor.
        upper = graph.create_variable(name="", value=None, is_parameter=False)
    else:
        upper = graph.create_variable(name="clip_max", value=torch.tensor(max_value), is_parameter=True)
    return [_activation(graph)[0], lower, upper]


def _clip_min_only(graph, min_value=0.0):
    # ONNX Clip(x, min, "") after trailing-empty strip: two inputs, no max.
    lower = graph.create_variable(name="clip_min", value=torch.tensor(min_value), is_parameter=True)
    return [_activation(graph)[0], lower]


def _clip_max_only(graph, max_value=1.0):
    # ONNX Clip(x, "", max): the empty min is not trailing, so it stays.
    lower = graph.create_variable(name="clip_min", value=None, is_parameter=True)
    upper = graph.create_variable(name="clip_max", value=torch.tensor(max_value), is_parameter=True)
    return [_activation(graph)[0], lower, upper]


def _export_with_lut(op, exponents, permutations=None, layouts=None):
    info = ExporterPatternInfo()
    info.reset()
    graph = BaseGraph(name="unused")
    graph.operations[op.name] = op
    for name, exponent in exponents.items():
        info.add_var_exponents(name, exponent)
    for name, perm in (permutations or {}).items():
        info.add_var_permute(name, perm)
    for name, layout in (layouts or {}).items():
        info.add_var_layout(name, layout)
    try:
        exported = AddLUTPattern(int16_step=256).export(op, graph)
        lut_name = exported.attributes.get("lut")
        lut = info.get_lut(lut_name)
        lut_copy = lut.detach().clone() if isinstance(lut, torch.Tensor) else None
        return exported, lut_copy
    finally:
        info.reset()


def _export(op, exponents, permutations=None, layouts=None):
    exported, _ = _export_with_lut(op, exponents, permutations, layouts)
    return exported


def test_s16_relu_stays_on_integer_kernel():
    _, op = _quantize("Relu", TargetPlatform.ESPDL_INT16, _activation, "y")
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_fused_relu_exports_stamped_table():
    _, op = _quantize("Relu", TargetPlatform.ESPDL_INT16, _activation, "y")
    op.type = "LUT"
    op.attributes["original_op_type"] = "Relu"
    op.attributes["int16_lut_step"] = 256
    exported = _export(op, {"x": -8, "y": -8})
    assert exported.attributes["lut"].startswith("Relu_lut_")
    assert exported.type == "LUT"


def test_s8_relu_keeps_step1_table():
    _, op = _quantize("Relu", TargetPlatform.ESPDL_INT8, _activation, "y")
    exported = _export(op, {"x": -4, "y": -4})
    assert exported.attributes["lut"].startswith("Relu_lut_")


def test_s8_clip_keeps_step1_table():
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT8, _clip_bounds, "y")
    exported = _export(op, {"x": -4, "clip_min": -4, "clip_max": -4, "y": -4})
    assert exported.attributes["lut"].startswith("Clip_lut_")

    _, runtime_op = _quantize("Clip", TargetPlatform.ESPDL_INT8, _runtime_min, "y")
    exported_runtime = _export(runtime_op, {"x": -4, "clip_min": -4, "y": -4})
    assert "lut" not in exported_runtime.attributes


def test_s16_clip_same_exponent_stays_on_integer_kernel():
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    exported = _export(op, {"x": -8, "clip_min": -8, "clip_max": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_clip_omitted_bound_stays_on_integer_kernel():
    _, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _clip_bounds(graph, omit_max=True),
        "y",
    )
    exported = _export(op, {"x": -8, "clip_min": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_clip_attribute_bounds_become_integer_inputs():
    from esp_ppq.IR.morph import GraphFormatter

    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _activation, "y")
    _install_quant_op(graph, op)
    op.attributes["min"] = -1.0
    op.attributes["max"] = 1.0
    op.attributes["int16_lut_step"] = 256
    x = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.equal(Clip_forward(op, [x]), torch.clamp(x, -1.0, 1.0))
    GraphFormatter(graph).format_clip()
    assert len(op.inputs) == 3

    def _scalar(value):
        return float(value.item() if isinstance(value, torch.Tensor) else value)

    assert _scalar(op.inputs[1].value) == -1.0
    assert _scalar(op.inputs[2].value) == 1.0
    assert "min" not in op.attributes
    assert "max" not in op.attributes
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_clip_attribute_bounds_require_format_clip_for_s16_export():
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _activation, "y")
    op.attributes["min"] = -1.0
    op.attributes["max"] = 1.0
    assert "int16_lut_step" not in op.attributes
    x = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.equal(Clip_forward(op, [x]), torch.clamp(x, -1.0, 1.0))
    with pytest.raises(ValueError, match="FORMAT_CLIP"):
        _export(op, {"x": -8, "y": -8})


def test_s16_clip_mismatched_exponent_moves_onto_input_grid():
    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-4)
    op.input_quant_config[2].scale = torch.tensor(2.0**-4)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    op.attributes["int16_lut_step"] = 256
    _passive_clip(graph, op)
    exported = _export(op, {"x": -8, "clip_min": -4, "clip_max": -4, "y": -8})
    assert "lut" not in exported.attributes
    assert float(exported.input_quant_config[1].scale) == float(exported.input_quant_config[0].scale)
    assert float(exported.input_quant_config[2].scale) == float(exported.input_quant_config[0].scale)


def test_int8_config_on_int16_platform_exports_as_s8():
    """Output bit width selects quant_type. An 8-bit Clip on an INT16 platform is S8."""
    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    for config in op.input_quant_config + op.output_quant_config:
        config.num_of_bits = 8
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-4)
    op.input_quant_config[2].scale = torch.tensor(2.0**-4)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    op.attributes["int16_lut_step"] = 256
    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == EspQuantType.S8
    _passive_clip(graph, op)
    exported = _export(op, {"x": -8, "clip_min": -4, "clip_max": -4, "y": -8})
    assert exported.attributes["quant_type"] == EspQuantType.S8
    assert exported.attributes["lut"].startswith("Clip_lut_")
    assert float(exported.input_quant_config[1].scale) == float(exported.input_quant_config[0].scale)


def _runtime_min(graph):
    x = graph.create_variable(name="x", shape=[1])
    lower = graph.create_variable(name="clip_min", value=None, is_parameter=False)
    return [x, lower]


def test_clip_does_not_read_lut_optimization_setting():
    from esp_ppq.api.setting import QuantizationSetting
    from esp_ppq.IR.morph import GraphFormatter
    from esp_ppq.quantization.optim.espdl_lut import EspdlLUTFusionPass

    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _activation, "y")
    _install_quant_op(graph, op)
    op.attributes["min"] = -1.0
    op.attributes["max"] = 1.0
    GraphFormatter(graph).format_clip()
    sigmoid_out = graph.create_variable(name="sig_y", shape=[1])
    graph.create_operation(
        op_type="Sigmoid",
        name="Sigmoid_op",
        platform=TargetPlatform.ESPDL_INT16,
        inputs=[graph.create_variable(name="sig_x", shape=[1])],
        outputs=[sigmoid_out],
    )
    setting = QuantizationSetting()
    setting.lut_optimization = False
    setting.lut_optimization_setting.step = 64
    pipeline = list(EspdlInt16Quantizer(graph).build_quant_pipeline(setting))
    assert not any(isinstance(item, EspdlLUTFusionPass) for item in pipeline)
    from esp_ppq.quantization.optim.parameters import PassiveParameterQuantizePass

    assert any(isinstance(item, PassiveParameterQuantizePass) for item in pipeline)
    assert graph.operations["Sigmoid_op"].type == "Sigmoid"
    assert "int16_lut_step" not in op.attributes
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_runtime_clip_bound_same_exponent_stays_on_integer_kernel():
    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _runtime_min, "y")
    _install_quant_op(graph, op)
    scale = torch.tensor(2.0**-8)
    for config in op.input_quant_config + op.output_quant_config:
        config.scale = scale
    assert "int16_lut_step" not in op.attributes
    exported = _export(op, {"x": -8, "clip_min": -8, "y": -8})
    assert "lut" not in exported.attributes
    x = torch.tensor([-1.0, -0.2, 0.2, 1.0])
    assert torch.equal(Clip_forward(op, [x, torch.tensor(0.0)]), torch.clamp(x, min=0.0))
    assert torch.equal(Clip_forward(op, [x, torch.tensor(0.5)]), torch.clamp(x, min=0.5))


def test_s16_runtime_clip_bound_without_quantized_producer_is_rejected():
    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _runtime_min, "y")
    _install_quant_op(graph, op)
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-4)
    assert "int16_lut_step" not in op.attributes
    x = torch.tensor([-1.0, -0.2, 0.2, 1.0])
    assert torch.equal(Clip_forward(op, [x, torch.tensor(0.0)]), torch.clamp(x, min=0.0))
    assert torch.equal(Clip_forward(op, [x, torch.tensor(0.5)]), torch.clamp(x, min=0.5))
    with pytest.raises(PermissionError, match="has no quantized producer output"):
        _passive_clip(graph, op)


def _prelu_inputs(graph, slope, shape=(1, 1, 1, 4)):
    # create_variable does not forward shape; the integer kernel reads it from the attribute.
    x = graph.create_variable(name="x")
    x.shape = list(shape)
    weight = graph.create_variable(name="slope", value=torch.tensor(slope), is_parameter=True)
    return [x, weight]


def _prelu_tensor_inputs(graph, slope, shape=(1, 1, 1, 4)):
    x = graph.create_variable(name="x")
    x.shape = list(shape)
    weight = graph.create_variable(name="slope", value=slope.clone(), is_parameter=True)
    return [x, weight]


def _prelu_per_channel_export_meta(activation_shape):
    """Export metadata required for per-channel PRelu without NCHW->NHWC perm."""
    shape = tuple(activation_shape)
    if len(shape) == 4 and shape[1] == 1 and shape[2] == 1:
        return {"layouts": {"x": LayoutAnnotation.NHWC}}
    return {"permutations": {"x": [0, 2, 3, 1]}}


def _prelu_runtime_inputs(graph):
    x = graph.create_variable(name="x", shape=[1, 4])
    slope = graph.create_variable(name="slope", shape=[4])
    return [x, slope]


def _prelu_one_input(graph):
    return [graph.create_variable(name="x", shape=[1, 4])]


def test_s16_scalar_prelu_uses_the_integer_kernel():
    """INT16 broadcasts a length-1 slope; a step-256 table would not match F.prelu."""
    _, op = _quantize("PRelu", TargetPlatform.ESPDL_INT16, lambda graph: _prelu_inputs(graph, [0.25]), "y")
    exported = _export(op, {"x": -8, "slope": -8, "y": -8})
    assert "lut" not in exported.attributes
    assert tuple(exported.inputs[1].value.shape) == (1, 1, 1)
    assert float(exported.inputs[1].value.reshape(-1)[0]) == pytest.approx(0.25)


def test_s16_scalar_prelu_falls_back_to_table_for_non_rank4_activation():
    """Linear [N, C] and LayerNorm [B, T, C] activations cannot use the rank-4 integer kernel."""
    for shape in ((2, 4), (1, 3, 4), ()):
        _, op = _quantize(
            "PRelu",
            TargetPlatform.ESPDL_INT16,
            lambda graph, s=shape: _prelu_inputs(graph, [0.25], shape=s),
            "y",
        )
        exported = _export(op, {"x": -8, "slope": -8, "y": -8})
        assert exported.attributes["lut"].startswith("PRelu_lut_"), shape


def test_s3_s16_scalar_prelu_uses_table_only_when_exponent_changes():
    """EE.VPRELU.S16 on S3 copies the positive branch without output_shift."""
    _, same_exp = _quantize("PRelu", TargetPlatform.ESPDL_S3_INT16, lambda graph: _prelu_inputs(graph, [0.25]), "y")
    exported_same = _export(same_exp, {"x": -8, "slope": -8, "y": -8})
    assert "lut" not in exported_same.attributes
    assert tuple(exported_same.inputs[1].value.shape) == (1, 1, 1)

    _, shifted_s3 = _quantize("PRelu", TargetPlatform.ESPDL_S3_INT16, lambda graph: _prelu_inputs(graph, [0.25]), "y")
    exported_shifted_s3 = _export(shifted_s3, {"x": -8, "slope": -8, "y": -7})
    assert exported_shifted_s3.attributes["lut"].startswith("PRelu_lut_")

    _, shifted_p4 = _quantize("PRelu", TargetPlatform.ESPDL_INT16, lambda graph: _prelu_inputs(graph, [0.25]), "y")
    exported_shifted_p4 = _export(shifted_p4, {"x": -8, "slope": -8, "y": -7})
    assert "lut" not in exported_shifted_p4.attributes
    assert tuple(exported_shifted_p4.inputs[1].value.shape) == (1, 1, 1)


@pytest.mark.parametrize("platform", (TargetPlatform.ESPDL_S3_INT16, TargetPlatform.ESPDL_S3_H_PRE_INT16))
def test_s3_s16_vector_prelu_rejects_exponent_change(platform):
    """Per-channel INT16 PRelu on S3 cannot requantize the positive branch."""
    _, op = _quantize(
        "PRelu",
        platform,
        lambda graph: _prelu_inputs(graph, [0.25, 0.5, 0.25, 0.5]),
        "y",
    )
    with pytest.raises(ValueError, match=r"per-channel slope on ESP32-S3"):
        meta = _prelu_per_channel_export_meta(op.inputs[0].shape)
        _export(op, {"x": -8, "slope": -8, "y": -7}, **meta)


@pytest.mark.parametrize("platform", (TargetPlatform.ESPDL_S3_INT16, TargetPlatform.ESPDL_S3_H_PRE_INT16))
def test_s3_s16_vector_prelu_keeps_integer_kernel_when_exponent_matches(platform):
    _, op = _quantize(
        "PRelu",
        platform,
        lambda graph: _prelu_inputs(graph, [0.25, 0.5, 0.25, 0.5]),
        "y",
    )
    meta = _prelu_per_channel_export_meta(op.inputs[0].shape)
    exported = _export(op, {"x": -8, "slope": -8, "y": -8}, **meta)
    assert "lut" not in exported.attributes
    assert tuple(exported.inputs[1].value.shape) == (4, 1, 1)


def test_s8_scalar_prelu_keeps_table():
    """The int8 kernel does not broadcast a length-1 slope, so the step-1 table stays."""
    _, op = _quantize("PRelu", TargetPlatform.ESPDL_INT8, lambda graph: _prelu_inputs(graph, [0.25]), "y")
    exported = _export(op, {"x": -4, "slope": -4, "y": -4})
    assert exported.attributes["lut"].startswith("PRelu_lut_")


def test_s16_vector_prelu_does_not_export_table():
    _, op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_inputs(graph, [0.25, 0.5, 0.25, 0.5]),
        "y",
    )
    meta = _prelu_per_channel_export_meta(op.inputs[0].shape)
    exported = _export(op, {"x": -8, "slope": -8, "y": -8}, **meta)
    assert "lut" not in exported.attributes
    assert tuple(exported.inputs[1].value.shape) == (4, 1, 1)


def test_prelu_slope_is_normalized_to_rank3_for_the_integer_kernel():
    """A per-channel slope of any rank is stored as [c, 1, 1]."""
    for platform, slope in (
        (TargetPlatform.ESPDL_INT8, torch.full((4,), 0.25)),
        (TargetPlatform.ESPDL_INT8, torch.full((1, 4, 1, 1), 0.25)),
        (TargetPlatform.ESPDL_INT16, torch.full((4,), 0.25)),
        (TargetPlatform.ESPDL_INT16, torch.full((1, 4, 1, 1), 0.25)),
    ):
        _, op = _quantize("PRelu", platform, lambda graph, s=slope: _prelu_tensor_inputs(graph, s), "y")
        meta = _prelu_per_channel_export_meta(op.inputs[0].shape)
        exported = _export(op, {"x": -8, "slope": -8, "y": -8}, **meta)
        assert "lut" not in exported.attributes, platform
        assert tuple(exported.inputs[1].value.shape) == (4, 1, 1), platform


def test_prelu_slope_must_match_the_channel_count_the_device_indexes():
    """NCHW [N, C, H, W] becomes NHWC on device; slope length must equal C, not H*W*C."""
    per_channel = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT8,
        lambda graph: _prelu_tensor_inputs(graph, per_channel, shape=(1, 4, 2, 2)),
        "y",
    )
    exported = _export(op, {"x": -4, "slope": -4, "y": -4}, permutations={"x": [0, 2, 3, 1]})
    assert tuple(exported.inputs[1].value.shape) == (4, 1, 1)

    elementwise = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _, op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT8,
        lambda graph: _prelu_tensor_inputs(graph, elementwise, shape=(1, 4, 2, 2)),
        "y",
    )
    with pytest.raises(ValueError, match=r"16 slope elements, but the activation has 4 channels"):
        _export(op, {"x": -4, "slope": -4, "y": -4}, permutations={"x": [0, 2, 3, 1]})

    short = torch.tensor([0.1, 0.2, 0.3])
    _, op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, short, shape=(1, 4, 2, 2)),
        "y",
    )
    with pytest.raises(ValueError, match=r"3 slope elements, but the activation has 4 channels"):
        _export(op, {"x": -8, "slope": -8, "y": -8}, permutations={"x": [0, 2, 3, 1]})


def test_prelu_per_channel_uses_export_layout_before_shape_heuristic():
    """NCHW layout metadata rejects a W-length slope even when axis 3 equals W."""
    slopes = torch.arange(8, dtype=torch.float32)
    _, op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, slopes, shape=(1, 4, 8, 8)),
        "y",
    )
    with pytest.raises(ValueError, match=r"permute unset"):
        _export(op, {"x": -8, "slope": -8, "y": -8}, layouts={"x": LayoutAnnotation.NCHW})

    _, packed_mistag = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, slopes, shape=(1, 4, 8, 8)),
        "y_packed",
    )
    with pytest.raises(ValueError, match=r"permute unset"):
        _export(
            packed_mistag,
            {"x": -8, "slope": -8, "y_packed": -8},
            layouts={"x": LayoutAnnotation.N8HWC8_UNALIGNED},
        )

    per_channel = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, ok_op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, per_channel, shape=(1, 1, 1, 4)),
        "y2",
    )
    exported = _export(
        ok_op,
        {"x": -8, "slope": -8, "y2": -8},
        layouts={"x": LayoutAnnotation.NHWC},
    )
    assert tuple(exported.inputs[1].value.shape) == (4, 1, 1)

    _, no_meta = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, per_channel, shape=(1, 4, 8, 8)),
        "y3",
    )
    with pytest.raises(ValueError, match=r"permute unset"):
        _export(no_meta, {"x": -8, "slope": -8, "y3": -8})


def test_identity_permute_refuses_per_channel_slope_on_nchw():
    """Axis 3 of an untouched NCHW tensor is W, even when C == W."""
    per_channel = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, per_channel, shape=(1, 4, 8, 8)),
        "y",
    )
    with pytest.raises(ValueError, match=r"indexes the slope along axis 3"):
        _export(op, {"x": -8, "slope": -8, "y": -8}, permutations={"x": [0, 1, 2, 3]})

    square = torch.arange(8, dtype=torch.float32)
    _, square_op = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_tensor_inputs(graph, square, shape=(1, 8, 8, 8)),
        "y2",
    )
    with pytest.raises(ValueError, match=r"indexes the slope along axis 3"):
        _export(square_op, {"x": -8, "slope": -8, "y2": -8}, permutations={"x": [0, 1, 2, 3]})


def test_s16_scalar_prelu_keeps_integer_kernel_on_rank4_permute():
    """Scalar INT16 PRelu uses the integer kernel whenever the device activation is rank 4."""
    _, nchw = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_inputs(graph, [0.25], shape=(1, 4, 8, 8)),
        "y",
    )
    exported_nchw = _export(nchw, {"x": -8, "slope": -8, "y": -8}, permutations={"x": [0, 1, 2, 3]})
    assert "lut" not in exported_nchw.attributes
    assert tuple(exported_nchw.inputs[1].value.shape) == (1, 1, 1)

    _, nhwc = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_inputs(graph, [0.25], shape=(1, 4, 8, 8)),
        "y2",
    )
    exported_nhwc = _export(nhwc, {"x": -8, "slope": -8, "y2": -8}, permutations={"x": [0, 2, 3, 1]})
    assert "lut" not in exported_nhwc.attributes
    assert tuple(exported_nhwc.inputs[1].value.shape) == (1, 1, 1)


def test_prelu_runtime_slope_is_rejected_with_a_clear_error():
    """esp-dl loads the PRelu slope as a model parameter; a runtime tensor can not deploy."""
    for platform in (TargetPlatform.ESPDL_INT8, TargetPlatform.ESPDL_INT16):
        _, op = _quantize("PRelu", platform, _prelu_runtime_inputs, "y")
        with pytest.raises(ValueError, match="no constant slope"):
            _export(op, {"x": -8, "slope": -8, "y": -8})


def test_prelu_without_slope_input_is_rejected_not_crash():
    _, op = _quantize("PRelu", TargetPlatform.ESPDL_INT8, _prelu_one_input, "y")
    with pytest.raises(ValueError, match="no constant slope"):
        _export(op, {"x": -8, "y": -8})


def test_s16_unfused_sigmoid_does_not_export_table():
    _, op = _quantize("Sigmoid", TargetPlatform.ESPDL_INT16, _activation, "y")
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_fused_sigmoid_exports_table():
    _, op = _quantize("Sigmoid", TargetPlatform.ESPDL_INT16, _activation, "y")
    op.type = "LUT"
    op.attributes["original_op_type"] = "Sigmoid"
    op.attributes["int16_lut_step"] = 256
    exported = _export(op, {"x": -8, "y": -8})
    assert exported.attributes["lut"].startswith("Sigmoid_lut_")


def test_s16_unfused_gelu_still_exports_table():
    """esp-dl registers Gelu as LUT; an unfused 16-bit Gelu still needs a table."""
    _, op = _quantize("Gelu", TargetPlatform.ESPDL_INT16, _activation, "y")
    exported = _export(op, {"x": -8, "y": -8})
    assert exported.attributes["lut"].startswith("Gelu_lut_")


def test_s16_gelu_refuses_missing_table():
    _, op = _quantize("Gelu", TargetPlatform.ESPDL_INT16, _activation, "y")
    op.input_quant_config[0].num_of_bits = 8
    with pytest.raises(ValueError, match="input and output must be 16-bit"):
        _export(op, {"x": -8, "y": -8})

    for bad_step in (-1, 0, 48, 96, 100, 131072):
        _, op = _quantize("Gelu", TargetPlatform.ESPDL_INT16, _activation, "y")
        op.attributes["int16_lut_step"] = bad_step
        with pytest.raises(ValueError, match="int16_lut_step must be a positive power of two"):
            _export(op, {"x": -8, "y": -8})

    _, prelu_fallback = _quantize(
        "PRelu",
        TargetPlatform.ESPDL_INT16,
        lambda graph: _prelu_inputs(graph, [0.25], shape=(2, 4)),
        "y",
    )
    prelu_fallback.attributes["int16_lut_step"] = 48
    with pytest.raises(ValueError, match="int16_lut_step must be a positive power of two"):
        _export(prelu_fallback, {"x": -8, "slope": -8, "y": -8})


def test_s16_same_exponent_clip_forward_stays_on_clamp():
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    scale = torch.tensor(2.0**-8)
    for config in op.input_quant_config + op.output_quant_config:
        config.scale = scale
    x = torch.linspace(-1.0, 1.5, 17)
    got = Clip_forward(op, [x, torch.tensor(0.0), torch.tensor(1.0)])
    assert torch.equal(got, torch.clamp(x, 0.0, 1.0))


def test_s16_unregistered_bound_exponent_moves_onto_input_grid():
    from esp_ppq.core import QuantizationVisibility

    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    for config in op.input_quant_config[1:]:
        config.scale = torch.tensor(2.0**-8)
        config.visibility = QuantizationVisibility.INTERNAL
    x = torch.arange(-512, 512, dtype=torch.float32) * (2.0**-8)
    assert "int16_lut_step" not in op.attributes
    ideal = torch.clamp(x, 0.0, 1.0)
    assert torch.equal(Clip_forward(op, [x, torch.tensor(0.0), torch.tensor(1.0)]), ideal)
    _passive_clip(graph, op)
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes
    assert float(exported.input_quant_config[1].scale) == float(exported.input_quant_config[0].scale)
    assert exported.input_quant_config[1].visibility != QuantizationVisibility.INTERNAL


def test_s16_attribute_step_does_not_drop_bounds():
    from esp_ppq.IR.morph import GraphFormatter

    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _activation, "y")
    op.attributes["min"] = -1.0
    op.attributes["max"] = 1.0
    op.attributes["int16_lut_step"] = -1
    x = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.equal(Clip_forward(op, [x]), torch.clamp(x, -1.0, 1.0))
    GraphFormatter(graph).format_clip()
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_two_input_clip_clamps_min_only():
    """clamp_min is Clip(x, min) after the parser strips the empty max name."""
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_min_only, "y")
    x = torch.arange(-512, 512, dtype=torch.float32) * (2.0**-8)
    lower = torch.tensor(0.0)
    got = Clip_forward(op, [x, lower])
    assert torch.equal(got, torch.clamp(x, min=0.0))
    exported = _export(op, {"x": -8, "clip_min": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_s16_max_only_clip_treats_missing_min_as_unbounded():
    """clamp_max keeps an empty min input between x and max."""
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_max_only, "y")
    x = torch.arange(-512, 512, dtype=torch.float32) * (2.0**-8)
    upper = torch.tensor(1.0)
    got = Clip_forward(op, [x, None, upper])
    assert torch.equal(got, torch.clamp(x, max=1.0))


def test_s16_attribute_clip_reads_both_bounds():
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _activation, "y")
    op.attributes["min"] = 0.0
    op.attributes["max"] = 1.0
    x = torch.arange(-512, 512, dtype=torch.float32) * (2.0**-8)
    got = Clip_forward(op, [x])
    explicit = Clip_forward(op, [x, torch.tensor(0.0), torch.tensor(1.0)])
    assert torch.equal(got, explicit)


def _install_quant_op(graph, op):
    """_quantize returns a wrapper that is not yet the graph's operation."""
    plain = graph.operations[op.name]
    graph.operations[op.name] = op
    for var in op.inputs + op.outputs:
        if var.source_op is plain:
            var.source_op = op
        for index, dest in enumerate(var.dest_ops):
            if dest is plain:
                var.dest_ops[index] = op
    return op


def _passive_clip(graph, op, visibility=QuantizationVisibility.EXPORT_WHEN_ACTIVE):
    """Master Clip bound scales the way the quant pipeline does."""
    from esp_ppq.quantization.optim.parameters import PassiveParameterQuantizePass

    for config in op.input_quant_config + op.output_quant_config:
        if config.state == QuantizationStates.INITIAL:
            config.state = QuantizationStates.ACTIVATED
    _install_quant_op(graph, op)
    PassiveParameterQuantizePass(clip_visiblity=visibility).optimize(graph)


def _quantize_parameters(op):
    info = ExporterPatternInfo()
    info.reset()
    graph = BaseGraph(name="to_int")
    graph.operations[op.name] = op
    QuantVariableToIntPattern().export(op, graph)
    return info


def _narrow_bound_to_int8(op):
    """Start the bound configs at 8-bit before PassiveParameterQuantizePass aligns them."""
    for config in op.input_quant_config[1:]:
        config.num_of_bits = 8
        config.quant_min = -128
        config.quant_max = 127


@pytest.mark.parametrize("platform", [TargetPlatform.ESPDL_INT16, TargetPlatform.ESPDL_H_PRE_INT16])
@pytest.mark.parametrize(
    ("visibility", "exports_exponent"),
    [
        (QuantizationVisibility.INTERNAL, False),
        (QuantizationVisibility.EXPORT_WHEN_ACTIVE, True),
    ],
)
def test_s16_mastered_clip_bound_is_stored_as_int16_codes(platform, visibility, exports_exponent):
    """1.0 at input scale 2^-8 is code 256, including when the bound config is int8."""
    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=-1.0, max_value=1.0),
        "y",
    )
    op.platform = platform
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-3)
    op.input_quant_config[2].scale = torch.tensor(2.0**-3)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    _narrow_bound_to_int8(op)
    _passive_clip(graph, op, visibility)
    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == EspQuantType.S16

    info = _quantize_parameters(op)
    assert op.inputs[1].value.dtype == torch.int16
    assert op.inputs[2].value.dtype == torch.int16
    assert int(op.inputs[1].value.item()) == -256
    assert int(op.inputs[2].value.item()) == 256
    assert op.input_quant_config[1].visibility == visibility
    assert op.input_quant_config[1].num_of_bits == 16
    assert op.input_quant_config[2].num_of_bits == 16
    assert (op.input_quant_config[1].quant_min, op.input_quant_config[1].quant_max) == (-32768, 32767)
    assert (op.input_quant_config[2].quant_min, op.input_quant_config[2].quant_max) == (-32768, 32767)
    if exports_exponent:
        assert info.get_var_exponents("clip_min") == [-8]
        assert info.get_var_exponents("clip_max") == [-8]
    else:
        assert info.get_var_exponents("clip_min") is None
        assert info.get_var_exponents("clip_max") is None

    _quantize_parameters(op)
    assert int(op.inputs[1].value.item()) == -256
    assert int(op.inputs[2].value.item()) == 256
    exported = _export(op, {"x": -8, "y": -8})
    assert "lut" not in exported.attributes


def test_custom_tqc_16bit_clip_on_int8_platform_aligns_simulation_and_export():
    """Promoting input_0 and output_0 to 16-bit widens the bounds so x=0.75 is not clamped at 127*2^-8."""
    from esp_ppq.executor import TorchExecutor

    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT8,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=-1.0, max_value=1.0),
        "y",
    )
    _install_quant_op(graph, op)
    graph.mark_variable_as_graph_input(op.inputs[0])
    graph.mark_variable_as_graph_output(op.outputs[0])

    quantizer = EspdlQuantizer(graph)
    quantizer.custom_tqc = {
        op.name: {
            "input_0": {"bit_width": 16},
            "output_0": {"bit_width": 16},
        }
    }
    quantizer._apply_custom_tqc(op, op.config)
    assert op.input_quant_config[1].num_of_bits == 8
    assert op.input_quant_config[2].num_of_bits == 8

    for config in (op.input_quant_config[0], op.output_quant_config[0]):
        config.state = QuantizationStates.ACTIVATED
        config.scale = torch.tensor(2.0**-8)
        config.offset = torch.tensor(0.0)
    _passive_clip(graph, op, QuantizationVisibility.INTERNAL)

    assert op.input_quant_config[1].num_of_bits == 16
    assert op.input_quant_config[2].num_of_bits == 16
    assert (op.input_quant_config[1].quant_min, op.input_quant_config[1].quant_max) == (-32768, 32767)
    assert (op.input_quant_config[2].quant_min, op.input_quant_config[2].quant_max) == (-32768, 32767)

    executor = TorchExecutor(graph, device="cpu")
    (sim_out,) = executor.forward({"x": torch.tensor([0.75])}, ["y"])
    assert torch.isclose(sim_out, torch.tensor([0.75]))

    InsertQuantTypePattern().export(op, graph)
    _quantize_parameters(op)
    assert int(op.inputs[1].value.item()) == -256
    assert int(op.inputs[2].value.item()) == 256


def test_mixed_or_dominated_activation_bitwidth_guard():
    """InsertQuantTypePattern and PassiveParameterQuantizePass read dominated_by.num_of_bits and reject mixed widths."""
    # 1. Only output_0 promoted to 16-bit while input_0 stays 8-bit.
    graph, out_only = _quantize("Clip", TargetPlatform.ESPDL_INT8, _clip_bounds, "y")
    out_only.output_quant_config[0].num_of_bits = 16
    with pytest.raises(ValueError, match="requires input and output activations to share one bit width"):
        _passive_clip(graph, out_only)
    with pytest.raises(ValueError, match="requires input and output activations to share one bit width"):
        InsertQuantTypePattern().export(out_only, graph)

    # 2. Input dominated by an 8-bit producer while Clip itself is 16-bit.
    graph, dom_in = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    master_8 = dom_in.input_quant_config[0].copy()
    master_8.num_of_bits = 8
    master_8.quant_min, master_8.quant_max = -128, 127
    master_8.state = QuantizationStates.ACTIVATED
    master_8.scale = torch.tensor(2.0**-4)
    master_8.offset = torch.tensor(0.0)
    dom_in.input_quant_config[0].dominated_by = master_8
    with pytest.raises(ValueError, match="requires input and output activations to share one bit width"):
        _passive_clip(graph, dom_in)
    with pytest.raises(ValueError, match="requires input and output activations to share one bit width"):
        InsertQuantTypePattern().export(dom_in, graph)

    # 3. Both input_0 and output_0 dominated by an 8-bit master on an INT16 op: effective width is 8 -> S8.
    graph, both_dom = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    both_dom.input_quant_config[0].dominated_by = master_8
    both_dom.output_quant_config[0].dominated_by = master_8
    _passive_clip(graph, both_dom)
    assert both_dom.input_quant_config[1].num_of_bits == 8
    assert (both_dom.input_quant_config[1].quant_min, both_dom.input_quant_config[1].quant_max) == (-128, 127)
    InsertQuantTypePattern().export(both_dom, graph)
    assert both_dom.attributes["quant_type"] == EspQuantType.S8

    # 4. ESP-DL quantized platform must not carry an FP32 activation output.
    graph, fp32_out = _quantize("Clip", TargetPlatform.ESPDL_INT8, _clip_bounds, "y")
    fp32_out.output_quant_config[0].state = QuantizationStates.FP32
    with pytest.raises(ValueError, match="output is FP32"):
        InsertQuantTypePattern().export(fp32_out, graph)


@pytest.mark.parametrize(
    ("op_type", "platform", "expected_quant_type"),
    [
        ("Softmax", TargetPlatform.ESPDL_INT8, EspQuantType.S8),
        ("LogSoftmax", TargetPlatform.ESPDL_INT16, EspQuantType.S16),
    ],
)
def test_softmax_family_with_fp32_output_uses_quantized_input_width(op_type, platform, expected_quant_type):
    graph, op = _quantize(op_type, platform, _activation, "y")
    op.output_quant_config[0].state = QuantizationStates.FP32

    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == expected_quant_type


def _two_activations(graph):
    return [
        graph.create_variable(name="x", shape=[1]),
        graph.create_variable(name="other", shape=[1]),
    ]


@pytest.mark.parametrize(
    "op_type",
    ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "And", "Or", "Xor", "NonZero"],
)
@pytest.mark.parametrize(
    ("platform", "expected_quant_type"),
    [
        (TargetPlatform.ESPDL_INT8, EspQuantType.S8),
        (TargetPlatform.ESPDL_INT16, EspQuantType.S16),
    ],
)
def test_socket_fp32_output_uses_quantized_input_width(op_type, platform, expected_quant_type):
    inputs = _activation if op_type == "NonZero" else _two_activations
    graph, op = _quantize(op_type, platform, inputs, "y")
    assert op.output_quant_config[0].state == QuantizationStates.FP32

    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == expected_quant_type


def test_greater_fp32_output_rejects_mixed_input_widths():
    graph, op = _quantize("Greater", TargetPlatform.ESPDL_INT8, _two_activations, "y")
    op.input_quant_config[1].num_of_bits = 16

    with pytest.raises(ValueError, match="share one bit width"):
        InsertQuantTypePattern().export(op, graph)


def test_every_socket_with_fp32_primary_output_selects_kernel_from_inputs():
    recognized = []
    for op_type in sorted(DEFAULT_SOCKET_TABLE):
        op = Operation(name=op_type, op_type=op_type, attributes={})
        out_plat = op.socket.out_plat or []
        if out_plat and out_plat[0] == TargetPlatform.FP32:
            assert _kernel_width_from_inputs(op, 0), op_type
            recognized.append(op_type)
    assert recognized == [
        "And",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "NonZero",
        "Or",
        "Xor",
    ]
    topk = Operation(name="TopK", op_type="TopK", attributes={})
    assert topk.socket.out_plat[1] == TargetPlatform.FP32
    assert topk.socket.out_plat[0] != TargetPlatform.FP32
    assert _kernel_width_from_inputs(topk, 1)
    assert not _kernel_width_from_inputs(topk, 0)
    for op_type in ("Softmax", "LogSoftmax"):
        softmax = Operation(name=op_type, op_type=op_type, attributes={})
        assert _kernel_width_from_inputs(softmax, 0)


def test_softmax_family_with_fp32_output_rejects_non_8_16_quantized_input():
    graph, op = _quantize("Softmax", TargetPlatform.ESPDL_INT8, _activation, "y")
    op.output_quant_config[0].state = QuantizationStates.FP32
    op.input_quant_config[0].num_of_bits = 4

    with pytest.raises(ValueError, match="requires 8- or 16-bit activation quantization"):
        InsertQuantTypePattern().export(op, graph)


def test_s16_unmastered_internal_clip_bound_is_baked_on_input_grid():
    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=-1.0, max_value=1.0),
        "y",
    )
    _install_quant_op(graph, op)
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    for config in op.input_quant_config[1:]:
        config.scale = torch.tensor(2.0**-3)
        config.state = QuantizationStates.ACTIVATED
        config.visibility = QuantizationVisibility.INTERNAL
    InsertQuantTypePattern().export(op, graph)

    _quantize_parameters(op)
    assert op.inputs[1].value.dtype == torch.int16
    assert op.inputs[2].value.dtype == torch.int16
    assert int(op.inputs[1].value.item()) == -256
    assert int(op.inputs[2].value.item()) == 256


def test_unquantized_integer_clip_bound_is_rejected_at_export():
    """quantize_passive_parameter=False leaves bounds at FP32. Do not bake codes the simulator never used."""
    _, s16 = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=0.1, max_value=1.1),
        "y",
    )
    assert s16.input_quant_config[1].state == QuantizationStates.FP32
    with pytest.raises(ValueError, match="quantize_passive_parameter"):
        _quantize_parameters(s16)
    assert s16.inputs[1].value.dtype.is_floating_point
    assert float(s16.inputs[1].value) == pytest.approx(0.1)

    def mixed(graph):
        x = graph.create_variable(name="x", shape=[1])
        clip_min = graph.create_variable(name="clip_min", value=None, is_parameter=False)
        clip_max = graph.create_variable(name="clip_max", value=torch.tensor(1.1), is_parameter=True)
        return [x, clip_min, clip_max]

    _, s8_runtime = _quantize("Clip", TargetPlatform.ESPDL_INT8, mixed, "y")
    with pytest.raises(ValueError, match="quantize_passive_parameter"):
        _quantize_parameters(s8_runtime)
    assert s8_runtime.inputs[2].value.dtype.is_floating_point

    _, s8_constants = _quantize("Clip", TargetPlatform.ESPDL_INT8, _clip_bounds, "y")
    _quantize_parameters(s8_constants)
    assert s8_constants.inputs[1].value.dtype.is_floating_point
    assert float(s8_constants.inputs[1].value) == 0.0


def test_s16_mastered_clip_integer_bound_dtype_is_requantized():
    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=-1.0, max_value=1.0),
        "y",
    )
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-3)
    op.input_quant_config[2].scale = torch.tensor(2.0**-3)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    _passive_clip(graph, op)
    InsertQuantTypePattern().export(op, graph)

    op.inputs[1].value = torch.tensor(-1, dtype=torch.int32)
    op.inputs[2].value = torch.tensor(1, dtype=torch.int64)

    _quantize_parameters(op)
    assert op.inputs[1].value.dtype == torch.int16
    assert op.inputs[2].value.dtype == torch.int16
    assert int(op.inputs[1].value.item()) == -256
    assert int(op.inputs[2].value.item()) == 256

    # QuantVariableToIntPattern can run multiple times during export preparation.
    _quantize_parameters(op)
    assert int(op.inputs[1].value.item()) == -256
    assert int(op.inputs[2].value.item()) == 256


def test_s16_infinite_clip_bound_bakes_to_the_extreme_code():
    """±inf is an unbounded side. The integer kernel stores quant_min or quant_max."""
    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=float("-inf"), max_value=float("inf")),
        "y",
    )
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-8)
    op.input_quant_config[2].scale = torch.tensor(2.0**-8)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    _passive_clip(graph, op)
    InsertQuantTypePattern().export(op, graph)

    _quantize_parameters(op)
    grid = op.input_quant_config[0].dominated_by
    assert op.inputs[1].value.dtype == torch.int16
    assert op.inputs[2].value.dtype == torch.int16
    assert int(op.inputs[1].value.item()) == int(grid.quant_min)
    assert int(op.inputs[2].value.item()) == int(grid.quant_max)


def test_s16_nan_clip_bound_is_rejected():
    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT16,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=float("nan"), max_value=1.0),
        "y",
    )
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-8)
    op.input_quant_config[2].scale = torch.tensor(2.0**-8)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    _passive_clip(graph, op)
    InsertQuantTypePattern().export(op, graph)

    with pytest.raises(ValueError, match="contains NaN"):
        _quantize_parameters(op)


def test_s8_internal_clip_bound_stays_float_for_lut():
    graph, op = _quantize(
        "Clip",
        TargetPlatform.ESPDL_INT8,
        lambda bound_graph: _clip_bounds(bound_graph, min_value=0.0, max_value=1.0),
        "y",
    )
    op.input_quant_config[0].scale = torch.tensor(2.0**-4)
    op.input_quant_config[1].scale = torch.tensor(2.0**-2)
    op.input_quant_config[2].scale = torch.tensor(2.0**-2)
    op.output_quant_config[0].scale = torch.tensor(2.0**-4)
    _passive_clip(graph, op, QuantizationVisibility.INTERNAL)
    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == EspQuantType.S8

    _quantize_parameters(op)
    assert op.inputs[1].value.dtype.is_floating_point
    assert op.inputs[2].value.dtype.is_floating_point
    assert float(op.inputs[1].value) == 0.0
    assert float(op.inputs[2].value) == 1.0
    exported, lut = _export_with_lut(op, {"x": -4, "y": -4})
    assert exported.attributes["lut"].startswith("Clip_lut_")
    assert lut is not None
    assert int(lut.min().item()) == 0
    assert int(lut.max().item()) == 16


@pytest.mark.parametrize(
    ("visibility", "exports_exponent"),
    [
        (QuantizationVisibility.INTERNAL, False),
        (QuantizationVisibility.EXPORT_WHEN_ACTIVE, True),
    ],
)
def test_s8_mixed_runtime_and_constant_clip_bound_bakes_int8_codes(visibility, exports_exponent):
    """An S8 Clip with a runtime bound runs on the int8 kernel, so a constant bound must be int8."""

    def inputs(graph):
        x = graph.create_variable(name="x", shape=[1])
        bound_src = graph.create_variable(name="bound_src", shape=[1])
        clip_min = graph.create_variable(name="clip_min", value=None, is_parameter=False)
        clip_max = graph.create_variable(name="clip_max", value=torch.tensor(1.5), is_parameter=True)
        graph.create_operation(
            op_type="Relu",
            name="bound_producer",
            platform=TargetPlatform.ESPDL_INT8,
            inputs=[bound_src],
            outputs=[clip_min],
        )
        return [x, clip_min, clip_max]

    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT8, inputs, "y")
    producer_raw = graph.operations["bound_producer"]
    producer_cfg = EspdlQuantizer(graph).init_quantize_config(producer_raw)
    producer = QuantableOperation(
        convert_from=producer_raw,
        quantize_config=producer_cfg,
        platform=TargetPlatform.ESPDL_INT8,
    )
    _install_quant_op(graph, producer)
    for config in producer.input_quant_config + producer.output_quant_config:
        config.state = QuantizationStates.ACTIVATED
        config.scale = torch.tensor(2.0**-4)
        config.offset = torch.tensor(0.0)

    op.input_quant_config[0].scale = torch.tensor(2.0**-4)
    op.input_quant_config[1].scale = torch.tensor(2.0**-2)
    op.input_quant_config[2].scale = torch.tensor(2.0**-2)
    op.output_quant_config[0].scale = torch.tensor(2.0**-4)
    op.input_quant_config[2].num_of_bits = 16
    op.input_quant_config[2].quant_min = -32768
    op.input_quant_config[2].quant_max = 32767
    _passive_clip(graph, op, visibility)
    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == EspQuantType.S8

    info = _quantize_parameters(op)
    assert op.inputs[2].value.dtype == torch.int8
    assert int(op.inputs[2].value.item()) == 24
    if exports_exponent:
        assert info.get_var_exponents("clip_max") == [-4]
    else:
        assert info.get_var_exponents("clip_max") is None

    exported = _export(op, {"x": -4, "clip_min": -4, "clip_max": -4, "y": -4})
    assert "lut" not in exported.attributes


def test_internal_pad_value_is_not_baked_with_clip_bounds():
    """Pad constant_value uses the same INTERNAL visibility and is not an integer Clip bound."""
    graph = BaseGraph(name="pad_internal", built_from=NetworkFramework.ONNX)
    activation = graph.create_variable(name="x", shape=[1])
    pads = graph.create_variable(name="pads", value=torch.tensor([0, 1], dtype=torch.float32), is_parameter=True)
    value = graph.create_variable(name="pad_value", value=torch.tensor(1.0), is_parameter=True)
    output = graph.create_variable(name="y", shape=[1])
    operation = graph.create_operation(
        op_type="Pad",
        name="Pad_op",
        platform=TargetPlatform.ESPDL_INT16,
        inputs=[activation, pads, value],
        outputs=[output],
    )
    # Pad links its output to the input, so the shared scale loop in _quantize cannot assign it.
    config = EspdlInt16Quantizer(graph).init_quantize_config(operation)
    op = QuantableOperation(convert_from=operation, quantize_config=config, platform=TargetPlatform.ESPDL_INT16)
    _install_quant_op(graph, op)
    op.attributes["quant_type"] = EspQuantType.S16
    input_cfg = op.input_quant_config[0]
    pad_cfg = op.input_quant_config[2]
    input_cfg.state = QuantizationStates.ACTIVATED
    input_cfg.scale = torch.tensor(2.0**-8)
    input_cfg.offset = torch.tensor(0.0)
    pad_cfg.num_of_bits = 8
    pad_cfg.quant_min = -128
    pad_cfg.quant_max = 127
    pad_cfg.master_by = input_cfg
    pad_cfg.visibility = QuantizationVisibility.INTERNAL

    _quantize_parameters(op)
    assert op.inputs[2].value.dtype.is_floating_point
    assert float(op.inputs[2].value) == 1.0


def test_export_does_not_write_clip_table(tmp_path):
    """Export does not write a Clip table or int16_lut_step. The exporter argument is ignored."""
    from esp_ppq.IR.morph import GraphFormatter

    graph, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _activation, "y")
    _install_quant_op(graph, op)
    op.attributes["min"] = -1.0
    op.attributes["max"] = 1.0
    GraphFormatter(graph).format_clip()
    assert len(op.inputs) == 3
    recorded = []
    original = AddLUTPattern.export

    class _StopExport(Exception):
        pass

    def _record(self, op, graph, **kwargs):
        result = original(self, op, graph, **kwargs)
        if op.type == "Clip":
            recorded.append((op.attributes.get("int16_lut_step"), "lut" in op.attributes))
            raise _StopExport
        return result

    AddLUTPattern.export = _record
    try:
        for step in (-1, 256):
            recorded.clear()
            SingletonMeta._instances.pop(AddLUTPattern, None)
            with pytest.raises(_StopExport):
                EspdlExporter().export(
                    str(tmp_path / f"clip_{step}.espdl"),
                    graph,
                    int16_lut_step=step,
                    export_config=False,
                )
            assert recorded == [(None, False)]
            assert "int16_lut_step" not in op.attributes
            assert "lut" not in op.attributes
            assert len(op.inputs) == 3

        op.attributes["int16_lut_step"] = 32
        recorded.clear()
        SingletonMeta._instances.pop(AddLUTPattern, None)
        with pytest.raises(_StopExport):
            EspdlExporter().export(
                str(tmp_path / "clip_keep.espdl"),
                graph,
                int16_lut_step=256,
                export_config=False,
            )
        assert op.attributes["int16_lut_step"] == 32
        assert recorded == [(32, False)]
        assert "lut" not in op.attributes
        assert len(op.inputs) == 3
    finally:
        AddLUTPattern.export = original
        SingletonMeta._instances.pop(AddLUTPattern, None)


def test_espdl_platform_and_num_of_bits_select_int16_lut():
    from esp_ppq.parser.espdl.export_patterns import InsertQuantTypePattern
    from esp_ppq.quantization.optim.espdl_lut import EspdlLUTFusionPass

    graph, op = _quantize("Sigmoid", TargetPlatform.ESPDL_INT8, _activation, "y")
    _install_quant_op(graph, op)
    for config in op.input_quant_config + op.output_quant_config:
        config.num_of_bits = 16
    EspdlLUTFusionPass(target_ops=["Sigmoid"], lut_step=32).optimize(graph)
    assert op.type == "LUT"
    assert op.attributes["int16_lut_step"] == 32
    InsertQuantTypePattern().export(op, graph)
    assert op.attributes["quant_type"] == EspQuantType.S16

    graph, non_espdl = _quantize("Sigmoid", TargetPlatform.ESPDL_INT16, _activation, "y2")
    _install_quant_op(graph, non_espdl)
    non_espdl.platform = TargetPlatform.FP32
    EspdlLUTFusionPass(target_ops=["Sigmoid"], lut_step=32).optimize(graph)
    assert non_espdl.type == "Sigmoid"

    graph, clip = _quantize("Clip", TargetPlatform.ESPDL_INT8, _activation, "clip_y")
    _install_quant_op(graph, clip)
    clip.type = "LUT"
    clip.attributes["int16_lut_step"] = 32
    clip.attributes["original_op_type"] = "Clip"
    with pytest.raises(ValueError, match="quant_type is S8"):
        _export(clip, {"x": -8, "clip_y": -8})


def test_s16_mismatched_clip_forward_stays_on_clamp():
    _, op = _quantize("Clip", TargetPlatform.ESPDL_INT16, _clip_bounds, "y")
    op.input_quant_config[0].scale = torch.tensor(2.0**-8)
    op.input_quant_config[1].scale = torch.tensor(2.0**-4)
    op.input_quant_config[2].scale = torch.tensor(2.0**-4)
    op.output_quant_config[0].scale = torch.tensor(2.0**-8)
    x = torch.arange(-512, 512, dtype=torch.float32) * (2.0**-8)
    got = Clip_forward(op, [x, torch.tensor(0.0), torch.tensor(1.0)])
    assert torch.equal(got, torch.clamp(x, 0.0, 1.0))


class _ReluClipSigmoid(nn.Module):
    """Bare Relu, ReLU6 (Clip), then Sigmoid. No Conv, so export cannot fuse them away."""

    def forward(self, x):
        x = torch.relu(x)
        x = torch.nn.functional.relu6(x)
        return torch.sigmoid(x)


def _exported_kind(op):
    return op.attributes.get("original_op_type", op.type)


def test_s16_export_pipeline_wires_relu_clip_sigmoid():
    """Quantize and export through EspdlExporter, then read the prepared graph.

    InsertQuantTypePattern and QuantVariableToIntPattern must run before
    AddLUTPattern. Relu and a same-exponent Clip stay on the integer kernel;
    Sigmoid keeps its table.
    """
    torch.manual_seed(0)
    model = _ReluClipSigmoid().eval()
    calib = [torch.randn(8) for _ in range(4)]
    loader = DataLoader(calib, batch_size=2)
    export_file = os.path.join(tempfile.gettempdir(), "int16_relu_clip_sigmoid.espdl")
    prepared = {}
    original = EspdlExporter.prepare_graph

    def _capture(self, graph, *args, **kwargs):
        graph = original(self, graph, *args, **kwargs)
        prepared["graph"] = graph
        return graph

    EspdlExporter.prepare_graph = _capture
    try:
        espdl_quantize_torch(
            model=model,
            espdl_export_file=export_file,
            calib_dataloader=loader,
            calib_steps=2,
            input_shape=[1, 8],
            target="esp32p4",
            num_of_bits=16,
            collate_fn=lambda batch: batch.to("cpu"),
            device="cpu",
            error_report=False,
            skip_export=False,
            export_config=False,
            verbose=0,
        )
    finally:
        EspdlExporter.prepare_graph = original
        for path in (export_file, export_file.rsplit(".", 1)[0] + ".onnx"):
            if os.path.exists(path):
                os.unlink(path)

    graph = prepared["graph"]
    relu = next(op for op in graph.operations.values() if _exported_kind(op) == "Relu")
    clip = next(op for op in graph.operations.values() if _exported_kind(op) == "Clip")
    sigmoid = next(op for op in graph.operations.values() if _exported_kind(op) == "Sigmoid")

    assert relu.attributes.get("quant_type") == EspQuantType.S16
    assert clip.attributes.get("quant_type") == EspQuantType.S16
    assert sigmoid.attributes.get("quant_type") == EspQuantType.S16
    assert "lut" not in relu.attributes
    assert "lut" not in clip.attributes
    assert sigmoid.attributes["lut"].startswith("Sigmoid_lut_")


def _device_clip_codes(values, op, in_config, out_config):
    """dl_module_clip.hpp: clip raw codes, rescale by the exponent ratio, truncate."""
    import math

    from esp_ppq.quantization.qfunction.linear import PPQLinearQuant_toInt
    from esp_ppq.utils.round import ppq_tensor_round

    codes = PPQLinearQuant_toInt(values, in_config).to(torch.int32)

    quant_type = op.attributes.get("quant_type")
    uses_integer_kernel = quant_type == EspQuantType.S16 or (
        quant_type == EspQuantType.S8 and any(bound.value is None and bound.name != "" for bound in list(op.inputs)[1:])
    )

    def _bound_code(bound_var, default):
        if bound_var is None or bound_var.name == "" or bound_var.value is None:
            return default
        bound = bound_var.value
        if isinstance(bound, torch.Tensor):
            scalar = bound.reshape(-1)[0]
            if uses_integer_kernel and scalar.dtype.is_floating_point:
                raise AssertionError(
                    "Clip integer-kernel test expects exported integer bound codes, got floating tensor"
                )
            if scalar.dtype.is_floating_point:
                numeric = float(scalar.item())
                if not math.isfinite(numeric):
                    return default
                return int(PPQLinearQuant_toInt(torch.tensor(numeric, dtype=torch.float32), in_config))
            return int(scalar.item())
        if isinstance(bound, int):
            return int(bound)
        numeric = float(bound)
        if not math.isfinite(numeric):
            return default
        return int(PPQLinearQuant_toInt(torch.tensor(numeric, dtype=torch.float32), in_config))

    low_var = op.inputs[1] if len(op.inputs) > 1 else None
    high_var = op.inputs[2] if len(op.inputs) > 2 else None
    low = _bound_code(low_var, int(in_config.quant_min))
    high = _bound_code(high_var, int(in_config.quant_max))
    clipped = codes.clamp(low, high).to(torch.float32)
    in_exp = int(round(math.log2(float(in_config.scale))))
    out_exp = int(round(math.log2(float(out_config.scale))))
    scaled = clipped * (2.0 ** (in_exp - out_exp))
    rounded = ppq_tensor_round(scaled, out_config.rounding)
    return rounded.clamp(int(out_config.quant_min), int(out_config.quant_max)).to(torch.int32)


def _sim_clip_codes(op, graph, values, output_name):
    """Executor path: fake-quant inputs, Clip_forward, fake-quant the output."""
    from esp_ppq.executor.torch import TorchExecutor
    from esp_ppq.quantization.qfunction.linear import PPQLinearQuant_toInt

    _passive_clip(graph, op)
    graph.outputs[output_name] = op.outputs[0]
    graph.inputs[op.inputs[0].name] = op.inputs[0]
    executor = TorchExecutor(graph, device="cpu")
    simulated = executor.forward({op.inputs[0].name: values}, output_names=[output_name])[0]
    return PPQLinearQuant_toInt(simulated, op.output_quant_config[0]).to(torch.int32)


def test_clip_sim_matches_device_integer_kernel():
    """Same exponent is exact. An INT16 output-scale shift stays within one code."""
    import math

    from esp_ppq.quantization.quantizer.EspdlQuantizer import (
        EspdlInt16Quantizer,
        EspdlQuantizer,
        EspdlS3Int16Quantizer,
        EspdlS3Quantizer,
    )

    cases = (
        (TargetPlatform.ESPDL_INT8, EspdlQuantizer, 0),
        (TargetPlatform.ESPDL_S3_INT8, EspdlS3Quantizer, 0),
        (TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer, 1),
        (TargetPlatform.ESPDL_S3_INT16, EspdlS3Int16Quantizer, 1),
    )
    scenarios = (
        ("same_scale", 0.0, 1.0, False, 0),
        ("mismatched_constant", 0.1, 1.1, False, 0),
        ("attribute", 0.1, 1.1, True, 0),
        ("finer_output", 0.1, 1.1, False, -2),
        ("coarser_output", 0.1, 1.1, True, 2),
    )
    for platform, quantizer_cls, shift_tolerance in cases:
        bits = 8 if "INT8" in platform.name and "INT16" not in platform.name else 16
        span = 1 << (bits - 1)
        for name, lower, upper, use_attribute, out_shift in scenarios:
            in_exp = -4
            out_exp = in_exp + out_shift
            codes = torch.arange(-span, span, dtype=torch.float32)
            values = codes * (2.0**in_exp)
            graph = BaseGraph(name=f"clip_align_{name}", built_from=NetworkFramework.ONNX)
            output = graph.create_variable(name="y", shape=[1])
            if use_attribute:
                from esp_ppq.IR.morph import GraphFormatter

                x = graph.create_variable(name="x", shape=[1])
                operation = graph.create_operation(
                    op_type="Clip",
                    name="Clip_op",
                    platform=platform,
                    inputs=[x],
                    outputs=[output],
                )
                operation.attributes["min"] = lower
                operation.attributes["max"] = upper
                GraphFormatter(graph).format_clip()
                for bound in operation.inputs[1:]:
                    if bound.value is not None and not isinstance(bound.value, torch.Tensor):
                        bound.value = torch.tensor(float(bound.value), dtype=torch.float32)
            else:
                operation = graph.create_operation(
                    op_type="Clip",
                    name="Clip_op",
                    platform=platform,
                    inputs=_clip_bounds(graph, lower, upper),
                    outputs=[output],
                )
            quantizer = quantizer_cls(graph)
            config = quantizer.init_quantize_config(operation)
            op = QuantableOperation(convert_from=operation, quantize_config=config, platform=platform)
            op.input_quant_config[0].scale = torch.tensor(2.0**in_exp)
            op.output_quant_config[0].scale = torch.tensor(2.0**out_exp)
            op.input_quant_config[0].offset = torch.tensor(0.0)
            op.output_quant_config[0].offset = torch.tensor(0.0)
            for bound_config in op.input_quant_config[1:]:
                bound_config.scale = torch.tensor(2.0 ** (in_exp + 4))
                bound_config.offset = torch.tensor(0.0)
            simulated = _sim_clip_codes(op, graph, values, "y")
            InsertQuantTypePattern().export(op, graph)
            _quantize_parameters(op)
            device = _device_clip_codes(values, op, op.input_quant_config[0], op.output_quant_config[0])
            delta = (simulated - device).abs().max().item()
            if out_shift == 0:
                assert torch.equal(simulated, device), f"{platform.name} {name} differs by {delta} codes"
            else:
                assert delta <= shift_tolerance, (
                    f"{platform.name} {name} max code delta {delta} exceeds {shift_tolerance}"
                )
            assert math.isclose(float(op.input_quant_config[1].scale), float(op.input_quant_config[0].scale))

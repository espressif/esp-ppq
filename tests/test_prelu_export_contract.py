import pytest
import torch

from esp_ppq.core import NetworkFramework, TargetPlatform
from esp_ppq.IR import BaseGraph
from esp_ppq.lib.quant import Quantizer
from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
from esp_ppq.parser.espdl.export_patterns import AddLUTPattern, InsertQuantTypePattern


@pytest.fixture(autouse=True)
def _reset_exporter_pattern_info():
    info = ExporterPatternInfo()
    info.reset()
    yield
    info.reset()


def _build_quantized_prelu(platform: TargetPlatform):
    graph = BaseGraph(name=f"prelu_contract_{platform.name}", built_from=NetworkFramework.ONNX)
    # IR shape is NCHW. Export converts it to NHWC, where the integer kernel
    # indexes the two channels on axis 3.
    # create_variable does not forward a shape keyword.
    x = graph.create_variable("x")
    x.shape = [1, 2, 1, 1]
    slope = graph.create_variable(
        "slope",
        value=torch.tensor([0.125, 0.25], dtype=torch.float32),
        is_parameter=True,
    )
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="PRelu",
        name="prelu",
        platform=platform,
        inputs=[x, slope],
        outputs=[y],
    )

    op = Quantizer(platform, graph).quantize_operation("prelu", platform=platform)
    InsertQuantTypePattern().export(op, graph)
    ExporterPatternInfo().add_var_permute(x.name, [0, 2, 3, 1])
    return graph, op


def test_prelu_s16_vector_slope_must_match_platform_bits():
    graph, op = _build_quantized_prelu(TargetPlatform.ESPDL_INT16)
    op.input_quant_config[1].num_of_bits = 8

    with pytest.raises(
        ValueError,
        match=r"exports PRelu as S16 without LUT, but slope uses 8-bit quantization",
    ):
        AddLUTPattern().export(op, graph)


def test_prelu_s8_vector_slope_must_match_platform_bits():
    graph, op = _build_quantized_prelu(TargetPlatform.ESPDL_INT8)
    op.input_quant_config[1].num_of_bits = 16

    with pytest.raises(
        ValueError,
        match=r"exports PRelu as S8 without LUT, but slope uses 16-bit quantization",
    ):
        AddLUTPattern().export(op, graph)


def test_prelu_vector_slope_with_matching_bits_exports_without_lut():
    graph, op = _build_quantized_prelu(TargetPlatform.ESPDL_INT16)
    op.input_quant_config[1].num_of_bits = 16

    exported = AddLUTPattern().export(op, graph)

    assert exported is op
    assert "lut" not in op.attributes
    # PRelu::deserialize asserts a rank-3 [c, 1, 1] slope on the integer kernel.
    assert tuple(op.inputs[1].value.shape) == (2, 1, 1)

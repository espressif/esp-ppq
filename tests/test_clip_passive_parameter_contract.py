import pytest
import torch

from esp_ppq.core import NetworkFramework, QuantizationStates, QuantizationVisibility, TargetPlatform
from esp_ppq.IR import BaseGraph
from esp_ppq.lib.quant import Quantizer
from esp_ppq.parser.espdl.export_patterns import EspdlQuantHelper, InsertQuantNodePattern, InsertRequantNodePattern
from esp_ppq.quantization.optim.parameters import PassiveParameterQuantizePass


def test_clip_runtime_bound_with_mismatched_scale_is_rejected():
    graph = BaseGraph(name="clip_runtime_bound_contract", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )
    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    # The Clip-side bound config can look aligned; esp-dl reads codes from the producer output.
    _activate(min_cfg, 0.125, 8)
    _activate(max_cfg, 0.125, 8)
    _activate(producer.output_quant_config[0], 0.25, 8)

    with pytest.raises(
        PermissionError,
        match=r"clip_min \(runtime tensor\) producer output does not share the input scale/offset",
    ):
        PassiveParameterQuantizePass().optimize(graph)


def _activate(config, scale: float, bits: int):
    config.state = QuantizationStates.ACTIVATED
    config.scale = torch.tensor([scale], dtype=torch.float32)
    config.offset = torch.tensor([0.0], dtype=torch.float32)
    config.num_of_bits = bits


@pytest.mark.parametrize(
    ("visibility", "expects_reject"),
    [
        (QuantizationVisibility.INTERNAL, True),
        (QuantizationVisibility.EXPORT_WHEN_ACTIVE, False),
    ],
)
def test_clip_runtime_bound_mismatch_depends_on_bound_exportability(visibility, expects_reject):
    """Quantizer leaves Clip bounds FP32 with no scale. Exportability follows the rewritten config."""
    graph = BaseGraph(name="clip_runtime_bound_producer_mismatch", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    assert min_cfg.state == QuantizationStates.FP32
    assert min_cfg.scale is None
    _activate(producer.output_quant_config[0], 0.25, 8)

    if expects_reject:
        with pytest.raises(
            PermissionError,
            match=r"clip_min \(runtime tensor\) producer output does not share the input scale/offset",
        ):
            PassiveParameterQuantizePass(clip_visiblity=visibility).optimize(graph)
        assert min_cfg.state == QuantizationStates.FP32
        assert min_cfg.scale is None
    else:
        PassiveParameterQuantizePass(clip_visiblity=visibility).optimize(graph)
        assert min_cfg.state == QuantizationStates.PASSIVE
        assert torch.allclose(min_cfg.scale, input_cfg.scale)
        assert min_cfg.master_by == input_cfg
        assert min_cfg.visibility == visibility
        assert max_cfg.master_by == input_cfg
        assert max_cfg.visibility == visibility


def test_mismatched_runtime_bound_follows_exporter_tqc(monkeypatch):
    """The pass does not keep its own exportability predicate. A negative answer from the exporter rejects the bound."""
    graph = BaseGraph(name="clip_runtime_bound_tqc_hook", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, _max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(producer.output_quant_config[0], 0.25, 8)
    seen = {}

    def reject_export(TQC, bounded_var):
        seen["state"] = TQC.state
        seen["visibility"] = TQC.visibility
        seen["bits"] = int(TQC.num_of_bits)
        seen["scale"] = TQC.scale
        return False

    monkeypatch.setattr(EspdlQuantHelper, "TQC_Exportable_Check", staticmethod(reject_export))

    with pytest.raises(PermissionError, match=r"clip_min \(runtime tensor\)"):
        PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE).optimize(graph)

    assert seen["state"] == QuantizationStates.PASSIVE
    assert seen["visibility"] == QuantizationVisibility.EXPORT_WHEN_ACTIVE
    assert seen["bits"] == int(clip.output_quant_config[0].num_of_bits)
    assert torch.allclose(seen["scale"], input_cfg.scale)
    assert min_cfg.state == QuantizationStates.FP32
    assert min_cfg.scale is None


def test_clip_runtime_bound_mismatch_is_rejected_when_aligned_config_can_not_export():
    """A widened bound range does not count. Export checks the range copied from the output."""
    graph = BaseGraph(name="clip_runtime_bound_non_exportable", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    # Pre-alignment this bound is 16-bit, so its own range would pass an INT8 check.
    min_cfg.num_of_bits = 16
    min_cfg.quant_min = -32768
    min_cfg.quant_max = 32767
    _activate(max_cfg, 0.125, 8)
    _activate(producer.output_quant_config[0], 0.25, 8)
    out_cfg = clip.output_quant_config[0].dominated_by
    out_cfg.quant_min = -200
    out_cfg.quant_max = 200

    with pytest.raises(
        PermissionError,
        match=r"clip_min \(runtime tensor\) producer output does not share the input scale/offset",
    ):
        PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE).optimize(graph)
    assert min_cfg.num_of_bits == 16
    assert min_cfg.state == QuantizationStates.FP32


def test_clip_runtime_bound_is_rejected_when_producer_bit_width_differs():
    graph = BaseGraph(name="clip_runtime_bound_bit_width", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT16,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT16)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(min_cfg, 0.25, 8)
    _activate(max_cfg, 0.125, 8)
    _activate(producer.output_quant_config[0], 0.125, 16)

    with pytest.raises(
        PermissionError,
        match=r"clip_min \(runtime tensor\) producer output bit width 16 differs from Clip input bit width 8",
    ):
        PassiveParameterQuantizePass().optimize(graph)


@pytest.mark.parametrize(
    ("input_scale", "producer_scale"),
    [
        (0.125, 0.25),
        (2.0**-21, 2.0**-20),
        (2.0**-28, 2.0**-27),
    ],
)
def test_clip_runtime_bound_exportable_mismatch_inserts_requantize_node(input_scale, producer_scale):
    graph = BaseGraph(name="clip_runtime_bound_requant_insert", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, input_scale, 8)
    assert min_cfg.state == QuantizationStates.FP32
    assert min_cfg.scale is None
    _activate(max_cfg, input_scale, 8)
    _activate(producer.output_quant_config[0], producer_scale, 8)

    PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE).optimize(graph)
    assert min_cfg.state == QuantizationStates.PASSIVE
    InsertRequantNodePattern().export(clip, graph)

    assert clip.inputs[1].source_op is not None
    assert clip.inputs[1].source_op.type == "RequantizeLinear"


def test_clip_runtime_bound_exportable_fp32_producer_inserts_quantize_node():
    graph = BaseGraph(name="clip_runtime_bound_quant_insert", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    assert min_cfg.state == QuantizationStates.FP32
    assert min_cfg.scale is None
    _activate(max_cfg, 0.125, 8)
    producer_out = producer.output_quant_config[0]
    producer_out.state = QuantizationStates.FP32
    producer_out.scale = None
    producer_out.offset = None

    PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE).optimize(graph)
    InsertQuantNodePattern().export(clip, graph)

    assert clip.inputs[1].source_op is not None
    assert clip.inputs[1].source_op.type == "QuantizeLinear"


@pytest.mark.parametrize(
    ("producer_state", "has_grid", "expected_reason"),
    [
        (
            QuantizationStates.PASSIVE_INIT,
            False,
            r"producer output has no quantized scale/offset",
        ),
        (
            QuantizationStates.DEQUANTIZED,
            True,
            r"producer output state DEQUANTIZED has no quantized codes",
        ),
    ],
)
def test_clip_runtime_bound_unready_quantized_producer_is_rejected_before_export(
    producer_state, has_grid, expected_reason
):
    graph = BaseGraph(
        name=f"clip_runtime_bound_{producer_state.name.lower()}_producer", built_from=NetworkFramework.ONNX
    )
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, _max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    producer_out = producer.output_quant_config[0]
    if has_grid:
        _activate(producer_out, 0.125, 8)
    producer_out.state = producer_state

    with pytest.raises(
        PermissionError,
        match=rf"clip_min \(runtime tensor\) {expected_reason}",
    ):
        PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE).optimize(graph)
    assert min_cfg.state == QuantizationStates.FP32
    assert min_cfg.scale is None


def test_clip_runtime_graph_input_bound_is_rejected_even_when_exportable():
    graph = BaseGraph(name="clip_runtime_graph_input_bound", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )
    clip = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, _max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    assert min_cfg.state == QuantizationStates.FP32
    assert clip_min.source_op is None

    with pytest.raises(
        PermissionError,
        match=r"clip_min \(runtime tensor\) has no quantized producer output",
    ):
        PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE).optimize(graph)
    assert min_cfg.state == QuantizationStates.FP32


def test_espdl_clip_rejects_fp32_output_before_bound_rewrite():
    graph = BaseGraph(name="clip_fp32_output_contract", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    clip_min = graph.create_variable("clip_min", value=torch.tensor([-1.0], dtype=torch.float32), is_parameter=True)
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )
    clip = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(min_cfg, 0.25, 8)
    _activate(max_cfg, 0.125, 8)
    clip.output_quant_config[0].state = QuantizationStates.FP32

    with pytest.raises(ValueError, match="output is FP32"):
        PassiveParameterQuantizePass().optimize(graph)


@pytest.mark.parametrize(
    "visibility",
    [QuantizationVisibility.INTERNAL, QuantizationVisibility.EXPORT_WHEN_ACTIVE],
)
def test_clip_omitted_runtime_bound_name_is_skipped(visibility):
    graph = BaseGraph(name="clip_omitted_runtime_bound", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    omitted_min = graph.create_variable("", value=None, is_parameter=False)
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, omitted_min, clip_max],
        outputs=[y],
    )

    clip = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(min_cfg, 0.25, 8)
    _activate(max_cfg, 0.125, 8)

    PassiveParameterQuantizePass(clip_visiblity=visibility).optimize(graph)

    assert min_cfg.master_by == input_cfg
    assert max_cfg.master_by == input_cfg
    assert min_cfg.visibility == visibility
    assert max_cfg.visibility == visibility


@pytest.mark.parametrize(
    "visibility",
    [QuantizationVisibility.INTERNAL, QuantizationVisibility.EXPORT_WHEN_ACTIVE],
)
def test_clip_runtime_bound_produced_on_the_input_grid_is_accepted(visibility):
    graph = BaseGraph(name="clip_runtime_bound_aligned_producer", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    bound_src = graph.create_variable("bound_src")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Relu",
        name="bound_producer",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[bound_src],
        outputs=[clip_min],
    )
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    producer = quantizer.quantize_operation("bound_producer", platform=TargetPlatform.ESPDL_INT8)
    clip = quantizer.quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    # The Clip-side config disagrees; the producer is what actually writes the codes.
    _activate(min_cfg, 0.25, 8)
    _activate(max_cfg, 0.125, 8)
    _activate(producer.output_quant_config[0], 0.125, 8)

    PassiveParameterQuantizePass(clip_visiblity=visibility).optimize(graph)

    assert min_cfg.master_by == input_cfg
    assert min_cfg.visibility == visibility


@pytest.mark.parametrize(
    "platform",
    [TargetPlatform.PPL_CUDA_INT8, TargetPlatform.TRT_INT8, TargetPlatform.ONNXRUNTIME],
)
def test_non_espdl_runtime_clip_bound_keeps_master_by(platform):
    graph = BaseGraph(name=f"clip_runtime_bound_{platform.name}", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    clip_min = graph.create_variable("clip_min")
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=platform,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )
    clip = Quantizer(platform, graph).quantize_operation("clip")
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(min_cfg, 0.25, 8)
    _activate(max_cfg, 0.125, 8)

    PassiveParameterQuantizePass().optimize(graph)

    assert min_cfg.master_by == input_cfg
    assert max_cfg.master_by == input_cfg


def test_unique_constant_clip_bound_is_remapped_onto_the_input_scale():
    graph = BaseGraph(name="clip_unique_constant_bound", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    clip_min = graph.create_variable("clip_min", value=torch.tensor([-1.0], dtype=torch.float32), is_parameter=True)
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )
    clip = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(min_cfg, 0.25, 8)
    _activate(max_cfg, 0.5, 8)

    PassiveParameterQuantizePass().optimize(graph)

    assert min_cfg.master_by == input_cfg
    assert max_cfg.master_by == input_cfg
    assert torch.allclose(min_cfg.scale, input_cfg.scale)
    assert torch.allclose(max_cfg.scale, input_cfg.scale)


@pytest.mark.parametrize("min_scale", [0.25, 0.125])
def test_clip_shared_bound_is_rejected_even_when_its_scale_matches(min_scale):
    graph = BaseGraph(name="clip_shared_bound_contract", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    clip_min = graph.create_variable("clip_min", value=torch.tensor([-1.0], dtype=torch.float32), is_parameter=True)
    clip_max = graph.create_variable("clip_max", value=torch.tensor([6.0], dtype=torch.float32), is_parameter=True)
    y = graph.create_variable("y")
    graph.create_operation(
        op_type="Clip",
        name="clip",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, clip_min, clip_max],
        outputs=[y],
    )

    aux = graph.create_variable("aux")
    aux_out = graph.create_variable("aux_out")
    graph.create_operation(
        op_type="Add",
        name="reuse_clip_min",
        platform=TargetPlatform.FP32,
        inputs=[clip_min, aux],
        outputs=[aux_out],
    )
    clip = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation("clip", platform=TargetPlatform.ESPDL_INT8)
    input_cfg, min_cfg, max_cfg = clip.config.input_quantization_config
    _activate(input_cfg, 0.125, 8)
    _activate(min_cfg, min_scale, 8)
    _activate(max_cfg, 0.125, 8)

    with pytest.raises(
        PermissionError,
        match=r"clip_min \(shared constant\) is used by more than one operation",
    ):
        PassiveParameterQuantizePass().optimize(graph)

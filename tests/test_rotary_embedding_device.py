"""Regression coverage for ESP-DL RoPE's deployment contract and P4 emulator."""

import re
from unittest.mock import patch

import pytest
import torch

from esp_ppq import TargetPlatform
from esp_ppq.core import (
    OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE,
    DataType,
    OperationQuantizationConfig,
    QuantizationStates,
    RoundingPolicy,
)
from esp_ppq.executor.op.torch.espdl_rotary_embedding import (
    _round_shift_half_even,
    _round_shift_half_up,
    espdl_rotary_embedding_forward,
    rotary_embedding_reference,
    simulate_espdl_rotary_embedding,
)
from esp_ppq.IR import BaseGraph
from esp_ppq.IR.base.opdef import RotaryEmbedding_Socket
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.lib import LinearQuantizationConfig
from esp_ppq.quantization.optim import (
    check_rotary_embedding,
    finalize_espdl_rotary_embedding_for_export,
    legalize_espdl_rotary_embedding_graph,
)
from esp_ppq.quantization.optim.rotary_embedding_fusion import (
    _finalize_espdl_rotary_embedding_quantization_contract,
    _rope_prequant_needs_trace,
)
from esp_ppq.quantization.quantizer import EspdlQuantizer


def _config(bits: int, exponent: int, rounding: RoundingPolicy):
    config = LinearQuantizationConfig(
        power_of_2=True,
        num_of_bits=bits,
        quant_min=-(1 << (bits - 1)),
        quant_max=(1 << (bits - 1)) - 1,
        rounding=rounding,
    )
    config.scale = torch.tensor(2.0**exponent)
    config.offset = torch.tensor(0.0)
    config.state = QuantizationStates.ACTIVATED
    return config


def _make_quantized_rope(
    *,
    x_bits=8,
    cache_bits=16,
    y_bits=8,
    x_exp=-7,
    cache_exp=-15,
    y_exp=-7,
    head_dim=32,
    seq_len=3,
    platform=TargetPlatform.ESPDL_INT8,
):
    graph = BaseGraph(name="rope_device_contract")
    x = graph.create_variable("x")
    x.shape, x.dtype = [1, 2, seq_len, head_dim], DataType.FP32
    cos = graph.create_variable(
        "cos",
        value=torch.ones(1, seq_len, head_dim // 2),
        is_parameter=True,
    )
    sin = graph.create_variable(
        "sin",
        value=torch.zeros(1, seq_len, head_dim // 2),
        is_parameter=True,
    )
    y = graph.create_variable("y")
    y.shape, y.dtype = [1, 2, seq_len, head_dim], DataType.FP32
    base = graph.create_operation(
        op_type="RotaryEmbedding",
        name="rope",
        attributes={},
        platform=platform,
        inputs=[x, cos, sin],
        outputs=[y],
    )
    half_even_platforms = {
        TargetPlatform.ESPDL_INT8,
        TargetPlatform.ESPDL_INT16,
        TargetPlatform.ESPDL_H_PRE_INT16,
        TargetPlatform.ESPDL_W8A16,
    }
    rounding = RoundingPolicy.ROUND_HALF_EVEN if platform in half_even_platforms else RoundingPolicy.ROUND_HALF_UP
    config = OperationQuantizationConfig(
        input_quantization_configs=[
            _config(x_bits, x_exp, rounding),
            _config(cache_bits, cache_exp, rounding),
            _config(cache_bits, cache_exp, rounding),
        ],
        output_quantization_configs=[_config(y_bits, y_exp, rounding)],
    )
    quantized = QuantableOperation(convert_from=base, quantize_config=config, platform=platform)
    graph.operations[base.name] = quantized
    for var in base.inputs:
        var.dest_ops[var.dest_ops.index(base)] = quantized
    y.source_op = quantized
    return graph, quantized


def test_rope_dynamic_cache_promotion_updates_kl_histogram_geometry():
    """Global KL calibration must remain valid after cache int8 -> int16 promotion."""
    graph = BaseGraph(name="dynamic_rope_calibration")
    shapes = ([1, 2, 3, 32], [1, 3, 16], [1, 3, 16])
    inputs = []
    for name, shape in zip(("x", "cos", "sin"), shapes):
        variable = graph.create_variable(name)
        variable.shape, variable.dtype = list(shape), DataType.FP32
        inputs.append(variable)
    output = graph.create_variable("y")
    output.shape, output.dtype = list(shapes[0]), DataType.FP32
    rope = graph.create_operation(
        op_type="RotaryEmbedding",
        name="rope",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=inputs,
        outputs=[output],
    )

    config = EspdlQuantizer(graph).init_quantize_config(rope)
    assert [item.num_of_bits for item in config.input_quantization_config] == [8, 16, 16]
    expected_bins = 32 * (1 << 15)
    for cache_config in config.input_quantization_config[1:3]:
        assert cache_config.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE] == expected_bins


def test_rope_reference_rejects_zero_head_dimension():
    _, rope = _make_quantized_rope(head_dim=2, seq_len=1)
    x = torch.empty(1, 2, 1, 0)
    cos = torch.empty(1, 1, 0)
    sin = torch.empty(1, 1, 0)

    with pytest.raises(ValueError, match="X dimensions must be positive"):
        rotary_embedding_reference(rope, [x, cos, sin])


def test_rope_export_finalizer_validates_actual_storage_contract():
    graph, rope = _make_quantized_rope()

    errors, hints = finalize_espdl_rotary_embedding_for_export(graph, platform=TargetPlatform.ESPDL_INT8, strict=True)
    assert not errors
    assert not hints  # D=32 -> half=16, the P4 mixed s8/s16 cache SIMD path is legal.

    # The C++ module requires equal cache storage types. This cannot be
    # deferred to its generic path because its input parser receives one
    # concrete dtype per tensor.
    rope.input_quant_config[2].num_of_bits = 8
    errors, _ = finalize_espdl_rotary_embedding_for_export(graph, platform=TargetPlatform.ESPDL_INT8, strict=False)
    assert any("cos=INT16 and sin=INT8" in error for error in errors)


def test_rope_postcalibration_finalizer_checks_only_quantization_contract():
    graph, rope = _make_quantized_rope()

    with patch("esp_ppq.quantization.optim.rotary_embedding_fusion._check_op") as check_op:
        errors, hints = _finalize_espdl_rotary_embedding_quantization_contract(
            graph,
            platform=TargetPlatform.ESPDL_INT8,
            strict=True,
        )
    check_op.assert_not_called()
    assert not errors
    assert not hints

    rope.input_quant_config[2].num_of_bits = 8
    errors, _ = _finalize_espdl_rotary_embedding_quantization_contract(
        graph,
        platform=TargetPlatform.ESPDL_INT8,
        strict=False,
    )
    assert any("cos=INT16 and sin=INT8" in error for error in errors)


def test_rope_target_aware_finalizer_suppresses_p4_only_hint_on_s3():
    graph, _ = _make_quantized_rope(head_dim=40, platform=TargetPlatform.ESPDL_S3_INT8)
    errors, hints = finalize_espdl_rotary_embedding_for_export(
        graph, platform=TargetPlatform.ESPDL_S3_INT8, strict=True
    )
    assert not errors
    assert not hints

    p4_graph, _ = _make_quantized_rope(head_dim=40, platform=TargetPlatform.ESPDL_INT8)
    _, p4_hints = finalize_espdl_rotary_embedding_for_export(p4_graph, platform=TargetPlatform.ESPDL_INT8, strict=True)
    assert p4_hints and "half=20" in p4_hints[0]


def test_rope_check_is_fail_closed_for_dynamic_shape_and_output_dtype():
    graph, rope = _make_quantized_rope()
    rope.inputs[0].shape = ["batch", 2, 3, 32]
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any("X shape is missing or dynamic" in error for error in errors)

    rope.inputs[0].shape = [1, 2, 3, 32]
    rope.outputs[0].dtype = DataType.FP16
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any("Y dtype FP16 is unsupported" in error for error in errors)
    assert any("X dtype FP32 != Y dtype FP16" in error for error in errors)

    # Malformed native attributes must produce an actionable validation error,
    # not a host-side int() exception that skips export validation entirely.
    rope.outputs[0].dtype = DataType.FP32
    rope.attributes["rotary_embedding_dim"] = "not-an-int"
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any("rotary_embedding_dim='not-an-int' is not an integer" in error for error in errors)

    rope.attributes.clear()
    rope.attributes["interleaved"] = "not-an-int"
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any("interleaved='not-an-int' is not an integer" in error for error in errors)


def test_four_input_native_rope_reaches_explicit_backend_validation():
    """Socket arity must not fail before we can explain position_ids is unsupported."""
    graph = BaseGraph(name="four_input_rope")
    x = graph.create_variable("x")
    x.shape, x.dtype = [1, 2, 3, 16], DataType.FP32
    cos = graph.create_variable("cos", value=torch.ones(1, 3, 8), is_parameter=True)
    sin = graph.create_variable("sin", value=torch.zeros(1, 3, 8), is_parameter=True)
    position_ids = graph.create_variable("position_ids", value=torch.zeros(1, 3, dtype=torch.int64), is_parameter=True)
    y = graph.create_variable("y")
    y.shape, y.dtype = [1, 2, 3, 16], DataType.FP32
    rope = graph.create_operation(
        op_type="RotaryEmbedding",
        name="rope_with_position_ids",
        attributes={},
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, cos, sin, position_ids],
        outputs=[y],
    )

    assert len(RotaryEmbedding_Socket(rope).in_plat) == 4
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any("position_ids is not supported" in error for error in errors)


def test_prequant_trace_request_includes_native_rope_output_metadata():
    graph, rope = _make_quantized_rope()
    rope.outputs[0].shape = None
    assert _rope_prequant_needs_trace(graph)


def test_native_identity_rope_is_legalized_before_strict_subset_check():
    graph = BaseGraph(name="identity_native_rope")
    x = graph.create_variable("x")
    x.shape, x.dtype = [1, 2, 3, 16], DataType.FP32
    cos = graph.create_variable("cos", value=torch.ones(1, 3, 4), is_parameter=True)
    sin = graph.create_variable("sin", value=torch.zeros(1, 3, 4), is_parameter=True)
    y = graph.create_variable("y")
    y.shape, y.dtype = [1, 2, 3, 16], DataType.FP32
    op = graph.create_operation(
        op_type="RotaryEmbedding",
        name="identity_partial_interleaved",
        attributes={"interleaved": 1, "rotary_embedding_dim": 8},
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, cos, sin],
        outputs=[y],
    )

    assert legalize_espdl_rotary_embedding_graph(graph) == 1
    assert op.type == "Identity"
    assert op.inputs == [x]
    assert "cos" not in graph.variables and "sin" not in graph.variables
    errors, _ = check_rotary_embedding(graph, strict=True)
    assert not errors


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda _x, y, _op: setattr(y, "shape", None), "Y shape is missing or dynamic"),
        (lambda _x, y, _op: setattr(y, "shape", [1, 2, 3, 8]), "Y shape (1, 2, 3, 8) != X shape (1, 2, 3, 16)"),
        (lambda _x, y, _op: setattr(y, "dtype", DataType.INT8), "X dtype FP32 != Y dtype INT8"),
        (lambda _x, y, _op: setattr(y, "dtype", DataType.FP16), "Y dtype FP16 is unsupported"),
        (
            lambda _x, _y, op: op.attributes.__setitem__("interleaved", "not-an-int"),
            "interleaved='not-an-int' is not an integer",
        ),
    ],
)
def test_native_identity_rope_does_not_bypass_output_or_attribute_validation(mutate, expected_error):
    graph = BaseGraph(name="invalid_identity_native_rope")
    x = graph.create_variable("x")
    x.shape, x.dtype = [1, 2, 3, 16], DataType.FP32
    cos = graph.create_variable("cos", value=torch.ones(1, 3, 8), is_parameter=True)
    sin = graph.create_variable("sin", value=torch.zeros(1, 3, 8), is_parameter=True)
    y = graph.create_variable("y")
    y.shape, y.dtype = [1, 2, 3, 16], DataType.FP32
    op = graph.create_operation(
        op_type="RotaryEmbedding",
        name="invalid_identity_rope",
        attributes={},
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, cos, sin],
        outputs=[y],
    )
    mutate(x, y, op)

    assert legalize_espdl_rotary_embedding_graph(graph) == 0
    assert op.type == "RotaryEmbedding"
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        check_rotary_embedding(graph, strict=True)


def test_exporter_legalizes_native_identity_rope_without_mutating_source(tmp_path):
    """The direct FP32 export path may skip prequant, but must preserve its input graph."""
    from esp_ppq.parser.espdl_exporter import EspdlExporter

    graph = BaseGraph(name="identity_native_rope_export")
    x = graph.create_variable("x")
    x.shape, x.dtype = [1, 2, 3, 16], DataType.FP32
    cos = graph.create_variable("cos", value=torch.ones(1, 3, 4), is_parameter=True)
    sin = graph.create_variable("sin", value=torch.zeros(1, 3, 4), is_parameter=True)
    y = graph.create_variable("y")
    y.shape, y.dtype = [1, 2, 3, 16], DataType.FP32
    op = graph.create_operation(
        op_type="RotaryEmbedding",
        name="identity_partial_interleaved_export",
        attributes={"interleaved": 1, "rotary_embedding_dim": 8},
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, cos, sin],
        outputs=[y],
    )
    graph.mark_variable_as_graph_input(x)
    graph.mark_variable_as_graph_output(y)

    output_path = tmp_path / "identity_rope.espdl"
    EspdlExporter().export(file_path=str(output_path), graph=graph, export_config=False)

    assert output_path.is_file() and output_path.stat().st_size > 0
    assert graph.operations[op.name] is op
    assert op.type == "RotaryEmbedding"
    assert op.inputs == [x, cos, sin]


def test_p4_rope_emulator_models_int_min_scalar_fallback():
    # Pure s8 SIMD uses saturating vsub for -x2: -(-128) becomes +127,
    # whereas scalar arithmetic sees +128. Production dispatch scans each
    # pure-width row and falls back to exact integer scalar arithmetic when
    # INT_MIN occurs, so module-level SIMD and scalar results must agree.
    _, pure_s8 = _make_quantized_rope(cache_bits=8, cache_exp=-7)
    x = torch.zeros(1, 2, 3, 32)
    x[..., 16] = -1.0  # -128 at X scale 2^-7
    cos = torch.zeros(1, 3, 16)
    sin = torch.ones(1, 3, 16) * (127.0 / 128.0)
    simd = simulate_espdl_rotary_embedding(pure_s8, [x, cos, sin])
    scalar = simulate_espdl_rotary_embedding(pure_s8, [x, cos, sin], force_scalar=True)
    assert simd is not None and scalar is not None
    assert torch.equal(simd, scalar)
    assert scalar[..., 0].eq(127.0 / 128.0).all()

    # The mixed kernel widens s8 X to s16 before its subtraction, so -128 is
    # representable and its SIMD result matches the scalar implementation.
    _, mixed = _make_quantized_rope(cache_bits=16, cache_exp=-15)
    mixed_sin = torch.ones(1, 3, 16) * (32767.0 / 32768.0)
    mixed_simd = simulate_espdl_rotary_embedding(mixed, [x, cos, mixed_sin])
    mixed_scalar = simulate_espdl_rotary_embedding(mixed, [x, cos, mixed_sin], force_scalar=True)
    assert torch.equal(mixed_simd, mixed_scalar)

    # Pure s16 has the same saturating-negate boundary and therefore the
    # same row-level fallback contract.
    _, pure_s16 = _make_quantized_rope(
        x_bits=16,
        cache_bits=16,
        y_bits=16,
        x_exp=-15,
        cache_exp=-15,
        y_exp=-15,
        head_dim=16,
    )
    x16 = torch.zeros(1, 2, 3, 16)
    x16[..., 8] = -1.0
    cos16 = torch.zeros(1, 3, 8)
    sin16 = torch.ones(1, 3, 8) * (32767.0 / 32768.0)
    s16_simd = simulate_espdl_rotary_embedding(pure_s16, [x16, cos16, sin16])
    s16_scalar = simulate_espdl_rotary_embedding(pure_s16, [x16, cos16, sin16], force_scalar=True)
    assert torch.equal(s16_simd, s16_scalar)
    assert s16_simd[..., 0].eq(32767.0 / 32768.0).all()

    # The registered executor entry point automatically selects the P4 model
    # once all TQCs are calibrated.
    dispatched = espdl_rotary_embedding_forward(pure_s8, [x, cos, sin])
    assert torch.equal(dispatched, simd)


def test_rope_integer_rne_handles_int64_shift_boundary():
    values = torch.tensor(
        [
            -1,
            0,
            1,
            -(1 << 62),
            1 << 62,
            -(1 << 62) - 1,
            (1 << 62) + 1,
            -(1 << 63),
            (1 << 63) - 1,
        ],
        dtype=torch.int64,
    )
    expected = torch.tensor([0, 0, 0, 0, 0, -1, 1, -1, 1], dtype=torch.int64)
    assert torch.equal(_round_shift_half_even(values, 63), expected)
    assert _round_shift_half_even(torch.tensor([-554942450], dtype=torch.int64), 15).item() == -16935


def test_s3_and_c_rope_emulator_use_half_up_integer_contract():
    ties = torch.tensor([-3, -1, 1, 3, -(1 << 62), 1 << 62], dtype=torch.int64)
    assert torch.equal(_round_shift_half_up(ties[:4], 1), torch.tensor([-1, 0, 1, 2], dtype=torch.int64))
    assert torch.equal(_round_shift_half_up(ties[4:], 63), torch.tensor([0, 1], dtype=torch.int64))

    _, p4_op = _make_quantized_rope(cache_bits=8, cache_exp=-1, x_exp=0, y_exp=0, head_dim=4, seq_len=1)
    _, s3_op = _make_quantized_rope(
        cache_bits=8,
        cache_exp=-1,
        x_exp=0,
        y_exp=0,
        head_dim=4,
        seq_len=1,
        platform=TargetPlatform.ESPDL_S3_INT8,
    )
    _, c_op = _make_quantized_rope(
        cache_bits=16,
        cache_exp=-15,
        x_bits=16,
        y_bits=16,
        x_exp=-15,
        y_exp=-15,
        head_dim=4,
        seq_len=1,
        platform=TargetPlatform.ESPDL_C_H_PRE_INT16,
    )
    x_tie = torch.ones(1, 2, 1, 4, dtype=torch.float32)
    cos_tie = torch.full((1, 1, 2), 0.5, dtype=torch.float32)
    sin_tie = torch.zeros(1, 1, 2, dtype=torch.float32)
    p4_out = espdl_rotary_embedding_forward(p4_op, [x_tie, cos_tie, sin_tie])
    s3_out = espdl_rotary_embedding_forward(s3_op, [x_tie, cos_tie, sin_tie])
    assert torch.all(p4_out[..., :2] == 0.0)
    assert torch.all(s3_out[..., :2] == 1.0)

    c_out = espdl_rotary_embedding_forward(c_op, [x_tie.double(), cos_tie.double(), sin_tie.double()])
    assert c_out.dtype == torch.float64

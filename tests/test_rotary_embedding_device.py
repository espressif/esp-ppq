"""Regression coverage for ESP-DL RoPE's executor contract and P4 emulator."""

import pytest
import torch

from esp_ppq import TargetPlatform
from esp_ppq.core import (
    OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE,
    DataType,
    OperationQuantizationConfig,
    QuantizationStates,
    QuantizationVisibility,
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
from esp_ppq.lib.quant import Quantizer
from esp_ppq.parser.espdl.espdl_typedef import EspQuantType, ExporterPatternInfo
from esp_ppq.parser.espdl.export_patterns import InsertQuantTypePattern, QuantVariableToIntPattern
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


def _replace_operation(graph: BaseGraph, op: QuantableOperation):
    plain = graph.operations[op.name]
    graph.operations[op.name] = op
    for var in op.inputs + op.outputs:
        if var.source_op is plain:
            var.source_op = op
        for index, dest in enumerate(var.dest_ops):
            if dest is plain:
                var.dest_ops[index] = op


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


def _make_runtime_rope():
    graph = BaseGraph(name="dynamic_rope_export_route")
    x = graph.create_variable("x")
    x.shape, x.dtype = [1, 2, 3, 32], DataType.FP32
    cos_src = graph.create_variable("cos_src")
    cos_src.shape, cos_src.dtype = [1, 3, 16], DataType.FP32
    sin_src = graph.create_variable("sin_src")
    sin_src.shape, sin_src.dtype = [1, 3, 16], DataType.FP32
    cos = graph.create_variable("cos")
    cos.shape, cos.dtype = [1, 3, 16], DataType.FP32
    sin = graph.create_variable("sin")
    sin.shape, sin.dtype = [1, 3, 16], DataType.FP32
    y = graph.create_variable("y")
    y.shape, y.dtype = [1, 2, 3, 32], DataType.FP32
    graph.create_operation(
        op_type="Relu",
        name="cos_runtime",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[cos_src],
        outputs=[cos],
    )
    graph.create_operation(
        op_type="Relu",
        name="sin_runtime",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[sin_src],
        outputs=[sin],
    )
    rope = graph.create_operation(
        op_type="RotaryEmbedding",
        name="rope",
        attributes={},
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[x, cos, sin],
        outputs=[y],
    )
    return graph, rope


def test_rope_runtime_cache_promotion_keeps_s8_quant_type():
    graph, rope = _make_runtime_rope()
    config = EspdlQuantizer(graph).init_quantize_config(rope)
    assert [item.num_of_bits for item in config.input_quantization_config] == [8, 16, 16]
    quantized = QuantableOperation(convert_from=rope, quantize_config=config, platform=TargetPlatform.ESPDL_INT8)
    _replace_operation(graph, quantized)

    InsertQuantTypePattern().export(quantized, graph)
    assert quantized.attributes["quant_type"] == EspQuantType.S8


def test_rope_constant_cache_width_may_differ_from_activation_width():
    graph, rope = _make_quantized_rope(x_bits=8, cache_bits=16, y_bits=8)

    InsertQuantTypePattern().export(rope, graph)

    assert rope.attributes["quant_type"] == EspQuantType.S8


@pytest.mark.parametrize("reason", ["fp32_state", "internal_visibility"])
def test_rope_constant_cache_dtypes_must_match(reason):
    graph, rope = _make_quantized_rope(x_bits=8, cache_bits=16, y_bits=8)
    if reason == "fp32_state":
        rope.input_quant_config[1].state = QuantizationStates.FP32
    else:
        rope.input_quant_config[1].visibility = QuantizationVisibility.INTERNAL

    with pytest.raises(ValueError, match=r"cos/sin cache inputs to share one dtype"):
        InsertQuantTypePattern().export(rope, graph)


def _make_runtime_rope_with_cache_configs(*, cos_visibility, sin_visibility, producer_fp32):
    """Runtime cos/sin producers plus hand-built S16 cache configs."""
    graph, rope = _make_runtime_rope()
    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    rounding = RoundingPolicy.ROUND_HALF_EVEN
    for producer_name in ("cos_runtime", "sin_runtime"):
        producer = quantizer.quantize_operation(producer_name, platform=TargetPlatform.ESPDL_INT8)
        if producer_fp32:
            producer.output_quant_config[0].state = QuantizationStates.FP32
        else:
            producer.output_quant_config[0] = _config(8, -7, rounding)
        _replace_operation(graph, producer)

    cache_configs = []
    for visibility in (cos_visibility, sin_visibility):
        cache_config = _config(16, -15, rounding)
        cache_config.visibility = visibility
        cache_configs.append(cache_config)
    config = OperationQuantizationConfig(
        input_quantization_configs=[_config(8, -7, rounding), *cache_configs],
        output_quantization_configs=[_config(8, -7, rounding)],
    )
    quantized = QuantableOperation(convert_from=rope, quantize_config=config, platform=TargetPlatform.ESPDL_INT8)
    _replace_operation(graph, quantized)
    return graph, quantized


def test_rope_skipped_runtime_cache_follows_its_producer_dtype():
    """An INTERNAL runtime cache keeps the producer's INT8 output, so mixing it with S16 must fail."""
    graph, rope = _make_runtime_rope_with_cache_configs(
        cos_visibility=QuantizationVisibility.INTERNAL,
        sin_visibility=QuantizationVisibility.EXPORT_WHEN_ACTIVE,
        producer_fp32=False,
    )

    with pytest.raises(ValueError, match=r"cos/sin cache inputs to share one dtype"):
        InsertQuantTypePattern().export(rope, graph)


def test_rope_skipped_runtime_cache_matches_a_fp32_producer():
    """Two skipped caches over FP32 producers stay FP32, which the device accepts."""
    graph, rope = _make_runtime_rope_with_cache_configs(
        cos_visibility=QuantizationVisibility.INTERNAL,
        sin_visibility=QuantizationVisibility.INTERNAL,
        producer_fp32=True,
    )

    InsertQuantTypePattern().export(rope, graph)

    assert rope.attributes["quant_type"] == EspQuantType.S8


def test_rope_runtime_cache_dtypes_must_match():
    graph, rope = _make_runtime_rope()
    config = EspdlQuantizer(graph).init_quantize_config(rope)
    config.input_quantization_config[1].state = QuantizationStates.FP32
    config.input_quantization_config[2] = _config(16, -15, RoundingPolicy.ROUND_HALF_EVEN)
    quantized = QuantableOperation(convert_from=rope, quantize_config=config, platform=TargetPlatform.ESPDL_INT8)
    _replace_operation(graph, quantized)

    with pytest.raises(ValueError, match=r"cos/sin cache inputs to share one dtype"):
        InsertQuantTypePattern().export(quantized, graph)


def _make_runtime_rope_with_producer_bits(*, cos_bits, sin_bits, cos_state, sin_state):
    """Runtime producers with explicit widths, plus cache configs in the given states."""
    graph, rope = _make_runtime_rope()
    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    rounding = RoundingPolicy.ROUND_HALF_EVEN
    for producer_name, bits in (("cos_runtime", cos_bits), ("sin_runtime", sin_bits)):
        producer = quantizer.quantize_operation(producer_name, platform=TargetPlatform.ESPDL_INT8)
        producer.output_quant_config[0] = _config(bits, -7 if bits == 8 else -15, rounding)
        _replace_operation(graph, producer)

    cache_configs = []
    for state in (cos_state, sin_state):
        cache_config = _config(16, -15, rounding)
        cache_config.state = state
        cache_configs.append(cache_config)
    config = OperationQuantizationConfig(
        input_quantization_configs=[_config(8, -7, rounding), *cache_configs],
        output_quantization_configs=[_config(8, -7, rounding)],
    )
    quantized = QuantableOperation(convert_from=rope, quantize_config=config, platform=TargetPlatform.ESPDL_INT8)
    _replace_operation(graph, quantized)
    return quantized


def _write_runtime_cache_dtypes(rope):
    """Dtype QuantVariableToIntPattern leaves on the two cache tensors."""
    ExporterPatternInfo().reset()
    try:
        for cache in rope.inputs[1:3]:
            producer = cache.source_op
            if isinstance(producer, QuantableOperation):
                QuantVariableToIntPattern().export(producer, None)
        QuantVariableToIntPattern().export(rope, None)
        return rope.inputs[1].dtype, rope.inputs[2].dtype
    finally:
        ExporterPatternInfo().reset()


def test_rope_fp32_cache_config_follows_mismatched_producer_dtypes():
    """A FP32 cache config is skipped, so INT8 and INT16 producers must not export."""
    rope = _make_runtime_rope_with_producer_bits(
        cos_bits=8,
        sin_bits=16,
        cos_state=QuantizationStates.FP32,
        sin_state=QuantizationStates.FP32,
    )

    with pytest.raises(ValueError, match=r"cos/sin cache inputs to share one dtype, got \['S8', 'S16'\]"):
        InsertQuantTypePattern().export(rope, None)
    assert _write_runtime_cache_dtypes(rope) == (DataType.INT8, DataType.INT16)


def test_rope_fp32_cache_config_matches_an_int16_producer():
    """A skipped FP32 config over an INT16 producer matches an exported S16 cache."""
    rope = _make_runtime_rope_with_producer_bits(
        cos_bits=16,
        sin_bits=16,
        cos_state=QuantizationStates.FP32,
        sin_state=QuantizationStates.ACTIVATED,
    )

    InsertQuantTypePattern().export(rope, None)
    assert rope.attributes["quant_type"] == EspQuantType.S8
    assert _write_runtime_cache_dtypes(rope) == (DataType.INT16, DataType.INT16)


def test_rope_reference_rejects_zero_head_dimension():
    _, rope = _make_quantized_rope(head_dim=2, seq_len=1)
    x = torch.empty(1, 2, 1, 0)
    cos = torch.empty(1, 1, 0)
    sin = torch.empty(1, 1, 0)

    with pytest.raises(ValueError, match="X dimensions must be positive"):
        rotary_embedding_reference(rope, [x, cos, sin])


def test_four_input_native_rope_reaches_explicit_backend_validation():
    """Socket arity must not fail before the executor explains 4 inputs are unsupported."""
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
    with pytest.raises(ValueError, match="expects exact 3 inputs"):
        espdl_rotary_embedding_forward(
            rope,
            [
                torch.zeros(1, 2, 3, 16),
                cos.value,
                sin.value,
                position_ids.value,
            ],
        )


def test_p4_rope_forward_keeps_device_values_and_float_backward():
    _, rope = _make_quantized_rope()
    x = torch.randn(1, 2, 3, 32, requires_grad=True)
    cos = torch.randn(1, 3, 16, requires_grad=True)
    sin = torch.randn(1, 3, 16, requires_grad=True)
    values = [x, cos, sin]
    with torch.no_grad():
        device_value = espdl_rotary_embedding_forward(rope, [tensor.detach() for tensor in values])
    out = espdl_rotary_embedding_forward(rope, values)
    assert torch.equal(out, device_value)
    out.sum().backward()
    references = [tensor.detach().requires_grad_(True) for tensor in values]
    rotary_embedding_reference(rope, references).sum().backward()
    for tensor, reference in zip(values, references):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.allclose(tensor.grad, reference.grad)


def test_p4_rope_ste_masks_nonfinite_reference_forward():
    _, rope = _make_quantized_rope()
    x = torch.randn(1, 2, 3, 32, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(1, 3, 16, dtype=torch.float32)
    cos[..., 0] = float("inf")
    cos.requires_grad_(True)
    sin = torch.randn(1, 3, 16, dtype=torch.float32, requires_grad=True)
    values = [x, cos, sin]

    with torch.no_grad():
        device_value = espdl_rotary_embedding_forward(rope, [tensor.detach() for tensor in values])
        reference = rotary_embedding_reference(rope, [tensor.detach() for tensor in values])
    assert not torch.isfinite(reference).all(), "test setup must produce non-finite fp32 reference"

    out = espdl_rotary_embedding_forward(rope, values)
    assert torch.equal(out.detach(), device_value)
    assert torch.isfinite(out).all()

    out.sum().backward()
    references = [tensor.detach().clone().requires_grad_(True) for tensor in values]
    rotary_embedding_reference(rope, references).sum().backward()
    for tensor, reference in zip(values, references):
        assert tensor.grad is not None
        assert reference.grad is not None
        assert torch.equal(~torch.isfinite(tensor.grad), ~torch.isfinite(reference.grad))


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

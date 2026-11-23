"""S3 and C LayerNorm use the shared integer kernel, with each platform's rounding."""

import pytest
import torch

from esp_ppq.core import NetworkFramework, QuantizationStates, RoundingPolicy, TargetPlatform
from esp_ppq.executor import TorchExecutor
from esp_ppq.executor.op.torch.default import LayerNorm_forward, _layernorm_quantization
from esp_ppq.IR import BaseGraph, QuantableOperation
from esp_ppq.parser.espdl.espdl_typedef import EspQuantType
from esp_ppq.parser.espdl.export_patterns import InsertQuantTypePattern
from esp_ppq.quantization.quantizer.EspdlQuantizer import (
    EspdlCHPreInt16Quantizer,
    EspdlCInt16Quantizer,
    EspdlCQuantizer,
    EspdlCW8A16Quantizer,
    EspdlHPreInt16Quantizer,
    EspdlInt16Quantizer,
    EspdlQuantizer,
    EspdlS3HPreInt16Quantizer,
    EspdlS3Int16Quantizer,
    EspdlS3Quantizer,
    EspdlS3W8A16Quantizer,
    EspdlW8A16Quantizer,
)


def _quantized_layernorm(platform, quantizer_cls, has_bias=True, scales=None):
    graph = BaseGraph(name="layernorm_s3", built_from=NetworkFramework.ONNX)
    x = graph.create_variable(name="x", shape=[1, 4])
    scale = graph.create_variable(name="scale", value=torch.ones(4), is_parameter=True)
    inputs = [x, scale]
    if has_bias:
        inputs.append(graph.create_variable(name="bias", value=torch.zeros(4), is_parameter=True))
    y = graph.create_variable(name="y", shape=[1, 4])
    operation = graph.create_operation(
        op_type="LayerNormalization",
        name="layernorm",
        platform=platform,
        inputs=inputs,
        outputs=[y],
        attributes={"axis": -1, "epsilon": 1e-5},
    )
    config = quantizer_cls(graph).init_quantize_config(operation)
    tensor_configs = config.input_quantization_config + config.output_quantization_config
    chosen = (0.125,) * len(tensor_configs) if scales is None else tuple(scales)
    if len(chosen) != len(tensor_configs):
        raise ValueError(f'expected {len(tensor_configs)} scales, got {len(chosen)}')
    for tensor_config, scale in zip(tensor_configs, chosen):
        tensor_config.scale = torch.tensor(scale)
        tensor_config.offset = torch.tensor(0.0)
        tensor_config.state = QuantizationStates.ACTIVATED
    return QuantableOperation(convert_from=operation, quantize_config=config, platform=platform)


def test_s3_int8_uses_integer_kernel_and_half_up():
    op = _quantized_layernorm(TargetPlatform.ESPDL_S3_INT8, EspdlS3Quantizer)
    quant = _layernorm_quantization(op, has_bias=True)
    assert quant is not None
    scales, configs, accum_int32 = quant
    assert scales == [0.125, 0.125, 0.125, 0.125]
    assert accum_int32 is True
    assert all(config.rounding == RoundingPolicy.ROUND_HALF_UP for config in configs)


def test_c_platforms_use_integer_kernel_and_half_up():
    cases = (
        (TargetPlatform.ESPDL_C_INT8, EspdlCQuantizer, True, 8),
        (TargetPlatform.ESPDL_C_INT16, EspdlCInt16Quantizer, False, 16),
        (TargetPlatform.ESPDL_C_H_PRE_INT16, EspdlCHPreInt16Quantizer, False, 16),
        (TargetPlatform.ESPDL_C_W8A16, EspdlCW8A16Quantizer, False, 16),
    )
    for platform, quantizer_cls, accum_int32, bits in cases:
        op = _quantized_layernorm(platform, quantizer_cls)
        quant = _layernorm_quantization(op, has_bias=True)
        assert quant is not None, platform
        _scales, configs, got_accum = quant
        assert got_accum is accum_int32, platform
        assert all(config.num_of_bits == bits for config in configs), platform
        assert all(config.rounding == RoundingPolicy.ROUND_HALF_UP for config in configs), platform


def test_s3_int16_keeps_int64_accumulation():
    op = _quantized_layernorm(TargetPlatform.ESPDL_S3_INT16, EspdlS3Int16Quantizer)
    quant = _layernorm_quantization(op, has_bias=True)
    assert quant is not None
    _scales, configs, accum_int32 = quant
    assert accum_int32 is False
    assert all(config.num_of_bits == 16 for config in configs)
    assert all(config.rounding == RoundingPolicy.ROUND_HALF_UP for config in configs)


def _plain_layernorm():
    graph = BaseGraph(name="layernorm_fp32", built_from=NetworkFramework.ONNX)
    x = graph.create_variable(name="x", shape=[2, 4])
    scale = graph.create_variable(name="scale", value=torch.ones(4), is_parameter=True)
    bias = graph.create_variable(name="bias", value=torch.zeros(4), is_parameter=True)
    y = graph.create_variable(name="y", shape=[2, 4])
    return graph.create_operation(
        op_type="LayerNormalization",
        name="layernorm",
        platform=TargetPlatform.FP32,
        inputs=[x, scale, bias],
        outputs=[y],
        attributes={"axis": -1, "epsilon": 1e-5},
    )


def test_float_layernorm_keeps_multidimensional_scale():
    op = _plain_layernorm()
    x = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3, 4)
    got = LayerNorm_forward(op, [x, weight, bias])
    expected = torch.nn.functional.layer_norm(x, weight.shape, weight, bias, 1e-5)
    assert torch.equal(got, expected)


def test_integer_layernorm_rejects_non_vector_scale():
    op = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    x = torch.randn(2, 4)
    weight = torch.randn(1, 4)
    bias = torch.randn(4)
    with pytest.raises(ValueError, match="only normalizes the last dimension"):
        LayerNorm_forward(op, [x, weight, bias])


def test_unquantized_layernorm_matches_torch_and_keeps_float64():
    op = _plain_layernorm()
    x = torch.randn(2, 4, dtype=torch.float64)
    weight = torch.randn(4, dtype=torch.float64)
    bias = torch.randn(4, dtype=torch.float64)
    got = LayerNorm_forward(op, [x, weight, bias])
    expected = torch.nn.functional.layer_norm(x, weight.shape, weight, bias, 1e-5)
    assert got.dtype == torch.float64
    assert torch.equal(got, expected)


def test_integer_forward_keeps_device_values_and_torch_backward():
    op = _quantized_layernorm(TargetPlatform.ESPDL_S3_INT8, EspdlS3Quantizer)
    x = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(4, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    with torch.no_grad():
        device_value = LayerNorm_forward(op, [x, weight, bias])
    y = LayerNorm_forward(op, [x, weight, bias])
    assert torch.equal(y, device_value)
    y.sum().backward()

    x_ref = x.detach().requires_grad_(True)
    weight_ref = weight.detach().requires_grad_(True)
    bias_ref = bias.detach().requires_grad_(True)
    torch.nn.functional.layer_norm(x_ref, weight_ref.shape, weight_ref, bias_ref, 1e-5).sum().backward()
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(weight.grad, weight_ref.grad)
    assert torch.allclose(bias.grad, bias_ref.grad)


def test_integer_layernorm_ste_masks_nonfinite_reference_forward():
    op = _quantized_layernorm(TargetPlatform.ESPDL_S3_INT8, EspdlS3Quantizer)
    x = torch.tensor([[0.5, float("inf"), -0.25, 0.0]], dtype=torch.float32, requires_grad=True)
    weight = torch.tensor([1.0, 0.5, 0.25, 1.0], dtype=torch.float32, requires_grad=True)
    bias = torch.tensor([0.125, 0.0, -0.125, 0.25], dtype=torch.float32, requires_grad=True)

    with torch.no_grad():
        device_value = LayerNorm_forward(op, [x.detach(), weight.detach(), bias.detach()])
        reference = torch.nn.functional.layer_norm(x.detach(), weight.shape, weight.detach(), bias.detach(), 1e-5)
    assert not torch.isfinite(reference).all(), "test setup must produce non-finite fp32 reference"

    y = LayerNorm_forward(op, [x, weight, bias])
    assert torch.equal(y.detach(), device_value)
    assert torch.isfinite(y).all()

    y.sum().backward()
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    ref = torch.nn.functional.layer_norm(x_ref, weight_ref.shape, weight_ref, bias_ref, 1e-5)
    ref.sum().backward()
    for tensor, reference_grad in ((x, x_ref.grad), (weight, weight_ref.grad), (bias, bias_ref.grad)):
        assert tensor.grad is not None
        assert reference_grad is not None
        assert torch.equal(~torch.isfinite(tensor.grad), ~torch.isfinite(reference_grad))


_LN_X = torch.tensor([[0.5, -0.25, 0.125, -0.125]])
_LN_WEIGHT = torch.tensor([1.0, 0.5, 0.25, 1.0])
_LN_BIAS = torch.tensor([0.125, 0.0, -0.125, 0.25])
# Frozen device codes at scale 0.125, half-even. Regenerating these from
# LayerNorm_forward would hide a change in the integer kernel.
_LN_CODES = {
    (8, True): torch.tensor([[1.625, -0.5, -0.125, -0.375]]),
    (8, False): torch.tensor([[1.5, -0.5, 0.0, -0.625]]),
    (16, True): torch.tensor([[1.625, -0.5, -0.125, -0.375]]),
    (16, False): torch.tensor([[1.5, -0.5, 0.0, -0.625]]),
}


def _assert_layernorm_ste(op, with_bias):
    x = _LN_X.clone().requires_grad_(True)
    weight = _LN_WEIGHT.clone().requires_grad_(True)
    bias = _LN_BIAS.clone().requires_grad_(True) if with_bias else None
    values = [x, weight] if bias is None else [x, weight, bias]
    with torch.no_grad():
        device_value = LayerNorm_forward(op, [t.detach() for t in values])
    y = LayerNorm_forward(op, values)
    assert torch.equal(y, device_value)
    y.sum().backward()
    tracked = [x, weight] if bias is None else [x, weight, bias]
    for tensor in tracked:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_p4_integer_layernorm_codes_and_ste():
    cases = (
        (8, True, TargetPlatform.ESPDL_INT8, EspdlQuantizer),
        (8, False, TargetPlatform.ESPDL_INT8, EspdlQuantizer),
        (16, True, TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer),
        (16, False, TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer),
    )
    for bits, with_bias, platform, quantizer_cls in cases:
        op = _quantized_layernorm(platform, quantizer_cls, has_bias=with_bias)
        values = [_LN_X, _LN_WEIGHT] if not with_bias else [_LN_X, _LN_WEIGHT, _LN_BIAS]
        got = LayerNorm_forward(op, values)
        assert torch.equal(got, _LN_CODES[(bits, with_bias)]), (bits, with_bias, got)
        _assert_layernorm_ste(op, with_bias)


# Halfway values at scale 0.125. half-even and half-up disagree on the third code.
# Frozen from the integer kernel; do not recompute them inside the test.
_LN_TIE_X = torch.tensor([[5.5625, 2.1875, 0.1875, -3.6875]])
_LN_TIE_EVEN = torch.tensor([[1.5, 0.125, -0.125, -1.125]])
_LN_TIE_UP = torch.tensor([[1.5, 0.125, -0.25, -1.125]])
# These codes sit outside the int8 range, so the int8 row clamps and the int16 row does not.
_LN_WIDE_X = torch.tensor([[80.0, -80.0, 20.0, -20.0]])
_LN_WIDE_WEIGHT = torch.ones(4)
_LN_WIDE_BIAS = torch.zeros(4)
_LN_WIDE_8 = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
_LN_WIDE_16 = torch.tensor([[1.375, -1.375, 0.375, -0.375]])


def test_platform_rounding_changes_layernorm_codes():
    cases = (
        (TargetPlatform.ESPDL_INT8, EspdlQuantizer, _LN_TIE_EVEN),
        (TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer, _LN_TIE_EVEN),
        (TargetPlatform.ESPDL_S3_INT8, EspdlS3Quantizer, _LN_TIE_UP),
        (TargetPlatform.ESPDL_S3_INT16, EspdlS3Int16Quantizer, _LN_TIE_UP),
    )
    for platform, quantizer_cls, expected in cases:
        op = _quantized_layernorm(platform, quantizer_cls, has_bias=True)
        got = LayerNorm_forward(op, [_LN_TIE_X, _LN_WEIGHT, _LN_BIAS])
        assert torch.equal(got, expected), (platform, got)


def test_int16_layernorm_codes_differ_from_int8_when_input_exceeds_int8():
    cases = (
        (8, TargetPlatform.ESPDL_INT8, EspdlQuantizer, _LN_WIDE_8),
        (16, TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer, _LN_WIDE_16),
        (8, TargetPlatform.ESPDL_S3_INT8, EspdlS3Quantizer, _LN_WIDE_8),
        (16, TargetPlatform.ESPDL_S3_INT16, EspdlS3Int16Quantizer, _LN_WIDE_16),
    )
    for bits, platform, quantizer_cls, expected in cases:
        op = _quantized_layernorm(platform, quantizer_cls, has_bias=True)
        got = LayerNorm_forward(op, [_LN_WIDE_X, _LN_WIDE_WEIGHT, _LN_WIDE_BIAS])
        assert torch.equal(got, expected), (bits, platform, got)
        assert not torch.equal(_LN_WIDE_8, _LN_WIDE_16)


# x, weight, bias, and output scales. Frozen from the integer kernel.
_LN_DISTINCT_SCALES = (0.03125, 0.0078125, 0.0625, 0.015625)
_LN_DISTINCT_X = torch.tensor([[0.03, 0.06, -0.02, -0.07]])
_LN_DISTINCT_WEIGHT = torch.tensor([0.3, 0.1, 0.2, 0.4])
_LN_DISTINCT_BIAS = torch.tensor([0.05, -0.05, 0.1, -0.1])
_LN_DISTINCT_CODES = torch.tensor([[0.25, 0.0625, 0.0, -0.625]])


def test_distinct_scales_are_read_from_the_matching_tensor():
    cases = (
        (TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer),
        (TargetPlatform.ESPDL_S3_INT16, EspdlS3Int16Quantizer),
    )
    for platform, quantizer_cls in cases:
        op = _quantized_layernorm(platform, quantizer_cls, scales=_LN_DISTINCT_SCALES)
        got = LayerNorm_forward(op, [_LN_DISTINCT_X, _LN_DISTINCT_WEIGHT, _LN_DISTINCT_BIAS])
        assert torch.equal(got, _LN_DISTINCT_CODES), (platform, got)
        swapped = (_LN_DISTINCT_SCALES[1], _LN_DISTINCT_SCALES[0], *_LN_DISTINCT_SCALES[2:])
        exchanged = _quantized_layernorm(platform, quantizer_cls, scales=swapped)
        changed = LayerNorm_forward(exchanged, [_LN_DISTINCT_X, _LN_DISTINCT_WEIGHT, _LN_DISTINCT_BIAS])
        assert not torch.equal(changed, got), platform


def _sqrt_newton_inv_std(arg: torch.Tensor) -> torch.Tensor:
    """Reference-only fallback: 1 / sqrt_newton(x) with EN=1e-5 from esp-dl."""
    x = arg.float()
    zero_mask = x == 0
    result = torch.where(zero_mask, torch.zeros_like(x), x)
    for _ in range(64):
        last = result
        result = torch.where(zero_mask, torch.zeros_like(result), (last + x / last) * 0.5)
        if torch.max(torch.abs(result - last)).item() <= 1e-5:
            break
    return torch.where(zero_mask, torch.zeros_like(result), 1.0 / result)


# This row keeps quantized LayerNorm on the edge where fast_inv_sqrt and
# sqrt_newton produce different integer outputs. It guards inv_std path choice.
_LN_INVSTD_X = torch.tensor([[0.6057000160, 0.6049000025, 0.6032000184, 0.6148999929]])
_LN_INVSTD_W = torch.tensor([0.9519000053, 0.5159999728, 1.4578000307, 0.5740000010])
_LN_INVSTD_B = torch.tensor([0.2108000070, 0.2378000021, 0.2072000057, 0.1017000005])
_LN_INVSTD_FAST_CODES = torch.tensor([[0.03515625, 0.14453125, -1.1328125, 0.84375]])
_LN_INVSTD_NEWTON_CODES = torch.tensor([[0.03515625, 0.14453125, -1.1328125, 0.83984375]])


def test_frozen_vector_distinguishes_fast_inv_sqrt_from_sqrt_newton(monkeypatch):
    op = _quantized_layernorm(TargetPlatform.ESPDL_INT16, EspdlInt16Quantizer, has_bias=True)
    for config in op.input_quant_config + op.output_quant_config:
        config.scale = torch.tensor(2.0**-8)
        config.offset = torch.tensor(0.0)
    op.attributes["epsilon"] = 1e-5

    fast = LayerNorm_forward(op, [_LN_INVSTD_X, _LN_INVSTD_W, _LN_INVSTD_B])
    assert torch.equal(fast, _LN_INVSTD_FAST_CODES)

    monkeypatch.setattr("esp_ppq.executor.op.torch.default._layernorm_inv_std", _sqrt_newton_inv_std)
    newton = LayerNorm_forward(op, [_LN_INVSTD_X, _LN_INVSTD_W, _LN_INVSTD_B])
    assert torch.equal(newton, _LN_INVSTD_NEWTON_CODES)
    assert not torch.equal(newton, fast)


@pytest.mark.parametrize(
    ("platform", "quantizer_cls", "dtype"),
    [
        (TargetPlatform.ESPDL_H_PRE_INT16, EspdlHPreInt16Quantizer, torch.float64),
        (TargetPlatform.ESPDL_W8A16, EspdlW8A16Quantizer, torch.float32),
        (TargetPlatform.ESPDL_S3_H_PRE_INT16, EspdlS3HPreInt16Quantizer, torch.float64),
        (TargetPlatform.ESPDL_S3_W8A16, EspdlS3W8A16Quantizer, torch.float32),
    ],
)
def test_remaining_platforms_match_distinct_scale_codes(platform, quantizer_cls, dtype):
    op = _quantized_layernorm(platform, quantizer_cls, scales=_LN_DISTINCT_SCALES)
    values = [tensor.to(dtype) for tensor in (_LN_DISTINCT_X, _LN_DISTINCT_WEIGHT, _LN_DISTINCT_BIAS)]
    got = LayerNorm_forward(op, values)
    assert got.dtype == dtype
    assert torch.equal(got, _LN_DISTINCT_CODES.to(dtype)), (platform, got)


def test_p4_int8_still_uses_half_even():
    op = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    quant = _layernorm_quantization(op, has_bias=True)
    assert quant is not None
    _scales, configs, accum_int32 = quant
    assert accum_int32 is True
    assert all(config.rounding == RoundingPolicy.ROUND_HALF_EVEN for config in configs)


def test_mixed_bitwidth_layernorm_raises_instead_of_silent_float_fallback():
    legal = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    exported_legal = InsertQuantTypePattern().export(legal, BaseGraph(name="unused"))
    assert exported_legal.attributes["quant_type"] == EspQuantType.S8

    op = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    op.input_quant_config[0].num_of_bits = 16
    with pytest.raises(ValueError, match="share one bit width"):
        LayerNorm_forward(op, [_LN_X, _LN_WEIGHT, _LN_BIAS])
    with pytest.raises(ValueError, match="share one bit width"):
        InsertQuantTypePattern().export(op, BaseGraph(name="unused"))

    output_wide = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    output_wide.output_quant_config[0].num_of_bits = 16
    with pytest.raises(ValueError, match="share one bit width"):
        InsertQuantTypePattern().export(output_wide, BaseGraph(name="unused"))

    op_unsupported = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    for config in op_unsupported.input_quant_config + op_unsupported.output_quant_config:
        config.num_of_bits = 32
    with pytest.raises(ValueError, match="share one bit width"):
        LayerNorm_forward(op_unsupported, [_LN_X, _LN_WEIGHT, _LN_BIAS])
    with pytest.raises(ValueError, match="share one bit width"):
        InsertQuantTypePattern().export(op_unsupported, BaseGraph(name="unused"))


def test_partial_fp32_layernorm_stays_float_in_simulation_and_refuses_export():
    """A tensor left at FP32 keeps the float reference. Export must not stamp an integer kernel."""
    op = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    op.input_quant_config[1].state = QuantizationStates.FP32
    x = torch.tensor([[0.5, -0.25, 1.0, -1.5]])
    weight = torch.tensor([1.0, 0.5, 1.5, 0.25])
    bias = torch.tensor([0.0, 0.1, -0.1, 0.2])
    got = LayerNorm_forward(op, [x, weight, bias])
    expected = torch.nn.functional.layer_norm(x, weight.shape, weight, bias, 1e-5)
    assert torch.equal(got, expected)
    with pytest.raises(ValueError, match="float reference"):
        InsertQuantTypePattern().export(op, BaseGraph(name="unused"))
    assert "quant_type" not in op.attributes


def test_custom_tqc_bitwidth_override_aligns_simulation_and_export():
    op = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    for config in op.input_quant_config + op.output_quant_config:
        config.num_of_bits = 16
        config.quant_min = -32768
        config.quant_max = 32767
    quant = _layernorm_quantization(op, has_bias=True)
    assert quant is not None
    _scales, _configs, accum_int32 = quant
    assert accum_int32 is False
    got = LayerNorm_forward(op, [_LN_WIDE_X, _LN_WIDE_WEIGHT, _LN_WIDE_BIAS])
    assert torch.equal(got, _LN_WIDE_16)
    exported = InsertQuantTypePattern().export(op, BaseGraph(name="unused"))
    assert exported.attributes["quant_type"] == EspQuantType.S16


def test_env_var_enables_layernorm_emulation_on_non_espdl_platform(monkeypatch):
    op = _quantized_layernorm(TargetPlatform.ESPDL_INT8, EspdlQuantizer)
    op.platform = TargetPlatform.PPL_CUDA_INT8

    monkeypatch.delenv("ESPDL_LAYERNORM_DEVICE_EMULATE", raising=False)
    assert _layernorm_quantization(op, has_bias=True) is None
    float_out = LayerNorm_forward(op, [_LN_X, _LN_WEIGHT, _LN_BIAS])
    expected_float = torch.nn.functional.layer_norm(_LN_X, _LN_WEIGHT.shape, _LN_WEIGHT, _LN_BIAS, 1e-5)
    assert torch.equal(float_out, expected_float)
    assert not torch.equal(float_out, _LN_CODES[(8, True)])

    monkeypatch.setenv("ESPDL_LAYERNORM_DEVICE_EMULATE", "1")
    assert _layernorm_quantization(op, has_bias=True) is not None
    emulated_out = LayerNorm_forward(op, [_LN_X, _LN_WEIGHT, _LN_BIAS])
    assert torch.equal(emulated_out, _LN_CODES[(8, True)])


def _build_layernorm_relu_graph(platform, quantizer_cls, scale_value=0.125):
    graph = BaseGraph(name="layernorm_executor", built_from=NetworkFramework.ONNX)
    x = graph.create_variable(name="x", shape=[1, 4])
    weight = graph.create_variable(name="scale", value=_LN_WEIGHT.clone(), is_parameter=True)
    bias = graph.create_variable(name="bias", value=_LN_BIAS.clone(), is_parameter=True)
    ln_out = graph.create_variable(name="ln_out", shape=[1, 4])
    y = graph.create_variable(name="y", shape=[1, 4])
    graph.inputs["x"] = x
    graph.outputs["ln_out"] = ln_out
    graph.outputs["y"] = y
    ln_op = graph.create_operation(
        op_type="LayerNormalization",
        name="layernorm",
        platform=platform,
        inputs=[x, weight, bias],
        outputs=[ln_out],
        attributes={"axis": -1, "epsilon": 1e-5},
    )
    relu_op = graph.create_operation(
        op_type="Relu",
        name="relu",
        platform=platform,
        inputs=[ln_out],
        outputs=[y],
    )
    quantizer = quantizer_cls(graph)
    for plain in (ln_op, relu_op):
        cfg = quantizer.init_quantize_config(plain)
        for tqc in cfg.input_quantization_config + cfg.output_quantization_config:
            tqc.scale = torch.tensor(scale_value)
            tqc.offset = torch.tensor(0.0)
            tqc.state = QuantizationStates.ACTIVATED
        quant_op = QuantableOperation(convert_from=plain, quantize_config=cfg, platform=platform)
        graph.operations[plain.name] = quant_op
        for var in quant_op.inputs + quant_op.outputs:
            if var.source_op is plain:
                var.source_op = quant_op
            for idx, dest in enumerate(var.dest_ops):
                if dest is plain:
                    var.dest_ops[idx] = quant_op
    return graph


def test_layernorm_runs_through_torch_executor_and_feeds_downstream_op():
    graph = _build_layernorm_relu_graph(TargetPlatform.ESPDL_INT8, EspdlQuantizer, scale_value=0.125)
    executor = TorchExecutor(graph=graph, device="cpu")
    # Perturb x within half a step so input fake-quant is exercised before LayerNorm_forward.
    ln_out, relu_out = executor.forward(inputs=[_LN_X + 0.02], output_names=["ln_out", "y"])
    expected = _LN_CODES[(8, True)]
    assert torch.equal(ln_out, expected)
    assert torch.equal(relu_out, torch.relu(expected))

    hpre_graph = _build_layernorm_relu_graph(
        TargetPlatform.ESPDL_H_PRE_INT16, EspdlHPreInt16Quantizer, scale_value=0.125
    )
    hpre_executor = TorchExecutor(graph=hpre_graph, device="cpu")
    hpre_ln, hpre_relu = hpre_executor.forward(inputs=[_LN_X + 0.02], output_names=["ln_out", "y"])
    expected_16 = _LN_CODES[(16, True)]
    assert torch.equal(hpre_ln, expected_16)
    assert torch.equal(hpre_relu, torch.relu(expected_16))

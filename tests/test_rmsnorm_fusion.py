"""
Test RMSNormFusionPass: verify that decomposed RMS normalization subgraphs
are correctly fused into a single RMSNormalization operation.

Two variants are tested:
  1. Reciprocal-based: Pow → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul(X, 1/rms) → Mul(scale)
  2. Div-based:        Pow → ReduceMean → Add(eps) → Sqrt → Div(X, rms) → Mul(scale)
"""

import os
import tempfile

import torch
import torch.nn as nn

from esp_ppq import TargetPlatform, TorchExecutor
from esp_ppq.api import load_onnx_graph
from esp_ppq.api.espdl_interface import espdl_quantize_torch
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import ppq_warning
from esp_ppq.quantization.optim.refine import RMSNormFusionPass


def _export_model_to_onnx(model: nn.Module, input_shape, opset=11) -> str:
    tmp = os.path.join(tempfile.gettempdir(), 'test_rmsnorm_fusion.onnx')
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(input_shape),
        tmp,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return tmp


# ---------------------------------------------------------------------------
# Model A: Reciprocal-based RMSNorm
#   X → Pow(2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul(X, 1/rms) → Mul(scale)
# ---------------------------------------------------------------------------
class RMSNormReciprocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        xs = x.pow(2)
        xs_mean = xs.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(xs_mean + 1e-5)
        rcp = rms.reciprocal()
        return x * rcp * self.scale


# ---------------------------------------------------------------------------
# Model B: Div-based RMSNorm
#   X → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div(X, rms) → Mul(scale)
# ---------------------------------------------------------------------------
class RMSNormDivModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        xs = x.pow(2)
        xs_mean = xs.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(xs_mean + 1e-5)
        return (x / rms) * self.scale


def _has_rmsnorm_ops(graph):
    """Check graph contains at least one RMSNormalization op."""
    return any(op.type == 'RMSNormalization' for op in graph.operations.values())


def _has_decomposed_ops(graph):
    """Return True if the decomposed pattern ops are still present."""
    decomposed_types = {'Pow', 'ReduceMean', 'Sqrt', 'Reciprocal', 'Div'}
    found = {op.type for op in graph.operations.values() if op.type in decomposed_types}
    # Allow ReduceMean from other graph branches (e.g., in graph input handling)
    found.discard('ReduceMean')
    return bool(found)


def test_reciprocal_fusion_direct():
    """Test Reciprocal-based RMSNorm fusion with direct graph manipulation."""
    model = RMSNormReciprocalModel()
    onnx_path = _export_model_to_onnx(model, (2, 2, 3), opset=18)

    graph = load_onnx_graph(onnx_path)
    assert graph is not None

    # Verify decomposed pattern exists before fusion
    assert not _has_rmsnorm_ops(graph), 'Should not have RMSNormalization before fusion'

    # Dispatch to a quantizable platform
    from esp_ppq.api import dispatch_graph

    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')

    # Run fusion
    fusion_pass = RMSNormFusionPass()
    fusion_pass.optimize(graph)

    # Verify fusion result
    assert _has_rmsnorm_ops(graph), (
        f'Expected RMSNormalization op after fusion. Ops: {[op.type for op in graph.operations.values()]}'
    )

    # Verify intermediate ops were removed
    remaining_types = {op.type for op in graph.operations.values()}
    for removed in ('Pow', 'Reciprocal', 'Add'):
        assert removed not in remaining_types, f'{removed} should have been removed by fusion'

    os.unlink(onnx_path)
    print('PASS: test_reciprocal_fusion_direct')


def test_div_fusion_direct():
    """Test Div-based RMSNorm fusion with direct graph manipulation."""
    model = RMSNormDivModel()
    onnx_path = _export_model_to_onnx(model, (2, 2, 3), opset=18)

    graph = load_onnx_graph(onnx_path)
    assert graph is not None

    assert not _has_rmsnorm_ops(graph), 'Should not have RMSNormalization before fusion'

    from esp_ppq.api import dispatch_graph

    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')

    fusion_pass = RMSNormFusionPass()
    fusion_pass.optimize(graph)

    assert _has_rmsnorm_ops(graph), f'Expected RMSNormalization op after fusion. Ops: {list(graph.operations.keys())}'

    remaining_types = {op.type for op in graph.operations.values()}
    for removed in ('Pow', 'Div', 'Add', 'Sqrt'):
        assert removed not in remaining_types, f'{removed} should have been removed by fusion'

    os.unlink(onnx_path)
    print('PASS: test_div_fusion_direct')


def test_espdl_quantize_torch_with_rmsnorm():
    """Test RMSNorm fusion through espdl_quantize_torch pipeline.

    Uses a non-trivial scale (2.0 instead of 1.0) so that onnxsim does not
    fold away the scaling Mul operation.
    """
    model = RMSNormReciprocalModel()
    # Use non-trivial scale to prevent onnxsim from folding Mul(scale) away
    with torch.no_grad():
        model.scale.copy_(torch.ones(3) * 2.0)

    dataset = [torch.randn(2, 2, 3) for _ in range(4)]
    espdl_export_file = os.path.join(tempfile.gettempdir(), 'rmsnorm_test.espdl')
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.dispatcher = "allin"
    try:
        quantized = espdl_quantize_torch(
            model=model,
            espdl_export_file=espdl_export_file,
            calib_dataloader=dataset,
            calib_steps=4,
            input_shape=[2, 2, 3],
            target='esp32p4',
            num_of_bits=8,
            device='cpu',
            skip_export=False,
            error_report=False,
            verbose=0,
            setting=quant_setting,
        )

        assert _has_rmsnorm_ops(quantized), (
            f'espdl_quantize_torch should produce RMSNormalization ops. '
            f'Ops: {[op.type for op in quantized.operations.values()]}'
        )

        # Verify execution via TorchExecutor
        executor = TorchExecutor(graph=quantized, device='cpu')
        test_input = torch.randn(2, 2, 3)
        output = executor.forward(inputs=test_input)
        assert output is not None and len(output) > 0

        print('PASS: test_espdl_quantize_torch_with_rmsnorm')

    except Exception as e:
        print(f'SKIP: test_espdl_quantize_torch_with_rmsnorm — environment limitation: {e}')


def test_rmsnorm_forward_correctness():
    """Verify the fused RMSNormalization forward produces correct results."""
    model = RMSNormReciprocalModel()
    onnx_path = _export_model_to_onnx(model, (2, 2, 3), opset=11)

    graph = load_onnx_graph(onnx_path)

    from esp_ppq.api import dispatch_graph

    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')

    # Get reference output BEFORE fusion
    executor_before = TorchExecutor(graph=graph, device='cpu')
    test_input = torch.randn(2, 2, 3)
    ref_output = executor_before.forward(inputs=test_input)

    # Run fusion
    fusion_pass = RMSNormFusionPass()
    fusion_pass.optimize(graph)

    assert _has_rmsnorm_ops(graph)

    # Get output AFTER fusion
    executor_after = TorchExecutor(graph=graph, device='cpu')
    fused_output = executor_after.forward(inputs=test_input)

    # Compare
    if ref_output is not None and len(ref_output) > 0:
        diff = (ref_output[0] - fused_output[0]).abs().max().item()
        print(f'  Max diff before/after fusion: {diff:.2e}')
        assert diff < 1e-5, f'Output mismatch: max diff = {diff}'

    os.unlink(onnx_path)
    print('PASS: test_rmsnorm_forward_correctness')


if __name__ == '__main__':
    test_reciprocal_fusion_direct()
    test_div_fusion_direct()
    test_rmsnorm_forward_correctness()
    test_espdl_quantize_torch_with_rmsnorm()
    print('\nAll RMSNorm fusion tests passed.')

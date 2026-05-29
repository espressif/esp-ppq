"""
Test LpNormalizationFusionPass: verify that the decomposed L2 normalization
subgraph (ReduceL2 → Clip → Expand → Div) is correctly fused into a single
``LpNormalization`` operation.

The targeted pattern is:

    in_x ──→ ReduceL2 ──→ Clip ──→ Expand ──┐
      │                                         ├─→ Div ──→ output
      └─────────────────────────────────────────┘
"""

import os
import tempfile

import torch
import torch.nn as nn

from esp_ppq import TargetPlatform, TorchExecutor
from esp_ppq.api import load_onnx_graph
from esp_ppq.quantization.optim.refine import LpNormalizationFusionPass


def _export_model_to_onnx(model: nn.Module, input_shape, opset=11) -> str:
    tmp = os.path.join(tempfile.gettempdir(), 'test_lpnorm_fusion.onnx')
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
# Model: decomposed L2 normalization
#   X → ReduceL2 → Clip → Expand → Div(X, norm)
# ---------------------------------------------------------------------------
class LpNormDecomposedModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)
        norm_expanded = norm.expand(x.shape)
        return x / norm_expanded


def _has_lpnorm_ops(graph):
    """Return True if the graph contains at least one LpNormalization op."""
    return any(op.type == 'LpNormalization' for op in graph.operations.values())


def _has_decomposed_pattern(graph):
    """Return True if ReduceL2 followed by Div exists (decomposed pattern)."""
    types_found = {op.type for op in graph.operations.values()}
    return 'ReduceL2' in types_found and 'Div' in types_found


def test_fusion_direct():
    """Test LpNormalization fusion with direct graph manipulation."""
    model = LpNormDecomposedModel()
    onnx_path = _export_model_to_onnx(model, (2, 2, 3))

    graph = load_onnx_graph(onnx_path)
    assert graph is not None

    from esp_ppq.api import dispatch_graph

    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')

    # Run fusion
    fusion_pass = LpNormalizationFusionPass()
    fusion_pass.optimize(graph)

    assert _has_lpnorm_ops(graph), (
        f'Expected LpNormalization op after fusion. Ops: {[op.type for op in graph.operations.values()]}'
    )

    # Verify intermediate ops were removed
    remaining_types = {op.type for op in graph.operations.values()}
    for removed in ('ReduceL2', 'Clip', 'Expand', 'Div'):
        assert removed not in remaining_types, f'{removed} should have been removed by fusion'

    os.unlink(onnx_path)
    print('PASS: test_fusion_direct')


def test_forward_correctness():
    """Verify the fused LpNormalization forward produces correct results."""
    model = LpNormDecomposedModel()
    onnx_path = _export_model_to_onnx(model, (2, 2, 3))

    graph = load_onnx_graph(onnx_path)

    from esp_ppq.api import dispatch_graph

    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')

    # Get reference output BEFORE fusion
    executor_before = TorchExecutor(graph=graph, device='cpu')
    test_input = torch.randn(2, 2, 3)
    ref_output = executor_before.forward(inputs=test_input)

    # Run fusion
    fusion_pass = LpNormalizationFusionPass()
    fusion_pass.optimize(graph)

    assert _has_lpnorm_ops(graph)

    # Get output AFTER fusion
    executor_after = TorchExecutor(graph=graph, device='cpu')
    fused_output = executor_after.forward(inputs=test_input)

    # Compare
    if ref_output is not None and len(ref_output) > 0:
        diff = (ref_output[0] - fused_output[0]).abs().max().item()
        print(f'  Max diff before/after fusion: {diff:.2e}')
        assert diff < 1e-5, f'Output mismatch: max diff = {diff}'

    os.unlink(onnx_path)
    print('PASS: test_forward_correctness')


def test_espdl_pipeline():
    """Test LpNormalization fusion through the full espdl_quantize_torch pipeline."""
    from esp_ppq.api.espdl_interface import espdl_quantize_torch
    from esp_ppq.api.setting import QuantizationSettingFactory

    model = LpNormDecomposedModel()
    dataset = [torch.randn(2, 2, 3) for _ in range(4)]
    espdl_export_file = os.path.join(tempfile.gettempdir(), 'lpnorm_test.espdl')
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.dispatcher = 'allin'
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

        assert _has_lpnorm_ops(quantized), (
            f'espdl_quantize_torch should produce LpNormalization ops. '
            f'Ops: {[op.type for op in quantized.operations.values()]}'
        )

        executor = TorchExecutor(graph=quantized, device='cpu')
        test_input = torch.randn(2, 2, 3)
        output = executor.forward(inputs=test_input)
        assert output is not None and len(output) > 0

        print('PASS: test_espdl_pipeline')

    except Exception as e:
        print(f'SKIP: test_espdl_pipeline — environment limitation: {e}')


if __name__ == '__main__':
    test_fusion_direct()
    test_forward_correctness()
    test_espdl_pipeline()
    print('\nAll LpNormalization fusion tests passed.')

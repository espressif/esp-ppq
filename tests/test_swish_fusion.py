"""
Test Swish fusion: verify that Sigmoid + Mul patterns are correctly fused
into a Swish operation across different layout / passthrough combinations.

Three fusion mechanisms are exercised:

  * operation-level:  ``GraphFormatter.fuse_swish()``, runs during
    ``load_onnx_graph()`` (``FORMATTER_FUSE_SWISH = True``) — replaces
    Sigmoid + Mul with a single Swish op.

  * quantization-level: ``QuantizeFusionPass`` (when ``'Swish'`` is in
    ``activation_types``) — dominates quantization configs so the fused
    ops share a single quantization point.

Tested variants:
  1. Direct:            Conv → Sigmoid → Mul
  2. Transpose-through: Conv → Transpose → Sigmoid → Mul
  3. Reshape-through:   Conv → Reshape → Sigmoid → Mul
  4. Squeeze-through:   Conv → Squeeze → Sigmoid → Mul
  5. Multi-passthrough: Conv → Transpose → Reshape → Sigmoid → Mul
  6. No computing op:   GraphInput → Sigmoid → Mul  (should NOT fuse)
  7. End-to-end:        espdl_quantize_torch pipeline
"""

import os
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper

from esp_ppq import TargetPlatform, TorchExecutor
from esp_ppq.api import dispatch_graph, load_onnx_graph
from esp_ppq.api.espdl_interface import espdl_quantize_torch
from esp_ppq.core import QuantizationStates
from esp_ppq.IR import QuantableOperation
from esp_ppq.quantization.optim.refine import QuantizeFusionPass


# ---------------------------------------------------------------------------
# ONNX model builders — manually constructed so PyTorch's exporter doesn't
# pre-fuse ``x * sigmoid(x)`` into a native Swish node.
# ---------------------------------------------------------------------------
def _make_swish_onnx(op_before_mul=None, op_before_sigmoid=None):
    """Build an ONNX model with decomposed Swish::

        [optional op_before_mul / op_before_sigmoid]  →  conv  →
            ├─→ Sigmoid ─→
            └─────────────→ Mul → output

    *op_before_mul* / *op_before_sigmoid* are inserted between conv and the
    swish subgraph when given.
    """
    nodes = []
    initializers = []
    value_infos = []

    conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    conv_b = np.zeros(4).astype(np.float32)
    initializers += [
        helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 3, 3, 3], conv_w.flatten()),
        helper.make_tensor('conv_b', TensorProto.FLOAT, [4], conv_b.flatten()),
    ]

    nodes.append(
        helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    )
    value_infos.append(helper.make_tensor_value_info('conv_out', TensorProto.FLOAT, [1, 4, 8, 8]))
    current_sig_var = 'conv_out'
    current_mul_var = 'conv_out'

    if op_before_sigmoid:
        op_type, attr = op_before_sigmoid
        n = helper.make_node(op_type, [current_sig_var], ['pre_sig_out'], **attr)
        nodes.append(n)
        value_infos.append(helper.make_tensor_value_info('pre_sig_out', TensorProto.FLOAT, None))
        current_sig_var = 'pre_sig_out'
        current_mul_var = 'pre_sig_out'

    if op_before_mul:
        op_type, attr = op_before_mul
        n = helper.make_node(op_type, [current_mul_var], ['pre_mul_out'], **attr)
        nodes.append(n)
        value_infos.append(helper.make_tensor_value_info('pre_mul_out', TensorProto.FLOAT, None))
        current_mul_var = 'pre_mul_out'

    nodes.append(helper.make_node('Sigmoid', [current_sig_var], ['sig_out']))
    value_infos.append(helper.make_tensor_value_info('sig_out', TensorProto.FLOAT, None))
    nodes.append(helper.make_node('Mul', [current_mul_var, 'sig_out'], ['output']))

    graph = helper.make_graph(
        nodes,
        'swish_test',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    return model


def _save_and_load(model):
    tmp = os.path.join(tempfile.gettempdir(), 'test_swish_fusion.onnx')
    onnx.save(model, tmp)
    graph = load_onnx_graph(tmp)
    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')
    os.unlink(tmp)
    return graph


def _quantize_ops(graph):
    """Quantize all applicable ops."""
    from esp_ppq.IR import GraphReplacer, QuantableGraph
    from esp_ppq.IR.base.command import QuantizeOperationCommand
    from esp_ppq.lib.quant import Quantizer

    quantizer = Quantizer(TargetPlatform.ESPDL_INT8, graph)
    for op_name, op in list(graph.operations.items()):
        if op.platform == TargetPlatform.UNSPECIFIED:
            if op.type in quantizer.quant_operation_types:
                op.platform = quantizer.target_platform
            else:
                op.platform = TargetPlatform.FP32
        if op.platform not in {TargetPlatform.FP32, TargetPlatform.SOI}:
            processor = QuantableGraph(GraphReplacer(graph))
            processor(
                QuantizeOperationCommand(
                    op_name=op_name,
                    target_platform=op.platform,
                    config=quantizer.init_quantize_config(op),
                )
            )


def _is_overlapped(config) -> bool:
    return config.state == QuantizationStates.OVERLAPPED


def _has_swish_op(graph) -> bool:
    return any(op.type == 'Swish' for op in graph.operations.values())


# ===================================================================
# Test 1: Direct Swish — Conv → Sigmoid → Mul
# ===================================================================
def test_direct_swish():
    onnx_model = _make_swish_onnx()
    graph = _save_and_load(onnx_model)

    # Operation-level fusion (format_graph) should have created a Swish op
    assert _has_swish_op(graph), (
        f'Expected Swish op after format_graph. Ops: {[op.type for op in graph.operations.values()]}'
    )
    assert not any(op.type == 'Sigmoid' for op in graph.operations.values()), (
        'Sigmoid should have been removed by op-level fusion'
    )
    assert not any(op.type == 'Mul' for op in graph.operations.values()), (
        'Mul should have been removed by op-level fusion'
    )

    # Quantize and verify quantization-level fusion
    _quantize_ops(graph)

    fusion_pass = QuantizeFusionPass(
        activation_type={'Relu', 'Clip', 'Swish'},
        fuse_activation=True,
        fuse_passive_op=True,
        fuse_relu_clip=False,
    )
    fusion_pass.optimize(graph)

    # Swish op should have its output config activated (not OVERLAPPED)
    swish = [op for op in graph.operations.values() if op.type == 'Swish'][0]
    assert isinstance(swish, QuantableOperation)
    # Conv output → dominated by Swish output
    conv = [op for op in graph.operations.values() if op.type == 'Conv'][0]
    assert _is_overlapped(conv.config.output_quantization_config[0]), (
        f'Conv output should be OVERLAPPED, got {conv.config.output_quantization_config[0].state}'
    )

    print('PASS: test_direct_swish')


# ===================================================================
# Test 2: Transpose-through — Conv → Transpose → Sigmoid → Mul
# ===================================================================
def test_transpose_swish():
    nodes = []
    initializers = []
    conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    conv_b = np.zeros(4).astype(np.float32)
    initializers += [
        helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 3, 3, 3], conv_w.flatten()),
        helper.make_tensor('conv_b', TensorProto.FLOAT, [4], conv_b.flatten()),
    ]
    nodes.append(
        helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    )
    # One shared Transpose feeds BOTH Sigmoid and Mul
    nodes.append(helper.make_node('Transpose', ['conv_out'], ['trans_out'], perm=[0, 2, 3, 1]))
    nodes.append(helper.make_node('Sigmoid', ['trans_out'], ['sig_out']))
    nodes.append(helper.make_node('Mul', ['trans_out', 'sig_out'], ['output']))

    graph_def = helper.make_graph(
        nodes,
        'transpose_swish',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=initializers,
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 18)])
    graph = _save_and_load(model)

    assert _has_swish_op(graph), (
        f'Expected Swish op after format_graph. Ops: {[op.type for op in graph.operations.values()]}'
    )
    assert not any(op.type == 'Sigmoid' for op in graph.operations.values())
    assert not any(op.type == 'Mul' for op in graph.operations.values())
    assert any(op.type == 'Transpose' for op in graph.operations.values()), (
        'Transpose should still be present between Conv and Swish'
    )

    print('PASS: test_transpose_swish')


# ===================================================================
# Test 3: Reshape-through — Conv → Reshape → Sigmoid → Mul
# ===================================================================
def test_reshape_swish():
    nodes = []
    initializers = []
    conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    conv_b = np.zeros(4).astype(np.float32)
    initializers += [
        helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 3, 3, 3], conv_w.flatten()),
        helper.make_tensor('conv_b', TensorProto.FLOAT, [4], conv_b.flatten()),
        helper.make_tensor('shape', TensorProto.INT64, [4], [1, 4, 64, 1]),
    ]
    nodes.append(
        helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    )
    nodes.append(helper.make_node('Reshape', ['conv_out', 'shape'], ['reshaped']))
    nodes.append(helper.make_node('Sigmoid', ['reshaped'], ['sig_out']))
    nodes.append(helper.make_node('Mul', ['reshaped', 'sig_out'], ['output']))

    graph_def = helper.make_graph(
        nodes,
        'reshape_swish',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 64, 1])],
        initializer=initializers,
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 18)])
    graph = _save_and_load(model)
    assert _has_swish_op(graph), (
        f'Expected Swish op after format_graph. Ops: {[op.type for op in graph.operations.values()]}'
    )
    assert not any(op.type == 'Sigmoid' for op in graph.operations.values())

    print('PASS: test_reshape_swish')


# ===================================================================
# Test 4: Squeeze-through — Conv → Squeeze → Sigmoid → Mul
# ===================================================================
def test_squeeze_swish():
    nodes = []
    initializers = []
    conv_w = np.random.randn(4, 3, 1, 1).astype(np.float32)
    conv_b = np.zeros(4).astype(np.float32)
    initializers += [
        helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 3, 1, 1], conv_w.flatten()),
        helper.make_tensor('conv_b', TensorProto.FLOAT, [4], conv_b.flatten()),
        helper.make_tensor('axes', TensorProto.INT64, [1], [2]),
    ]
    nodes.append(
        helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    )
    nodes.append(helper.make_node('Squeeze', ['conv_out', 'axes'], ['squeezed']))
    nodes.append(helper.make_node('Sigmoid', ['squeezed'], ['sig_out']))
    nodes.append(helper.make_node('Mul', ['squeezed', 'sig_out'], ['output']))

    graph_def = helper.make_graph(
        nodes,
        'squeeze_swish',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 1, 1])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 1])],
        initializer=initializers,
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 18)])
    graph = _save_and_load(model)
    assert _has_swish_op(graph), (
        f'Expected Swish op after format_graph. Ops: {[op.type for op in graph.operations.values()]}'
    )

    print('PASS: test_squeeze_swish')


# ===================================================================
# Test 5: Multi-passthrough — Conv → Transpose → Reshape → Sigmoid → Mul
# ===================================================================
def test_multi_passthrough_swish():
    # Build: Conv → Transpose → Reshape → Sigmoid → Mul(Reshape_out, Sigmoid_out)
    nodes = []
    initializers = []
    conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    conv_b = np.zeros(4).astype(np.float32)
    initializers += [
        helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 3, 3, 3], conv_w.flatten()),
        helper.make_tensor('conv_b', TensorProto.FLOAT, [4], conv_b.flatten()),
        helper.make_tensor('shape', TensorProto.INT64, [3], [1, 32, 4]),
    ]
    nodes.append(
        helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    )
    nodes.append(helper.make_node('Transpose', ['conv_out'], ['trans_out'], perm=[0, 2, 3, 1]))
    nodes.append(helper.make_node('Reshape', ['trans_out', 'shape'], ['reshaped']))
    nodes.append(helper.make_node('Sigmoid', ['reshaped'], ['sig_out']))
    nodes.append(helper.make_node('Mul', ['reshaped', 'sig_out'], ['output']))

    graph_def = helper.make_graph(
        nodes,
        'multi_passthrough',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 4])],
        initializer=initializers,
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 18)])
    graph = _save_and_load(model)

    assert _has_swish_op(graph), (
        f'Expected Swish op after format_graph. Ops: {[op.type for op in graph.operations.values()]}'
    )
    assert not any(op.type == 'Sigmoid' for op in graph.operations.values())
    assert not any(op.type == 'Mul' for op in graph.operations.values())

    print('PASS: test_multi_passthrough_swish')


# ===================================================================
# Test 6: No computing op upstream — GraphInput → Sigmoid → Mul
# ===================================================================
def test_no_computing_op_swish():
    nodes = [
        helper.make_node('Sigmoid', ['input'], ['sig_out']),
        helper.make_node('Mul', ['input', 'sig_out'], ['output']),
    ]
    graph_def = helper.make_graph(
        nodes,
        'no_comp_swish',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 18)])
    graph = _save_and_load(model)

    # Should NOT fuse — no computing/passthrough op upstream of Sigmoid
    assert not _has_swish_op(graph), (
        f'Should NOT fuse Swish with graph input. Ops: {[op.type for op in graph.operations.values()]}'
    )
    assert any(op.type == 'Sigmoid' for op in graph.operations.values())

    print('PASS: test_no_computing_op_swish')


# ===================================================================
# Test 7: End-to-end pipeline via espdl_quantize_torch
# ===================================================================
def test_espdl_pipeline_swish():
    """End-to-end test using espdl_quantize_onnx (the ONNX path)."""
    from esp_ppq.api.espdl_interface import espdl_quantize_onnx

    nodes = []
    initializers = []
    conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    conv_b = np.zeros(4).astype(np.float32)
    initializers += [
        helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 3, 3, 3], conv_w.flatten()),
        helper.make_tensor('conv_b', TensorProto.FLOAT, [4], conv_b.flatten()),
    ]
    nodes.append(
        helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    )
    nodes.append(helper.make_node('Transpose', ['conv_out'], ['trans_out'], perm=[0, 2, 3, 1]))
    nodes.append(helper.make_node('Sigmoid', ['trans_out'], ['sig_out']))
    nodes.append(helper.make_node('Mul', ['trans_out', 'sig_out'], ['output']))

    graph_def = helper.make_graph(
        nodes,
        'pipeline_swish',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=initializers,
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 18)])
    tmp = os.path.join(tempfile.gettempdir(), 'swish_pipeline.onnx')
    onnx.save(model, tmp)

    dataset = [torch.randn(1, 3, 8, 8) for _ in range(4)]
    espdl_file = os.path.join(tempfile.gettempdir(), 'swish_pipeline.espdl')

    quantized = espdl_quantize_onnx(
        onnx_import_file=tmp,
        espdl_export_file=espdl_file,
        calib_dataloader=dataset,
        calib_steps=4,
        input_shape=[1, 3, 8, 8],
        target='esp32p4',
        num_of_bits=8,
        device='cpu',
        skip_export=True,
        error_report=False,
        verbose=0,
    )

    assert _has_swish_op(quantized), (
        f'Expected Swish op after full pipeline. Ops: {[op.type for op in quantized.operations.values()]}'
    )
    assert not any(op.type == 'Sigmoid' for op in quantized.operations.values())

    for op in quantized.operations.values():
        if op.type in ('Transpose',):
            assert isinstance(op, QuantableOperation)
            assert _is_overlapped(op.config.output_quantization_config[0]), (
                f'{op.type} output should be OVERLAPPED, got {op.config.output_quantization_config[0].state}'
            )

    executor = TorchExecutor(graph=quantized, device='cpu')
    test_input = torch.randn(1, 3, 8, 8)
    output = executor.forward(inputs=test_input)
    assert output is not None and len(output) > 0

    os.unlink(tmp)
    print('PASS: test_espdl_pipeline_swish')


if __name__ == '__main__':
    test_direct_swish()
    test_transpose_swish()
    test_reshape_swish()
    test_squeeze_swish()
    test_multi_passthrough_swish()
    test_no_computing_op_swish()
    test_espdl_pipeline_swish()
    print('\nAll Swish fusion tests passed.')

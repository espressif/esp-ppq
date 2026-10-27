"""Tests for format-stage InverseSigmoid fusion and ESP-DL export."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from esp_ppq import TorchExecutor
from esp_ppq.api import load_graph, load_onnx_graph, quantize_onnx_model
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import DataType, TargetPlatform
from esp_ppq.executor.op.torch.default import inverse_sigmoid_reference
from esp_ppq.IR import BaseGraph, GraphCommand, GraphCommandType, GraphFormatter, GraphMerger
from esp_ppq.parser.espdl_exporter import EspdlExporter

EPS = 1e-5


def _const(name, value, tensor_type, shape=()):
    count = int(np.prod(shape)) if shape else 1
    return helper.make_tensor(name, tensor_type, list(shape), [float(value)] * count)


def _inverse_sigmoid_onnx(
    *,
    eps=EPS,
    denom_eps=None,
    one=1.0,
    outer_clip=True,
    outer_min=0.0,
    outer_max=1.0,
    inner_max=None,
    dynamic_inner_max=False,
    explicit_missing_inner_max=False,
    fanout_on_ratio=False,
    extra_xc_consumer=False,
    extra_one_consumer=False,
    shared_eps=False,
    one_is_graph_output=False,
    input_shape=(1, 8),
    one_shape=(),
    tensor_type=TensorProto.FLOAT,
    name='inverse_sigmoid_fusion_test',
):
    """Build the DETR ``inverse_sigmoid`` lowering, with optional mutations."""
    if dynamic_inner_max and (inner_max is not None or explicit_missing_inner_max):
        raise ValueError("dynamic_inner_max cannot be combined with another inner max form")
    if denom_eps is None:
        denom_eps = eps
    eps_n_name = 'ceps' if shared_eps else 'ceps_n'
    eps_d_name = 'ceps' if shared_eps else 'ceps_d'
    inits = [
        _const('c0', outer_min, tensor_type),
        _const('c1', outer_max, tensor_type),
        _const(eps_n_name, eps, tensor_type),
        _const('one', one, tensor_type, shape=one_shape),
    ]
    if not shared_eps:
        inits.append(_const(eps_d_name, denom_eps, tensor_type))
    if inner_max is not None:
        inits.append(_const('cmax', inner_max, tensor_type))

    nodes = []
    if outer_clip:
        nodes.append(helper.make_node('Clip', ['input', 'c0', 'c1'], ['xc']))
        xc = 'xc'
    else:
        xc = 'input'

    if dynamic_inner_max:
        max_input = ['dynamic_max']
    elif inner_max is not None:
        max_input = ['cmax']
    else:
        max_input = [''] if explicit_missing_inner_max else []
    nodes.append(helper.make_node('Clip', [xc, eps_n_name, *max_input], ['num']))
    nodes.append(helper.make_node('Sub', ['one', xc], ['one_sub']))
    nodes.append(helper.make_node('Clip', ['one_sub', eps_d_name, *max_input], ['denom']))
    nodes.append(helper.make_node('Div', ['num', 'denom'], ['ratio']))
    nodes.append(helper.make_node('Log', ['ratio'], ['inverse_sigmoid_out']))

    output_shape = list(np.broadcast_shapes(input_shape, one_shape))
    outputs = [helper.make_tensor_value_info('inverse_sigmoid_out', tensor_type, output_shape)]
    if fanout_on_ratio:
        nodes.append(helper.make_node('Identity', ['ratio'], ['ratio_tap']))
        outputs.append(helper.make_tensor_value_info('ratio_tap', tensor_type, output_shape))
    if extra_xc_consumer:
        outputs.append(helper.make_tensor_value_info(xc, tensor_type, list(input_shape)))
    if extra_one_consumer:
        inits.append(_const('one_tap_bias', one, tensor_type, shape=one_shape))
        nodes.append(helper.make_node('Add', ['input', 'one_tap_bias'], ['one_tap']))
        outputs.append(helper.make_tensor_value_info('one_tap', tensor_type, output_shape))
    if one_is_graph_output:
        outputs.append(helper.make_tensor_value_info('one', tensor_type, list(one_shape)))

    inputs = [helper.make_tensor_value_info('input', tensor_type, list(input_shape))]
    if dynamic_inner_max:
        inputs.append(helper.make_tensor_value_info('dynamic_max', tensor_type, []))

    graph = helper.make_graph(
        nodes,
        name,
        inputs,
        outputs,
        initializer=inits,
    )
    opset = helper.make_opsetid('', 18)
    return helper.make_model(graph, opset_imports=[opset], producer_name=name)


def _load(model):
    tmp = os.path.join(tempfile.gettempdir(), f'{model.producer_name}.onnx')
    onnx.save(model, tmp)
    graph = load_graph(tmp)
    GraphFormatter(graph)(GraphCommand(GraphCommandType.CONVERT_TO_TENSOR))
    os.unlink(tmp)
    return graph


def _load_formatted(model):
    tmp = os.path.join(tempfile.gettempdir(), f'{model.producer_name}_formatted.onnx')
    onnx.save(model, tmp)
    graph = load_onnx_graph(tmp)
    os.unlink(tmp)
    return graph


def _op_types(graph):
    return {op.type for op in graph.operations.values()}


def _assert_graph_coherent(graph):
    for name, op in graph.operations.items():
        assert op.name == name
        for var in op.inputs:
            assert graph.variables.get(var.name) is var
            assert op in var.dest_ops
        for var in op.outputs:
            assert graph.variables.get(var.name) is var
            assert var.source_op is op
    for name, var in graph.variables.items():
        assert var.name == name
        if var.source_op is not None:
            assert graph.operations.get(var.source_op.name) is var.source_op
            assert var in var.source_op.outputs
        for op in var.dest_ops:
            assert graph.operations.get(op.name) is op
            assert var in op.inputs
    for boundary in (graph.inputs, graph.outputs):
        for name, var in boundary.items():
            assert graph.variables.get(name) is var


def _share_parameter_input(graph, canonical, consumer, input_index):
    previous = consumer.inputs[input_index]
    if consumer in previous.dest_ops:
        previous.dest_ops.remove(consumer)
    consumer.inputs[input_index] = canonical
    if consumer not in canonical.dest_ops:
        canonical.dest_ops.append(consumer)
    if (
        previous.name in graph.variables
        and previous.source_op is None
        and not previous.dest_ops
        and previous.name not in graph.inputs
        and previous.name not in graph.outputs
    ):
        graph.remove_variable(previous)


def _probe_input():
    """Values that hit every branch of clip / eps-floor / log."""
    return torch.tensor(
        [[-0.2, 0.0, 1e-6, EPS, 0.5, 1.0 - EPS, 1.0, 1.5]],
        dtype=torch.float32,
    )


def _native_inverse_sigmoid_graph(dtype, eps=EPS, platform=TargetPlatform.UNSPECIFIED):
    graph = BaseGraph(name='native_inverse_sigmoid')
    x = graph.create_variable(name='input')
    y = graph.create_variable(name='output')
    x.dtype = dtype
    y.dtype = dtype
    x.shape = [1, 8]
    y.shape = [1, 8]
    graph.create_operation(
        op_type='InverseSigmoid',
        name='inverse_sigmoid',
        attributes={'eps': eps},
        platform=platform,
        inputs=[x],
        outputs=[y],
    )
    graph.mark_variable_as_graph_input(x)
    graph.mark_variable_as_graph_output(y)
    return graph


def test_fusion_rewrites_canonical_pattern():
    graph = _load(_inverse_sigmoid_onnx())
    GraphMerger(graph).fuse_inverse_sigmoid()
    types = _op_types(graph)
    assert 'InverseSigmoid' in types, f'expected InverseSigmoid, got {types}'
    for leftover in ('Clip', 'Div', 'Log', 'Sub'):
        assert leftover not in types, f'{leftover} should have been folded: {types}'
    print('PASS: test_fusion_rewrites_canonical_pattern')


def test_fusion_survives_fused_name_collision():
    graph = _load(_inverse_sigmoid_onnx(name='inverse_sigmoid_name_collision'))
    log_op = next(op for op in graph.operations.values() if op.type == 'Log')
    colliding_name = f'InverseSigmoid_{log_op.name}'
    graph.create_operation(op_type='Identity', name=colliding_name)

    GraphMerger(graph).fuse_inverse_sigmoid()

    fused = next(op for op in graph.operations.values() if op.type == 'InverseSigmoid')
    assert fused.name != colliding_name
    assert graph.operations[colliding_name].type == 'Identity'
    assert 'Log' not in _op_types(graph)


def test_format_graph_fuses_canonical_pattern():
    graph = _load_formatted(_inverse_sigmoid_onnx(name='format_graph_inverse_sigmoid'))
    assert _op_types(graph) == {'InverseSigmoid'}
    _assert_graph_coherent(graph)


def test_fused_node_inherits_uniform_platform():
    graph = _load(_inverse_sigmoid_onnx(name='uniform_fp32_platform'))
    for op in graph.operations.values():
        op.platform = TargetPlatform.FP32
    GraphMerger(graph).fuse_inverse_sigmoid()
    fused = next(op for op in graph.operations.values() if op.type == 'InverseSigmoid')
    assert fused.platform == TargetPlatform.FP32


def test_fused_matches_reference_and_unfused():
    model = _inverse_sigmoid_onnx()
    graph = _load(model)
    x = _probe_input()

    before = TorchExecutor(graph=graph, device='cpu').forward(inputs=x)[0]
    GraphMerger(graph).fuse_inverse_sigmoid()
    after = TorchExecutor(graph=graph, device='cpu').forward(inputs=x)[0]
    ref = inverse_sigmoid_reference(x, EPS)

    d_fuse = (after - before).abs().max().item()
    d_ref = (after - ref).abs().max().item()
    print(f'  max|fused-unfused|={d_fuse:.3e}  max|fused-ref|={d_ref:.3e}')
    assert d_fuse == 0.0, f'fusion changed values: {d_fuse}'
    assert d_ref == 0.0, f'fused InverseSigmoid drifted from reference: {d_ref}'
    print('PASS: test_fused_matches_reference_and_unfused')


def test_cpp_kernel_formula_matches_reference():
    """Replicate dl_module_inverse_sigmoid.cpp in float32 numpy."""
    x = _probe_input().numpy()
    eps = np.float32(EPS)
    xc = np.clip(x, np.float32(0.0), np.float32(1.0))
    num = np.where(xc > eps, xc, eps).astype(np.float32)
    one_sub = (np.float32(1.0) - xc).astype(np.float32)
    denom = np.where(one_sub > eps, one_sub, eps).astype(np.float32)
    cpp = np.log(num / denom).astype(np.float32)
    ref = inverse_sigmoid_reference(torch.from_numpy(x), float(EPS)).numpy()
    d = np.max(np.abs(cpp - ref))
    print(f'  max|cpp-formula - inverse_sigmoid_reference|={d:.3e}')
    assert d == 0.0, f'C++ formula drifted from inverse_sigmoid_reference: {d}'
    print('PASS: test_cpp_kernel_formula_matches_reference')


def test_refuse_fanout_on_ratio():
    graph = _load(_inverse_sigmoid_onnx(fanout_on_ratio=True))
    GraphMerger(graph).fuse_inverse_sigmoid()
    types = _op_types(graph)
    assert 'InverseSigmoid' not in types, f'fan-out on Div output must not fuse: {types}'
    assert 'Div' in types and 'Log' in types
    print('PASS: test_refuse_fanout_on_ratio')


def test_refuse_restrictive_inner_clip_max():
    graph = _load(_inverse_sigmoid_onnx(inner_max=0.5))
    GraphMerger(graph).fuse_inverse_sigmoid()
    types = _op_types(graph)
    assert 'InverseSigmoid' not in types, f'Clip(eps, 0.5) must not fuse: {types}'
    print('PASS: test_refuse_restrictive_inner_clip_max')


def test_inner_clip_max_one_is_noop_and_fuses():
    graph = _load(_inverse_sigmoid_onnx(inner_max=1.0))
    GraphMerger(graph).fuse_inverse_sigmoid()
    types = _op_types(graph)
    assert 'InverseSigmoid' in types, f'Clip(eps, 1) is a no-op on [0,1] and must fuse: {types}'
    print('PASS: test_inner_clip_max_one_is_noop_and_fuses')


def test_refuse_dynamic_inner_clip_max_without_changing_values():
    graph = _load(
        _inverse_sigmoid_onnx(
            dynamic_inner_max=True,
            name='dynamic_inner_max',
        )
    )
    x = _probe_input()
    dynamic_max = torch.tensor(0.5, dtype=torch.float32)
    before = TorchExecutor(graph=graph, device='cpu').forward(inputs=[x, dynamic_max])[0]

    GraphMerger(graph).fuse_inverse_sigmoid()

    assert 'InverseSigmoid' not in _op_types(graph)
    assert {'Clip', 'Sub', 'Div', 'Log'} <= _op_types(graph)
    after = TorchExecutor(graph=graph, device='cpu').forward(inputs=[x, dynamic_max])[0]
    assert torch.equal(after, before)
    fused_semantics = inverse_sigmoid_reference(x, EPS)
    assert not torch.equal(before, fused_semantics)
    _assert_graph_coherent(graph)


def test_explicit_missing_inner_clip_max_still_fuses():
    graph = _load(
        _inverse_sigmoid_onnx(
            explicit_missing_inner_max=True,
            name='explicit_missing_inner_max',
        )
    )
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' in _op_types(graph)


def test_refuse_missing_outer_clip():
    graph = _load(_inverse_sigmoid_onnx(outer_clip=False))
    GraphMerger(graph).fuse_inverse_sigmoid()
    types = _op_types(graph)
    assert 'InverseSigmoid' not in types, f'missing Clip(0,1) must not fuse: {types}'
    print('PASS: test_refuse_missing_outer_clip')


def test_refuse_approximate_one():
    graph = _load(_inverse_sigmoid_onnx(one=1.0 - 5e-7, name='approximate_one'))
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' not in _op_types(graph)


def test_refuse_approximately_equal_eps():
    graph = _load(_inverse_sigmoid_onnx(denom_eps=EPS + 5e-10, name='approximate_eps'))
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' not in _op_types(graph)


def test_refuse_approximate_outer_clip_bound():
    graph = _load(_inverse_sigmoid_onnx(outer_min=5e-10, name='approximate_outer_bound'))
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' not in _op_types(graph)


def test_refuse_non_scalar_one_even_when_shape_preserving():
    graph = _load(_inverse_sigmoid_onnx(one_shape=(1, 8), name='shape_preserving_one'))
    x = _probe_input()
    executor = TorchExecutor(graph=graph, device='cpu')
    executor.tracing_operation_meta(inputs=x)
    before = executor.forward(inputs=x)[0]
    GraphMerger(graph).fuse_inverse_sigmoid()
    after = TorchExecutor(graph=graph, device='cpu').forward(inputs=x)[0]
    assert 'InverseSigmoid' not in _op_types(graph)
    assert {'Clip', 'Sub', 'Div', 'Log'} <= _op_types(graph)
    assert torch.equal(after, before)
    _assert_graph_coherent(graph)


def test_refuse_shape_expanding_all_one_tensor():
    graph = _load(
        _inverse_sigmoid_onnx(
            input_shape=(1, 1),
            one_shape=(1, 4),
            name='shape_expanding_one',
        )
    )
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' not in _op_types(graph)
    assert {'Clip', 'Sub', 'Div', 'Log'} <= _op_types(graph)
    _assert_graph_coherent(graph)


def test_two_phase_cleanup_preserves_shared_one_parameter():
    graph = _load(_inverse_sigmoid_onnx(extra_one_consumer=True, name='shared_one'))
    x = _probe_input()
    before = TorchExecutor(graph=graph, device='cpu').forward(inputs=x)
    sub = next(op for op in graph.operations.values() if op.type == 'Sub')
    add = next(op for op in graph.operations.values() if op.type == 'Add')
    one = sub.inputs[0]
    _share_parameter_input(graph, one, add, 1)
    GraphMerger(graph).fuse_inverse_sigmoid()
    after = TorchExecutor(graph=graph, device='cpu').forward(inputs=x)
    assert all(torch.equal(actual, expected) for actual, expected in zip(after, before))
    assert graph.variables[one.name] is one
    assert one.value is not None
    assert one.dest_ops == [add]
    _assert_graph_coherent(graph)


def test_two_phase_cleanup_removes_shared_dead_eps_once():
    graph = _load(_inverse_sigmoid_onnx(shared_eps=True, name='shared_eps'))
    clips = [op for op in graph.operations.values() if op.type == 'Clip']
    num_clip = next(op for op in clips if op.inputs[0].source_op is not None and op.inputs[0].source_op.type == 'Clip')
    denom_clip = next(op for op in clips if op.inputs[0].source_op is not None and op.inputs[0].source_op.type == 'Sub')
    eps = num_clip.inputs[1]
    _share_parameter_input(graph, eps, denom_clip, 1)
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' in _op_types(graph)
    assert eps.name not in graph.variables
    _assert_graph_coherent(graph)


def test_two_phase_cleanup_preserves_parameter_graph_output():
    graph = _load(_inverse_sigmoid_onnx(one_is_graph_output=True, name='parameter_graph_output'))
    GraphMerger(graph).fuse_inverse_sigmoid()
    assert 'InverseSigmoid' in _op_types(graph)
    assert graph.outputs['one'] is graph.variables['one']
    assert graph.variables['one'].value is not None
    assert graph.variables['one'].dest_ops == []
    _assert_graph_coherent(graph)


def test_keep_outer_clip_when_xc_has_extra_consumer():
    graph = _load(_inverse_sigmoid_onnx(extra_xc_consumer=True))
    GraphMerger(graph).fuse_inverse_sigmoid()
    types = _op_types(graph)
    assert 'InverseSigmoid' in types, f'expected InverseSigmoid, got {types}'
    assert 'Clip' in types, 'outer Clip(0,1) must stay when xc is also a graph output'
    for leftover in ('Div', 'Log', 'Sub'):
        assert leftover not in types, f'{leftover} should have been folded: {types}'
    print('PASS: test_keep_outer_clip_when_xc_has_extra_consumer')


def test_eps_float32_canonical():
    """Python 1e-5 (float64) must clamp as 1e-5f, matching the device attribute."""
    x = torch.tensor([[1e-5]], dtype=torch.float32)
    # Values strictly between float32(1e-5) and float64(1e-5) do not exist
    # as float32, so the interesting check is: clamp(min=1e-5) on a tensor
    # that equals 1e-5f is a no-op, and the reference uses that bit pattern.
    y = inverse_sigmoid_reference(x, 1e-5)
    y32 = inverse_sigmoid_reference(x, float(np.float32(1e-5)))
    assert (y - y32).abs().max().item() == 0.0
    print('PASS: test_eps_float32_canonical')


def test_executor_rejects_non_fp32_inverse_sigmoid():
    graph = _native_inverse_sigmoid_graph(DataType.FP16)
    executor = TorchExecutor(graph=graph, device='cpu')
    with pytest.raises(RuntimeError) as error:
        executor.forward(inputs=_probe_input().half())
    assert isinstance(error.value.__cause__, TypeError)
    assert 'unsupported execution dtype' in str(error.value.__cause__)


@pytest.mark.parametrize(
    'platform',
    [TargetPlatform.ESPDL_H_PRE_INT16, TargetPlatform.FP32],
)
def test_executor_keeps_h_pre_int16_float64_simulation_convention(platform):
    graph = _native_inverse_sigmoid_graph(
        DataType.FP32,
        platform=platform,
    )
    output = TorchExecutor(graph=graph, device='cpu').forward(inputs=_probe_input().double())[0]
    assert output.dtype == torch.float64
    expected = inverse_sigmoid_reference(_probe_input(), EPS).double()
    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    'platform',
    [TargetPlatform.ESPDL_INT8, TargetPlatform.ESPDL_H_PRE_INT16],
)
def test_quantized_inverse_sigmoid_keeps_fp32_logical_contract(platform):
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, f'inverse_sigmoid_{platform.name}.onnx')
        onnx.save(
            _inverse_sigmoid_onnx(
                name=f'quantized_{platform.name}',
            ),
            onnx_path,
        )
        sample = [_probe_input()]
        graph = quantize_onnx_model(
            onnx_import_file=onnx_path,
            calib_dataloader=[sample, [sample[0].clone()]],
            calib_steps=2,
            input_shape=None,
            platform=platform,
            inputs=sample,
            setting=QuantizationSettingFactory.espdl_setting(),
            device='cpu',
            verbose=0,
        )
        inverse_sigmoid = next(op for op in graph.operations.values() if op.type == 'InverseSigmoid')
        assert inverse_sigmoid.inputs[0].dtype == DataType.FP32
        assert inverse_sigmoid.outputs[0].dtype == DataType.FP32
        EspdlExporter().export(
            file_path=os.path.join(tmp, 'inverse_sigmoid.espdl'),
            graph=graph,
            export_config=False,
        )


def test_inverse_sigmoid_fusion_is_owned_by_format_graph():
    graph = _load_formatted(_inverse_sigmoid_onnx(name='inverse_sigmoid_fp32_prepare'))
    assert 'InverseSigmoid' in _op_types(graph)


if __name__ == '__main__':
    test_fusion_rewrites_canonical_pattern()
    test_format_graph_fuses_canonical_pattern()
    test_fused_node_inherits_uniform_platform()
    test_fused_matches_reference_and_unfused()
    test_cpp_kernel_formula_matches_reference()
    test_refuse_fanout_on_ratio()
    test_refuse_restrictive_inner_clip_max()
    test_inner_clip_max_one_is_noop_and_fuses()
    test_refuse_dynamic_inner_clip_max_without_changing_values()
    test_explicit_missing_inner_clip_max_still_fuses()
    test_refuse_missing_outer_clip()
    test_refuse_approximate_one()
    test_refuse_approximately_equal_eps()
    test_refuse_approximate_outer_clip_bound()
    test_refuse_non_scalar_one_even_when_shape_preserving()
    test_refuse_shape_expanding_all_one_tensor()
    test_two_phase_cleanup_preserves_shared_one_parameter()
    test_two_phase_cleanup_removes_shared_dead_eps_once()
    test_two_phase_cleanup_preserves_parameter_graph_output()
    test_keep_outer_clip_when_xc_has_extra_consumer()
    test_eps_float32_canonical()
    test_executor_rejects_non_fp32_inverse_sigmoid()
    test_executor_keeps_h_pre_int16_float64_simulation_convention(TargetPlatform.ESPDL_H_PRE_INT16)
    test_executor_keeps_h_pre_int16_float64_simulation_convention(TargetPlatform.FP32)
    test_inverse_sigmoid_fusion_is_owned_by_format_graph()
    print('\nAll InverseSigmoid fusion / alignment tests passed.')

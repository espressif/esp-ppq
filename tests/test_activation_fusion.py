import pytest
import torch

from esp_ppq.core import TargetPlatform
from esp_ppq.IR import BaseGraph
from esp_ppq.parser.espdl.export_patterns import FuseReluLikePattern
from esp_ppq.quantization.optim.refine import QuantizeFusionPass
from esp_ppq.quantization.quantizer.EspdlQuantizer import EspdlQuantizer


def _make_quantized_conv_relu_graph(expose_pre_activation: bool):
    graph = BaseGraph(name='conv_relu_graph_output')
    graph_input = graph.create_variable(name='input')
    weight = graph.create_variable(
        name='weight',
        value=torch.ones(1, 1, 1, 1),
        is_parameter=True,
    )
    pre_activation = graph.create_variable(name='pre_activation')
    post_activation = graph.create_variable(name='post_activation')
    graph.create_operation(
        op_type='Conv',
        name='conv',
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[graph_input, weight],
        outputs=[pre_activation],
    )
    graph.create_operation(
        op_type='Relu',
        name='relu',
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[pre_activation],
        outputs=[post_activation],
    )
    graph.mark_variable_as_graph_input(graph_input)
    if expose_pre_activation:
        graph.mark_variable_as_graph_output(pre_activation)
    graph.mark_variable_as_graph_output(post_activation)

    quantizer = EspdlQuantizer(graph)
    conv = quantizer.quantize_operation('conv', platform=TargetPlatform.ESPDL_INT8)
    relu = quantizer.quantize_operation('relu', platform=TargetPlatform.ESPDL_INT8)
    return graph, conv, relu


@pytest.mark.parametrize(
    ('fuse_activation', 'fuse_relu_clip'),
    [
        (True, False),
        (False, True),
    ],
)
def test_quantize_fusion_preserves_pre_activation_graph_output(fuse_activation, fuse_relu_clip):
    graph, conv, relu = _make_quantized_conv_relu_graph(expose_pre_activation=True)
    conv_output_config = conv.config.output_quantization_config[0]
    relu_input_config = relu.config.input_quantization_config[0]

    QuantizeFusionPass(
        activation_type={'Relu'},
        fuse_activation=fuse_activation,
        fuse_passive_op=False,
        fuse_relu_clip=fuse_relu_clip,
    ).optimize(graph)

    assert conv_output_config.dominated_by is conv_output_config
    assert relu_input_config.dominated_by is relu_input_config
    assert list(graph.outputs) == ['pre_activation', 'post_activation']
    assert graph.outputs['pre_activation'] is conv.outputs[0]
    assert graph.outputs['post_activation'] is relu.outputs[0]


def test_quantize_fusion_still_merges_non_boundary_activation():
    graph, conv, relu = _make_quantized_conv_relu_graph(expose_pre_activation=False)
    relu_output_config = relu.config.output_quantization_config[0]

    QuantizeFusionPass(
        activation_type={'Relu'},
        fuse_activation=True,
        fuse_passive_op=False,
        fuse_relu_clip=False,
    ).optimize(graph)

    assert conv.config.output_quantization_config[0].dominated_by is relu_output_config
    assert relu.config.input_quantization_config[0].dominated_by is relu_output_config


def test_export_fusion_preserves_pre_activation_graph_output():
    graph, conv, relu = _make_quantized_conv_relu_graph(expose_pre_activation=True)
    original_config = conv.config

    FuseReluLikePattern().export(conv, graph)

    assert set(graph.operations) == {'conv', 'relu'}
    assert list(graph.outputs) == ['pre_activation', 'post_activation']
    assert graph.outputs['pre_activation'] is conv.outputs[0]
    assert graph.outputs['post_activation'] is relu.outputs[0]
    assert conv.config is original_config


def test_export_fusion_still_merges_non_boundary_relu():
    graph, conv, _ = _make_quantized_conv_relu_graph(expose_pre_activation=False)

    FuseReluLikePattern().export(conv, graph)

    assert set(graph.operations) == {'conv'}
    assert set(graph.outputs) == {'post_activation'}
    assert 'pre_activation' not in graph.variables
    assert conv.outputs == [graph.outputs['post_activation']]
    assert graph.outputs['post_activation'].source_op is conv
    assert conv.attributes['activation'] == 'Relu'

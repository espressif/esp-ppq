from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from esp_ppq.api import load_onnx_graph, quantize_onnx_model
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import (
    DataType,
    NetworkFramework,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    QuantizationVisibility,
    RoundingPolicy,
    TargetPlatform,
)
from esp_ppq.executor.op.torch.default import TopK_forward
from esp_ppq.IR import BaseGraph, Operation, Opset
from esp_ppq.parser.espdl_exporter import EspdlExporter
from esp_ppq.quantization.quantizer.base import BaseQuantizer
from esp_ppq.quantization.quantizer.EspdlQuantizer import EspdlQuantizer, finalize_espdl_topk_for_export


def _topk_graph(
    *,
    opset=13,
    shape=(1, 8),
    k=3,
    k_parameter=True,
    attributes=None,
    platform=TargetPlatform.ESPDL_INT8,
):
    graph = BaseGraph(name="topk_contract", built_from=NetworkFramework.ONNX)
    x = graph.create_variable("x")
    x.shape, x.dtype = list(shape), DataType.FP32

    if k_parameter:
        k_var = graph.create_variable(
            "k",
            value=torch.tensor([k], dtype=torch.int64),
            is_parameter=True,
        )
    else:
        k_var = graph.create_variable("k")
        k_var.shape, k_var.dtype = [1], DataType.INT64

    axis = (attributes or {}).get("axis", -1)
    normalized_axis = axis % len(shape) if isinstance(axis, int) and shape else 0
    output_shape = list(shape)
    output_shape[normalized_axis] = k
    values = graph.create_variable("values")
    values.shape, values.dtype = output_shape, DataType.FP32
    indices = graph.create_variable("indices")
    indices.shape, indices.dtype = output_shape, DataType.INT64

    op_inputs = [x] if opset < 10 else [x, k_var]
    op_attributes = dict(attributes or {})
    if opset < 10:
        op_attributes.setdefault("k", k)
    op = graph.create_operation(
        op_type="TopK",
        name="topk",
        attributes=op_attributes,
        platform=platform,
        inputs=op_inputs,
        outputs=[values, indices],
    )
    op.opset = Opset(version=opset)
    graph.mark_variable_as_graph_input(x)
    if not k_parameter and opset >= 10:
        graph.mark_variable_as_graph_input(k_var)
    graph.mark_variable_as_graph_output(values)
    graph.mark_variable_as_graph_output(indices)
    return graph, op


def _topk_model():
    nodes = [
        helper.make_node("Sigmoid", ["input"], ["scores"]),
        helper.make_node(
            "TopK",
            ["scores", "k"],
            ["values", "indices"],
            axis=-1,
            largest=1,
            sorted=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "topk_quantization",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [
            helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 3]),
        ],
        [helper.make_tensor("k", TensorProto.INT64, [1], [3])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_topk_float_export_preparation_uses_static_contract():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "topk.onnx")
        onnx.save(_topk_model(), model_path)
        graph = load_onnx_graph(model_path)

    EspdlExporter.prepare_graph_for_float_export(
        graph=graph,
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[torch.randn(1, 8)],
    )
    assert any(op.type == "TopK" for op in graph.operations.values())


def test_topk_forward_supports_opset1_attribute_k():
    _, op = _topk_graph(opset=1, k=2, attributes={"axis": -1})
    x = torch.tensor([[3.0, 3.0, 2.0, 3.0]])

    values, indices = TopK_forward(op, [x])

    assert torch.equal(values, torch.tensor([[3.0, 3.0]]))
    assert torch.equal(indices, torch.tensor([[0, 1]], dtype=torch.int64))


def test_topk_forward_rejects_newer_attributes_on_opset1():
    _, op = _topk_graph(opset=1, k=2, attributes={"axis": -1, "largest": 0})
    with pytest.raises(ValueError, match="require ONNX opset 11"):
        TopK_forward(op, [torch.randn(1, 8)])


def test_topk_forward_supports_dynamic_k_outside_espdl():
    _, op = _topk_graph(k_parameter=False, attributes={"axis": -1, "largest": 0, "sorted": 0})
    x = torch.tensor([[2.0, 1.0, 1.0, 3.0]])

    values, indices = TopK_forward(op, [x, torch.tensor([2], dtype=torch.int64)])

    assert torch.equal(values, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(indices, torch.tensor([[1, 2]], dtype=torch.int64))


@pytest.mark.parametrize(
    ("k_value", "message"),
    [
        (torch.tensor(2, dtype=torch.int64), "shape \\[1\\]"),
        (torch.tensor([2, 1], dtype=torch.int64), "shape \\[1\\]"),
        (torch.tensor([2.0], dtype=torch.float32), "dtype int64"),
        (torch.tensor([0], dtype=torch.int64), "out of range"),
        (torch.tensor([-1], dtype=torch.int64), "out of range"),
        (torch.tensor([9], dtype=torch.int64), "out of range"),
    ],
)
def test_topk_forward_rejects_invalid_k(k_value, message):
    _, op = _topk_graph(attributes={"axis": -1, "largest": 1, "sorted": 1})
    with pytest.raises(ValueError, match=message):
        TopK_forward(op, [torch.randn(1, 8), k_value])


@pytest.mark.parametrize(
    ("opset", "attributes"),
    [
        (10, {"axis": -1}),
        (11, {"axis": -1, "largest": 1, "sorted": 0}),
        (23, {"axis": -1, "largest": 0, "sorted": 1}),
    ],
)
def test_espdl_topk_contract_accepts_supported_opsets(opset, attributes):
    graph, op = _topk_graph(opset=opset, attributes=attributes)
    finalize_espdl_topk_for_export(graph)
    values, indices = TopK_forward(op, [torch.arange(8.0).reshape(1, 8), op.inputs[1].value])
    assert values.shape == indices.shape == (1, 3)
    assert indices.dtype == torch.int64


def test_topk_integer_attributes_are_serialized_as_scalars():
    _, op = _topk_graph(
        attributes={
            "axis": torch.tensor([-1], dtype=torch.int64),
            "largest": torch.tensor([1], dtype=torch.int64),
            "sorted": torch.tensor([0], dtype=torch.int64),
        }
    )
    op.outputs[0].shape = [1, 3]
    op.outputs[1].shape = [1, 3]

    with patch("esp_ppq.parser.espdl_exporter.helper.make_node", return_value=0) as make_node:
        EspdlExporter().build_operator_proto(op)
    attributes = make_node.call_args.kwargs
    assert {name: attributes[name] for name in ("axis", "largest", "sorted")} == {
        "axis": -1,
        "largest": 1,
        "sorted": 0,
    }
    assert all(isinstance(attributes[name], int) for name in ("axis", "largest", "sorted"))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("old_opset", "use opset 10-23"),
        ("opset10_new_attributes", "require ONNX opset 11"),
        ("dynamic_k", "constant parameter"),
        ("missing_k", "constant value"),
        ("float_k", "K dtype must be INT64"),
        ("scalar_k", "K shape must be \\[1\\]"),
        ("zero_k", "out of range"),
        ("large_k", "out of range"),
        ("bad_axis", "out of range for rank"),
        ("bad_largest", "largest must be 0 or 1"),
        ("bad_sorted", "sorted must be 0 or 1"),
        ("scalar_x", "rank must be at least 1"),
        ("zero_dimension", "positive dimensions"),
        ("dynamic_x", "X shape is missing or dynamic"),
        ("unsupported_x_dtype", "X dtype FP16 is unsupported"),
        ("bad_values_shape", "Values shape must be"),
        ("bad_values_dtype", "Values dtype"),
        ("bad_indices_dtype", "Indices dtype must be INT64"),
        ("wrong_arity", "expected 2 inputs and 2 outputs"),
        ("too_many_elements", "element count exceeds"),
    ],
)
def test_espdl_topk_contract_rejects_unsupported_graphs(case, message):
    graph, op = _topk_graph(attributes={"axis": -1, "largest": 1, "sorted": 1})
    if case == "old_opset":
        op.opset = Opset(version=9)
    elif case == "opset10_new_attributes":
        op.opset = Opset(version=10)
    elif case == "dynamic_k":
        op.inputs[1].is_parameter = False
        op.inputs[1].value = None
        op.inputs[1].shape, op.inputs[1].dtype = [1], DataType.INT64
    elif case == "missing_k":
        op.inputs[1].value = None
        op.inputs[1].shape, op.inputs[1].dtype = [1], DataType.INT64
    elif case == "float_k":
        op.inputs[1].value = torch.tensor([3.0])
    elif case == "scalar_k":
        op.inputs[1].value = torch.tensor(3, dtype=torch.int64)
    elif case == "zero_k":
        op.inputs[1].value = torch.tensor([0], dtype=torch.int64)
    elif case == "large_k":
        op.inputs[1].value = torch.tensor([9], dtype=torch.int64)
    elif case == "bad_axis":
        op.attributes["axis"] = 2
    elif case == "bad_largest":
        op.attributes["largest"] = 2
    elif case == "bad_sorted":
        op.attributes["sorted"] = 2
    elif case == "scalar_x":
        op.inputs[0].shape = []
    elif case == "zero_dimension":
        op.inputs[0].shape = [1, 0]
    elif case == "dynamic_x":
        op.inputs[0].shape = ["batch", 8]
    elif case == "unsupported_x_dtype":
        op.inputs[0].dtype = DataType.FP16
        op.outputs[0].dtype = DataType.FP16
    elif case == "bad_values_shape":
        op.outputs[0].shape = [1, 4]
    elif case == "bad_values_dtype":
        op.outputs[0].dtype = DataType.FP16
    elif case == "bad_indices_dtype":
        op.outputs[1].dtype = DataType.INT32
    elif case == "wrong_arity":
        op.outputs.pop()
    elif case == "too_many_elements":
        op.inputs[0].shape = [50000, 50000]
        op.outputs[0].shape = [50000, 3]
        op.outputs[1].shape = [50000, 3]

    with pytest.raises(ValueError, match=message):
        finalize_espdl_topk_for_export(graph)


def test_topk_quantizer_preflights_static_contract_before_shared_pipeline():
    graph, op = _topk_graph(k_parameter=False, attributes={"axis": -1, "largest": 1, "sorted": 1})
    op.inputs[1].shape, op.inputs[1].dtype = [1], DataType.INT64

    with patch.object(BaseQuantizer, "quantize") as base_quantize:
        with pytest.raises(ValueError, match="constant parameter"):
            EspdlQuantizer(graph).quantize(
                inputs=[torch.randn(1, 8)],
                calib_dataloader=[],
                executor=None,
                setting=QuantizationSettingFactory.espdl_setting(),
            )
    base_quantize.assert_not_called()


def test_topk_quantizer_aligns_values_even_across_upstream_platforms():
    graph, op = _topk_graph(attributes={"axis": -1, "largest": 1, "sorted": 1})
    producer_input = graph.create_variable("producer_input")
    producer_input.shape, producer_input.dtype = [1, 8], DataType.FP32
    producer = graph.create_operation(
        op_type="Identity",
        name="producer",
        platform=TargetPlatform.FP32,
        inputs=[producer_input],
        outputs=[op.inputs[0]],
    )
    assert producer.platform != op.platform

    config = EspdlQuantizer(graph).init_quantize_config(op)
    assert config.input_quantization_config[1].state == QuantizationStates.FP32
    assert config.output_quantization_config[1].state == QuantizationStates.FP32
    assert config.output_quantization_config[0].dominated_by is config.input_quantization_config[0].dominated_by


@pytest.mark.parametrize(
    "platform",
    [TargetPlatform.ESPDL_INT8, TargetPlatform.ESPDL_H_PRE_INT16],
)
def test_topk_quantization_and_export_preserve_the_contract(platform):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "topk.onnx")
        export_path = os.path.join(tmp, "topk.espdl")
        onnx.save(_topk_model(), model_path)
        sample = [torch.randn(1, 8)]
        graph = quantize_onnx_model(
            onnx_import_file=model_path,
            calib_dataloader=[sample, [sample[0].clone()]],
            calib_steps=2,
            input_shape=None,
            platform=platform,
            inputs=sample,
            setting=QuantizationSettingFactory.espdl_setting(),
            device="cpu",
            verbose=0,
        )
        topk = next(op for op in graph.operations.values() if op.type == "TopK")

        assert topk.config.input_quantization_config[1].state == QuantizationStates.FP32
        assert topk.config.output_quantization_config[1].state == QuantizationStates.FP32
        assert (
            topk.config.output_quantization_config[0].dominated_by
            is topk.config.input_quantization_config[0].dominated_by
        )
        finalize_espdl_topk_for_export(graph)
        EspdlExporter().export(export_path, graph, export_config=False)
        assert os.path.getsize(export_path) > 0


def test_topk_export_rechecks_quantization_state():
    graph, op = _topk_graph(attributes={"axis": -1, "largest": 1, "sorted": 1})
    quantizer = EspdlQuantizer(graph)
    quantized = quantizer.quantize_operation(op.name, platform=TargetPlatform.ESPDL_INT8)
    quantized.config.output_quantization_config[1].state = QuantizationStates.ACTIVATED

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="Indices quantization state must be FP32"):
            EspdlExporter().export(
                os.path.join(tmp, "invalid.espdl"),
                graph,
                export_config=False,
            )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("values_bit_width", "same storage quantization"),
        ("values_fp32_only", "same storage quantization"),
        ("non_pot_policy", "symmetric per-tensor linear POWER_OF_2"),
        ("unsupported_4bit", "8-bit or 16-bit integer"),
        ("k_force_export", "K quantization config must not be exportable"),
        ("indices_force_export", "Indices quantization config must not be exportable"),
    ],
)
def test_topk_validation_rejects_unsafe_quantization_configs(case, message):
    graph, op = _topk_graph(attributes={"axis": -1, "largest": 1, "sorted": 1})
    quantized = EspdlQuantizer(graph).quantize_operation(op.name, platform=TargetPlatform.ESPDL_INT8)
    if case == "values_bit_width":
        quantized.config.output_quantization_config[0].num_of_bits = 16
    elif case == "values_fp32_only":
        quantized.config.output_quantization_config[0].state = QuantizationStates.FP32
    elif case == "non_pot_policy":
        policy = QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL + QuantizationProperty.LINEAR + QuantizationProperty.PER_TENSOR
        )
        quantized.config.input_quantization_config[0].policy = policy
        quantized.config.output_quantization_config[0].policy = policy
    elif case == "unsupported_4bit":
        for cfg in (quantized.config.input_quantization_config[0], quantized.config.output_quantization_config[0]):
            cfg.num_of_bits = 4
            cfg.quant_min, cfg.quant_max = -8, 7
    elif case == "k_force_export":
        config = quantized.config.input_quantization_config[1]
        config.visibility = QuantizationVisibility.FORCE_EXPORT
        config.scale, config.offset = torch.tensor(1.0), torch.tensor(0.0)
    elif case == "indices_force_export":
        config = quantized.config.output_quantization_config[1]
        config.visibility = QuantizationVisibility.FORCE_EXPORT
        config.scale, config.offset = torch.tensor(1.0), torch.tensor(0.0)

    with pytest.raises(ValueError, match=message):
        finalize_espdl_topk_for_export(graph)


def test_conv_to_topk_quantizes_and_exports_with_layout_restore():
    weight = helper.make_tensor(
        "conv_w",
        TensorProto.FLOAT,
        [4, 2, 1, 1],
        torch.linspace(-0.5, 0.5, steps=8, dtype=torch.float32).tolist(),
    )
    k = helper.make_tensor("k", TensorProto.INT64, [1], [2])
    graph_def = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "conv_w"], ["conv_out"]),
            helper.make_node("TopK", ["conv_out", "k"], ["values", "indices"], axis=1, largest=1, sorted=1),
        ],
        "conv_topk_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 3])],
        [
            helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 2, 2, 3]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2, 2, 3]),
        ],
        initializer=[weight, k],
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 13)])

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "conv_topk.onnx")
        espdl_path = os.path.join(tmp, "conv_topk.espdl")
        onnx.save(model, onnx_path)
        sample = [torch.randn(1, 2, 2, 3, dtype=torch.float32)]
        graph = quantize_onnx_model(
            onnx_import_file=onnx_path,
            calib_dataloader=[sample, [sample[0].clone()]],
            calib_steps=2,
            input_shape=None,
            platform=TargetPlatform.ESPDL_INT8,
            inputs=sample,
            setting=QuantizationSettingFactory.espdl_setting(),
            device="cpu",
            verbose=0,
        )
        EspdlExporter().export(espdl_path, graph, export_config=False)
        assert os.path.isfile(espdl_path)


def test_topk_ignores_non_exportable_force_visibility_and_rounding_differences():
    graph, op = _topk_graph(attributes={"axis": -1, "largest": 1, "sorted": 1})
    quantized = EspdlQuantizer(graph).quantize_operation(op.name, platform=TargetPlatform.ESPDL_INT8)
    quantized.config.input_quantization_config[1].visibility = QuantizationVisibility.FORCE_EXPORT
    quantized.config.output_quantization_config[1].visibility = QuantizationVisibility.FORCE_EXPORT
    quantized.config.output_quantization_config[0].rounding = RoundingPolicy.ROUND_HALF_UP
    assert (
        quantized.config.output_quantization_config[0].rounding
        != quantized.config.input_quantization_config[0].rounding
    )

    finalize_espdl_topk_for_export(graph)


def test_auto_generated_variable_names_skip_existing_round_trip_names():
    graph = BaseGraph(name="round_trip_var_names")
    existing = graph.create_variable(name="PPQ_Variable_0")
    existing_next = graph.create_variable(name="PPQ_Variable_1")
    out = graph.create_variable(name="out")
    graph.create_operation(op_type="Relu", name="relu", inputs=[existing], outputs=[out])

    created = graph.create_variable()
    assert created.name == "PPQ_Variable_2"
    assert existing_next.name in graph.variables

    inserted = Operation(name="identity", op_type="Identity", attributes={})
    graph.insert_op_on_var(inserting_op=inserted, var="PPQ_Variable_0")
    assert inserted.outputs[0].name == "PPQ_Variable_3"

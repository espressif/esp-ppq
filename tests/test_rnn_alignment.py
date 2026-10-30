import json

import pytest
import torch
from onnx import TensorProto, checker, helper, numpy_helper

from esp_ppq.api import load_onnx_graph
from esp_ppq.core import (
    NetworkFramework,
    QuantizationStates,
    TargetPlatform,
)
from esp_ppq.executor import TorchExecutor
from esp_ppq.IR import BaseGraph, GraphCommand, GraphCommandType, GraphFormatter
from esp_ppq.lib.quant import Quantizer
from esp_ppq.parser import espdl_exporter
from esp_ppq.parser.espdl_exporter import EspdlExporter
from esp_ppq.parser.onnx_exporter import OnnxExporter
from esp_ppq.parser.onnx_parser import OnnxParser
from esp_ppq.quantization.optim import QuantAlignmentPass


def _normalize_rnn_outputs(graph):
    GraphFormatter(graph)(GraphCommand(GraphCommandType.FORMAT_RNN_OUTPUT_ARITY))


def _make_quantized_rnn(op_type, num_inputs, num_outputs):
    graph = BaseGraph(name=f"{op_type}_graph", built_from=NetworkFramework.ONNX)
    input_names = ["" if index == 4 else f"input_{index}" for index in range(num_inputs)]
    inputs = [graph.create_variable(name=name) for name in input_names]
    outputs = [graph.create_variable(name=f"output_{index}") for index in range(num_outputs)]
    graph.create_operation(
        op_type=op_type,
        name=op_type,
        platform=TargetPlatform.ESPDL_INT8,
        inputs=inputs,
        outputs=outputs,
    )
    _normalize_rnn_outputs(graph)
    operation = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation(op_type)
    return graph, operation


@pytest.mark.parametrize(
    ("op_type", "node_inputs", "expected_inputs"),
    [
        ("GRU", ["X", "W", "R", "B"], ["X", "W", "R", "B"]),
        ("GRU", ["X", "W", "R", "B", "", ""], ["X", "W", "R", "B"]),
        ("LSTM", ["X", "W", "R", "B"], ["X", "W", "R", "B"]),
        ("LSTM", ["X", "W", "R", "B", "", "", "C0"], ["X", "W", "R", "B", "", "", "C0"]),
    ],
)
def test_onnx_parser_strips_trailing_empty_inputs_and_preserves_interior_ones(
    tmp_path, op_type, node_inputs, expected_inputs
):
    output_names = ["Y", "Y_h"] if op_type == "GRU" else ["Y", "Y_h", "Y_c"]
    node = helper.make_node(op_type, node_inputs, output_names, name="rnn")
    model = helper.make_model(
        helper.make_graph([node], f"{op_type}_graph", [], []),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_path = tmp_path / f"{op_type.lower()}.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = OnnxParser().build(str(model_path))
    operation = graph.operations["rnn"]
    assert [variable.name for variable in operation.inputs] == expected_inputs
    assert [variable.name for variable in operation.outputs] == output_names


@pytest.mark.parametrize(
    ("op_type", "node_outputs", "expected_arity"),
    [
        ("GRU", ["Y"], 2),
        ("LSTM", ["Y", "YH"], 3),
    ],
)
def test_format_graph_restores_trailing_optional_rnn_output_slots(tmp_path, op_type, node_outputs, expected_arity):
    node = helper.make_node(op_type, ["X", "W", "R"], node_outputs, name="rnn")
    model = helper.make_model(
        helper.make_graph(
            [node],
            f"{op_type}_graph",
            [],
            [helper.make_tensor_value_info(node_outputs[0], TensorProto.FLOAT, None)],
        ),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_path = tmp_path / f"{op_type.lower()}_outputs.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = load_onnx_graph(str(model_path))
    operation = graph.operations["rnn"]

    assert [variable.name for variable in operation.inputs] == ["X", "W", "R"]
    assert operation.num_of_output == expected_arity
    assert [variable.name for variable in operation.outputs[: len(node_outputs)]] == node_outputs
    assert all(variable.name for variable in operation.outputs[len(node_outputs) :])

    output_names = [variable.name for variable in operation.outputs]
    _normalize_rnn_outputs(graph)
    assert [variable.name for variable in operation.outputs] == output_names

    operation = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation(
        "rnn", platform=TargetPlatform.ESPDL_INT8
    )
    QuantAlignmentPass().optimize(graph)
    master = operation.config.output_quantization_config[0]
    assert operation.config.output_quantization_config[1].master_by is master
    if op_type == "LSTM":
        cell_output = operation.config.output_quantization_config[2]
        assert cell_output.master_by is cell_output


@pytest.mark.parametrize(
    ("op_type", "num_inputs", "num_outputs"),
    [
        ("GRU", 3, 2),
        ("GRU", 4, 2),
        ("LSTM", 3, 3),
        ("LSTM", 4, 3),
    ],
)
def test_rnn_alignment_accepts_omitted_initial_state(op_type, num_inputs, num_outputs):
    graph, operation = _make_quantized_rnn(op_type, num_inputs, num_outputs)

    assert operation.num_of_input == num_inputs
    QuantAlignmentPass().optimize(graph)

    master = operation.config.output_quantization_config[0]
    assert operation.config.output_quantization_config[1].master_by is master
    if op_type == "LSTM":
        cell_output = operation.config.output_quantization_config[2]
        assert cell_output.master_by is cell_output


@pytest.mark.parametrize(
    ("op_type", "num_inputs", "num_outputs"),
    [("GRU", 6, 2), ("LSTM", 7, 3)],
)
def test_rnn_alignment_aligns_explicit_initial_hidden_state(op_type, num_inputs, num_outputs):
    graph, operation = _make_quantized_rnn(op_type, num_inputs, num_outputs)

    assert operation.config.input_quantization_config[4].state == QuantizationStates.FP32
    QuantAlignmentPass().optimize(graph)

    master = operation.config.output_quantization_config[0]
    assert operation.config.output_quantization_config[1].master_by is master
    assert operation.config.input_quantization_config[5].master_by is master
    if op_type == "LSTM":
        cell_input = operation.config.input_quantization_config[6]
        cell_output = operation.config.output_quantization_config[2]
        assert cell_input.master_by is cell_input
        assert cell_output.master_by is cell_output


def test_missing_rnn_outputs_are_unique_and_do_not_create_cycles(tmp_path):
    nodes = [
        helper.make_node("GRU", ["X0", "W0", "R0"], ["", "YH0"], name="gru0"),
        helper.make_node("GRU", ["X1", "W1", "R1"], ["", "YH1"], name="gru1"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "two_grus", [], []),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_path = tmp_path / "two_grus.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = OnnxParser().build(str(model_path))
    _normalize_rnn_outputs(graph)
    gru0, gru1 = graph.operations["gru0"], graph.operations["gru1"]
    output_sinks = [gru0.outputs[0], gru1.outputs[0]]

    assert "" not in graph.variables
    assert len({variable.name for variable in output_sinks}) == 2
    assert all(variable.name for variable in output_sinks)
    assert {operation.name for operation in graph.topological_sort()} == {"gru0", "gru1"}


def test_rnn_output_formatter_rejects_invalid_arity_or_post_quant_mutation():
    graph = BaseGraph(name="gru_invalid", built_from=NetworkFramework.ONNX)
    graph.create_operation(
        op_type="GRU",
        name="gru_too_many",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[graph.create_variable(name=f"in_{i}") for i in range(3)],
        outputs=[graph.create_variable(name=f"out_{i}") for i in range(3)],
    )
    with pytest.raises(ValueError, match="expected at most 2 outputs"):
        _normalize_rnn_outputs(graph)

    quant_graph = BaseGraph(name="gru_post_quant", built_from=NetworkFramework.ONNX)
    quant_graph.create_operation(
        op_type="GRU",
        name="gru",
        platform=TargetPlatform.ESPDL_INT8,
        inputs=[quant_graph.create_variable(name=f"in_{i}") for i in range(3)],
        outputs=[quant_graph.create_variable(name="out_0")],
    )
    Quantizer(TargetPlatform.ESPDL_INT8, quant_graph).quantize_operation("gru")
    with pytest.raises(ValueError, match="cannot normalize outputs after quantization"):
        _normalize_rnn_outputs(quant_graph)


def test_onnx_parser_strips_trailing_empty_slots_but_keeps_interior_ones(tmp_path):
    scales = numpy_helper.from_array(torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32).numpy(), name="scales")
    nodes = [
        helper.make_node("Attention", ["Q", "K", "V", ""], ["attn_out"], name="attn", num_heads=1),
        helper.make_node("RotaryEmbedding", ["attn_out", "cos", "sin", ""], ["rope_out"], name="rope"),
        helper.make_node("Resize", ["rope_out", "", "scales"], ["Y"], name="resize", mode="nearest"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "mixed_optional_slots",
            [
                helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 1, 2, 2]),
                helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 1, 2, 2]),
                helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 1, 2, 2]),
                helper.make_tensor_value_info("cos", TensorProto.FLOAT, [2, 1]),
                helper.make_tensor_value_info("sin", TensorProto.FLOAT, [2, 1]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])],
            initializer=[scales],
        ),
        opset_imports=[helper.make_opsetid("", 23)],
    )
    model_path = tmp_path / "mixed_optional_slots.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = OnnxParser().build(str(model_path))
    assert [var.name for var in graph.operations["attn"].inputs] == ["Q", "K", "V"]
    assert [var.name for var in graph.operations["rope"].inputs] == ["attn_out", "cos", "sin"]

    resize_op = graph.operations["resize"]
    assert [var.name for var in resize_op.inputs] == ["rope_out", "", "scales"]

    for var_name, shape in (("Q", [1, 1, 2, 2]), ("attn_out", [1, 1, 2, 2]), ("rope_out", [1, 1, 2, 2])):
        graph.variables[var_name].shape = shape
    graph.variables["Y"].shape = [1, 1, 4, 4]
    scales_var = graph.variables["scales"]
    scales_var.value = torch.tensor(scales_var.value)
    exported = OnnxExporter().export_graph(graph.copy())
    exported_resize = next(node for node in exported.graph.node if node.name == "resize")
    assert list(exported_resize.input) == ["rope_out", "", "scales"]
    exported_value_info_names = {value.name for value in exported.graph.value_info}
    assert "" not in exported_value_info_names


@pytest.mark.parametrize(
    ("op_type", "node_outputs", "gate_count"),
    [
        ("GRU", ["", "YH"], 3),
        ("LSTM", ["", "YH"], 4),
    ],
)
def test_omitted_rnn_outputs_use_named_sinks_in_onnx_and_espdl_export(
    tmp_path, monkeypatch, op_type, node_outputs, gate_count
):
    operation_name = op_type.lower()
    node = helper.make_node(
        op_type,
        ["X", "W", "R"],
        node_outputs,
        name=operation_name,
        hidden_size=2,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            f"{operation_name}_slots",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 3]),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, gate_count * 2, 3]),
                helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, gate_count * 2, 2]),
            ],
            [helper.make_tensor_value_info("YH", TensorProto.FLOAT, [1, 1, 2])],
        ),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_path = tmp_path / f"{operation_name}_slots.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = OnnxParser().build(str(model_path))
    _normalize_rnn_outputs(graph)
    operation = graph.operations[operation_name]
    output_sinks = [var for var in operation.outputs if var.name != "YH"]
    assert len(output_sinks) == (1 if op_type == "GRU" else 2)
    assert all(sink.name for sink in output_sinks)

    graph.variables["X"].shape = [2, 1, 3]
    graph.variables["W"].shape = [1, gate_count * 2, 3]
    graph.variables["R"].shape = [1, gate_count * 2, 2]
    graph.variables["YH"].shape = [1, 1, 2]
    for sink in output_sinks:
        sink.shape = [2, 1, 1, 2] if sink is operation.outputs[0] else [1, 1, 2]

    exported = OnnxExporter().export_graph(graph.copy())
    checker.check_model(exported)
    exported_node = exported.graph.node[0]
    assert list(exported_node.input) == ["X", "W", "R"]
    assert list(exported_node.output) == [var.name for var in operation.outputs]

    espdl_path = tmp_path / f"{operation_name}_slots.espdl"
    EspdlExporter().export(str(espdl_path), graph, export_config=False)
    espdl_bytes = espdl_path.read_bytes()
    assert all(output_sink.name.encode() in espdl_bytes for output_sink in output_sinks)

    captured = {}

    def capture_node(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(espdl_exporter.helper, "make_node", capture_node)
    EspdlExporter().build_operator_proto(operation)
    assert captured["inputs"] == ["X", "W", "R"]
    assert captured["outputs"] == [var.name for var in operation.outputs]


def test_omitted_trailing_rnn_inputs_execute_correctly(tmp_path):
    hidden_size, input_size = 2, 3
    weight = torch.arange(6 * input_size, dtype=torch.float32).reshape(1, 6, input_size) / 100
    recurrent = torch.arange(6 * hidden_size, dtype=torch.float32).reshape(1, 6, hidden_size) / 100
    node = helper.make_node(
        "GRU",
        ["X", "W", "R"],
        ["Y"],
        name="gru",
        hidden_size=hidden_size,
        linear_before_reset=1,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "gru_execution",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, input_size])],
            [
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 1, hidden_size]),
            ],
            initializer=[
                numpy_helper.from_array(weight.numpy(), name="W"),
                numpy_helper.from_array(recurrent.numpy(), name="R"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_path = tmp_path / "gru_execution.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = OnnxParser().build(str(model_path))
    _normalize_rnn_outputs(graph)
    operation = graph.operations["gru"]
    assert operation.num_of_input == 3
    assert operation.num_of_output == 2

    outputs = TorchExecutor(graph=graph, device="cpu").forward(
        inputs=[torch.arange(6, dtype=torch.float32).reshape(2, 1, input_size) / 10]
    )
    assert tuple(outputs[0].shape) == (2, 1, 1, hidden_size)


def test_empty_input_placeholders_are_not_exported_as_quantized_tensors(tmp_path):
    node = helper.make_node("GRU", ["X", "W", "R", "B", "", "H0"], ["", "YH"], name="gru")
    model = helper.make_model(
        helper.make_graph([node], "gru_config", [], []),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_path = tmp_path / "gru_config.onnx"
    model_path.write_bytes(model.SerializeToString())

    graph = OnnxParser().build(str(model_path))
    _normalize_rnn_outputs(graph)
    operation = Quantizer(TargetPlatform.ESPDL_INT8, graph).quantize_operation(
        "gru", platform=TargetPlatform.ESPDL_INT8
    )
    QuantAlignmentPass().optimize(graph)
    output_sink = operation.outputs[0]
    onnx_config_path = tmp_path / "onnx.json"
    espdl_config_path = tmp_path / "espdl.json"
    OnnxExporter().export_quantization_config(str(onnx_config_path), graph)
    EspdlExporter().export_quantization_config(str(espdl_config_path), graph)

    onnx_payload = json.loads(onnx_config_path.read_text())
    espdl_payload = json.loads(espdl_config_path.read_text())
    onnx_configs = onnx_payload["configs"]["gru"]
    espdl_configs = espdl_payload["configs"]["gru"]
    assert "" not in onnx_configs
    assert output_sink.name in onnx_configs
    assert str(onnx_configs["YH"]["dominator"]) in onnx_payload["values"]
    assert "" not in espdl_configs
    assert output_sink.name in espdl_configs
    assert str(espdl_configs["YH"]["dominator"]) in espdl_payload["values"]

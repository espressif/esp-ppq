"""Normalize GRU/LSTM output arity after ONNX parsing.

ONNX simplifiers such as ``onnxsim`` drop trailing unused RNN outputs (e.g.
``GRU`` with only ``Y``). However, ``GRU_forward`` / ``LSTM_forward`` always
return the full output tuple, ``QuantAlignmentPass`` indexes ``outputs[1]``,
and ESP-DL's C++ ``GRU`` / ``LSTM`` modules unconditionally dereference all 2
or 3 output tensors. This pass appends named sink variables for any omitted
trailing RNN outputs before quantization or float export.
"""

from typing import Dict

from esp_ppq.IR import BaseGraph
from esp_ppq.IR.quantize import QuantableOperation

from .base import QuantizationOptimizationPass

_RNN_OUTPUT_ARITY: Dict[str, int] = {
    "GRU": 2,  # Y, Y_h
    "LSTM": 3,  # Y, Y_h, Y_c
}


class RNNArityNormalizationPass(QuantizationOptimizationPass):
    """Restore canonical GRU/LSTM output arity (GRU: 2, LSTM: 3)."""

    def __init__(self) -> None:
        super().__init__(name="RNN Arity Normalization Pass")

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for op in list(graph.operations.values()):
            target_outputs = _RNN_OUTPUT_ARITY.get(op.type)
            if target_outputs is None:
                continue
            if op.num_of_output > target_outputs:
                raise ValueError(
                    f"RNN {op.name}: expected at most {target_outputs} outputs for {op.type}, got {op.num_of_output}."
                )
            needs_output_update = op.num_of_output < target_outputs or any(not var.name for var in op.outputs)
            if needs_output_update and isinstance(op, QuantableOperation):
                raise ValueError(
                    f"RNN {op.name}: cannot normalize outputs after quantization; "
                    "run RNNArityNormalizationPass before quantize_operation."
                )
            for index, var in enumerate(op.outputs):
                if not var.name:
                    if var.source_op is op:
                        var.source_op = None
                    if not var.dest_ops and var.source_op is None:
                        graph.variables.pop("", None)
                    sink = graph.create_variable()
                    sink.source_op = op
                    op.outputs[index] = sink
            while op.num_of_output < target_outputs:
                graph.create_variable(source_op=op)

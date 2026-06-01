"""Regression test for scalar (rank-0) tensor serialization in the ESP-DL exporter.

`helper.make_tensor` must serialize a scalar (empty `dims`) tensor with shape `[1]`, not `[]`.
esp-dl's fbs loader (`FbsModel::get_operation_parameter`) reads a parameter's data using its dims
and dereferences a null pointer (LoadProhibited, EXCVADDR=0x0) during `Model::load()` on-device when
a parameter has empty dims. Such scalar params occur for e.g. the scale / zero_point of
QuantizeLinear / RequantizeLinear ops that esp-ppq inserts at branch-boundary scale changes, so any
branchy / multi-output quantized model can trigger the crash.

This is a pure-CPU unit test (no CUDA needed).
"""
import os
import sys

# helper.py imports the generated flatbuffers as top-level `FlatBuffers.Dl.*`, so the espdl parser
# directory must be importable.
_ESPDL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "esp_ppq", "parser", "espdl"))
if _ESPDL_DIR not in sys.path:
    sys.path.insert(0, _ESPDL_DIR)

from esp_ppq.parser.espdl import helper  # noqa: E402
import FlatBuffers.Dl.Tensor as Tensor  # noqa: E402
import FlatBuffers.Dl.TensorDataType as TensorDataType  # noqa: E402


def _make_and_read(dims, vals):
    helper.reset_global_fbs_builder()
    builder = helper.get_global_fbs_builder()
    offset = helper.make_tensor("t", TensorDataType.TensorDataType().FLOAT, dims, vals)
    builder.Finish(offset)
    t = Tensor.Tensor.GetRootAs(builder.Output(), 0)
    read_dims = [t.Dims(i) for i in range(t.DimsLength())]
    helper.reset_global_fbs_builder()
    return read_dims


def test_scalar_tensor_dims_coerced_to_one():
    # rank-0 scalar -> must become [1] (else esp-dl crashes on load)
    assert _make_and_read([], [0.5]) == [1]


def test_nonscalar_tensor_dims_unchanged():
    # regular tensors must be left exactly as-is
    assert _make_and_read([3], [1.0, 2.0, 3.0]) == [3]
    assert _make_and_read([2, 2], [1.0, 2.0, 3.0, 4.0]) == [2, 2]


if __name__ == "__main__":
    test_scalar_tensor_dims_coerced_to_one()
    test_nonscalar_tensor_dims_unchanged()
    print("PASS")

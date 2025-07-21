import os
import sys
import warnings
from typing import Any, Dict, NamedTuple, Union, cast

import numpy as np

sys.path.append(os.path.dirname(__file__))
import FlatBuffers.Dl.OptionalType as OptionalType
import FlatBuffers.Dl.SequenceType as SequenceType
import FlatBuffers.Dl.Tensor
import FlatBuffers.Dl.Tensor as Tensor
import FlatBuffers.Dl.TensorDataType as TensorDataType


class TensorDtypeMap(NamedTuple):
    np_dtype: np.dtype
    storage_dtype: int
    name: str


# tensor_dtype: (numpy type, storage type, string name)
TENSOR_TYPE_MAP = {
    int(TensorDataType.TensorDataType().FLOAT): TensorDtypeMap(
        np.dtype("float32"), int(TensorDataType.TensorDataType().FLOAT), "TensorDataType.FLOAT"
    ),
    int(TensorDataType.TensorDataType().UINT8): TensorDtypeMap(
        np.dtype("uint8"), int(TensorDataType.TensorDataType().INT32), "TensorDataType.UINT8"
    ),
    int(TensorDataType.TensorDataType().INT8): TensorDtypeMap(
        np.dtype("int8"), int(TensorDataType.TensorDataType().INT32), "TensorDataType.INT8"
    ),
    int(TensorDataType.TensorDataType().UINT16): TensorDtypeMap(
        np.dtype("uint16"), int(TensorDataType.TensorDataType().INT32), "TensorDataType.UINT16"
    ),
    int(TensorDataType.TensorDataType().INT16): TensorDtypeMap(
        np.dtype("int16"), int(TensorDataType.TensorDataType().INT32), "TensorDataType.INT16"
    ),
    int(TensorDataType.TensorDataType().INT32): TensorDtypeMap(
        np.dtype("int32"), int(TensorDataType.TensorDataType().INT32), "TensorDataType.INT32"
    ),
    int(TensorDataType.TensorDataType().INT64): TensorDtypeMap(
        np.dtype("int64"), int(TensorDataType.TensorDataType().INT64), "TensorDataType.INT64"
    ),
    int(TensorDataType.TensorDataType().BOOL): TensorDtypeMap(
        np.dtype("bool"), int(TensorDataType.TensorDataType().INT32), "TensorDataType.BOOL"
    ),
    int(TensorDataType.TensorDataType().FLOAT16): TensorDtypeMap(
        np.dtype("float16"), int(TensorDataType.TensorDataType().UINT16), "TensorDataType.FLOAT16"
    ),
    int(TensorDataType.TensorDataType().DOUBLE): TensorDtypeMap(
        np.dtype("float64"), int(TensorDataType.TensorDataType().DOUBLE), "TensorDataType.DOUBLE"
    ),
    int(TensorDataType.TensorDataType().UINT32): TensorDtypeMap(
        np.dtype("uint32"), int(TensorDataType.TensorDataType().UINT32), "TensorDataType.UINT32"
    ),
    int(TensorDataType.TensorDataType().UINT64): TensorDtypeMap(
        np.dtype("uint64"), int(TensorDataType.TensorDataType().UINT64), "TensorDataType.UINT64"
    ),
    int(TensorDataType.TensorDataType().STRING): TensorDtypeMap(
        np.dtype("object"), int(TensorDataType.TensorDataType().STRING), "TensorDataType.STRING"
    ),
}


class DeprecatedWarningDict(dict):  # type: ignore
    def __init__(
        self,
        dictionary: Dict[int, Union[int, str, np.dtype]],
        original_function: str,
        future_function: str = "",
    ) -> None:
        super().__init__(dictionary)
        self._origin_function = original_function
        self._future_function = future_function

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeprecatedWarningDict):
            return False
        return self._origin_function == other._origin_function and self._future_function == other._future_function

    def __getitem__(self, key: Union[int, str, np.dtype]) -> Any:
        if not self._future_function:
            warnings.warn(
                str(
                    f"`mapping.{self._origin_function}` is now deprecated and will be removed in a future release."
                    "To silence this warning, please simply use if-else statement to get the corresponding value."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                str(
                    f"`mapping.{self._origin_function}` is now deprecated and will be removed in a future release."
                    f"To silence this warning, please use `helper.{self._future_function}` instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)


# This map is used for converting TensorProto values into numpy arrays
TENSOR_TYPE_TO_NP_TYPE = DeprecatedWarningDict(
    {tensor_dtype: value.np_dtype for tensor_dtype, value in TENSOR_TYPE_MAP.items()},
    "TENSOR_TYPE_TO_NP_TYPE",
    "tensor_dtype_to_np_dtype",
)


_NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}


_STORAGE_TENSOR_TYPE_TO_FIELD = {
    int(TensorDataType.TensorDataType().FLOAT): Tensor.AddFloatData,
    int(TensorDataType.TensorDataType().INT32): Tensor.AddInt32Data,
    int(TensorDataType.TensorDataType().INT64): Tensor.AddInt64Data,
    int(TensorDataType.TensorDataType().UINT8): Tensor.AddInt32Data,
    int(TensorDataType.TensorDataType().UINT16): Tensor.AddInt32Data,
    int(TensorDataType.TensorDataType().DOUBLE): Tensor.AddDoubleData,
    int(TensorDataType.TensorDataType().UINT32): Tensor.AddUint64Data,
    int(TensorDataType.TensorDataType().UINT64): Tensor.AddUint64Data,
    int(TensorDataType.TensorDataType().STRING): Tensor.AddStringData,
    int(TensorDataType.TensorDataType().BOOL): Tensor.AddInt32Data,
}

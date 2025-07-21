from abc import ABCMeta, abstractmethod
from typing import Any, Callable

from esp_ppq.core import TensorQuantizationConfig


class BaseQuantFunction(Callable, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, input_tensor: Any, quantization_config: TensorQuantizationConfig, **kwargs) -> Any:
        raise NotImplementedError('Implement this first.')

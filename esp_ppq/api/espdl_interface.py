# -*- coding: utf-8 -*-
import os
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import onnx
import torch
from onnxsim import simplify
from torch.utils.data import DataLoader

import esp_ppq.lib as PFL
from esp_ppq.api.interface import load_onnx_graph, quantize_onnx_model
from esp_ppq.api.setting import QuantizationSetting, QuantizationSettingFactory
from esp_ppq.core import (
    QuantizationVisibility,
    TargetPlatform,
    empty_ppq_cache,
    ppq_warning,
)
from esp_ppq.executor import BaseGraphExecutor, TorchExecutor
from esp_ppq.IR import BaseGraph
from esp_ppq.log import NaiveLogger
from esp_ppq.parser.espdl_exporter import EspdlExporter
from esp_ppq.quantization.analyse import graphwise_error_analyse
from esp_ppq.quantization.analyse.layerwise import layerwise_error_analyse
from esp_ppq.quantization.optim import *

logger = NaiveLogger.get_logger("ESPDL")

_PIE_NONE_TARGETS = {"esp32", "esp32s2", "esp32c2", "esp32c3", "esp32c5", "esp32c6", "c"}
_PIE_V1_TARGETS = {"esp32s3"}
_PIE_V2_TARGETS = {"esp32p4", "esp32s31"}

_VALID_QUANT_TYPES = {"w8a8", "w8a16", "w16a16", "none"}
_PLATFORM_BY_QUANT = {
    "w8a8": {
        "v2": TargetPlatform.ESPDL_INT8,
        "v1": TargetPlatform.ESPDL_S3_INT8,
        "c": TargetPlatform.ESPDL_C_INT8,
    },
    "w8a16": {
        "v2": TargetPlatform.ESPDL_W8A16,
        "v1": TargetPlatform.ESPDL_S3_W8A16,
        "c": TargetPlatform.ESPDL_C_W8A16,
    },
    "w16a16": {
        "v2": TargetPlatform.ESPDL_INT16,
        "v1": TargetPlatform.ESPDL_S3_INT16,
        "c": TargetPlatform.ESPDL_C_INT16,
    },
    "w16a16_debug": {
        "v2": TargetPlatform.ESPDL_H_PRE_INT16,
        "v1": TargetPlatform.ESPDL_S3_H_PRE_INT16,
        "c": TargetPlatform.ESPDL_C_H_PRE_INT16,
    },
}


def _target_group(target: str) -> Optional[str]:
    if target in _PIE_V2_TARGETS:
        return "v2"
    if target in _PIE_V1_TARGETS:
        return "v1"
    if target in _PIE_NONE_TARGETS:
        return "c"
    return None


_QUANT_TYPE_ACT_BITS = {"w8a8": 8, "w8a16": 16, "w16a16": 16, "none": 8}


def _quant_type_from_bits(num_of_bits: int) -> Optional[str]:
    if num_of_bits == 8:
        return "w8a8"
    if num_of_bits == 16:
        return "w16a16"
    return None


def _resolve_quant_type_and_bits(
    quant_type: Optional[str],
    num_of_bits: int,
):
    """Resolve platform quant_type and activation bits for settings.

    ``quant_type`` wins when given; otherwise fall back to ``num_of_bits``.
    """
    if quant_type is None:
        return num_of_bits, num_of_bits

    resolved = quant_type.lower()
    if resolved not in _VALID_QUANT_TYPES:
        raise ValueError(f"Unsupported quant_type '{quant_type}'. Expected one of {sorted(_VALID_QUANT_TYPES)}.")
    return resolved, _QUANT_TYPE_ACT_BITS[resolved]


def get_target_platform(
    target: str,
    quant_type: Union[str, int] = "w8a8",
    debug: bool = False,
    **kwargs: Any,
):
    """Return the ESP-DL ``TargetPlatform`` for a chip and quantization scheme.

    New API::

        get_target_platform(target, quant_type="w8a8", debug=False)

    Legacy API (still supported)::

        get_target_platform(target, num_of_bits=8, float=False, hi_precision=False)
        get_target_platform(target, 16)

    Args:
        target: Chip name, e.g. ``esp32p4``, ``esp32s3``, ``esp32c3``, ``c``.
        quant_type: Quantization scheme: ``w8a8``, ``w8a16``, ``w16a16``, or ``none`` (FP32).
            An ``int`` is accepted as the legacy ``num_of_bits`` positional argument.
        debug: Use high-precision INT16 simulation (PPQ double-precision forward).
            Only affects ``w16a16``. Alias of the legacy ``hi_precision`` argument.
        num_of_bits (int, optional): Legacy activation bit width, 8 or 16.
        float (bool, optional): Legacy switch for FP32. Prefer ``quant_type="none"``.
        hi_precision (bool, optional): Legacy alias of ``debug``. When the operator is
            quantified at 16-bit, PPQ performs forward calculations with double precision,
            which, although maintaining high precision at 16-bit, may differ in precision
            from ESP-DL's operator. Currently, conv2d, gemm, and matmul are consistent
            in precision with PPQ's quantized results when calculated with double
            precision during 16-bit operations.
    """

    target = target.lower()
    use_float = kwargs.get("float", False)
    hi_precision = bool(debug) or bool(kwargs.get("hi_precision", False))
    num_of_bits = kwargs.get("num_of_bits", None)

    resolved = None
    if use_float:
        resolved = "none"
    elif isinstance(quant_type, int):
        num_of_bits = quant_type
        resolved = _quant_type_from_bits(num_of_bits)
    elif num_of_bits is not None and quant_type == "w8a8":
        # Legacy keyword call: get_target_platform(target, num_of_bits=16)
        resolved = _quant_type_from_bits(num_of_bits)
    elif isinstance(quant_type, str):
        resolved = quant_type.lower()
        if resolved not in _VALID_QUANT_TYPES:
            raise ValueError(f"Unsupported quant_type '{quant_type}'. Expected one of {sorted(_VALID_QUANT_TYPES)}.")
    else:
        raise TypeError(f"quant_type must be str or int, got {type(quant_type).__name__}.")

    if resolved == "none":
        return TargetPlatform.FP32

    group = _target_group(target)
    table_key = "w16a16_debug" if resolved == "w16a16" and hi_precision else resolved
    platform = _PLATFORM_BY_QUANT.get(table_key, {}).get(group)
    if platform is None:
        platform = TargetPlatform.FP32
        logger.warning(
            f"quant_type:{resolved}, num_of_bits:{num_of_bits}, target:{target}, "
            f"and debug/hi_precision:{hi_precision} will return TargetPlatform.FP32"
        )
    return platform


def insert_streaming_cache_on_var(
    var_name: str, window_size: int, op_name: str = None, frame_axis: int = 1
) -> Dict[str, Any]:
    """
    Get streaming attributes dictionary.
    Args:
        var_name (str): Name of the variable.
        window_size (int): Size of the streaming window.
        op_name (str, optional): Name of the src or dest operation. Defaults to None.
        frame_axis (int, optional): Axis representing the frame dimension. Defaults to 1. (NCH/NCHW: 2, NHWC: 1)
    Returns:
        Dict[str, Any]: Dictionary containing streaming attributes.
    """
    attributes = {
        "window_size": window_size,
        "op_name": op_name,
        "frame_axis": frame_axis,
    }
    return {var_name: attributes}


def get_random_inputs(input_shape: List[Any], dtype=torch.float32, device="cpu") -> List[Any]:
    if not isinstance(input_shape[0], list):
        input_shape = [input_shape]

    inputs = [torch.randn(size=shape, device=device, dtype=dtype) for shape in input_shape]

    return inputs


def generate_test_value(
    graph: BaseGraph,
    running_device: str,
    inputs: Union[dict, list, torch.Tensor],
    output_names: List[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    test_inputs_value = {}
    test_outputs_value = {}

    executor = TorchExecutor(graph=graph, device=running_device)
    outputs = executor.forward(inputs=inputs, output_names=output_names)
    # get test_inputs_value
    if isinstance(inputs, dict):
        for name, value in inputs.items():
            if name in graph.inputs:
                test_inputs_value[name] = value.clone().detach().to(running_device)
            else:
                logger.error(f"Can not find input {name} in your graph inputs, please check.")
    else:
        inputs_tmp = executor.prepare_input(inputs=inputs)
        test_inputs_value = {name: value.clone().detach().to(running_device) for name, value in inputs_tmp.items()}

    # get test_outputs_value
    if output_names is None:
        outputs_dictionary = graph.outputs
        test_outputs_value = {
            key: outputs[idx].clone().detach().to(running_device) for idx, key in enumerate(outputs_dictionary)
        }
    else:
        test_outputs_value = {
            output_name: output.clone().detach().to(running_device)
            for output_name, output in zip(output_names, outputs)
        }

    return {"inputs": test_inputs_value, "outputs": test_outputs_value}


def collate_fn_template(batch: Union[torch.Tensor, List[torch.Tensor]], dtype=torch.float32, device="cpu"):
    def _convert(tensor: torch.Tensor) -> torch.Tensor:
        # A bool input is a condition (Where, logical ops). Casting it to float
        # makes torch.where reject the tensor during calibration.
        if tensor.dtype == torch.bool:
            return tensor.to(device)
        return tensor.type(dtype).to(device)

    if isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        return [_convert(x) for x in batch]
    elif isinstance(batch, torch.Tensor):
        return _convert(batch)
    else:
        logger.error("please provide a valid collate_fn.")


@empty_ppq_cache
def espdl_quantize_onnx(
    onnx_import_file: str,
    espdl_export_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[Any],
    inputs: List[Any] = None,
    target: str = "esp32p4",
    num_of_bits: int = 8,
    collate_fn: Callable = None,
    dispatching_override: Dict[str, TargetPlatform] = None,
    dispatching_method: str = "allin",
    setting: QuantizationSetting = None,
    device: str = "cpu",
    error_report: bool = True,
    skip_export: bool = False,
    export_config: bool = True,
    export_test_values: bool = False,
    test_output_names: List[str] = None,
    auto_streaming: bool = False,
    streaming_table: Dict[str, Dict[str, Any]] = None,
    streaming_input_shape: List[Any] = None,
    streaming_custom_passes: Any = None,
    verbose: int = 0,
    quant_type: Optional[str] = None,
    debug: bool = False,
    **kwargs: Any,
) -> BaseGraph:
    """Quantize onnx model and return quantized ppq graph and executor .

    Args:
        onnx_import_file (str): onnx model file path
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4", "esp32s31", "esp32s3" or "c"
        num_of_bits: legacy quantizer bit width, 8 or 16. Prefer ``quant_type``.
        quant_type: quantization scheme: ``w8a8``, ``w8a16``, ``w16a16``, or ``none`` (FP32).
                    When set, it overrides ``num_of_bits``.
        debug: use high-precision INT16 simulation. Only affects ``w16a16``.
               Alias of the legacy ``hi_precision`` argument.
        collate_fn (Callable): batch collate func for preprocessing
        dispatching_override (deprecated): override dispatching result.
                                           It is recommended to use the `setting` parameter, as this argument will be removed in the next version.
        dispatching_method (deprecated): Refer to https://github.com/espressif/esp-ppq/blob/master/ppq/scheduler/__init__.py#L8.
                                         It is recommended to use the `setting` parameter, as this argument will be removed in the next version.
        setting (QuantizationSetting): Quantization setting, default espdl setting will be used when set None.
                                      ``espdl_attention_fusion`` controls primitive Attention fusion in both
                                      quantized and ``quant_type="none"`` FP32 export paths.
        device (str, optional):  execution device, defaults to 'cpu'.
        error_report (bool, optional): whether to print error report, defaults to True.
        skip_export (bool, optional): whether to export the quantized model, defaults to False.
        export_config (bool, optional): whether to export the quantization configuration, defaults to True.
        export_test_values (bool, optional): whether to export the test values, defaults to False.
        test_output_names (List[str], optional): tensor names of the model want to test, defaults to None.
        verbose (int, optional): whether to print details, defaults to 0.
        hi_precision (bool, optional): It's in kwargs. Legacy alias of ``debug``.
        metadata_props (Dict[str, str], optional): It's in kwargs. You can add custom key-value pairs to the model.

    Returns:
        BaseGraph:      The Quantized Graph, containing all information needed for backend execution
    """

    model = onnx.load(onnx_import_file)
    try:
        model_sim, check = simplify(model)
        if check:
            onnx.save(model_sim, onnx_import_file)
    except RuntimeError as e:
        if "No Op registered for" in str(e):
            ppq_warning(f"ONNX simplifier skipped due to custom operator: {e}")
        else:
            raise

    export_path = os.path.dirname(os.path.abspath(espdl_export_file))
    os.makedirs(export_path, exist_ok=True)

    # ------------------------------------------------------------
    #
    # 1: Quantize ONNX Model.
    #
    #  ------------------------------------------------------------
    if calib_dataloader is None or calib_steps is None:
        raise TypeError("Quantization needs a valid calib_dataloader and calib_steps setting.")
    quant_type, num_of_bits = _resolve_quant_type_and_bits(quant_type, num_of_bits)
    target_platform = get_target_platform(target, quant_type, debug=debug, **kwargs)
    input_dtype = torch.float32

    if not collate_fn:
        collate_fn = partial(collate_fn_template, dtype=input_dtype, device=device)

    ppq_graph = None
    if inputs:
        dummy_inputs = inputs
    else:
        dummy_inputs = get_random_inputs(input_shape, input_dtype, device)

    if target_platform != TargetPlatform.FP32:
        if dispatching_override is not None or dispatching_method != "allin":
            logger.warning(
                "It is recommended to use the setting parameter. The dispatching_override and dispatching_method will be deprecated."
            )

        if setting is None:
            setting = QuantizationSettingFactory.espdl_setting(num_of_bits=num_of_bits)

        if dispatching_method != "allin":
            setting.dispatcher = dispatching_method

        if dispatching_override is not None:
            for opname, platform in dispatching_override.items():
                setting.dispatching_table.append(opname, platform)

        ppq_graph = quantize_onnx_model(
            onnx_import_file=onnx_import_file,
            calib_dataloader=calib_dataloader,
            calib_steps=calib_steps,
            input_shape=None,
            platform=target_platform,
            input_dtype=input_dtype,
            inputs=dummy_inputs,
            setting=setting,
            collate_fn=collate_fn,
            device=device,
            verbose=verbose,
            do_quantize=True,
        )

        # ------------------------------------------------------------
        #
        # 2: Analyze Quantization Errors.
        #
        # ------------------------------------------------------------
        if error_report:
            graphwise_error_analyse(
                graph=ppq_graph,
                running_device=device,
                collate_fn=collate_fn,
                dataloader=calib_dataloader,
            )

            layerwise_error_analyse(
                graph=ppq_graph,
                running_device=device,
                collate_fn=collate_fn,
                dataloader=calib_dataloader,
            )
    else:
        # support TargetPlatform.FP32
        ppq_graph = EspdlExporter.prepare_graph_for_float_export(
            load_onnx_graph(onnx_import_file=onnx_import_file),
            platform=get_target_platform(target, "w8a8"),
            inputs=dummy_inputs,
            device=device,
            attention_fusion=bool(getattr(setting, "espdl_attention_fusion", True)),
        )
        target_platform = TargetPlatform.ESPDL_INT8

    # ------------------------------------------------------------
    #
    # 3: Export ESPDL Model.
    #
    # ------------------------------------------------------------
    if not skip_export:
        values_for_test = None
        if export_test_values:
            values_for_test = generate_test_value(ppq_graph, device, dummy_inputs, test_output_names)

        PFL.Exporter(platform=target_platform).export(
            file_path=espdl_export_file,
            graph=ppq_graph,
            values_for_test=values_for_test,
            export_config=export_config,
            auto_streaming=auto_streaming,
            streaming_table=streaming_table,
            streaming_input_shape=streaming_input_shape,
            streaming_custom_passes=streaming_custom_passes,
            **kwargs,
        )
    return ppq_graph


def espdl_quantize_torch(
    model: torch.nn.Module,
    espdl_export_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[Any],
    inputs: List[Any] = None,
    target: str = "esp32p4",
    num_of_bits: int = 8,
    collate_fn: Callable = None,
    dispatching_override: Dict[str, TargetPlatform] = None,
    dispatching_method: str = "allin",
    setting: QuantizationSetting = None,
    device: str = "cpu",
    error_report: bool = True,
    skip_export: bool = False,
    export_config: bool = True,
    export_test_values: bool = False,
    test_output_names: List[str] = None,
    verbose: int = 0,
    opset_version: int = 18,
    auto_streaming: bool = False,
    streaming_table: Dict[str, Dict[str, Any]] = None,
    streaming_input_shape: Dict[str, List[int]] = None,
    streaming_custom_passes: Any = None,
    quant_type: Optional[str] = None,
    debug: bool = False,
    **kwargs: Any,
) -> BaseGraph:
    """Quantize torch model and return quantized ppq graph and executor .

    Args:
        model (torch.nn.Module): torch model
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4", "esp32s31", "esp32s3" or "c"
        num_of_bits: legacy quantizer bit width, 8 or 16. Prefer ``quant_type``.
        quant_type: quantization scheme: ``w8a8``, ``w8a16``, ``w16a16``, or ``none`` (FP32).
                    When set, it overrides ``num_of_bits``.
        debug: use high-precision INT16 simulation. Only affects ``w16a16``.
               Alias of the legacy ``hi_precision`` argument.
        collate_fn (Callable): batch collate func for preprocessing
        dispatching_override (deprecated): override dispatching result.
                                        It is recommended to use the `setting` parameter, as this argument will be removed in the next version.
        dispatching_method (deprecated): Refer to https://github.com/espressif/esp-ppq/blob/master/ppq/scheduler/__init__.py#L8.
                                        It is recommended to use the `setting` parameter, as this argument will be removed in the next version.
        setting (QuantizationSetting): Quantization setting, default espdl setting will be used when set None
        device (str, optional):  execution device, defaults to 'cpu'.
        error_report (bool, optional): whether to print error report, defaults to True.
        skip_export (bool, optional): whether to export the quantized model, defaults to False.
        export_config (bool, optional): whether to export the quantization configuration, defaults to True.
        export_test_values (bool, optional): whether to export the test values, defaults to False.
        test_output_names (List[str], optional): tensor names of the model want to test, defaults to None.
        verbose (int, optional): whether to print details, defaults to 0.
        hi_precision (bool, optional): It's in kwargs. Legacy alias of ``debug``.
        metadata_props (Dict[str, str], optional): It's in kwargs. You can add custom key-value pairs to the model.

    Returns:
        BaseGraph:      The Quantized Graph, containing all information needed for backend execution
    """
    if not isinstance(input_shape[0], list):
        input_shape = [input_shape]
    export_path = os.path.dirname(os.path.abspath(espdl_export_file))
    os.makedirs(export_path, exist_ok=True)

    # step1: export onnx model
    model = model.eval()
    model = model.to(device)

    base_file_name, _ = os.path.splitext(espdl_export_file)
    onnx_file_path = base_file_name + ".onnx"
    if torch.__version__ >= "2.9.0":
        torch.onnx.export(
            model=model,
            args=tuple(
                [
                    torch.zeros(
                        size=shape,
                        device=device,
                        dtype=torch.float32,
                    )
                    for shape in input_shape
                ]
            ),
            f=onnx_file_path,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            model=model,
            args=tuple(
                [
                    torch.zeros(
                        size=shape,
                        device=device,
                        dtype=torch.float32,
                    )
                    for shape in input_shape
                ]
            ),
            f=onnx_file_path,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    # step2: quantize onnx model and export espdl model
    return espdl_quantize_onnx(
        onnx_import_file=onnx_file_path,
        espdl_export_file=espdl_export_file,
        calib_dataloader=calib_dataloader,
        calib_steps=calib_steps,
        input_shape=input_shape,
        inputs=inputs,
        target=target,
        num_of_bits=num_of_bits,
        quant_type=quant_type,
        debug=debug,
        collate_fn=collate_fn,
        dispatching_override=dispatching_override,
        dispatching_method=dispatching_method,
        setting=setting,
        device=device,
        error_report=error_report,
        skip_export=skip_export,
        export_config=export_config,
        export_test_values=export_test_values,
        test_output_names=test_output_names,
        auto_streaming=auto_streaming,
        streaming_table=streaming_table,
        streaming_input_shape=streaming_input_shape,
        streaming_custom_passes=streaming_custom_passes,
        verbose=verbose,
        **kwargs,
    )

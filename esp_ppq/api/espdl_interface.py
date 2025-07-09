# -*- coding: utf-8 -*-
import os
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
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
from esp_ppq.core import QuantizationVisibility, TargetPlatform, empty_ppq_cache
from esp_ppq.executor import BaseGraphExecutor, TorchExecutor
from esp_ppq.IR import BaseGraph
from esp_ppq.log import NaiveLogger
from esp_ppq.quantization.analyse import graphwise_error_analyse
from esp_ppq.quantization.analyse.layerwise import layerwise_error_analyse
from esp_ppq.quantization.optim import *

logger = NaiveLogger.get_logger('ESPDL')


def get_target_platform(
    target: str,
    num_of_bits: int = 8,
    float: bool = False,
    **kwargs: Any,
):
    """Quantize onnx model and return quantized ppq graph and executor .

    Args:
        hi_precision (bool, optional): When the operator is quantified at 16-bit, does PPQ perform forward calculations with double precision,
                                    which, although maintaining high precision at 16-bit, may differ in precision from ESP-DL's operator.
                                    Currently, conv2d, gemm, and matmal are consistent in precision with PPQ's quantized results when calculated
                                    with double precision during 16-bit operations.
    """

    platform = None
    target = target.lower()
    hi_precision = kwargs.get('hi_precision', False)

    if float:
        platform = TargetPlatform.FP32
    else:
        if num_of_bits == 8 and target == "esp32p4":
            platform = TargetPlatform.ESPDL_INT8
        elif num_of_bits == 16 and target == "esp32p4" and not hi_precision:
            platform = TargetPlatform.ESPDL_INT16
        elif num_of_bits == 8 and target == "esp32s3":
            platform = TargetPlatform.ESPDL_S3_INT8
        elif num_of_bits == 16 and target == "esp32s3" and not hi_precision:
            platform = TargetPlatform.ESPDL_S3_INT16
        elif num_of_bits == 16 and target == "esp32p4" and hi_precision:
            platform = TargetPlatform.ESPDL_H_PRE_INT16
        elif num_of_bits == 16 and target == "esp32s3" and hi_precision:
            platform = TargetPlatform.ESPDL_S3_H_PRE_INT16
        elif num_of_bits == 8 and target == "c":
            platform = TargetPlatform.ESPDL_C_INT8
        elif num_of_bits == 16 and target == "c" and not hi_precision:
            platform = TargetPlatform.ESPDL_C_INT16
        elif num_of_bits == 16 and target == "c" and hi_precision:
            platform = TargetPlatform.ESPDL_C_H_PRE_INT16
        else:
            platform = TargetPlatform.FP32
            logger.warning(f"Do not support num_of_bits:{num_of_bits}, will change to TargetPlatform.FP32")

    return platform


def get_random_inputs(input_shape: List[Any], dtype=torch.float32, device='cpu') -> List[Any]:
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
                test_inputs_value[name] = value.clone().detach().cpu()
            else:
                logger.error(f"Can not find input {name} in your graph inputs, please check.")
    else:
        inputs_tmp = executor.prepare_input(inputs=inputs)
        test_inputs_value = {name: value.clone().detach().cpu() for name, value in inputs_tmp.items()}

    # get test_outputs_value
    if output_names is None:
        outputs_dictionary = graph.outputs
        test_outputs_value = {key: outputs[idx].clone().detach().cpu() for idx, key in enumerate(outputs_dictionary)}
    else:
        test_outputs_value = {
            output_name: output.clone().detach().cpu() for output_name, output in zip(output_names, outputs)
        }

    return {"inputs": test_inputs_value, "outputs": test_outputs_value}


def collate_fn_template(batch: Union[torch.Tensor, List[torch.Tensor]], dtype=torch.float32, device='cpu'):
    if isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        return [x.type(dtype).to(device) for x in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.type(dtype).to(device)
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
    dispatching_method: str = "conservative",
    setting: QuantizationSetting = None,
    device: str = "cpu",
    error_report: bool = True,
    skip_export: bool = False,
    export_config: bool = True,
    export_test_values: bool = False,
    test_output_names: List[str] = None,
    verbose: int = 0,
    **kwargs: Any,
) -> BaseGraph:
    """Quantize onnx model and return quantized ppq graph and executor .

    Args:
        onnx_import_file (str): onnx model file path
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4", "esp32s3" or "c"
        num_of_bits: the number of quantizer bits, 8 or 16
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
        hi_precision (bool, optional): It's in kwargs. When the operator is quantified at 16-bit, does PPQ perform forward calculations with double precision,
                                       which, although maintaining high precision at 16-bit, may differ in precision from ESP-DL's operator.
                                       Currently, conv2d, gemm, and matmal are consistent in precision with PPQ's quantized results when calculated
                                       with double precision during 16-bit operations.
        metadata_props (Dict[str, str], optional): It's in kwargs. You can add custom key-value pairs to the model.

    Returns:
        BaseGraph:      The Quantized Graph, containing all information needed for backend execution
    """

    model = onnx.load(onnx_import_file)
    model_sim, check = simplify(model)
    if check:
        onnx.save(model_sim, onnx_import_file)

    export_path = os.path.dirname(os.path.abspath(espdl_export_file))
    os.makedirs(export_path, exist_ok=True)

    # ------------------------------------------------------------
    #
    # 1: Quantize ONNX Model.
    #
    #  ------------------------------------------------------------
    if calib_dataloader is None or calib_steps is None:
        raise TypeError("Quantization needs a valid calib_dataloader and calib_steps setting.")
    target_platform = get_target_platform(target, num_of_bits, **kwargs)
    input_dtype = torch.float32

    if not collate_fn:
        collate_fn = partial(collate_fn_template, dtype=input_dtype, device=device)

    ppq_graph = None
    if inputs:
        dummy_inputs = inputs
    else:
        dummy_inputs = get_random_inputs(input_shape, input_dtype, device)

    if target_platform != TargetPlatform.FP32:
        if dispatching_override is not None or dispatching_method != "conservative":
            logger.warning(
                "It is recommended to use the setting parameter. The dispatching_override and dispatching_method will be deprecated."
            )

        if setting is None:
            setting = QuantizationSettingFactory.espdl_setting()

        if dispatching_method != "conservative":
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
        ppq_graph = load_onnx_graph(onnx_import_file=onnx_import_file)
        executor = TorchExecutor(graph=ppq_graph, device=device)
        executor.tracing_operation_meta(inputs=dummy_inputs)
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
    dispatching_method: str = "conservative",
    setting: QuantizationSetting = None,
    device: str = "cpu",
    error_report: bool = True,
    skip_export: bool = False,
    export_config: bool = True,
    export_test_values: bool = False,
    test_output_names: List[str] = None,
    verbose: int = 0,
    **kwargs: Any,
) -> BaseGraph:
    """Quantize torch model and return quantized ppq graph and executor .

    Args:
        model (torch.nn.Module): torch model
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4", "esp32s3" or "c"
        num_of_bits: the number of quantizer bits, 8 or 16
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
        hi_precision (bool, optional): It's in kwargs. When the operator is quantified at 16-bit, does PPQ perform forward calculations with double precision,
                                       which, although maintaining high precision at 16-bit, may differ in precision from ESP-DL's operator.
                                       Currently, conv2d, gemm, and matmal are consistent in precision with PPQ's quantized results when calculated
                                       with double precision during 16-bit operations.
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
        opset_version=13,
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
        verbose=verbose,
        **kwargs,
    )

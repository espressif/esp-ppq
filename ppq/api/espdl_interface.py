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

import ppq.lib as PFL
from ppq.api.interface import load_onnx_graph
from ppq.core import QuantizationVisibility, TargetPlatform, empty_ppq_cache
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph
from ppq.log import NaiveLogger
from ppq.quantization.analyse import graphwise_error_analyse
from ppq.quantization.analyse.layerwise import layerwise_error_analyse
from ppq.quantization.optim import *

logger = NaiveLogger.get_logger('ESPDL')

def get_target_platform(target: str, num_of_bits: int = 8, float: bool = False):
    platform = None
    if float:
        platform = TargetPlatform.FP32
    else:
        if num_of_bits == 8 and target == "esp32p4":
            platform = TargetPlatform.ESPDL_INT8
        elif num_of_bits == 16 and target == "esp32p4":
            platform = TargetPlatform.ESPDL_INT16
        elif num_of_bits == 8 and target == "esp32s3":
            platform = TargetPlatform.ESPDL_S3_INT8
        else:
            platform = TargetPlatform.FP32
            logger.warning(f"Do not support num_of_bits:{num_of_bits}, will change to TargetPlatform.FP32")
    
    return platform

def get_random_inputs(input_shape: List[Any], dtype=torch.float32, device='cpu') -> List[Any]:
    if not isinstance(input_shape[0], list):
        input_shape = [input_shape]

    inputs = [
        torch.rand(size=shape, device=device, dtype=dtype)
        for shape in input_shape
    ]
    
    return inputs


def generate_test_value(
    graph: BaseGraph,
    executor: BaseGraphExecutor,
    inputs: Union[dict, list, torch.Tensor],
    output_names: List[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    test_inputs_value = {}
    test_outputs_value = {}

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
        test_inputs_value = {
            name: value.clone().detach().cpu() for name, value in inputs_tmp.items()
        }

    # get test_outputs_value
    if output_names is None:
        outputs_dictionary = graph.outputs
        test_outputs_value = {
            key: outputs[idx].clone().detach().cpu()
            for idx, key in enumerate(outputs_dictionary)
        }
    else:
        test_outputs_value = {
            output_name: output.clone().detach().cpu()
            for output_name, output in zip(output_names, outputs)
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
    target:str = "esp32p4",
    num_of_bits:int = 8,
    collate_fn: Callable = None,
    dispatching_override: Dict[str, TargetPlatform] = None,
    dispatching_method: str = "conservative",
    device: str = "cpu",
    error_report: bool = True,
    skip_export: bool = False,
    export_config: bool = True,
    export_test_values: bool = False,
    test_output_names: List[str] = None,
    verbose: int = 0,
) -> Tuple[BaseGraph, TorchExecutor] :
    """Quantize onnx model and return quantized ppq graph and executor .
    
    Args:
        onnx_import_file (str): onnx model file path
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4" and "esp32s3"
        num_of_bits: the number of quantizer bits, 8 or 16
        collate_fn (Callable): batch collate func for preprocessing
        dispatching_override: override dispatching result.
        dispatching_method: Refer to https://github.com/espressif/esp-ppq/blob/master/ppq/scheduler/__init__.py#L8
        device (str, optional):  execution device, defaults to 'cpu'.
        error_report (bool, optional): whether to print error report, defaults to True.
        skip_export (bool, optional): whether to export the quantized model, defaults to False.
        export_config (bool, optional): whether to export the quantization configuration, defaults to True.
        export_test_values (bool, optional): whether to export the test values, defaults to False.
        test_output_names (List[str], optional): tensor names of the model want to test, defaults to None.
        verbose (int, optional): whether to print details, defaults to 0.

    Returns:
        BaseGraph:      The Quantized Graph, containing all information needed for backend execution
        TorchExecutor:  PPQ Graph Executor 
    """    
    
    export_path = os.path.dirname(os.path.abspath(espdl_export_file))
    os.makedirs(export_path, exist_ok=True)

    # ------------------------------------------------------------
    #
    # 1: Quantize ONNX Model.
    #
    #  ------------------------------------------------------------
    if calib_dataloader is None or calib_steps is None:
        raise TypeError(
            "Quantization needs a valid calib_dataloader and calib_steps setting."
        )
    target_platform = get_target_platform(target, num_of_bits)
    input_dtype = torch.float32
    if num_of_bits == 16:
        input_dtype = torch.float64
    
    if not collate_fn:
        collate_fn = partial(collate_fn_template, dtype=input_dtype, device=device)

    ppq_graph = load_onnx_graph(onnx_import_file=onnx_import_file)
    if inputs:
        dummy_inputs = inputs
    else:
        dummy_inputs = get_random_inputs(input_shape, input_dtype, device)
    
    if target_platform != TargetPlatform.FP32:
        quantizer = PFL.Quantizer(platform=target_platform, graph=ppq_graph)
        dispatching_table = PFL.Dispatcher(
            graph=ppq_graph, method=dispatching_method
        ).dispatch(quantizer.quant_operation_types)

        # Override dispatching result
        if dispatching_override is not None:
            for opname, platform in dispatching_override.items():
                if opname not in ppq_graph.operations:
                    continue
                assert isinstance(platform, int) or isinstance(platform, TargetPlatform), (
                    f"Your dispatching_override table contains a invalid setting of operation {opname}, "
                    "All platform setting given in dispatching_override table is expected given as int or TargetPlatform, "
                    f"however {type(platform)} was given."
                )
                dispatching_table[opname] = TargetPlatform(platform)

        for opname, platform in dispatching_table.items():
            if platform == TargetPlatform.UNSPECIFIED:
                dispatching_table[opname] = target_platform

        # initial quantizer
        for op in ppq_graph.operations.values():
            quantizer.quantize_operation(
                op_name=op.name, platform=dispatching_table[op.name]
            )
        executor = TorchExecutor(graph=ppq_graph, device=device)
        executor.tracing_operation_meta(inputs=dummy_inputs)

        # Create the optimization pipeline, 
        pipeline = PFL.Pipeline(
            [
                QuantizeSimplifyPass(),
                QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
                ParameterQuantizePass(),
                RuntimeCalibrationPass(method="kl"),
                PassiveParameterQuantizePass(
                    clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE
                ),
                QuantAlignmentPass(elementwise_alignment="Align to Output"),
                # LearnedStepSizePass(steps=500, block_size=5)
            ]
        )
        logger.info(
            f"Calibration dataset samples: {len(calib_dataloader.dataset)}, len(Calibrate iter): {len(calib_dataloader)}"
        )
        pipeline.optimize(
            calib_steps=calib_steps,
            collate_fn=collate_fn,
            graph=ppq_graph,
            dataloader=calib_dataloader,
            executor=executor,
        )
        if verbose:
            logger.info(quantizer.report())
        logger.info("Network Quantization Finished.")

        
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
            values_for_test = generate_test_value(ppq_graph, executor, dummy_inputs, test_output_names)

        PFL.Exporter(platform=target_platform).export(
            file_path=espdl_export_file,
            graph=ppq_graph,
            values_for_test=values_for_test,
            export_config=export_config,
        )
    return ppq_graph, executor


def espdl_quantize_torch(
    model: torch.nn.Module,
    espdl_export_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[Any],
    inputs: List[Any] = None,
    target:str = "esp32p4",
    num_of_bits:int = 8,
    collate_fn: Callable = None,
    dispatching_override: Dict[str, TargetPlatform] = None,
    dispatching_method: str = "conservative",
    device: str = "cpu",
    error_report: bool = True,
    skip_export: bool = False,
    export_config: bool = True,
    export_test_values: bool = False,
    test_output_names: List[str] = None,
    verbose: int = 0,
) -> Tuple[BaseGraph, TorchExecutor]:
    """Quantize torch model and return quantized ppq graph and executor .
    
    Args:
        model (torch.nn.Module): torch model
        calib_dataloader (DataLoader): calibration data loader
        calib_steps (int): calibration steps
        input_shape (List[int]):a list of ints indicating size of inputs and batch size must be 1
        inputs (List[str]): a list of Tensor and batch size must be 1
        target: target chip, support "esp32p4" and "esp32s3"
        num_of_bits: the number of quantizer bits, 8 or 16
        collate_fn (Callable): batch collate func for preprocessing
        dispatching_override: override dispatching result.
        dispatching_method: Refer to https://github.com/espressif/esp-ppq/blob/master/ppq/scheduler/__init__.py#L8
        device (str, optional):  execution device, defaults to 'cpu'.
        error_report (bool, optional): whether to print error report, defaults to True.
        skip_export (bool, optional): whether to export the quantized model, defaults to False.
        export_config (bool, optional): whether to export the quantization configuration, defaults to True.
        export_test_values (bool, optional): whether to export the test values, defaults to False.
        test_output_names (List[str], optional): tensor names of the model want to test, defaults to None.
        verbose (int, optional): whether to print details, defaults to 0.

    Returns:
        BaseGraph:      The Quantized Graph, containing all information needed for backend execution
        TorchExecutor:  PPQ Graph Executor 
    """   
    if not isinstance(input_shape[0], list):
        input_shape = [input_shape]
    export_path = os.path.dirname(os.path.abspath(espdl_export_file))
    os.makedirs(export_path, exist_ok=True)

    # step1: export onnx model    
    model = model.eval()
    model = model.to(device)
    if num_of_bits == 16:
        model = model.double()

    base_file_name, _ = os.path.splitext(espdl_export_file)
    onnx_file_path = base_file_name + ".onnx"
    torch.onnx.export(
        model=model,
        args=tuple(
            [
                torch.zeros(
                    size=[1] + shape[1:],  # reset batch_size=1
                    device=device,
                    dtype=torch.float32 if num_of_bits == 8 else torch.float64,
                )
                for shape in input_shape
            ]
        ),
        f=onnx_file_path,
        opset_version=13,
        do_constant_folding=True,
    )

    model = onnx.load(onnx_file_path)
    model_sim, check = simplify(model)
    if check:
        onnx.save(model_sim, onnx_file_path)
    
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
        device=device,
        error_report=error_report,
        skip_export=skip_export,
        export_config=export_config,
        export_test_values=export_test_values,
        test_output_names=test_output_names,
        verbose=verbose,
    )

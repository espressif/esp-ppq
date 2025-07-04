import json
import os
from typing import Any, Callable, Iterable, List, Union

import torch
from torch.utils.data import DataLoader

import ppq.lib as PFL
from ppq.api.setting import (DispatchingTable, QuantizationSetting,
                             QuantizationSettingFactory)
from ppq.core import (PPQ_CONFIG, NetworkFramework, TargetPlatform,
                      empty_ppq_cache, ppq_warning)
from ppq.core.common import (FORMATTER_FORMAT_CONSTANT_INPUT,
                             FORMATTER_FUSE_BIAS_ADD, FORMATTER_FUSE_BN,
                             FORMATTER_REPLACE_BN_TO_CONV,
                             FORMATTER_REMOVE_IDENTITY,
                             FORMATTER_REMOVE_ISOLATED,
                             FORMATTER_FUSE_SWISH)
from ppq.executor import TorchExecutor
from ppq.executor.base import BaseGraphExecutor
from ppq.IR import (BaseGraph, GraphBuilder, GraphCommand, GraphCommandType,
                    GraphFormatter, GraphMerger, GraphReplacer)
from ppq.IR.quantize import QuantableOperation
from ppq.quantization.optim.base import QuantizationOptimizationPass
from ppq.quantization.quantizer import BaseQuantizer
from ppq.scheduler import DISPATCHER_TABLE, GraphDispatcher


def load_graph(file_path: str, from_framework: NetworkFramework=NetworkFramework.ONNX, **kwargs) -> BaseGraph:
    parser = PFL.Parser(from_framework)
    assert isinstance(parser, GraphBuilder), 'Unexpected Parser found.'
    if from_framework == NetworkFramework.CAFFE:
        assert 'caffemodel_path' in kwargs, ('parameter "caffemodel_path" is required here for loading caffe model from file, '
                                             'however it is missing from your invoking.')
        graph = parser.build(prototxt_path=file_path, caffemodel_path=kwargs['caffemodel_path'])
    else:
        graph = parser.build(file_path)
    return graph

def load_onnx_graph(onnx_import_file: str) -> BaseGraph:
    """
        从一个指定位置加载 onnx 计算图，注意该加载的计算图尚未经过调度，此时所有算子被认为是可量化的
        load onnx graph from the specified location
    Args:
        onnx_import_file (str): onnx 计算图的保存位置 the specified location

    Returns:
        BaseGraph: 解析 onnx 获得的 ppq 计算图对象 the parsed ppq IR graph
    """
    ppq_ir = load_graph(onnx_import_file, from_framework=NetworkFramework.ONNX)
    return format_graph(graph=ppq_ir)

def load_caffe_graph(prototxt_path: str, caffemodel_path: str) -> BaseGraph:
    """
        从一个指定位置加载 caffe 计算图，注意该加载的计算图尚未经过调度，此时所有算子被认为是可量化的
        load caffe graph from the specified location
    Args:
        prototxt_path (str): caffe prototxt的保存位置 the specified location of caffe prototxt
        caffemodel_path (str): caffe weight的保存位置 the specified location of caffe weight

    Returns:
        BaseGraph: 解析 caffe 获得的 ppq 计算图对象 the parsed ppq IR graph
    """
    ppq_ir = load_graph(file_path=prototxt_path, caffemodel_path=caffemodel_path, from_framework=NetworkFramework.CAFFE)
    return format_graph(graph=ppq_ir)

def load_native_graph(import_file: str) -> BaseGraph:
    """
        从一个指定位置加载 原生计算图，原生计算图中包含了所有量化与调度信息
        load native graph from the specified location
    Args:
        import_file (str): 计算图的保存位置 the specified location

    Returns:
        BaseGraph: 解析获得的 ppq 计算图对象 the parsed ppq IR graph
    """
    return load_graph(import_file, from_framework=NetworkFramework.NATIVE)

def load_torch_model(
    model: torch.nn.Module, onnx_file_name: str = 'MyModel.onnx', 
    sample: Union[dict, list, torch.Tensor] = None, device = 'cuda', 
    verbose: bool = False, opset: int = 11, do_constant_folding: bool = None, 
    dynamic_axes: dict = None, training: bool = False) -> BaseGraph:
    """
    Convert a Pytorch Module to PPQ graph, 
        this conversation exports pytorch model to onnx first,
        then PPQ will read from onnx file to build graph.

    Args:
        model (torch.nn.Module): Converting Model.
    
        onnx_file_name (str, optional): Exporting Onnx File name, by default it is 'MyModel.onnx'.
        
        sample (Union[dict, list, torch.Tensor], optional): 
            A sample input for tracing your pytorch model.
            
            Pytorch will feed this sample input to your model,
            recording record every layer and functions during executing.
            
            Each function recorded will be translated into onnx.
            The input and output shape will also depends on this sample input.

        device (str, optional): _description_. Defaults to 'cuda'.
        
        verbose (bool, optional): _description_. Defaults to True.
        
        opset (int, optional): _description_. Defaults to 11.
        
        do_constant_folding (bool, optional): 
            Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes.
        
        dynamic_axes (dict, optional): 
            (dict<string, dict<int, string>> or dict<string, list(int)>, default empty dict):
            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args``. To specify axes of tensors as
            dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:

    Returns:
        BaseGraph: Converted PPQ Graph Instance.
    """
    
    if opset not in {11, 13}:
        ppq_warning(f'The Opset of Exporting Model should be 11 or 13, however {opset} was given.')

    model = model.eval()
    model = model.to(device)
    
    _training = torch.onnx.TrainingMode.TRAINING if training else torch.onnx.TrainingMode.EVAL
    torch.onnx.export(
        model=model, args=sample,
        verbose=verbose, f=onnx_file_name, opset_version=opset,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes, 
        training=_training)
    
    return load_onnx_graph(onnx_import_file=onnx_file_name)

def dump_torch_to_onnx(
    model: torch.nn.Module,
    onnx_export_file: str,
    input_shape: List[int],
    input_dtype: torch.dtype = torch.float,
    inputs: List[Any] = None,
    device: str = 'cuda'):
    """
        转换一个 torch 模型到 onnx，并保存到指定位置
        convert a torch model to onnx and save to the specified location
    Args:
        model (torch.nn.Module): 被转换的 torch 模型 torch model used for conversion

        onnx_export_file (str): 保存文件的路径 the path to save onnx model

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
            a list of ints indicating size of input, for multiple inputs, please use keyword arg inputs for
            direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                   the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                   for direct parameter passing and this should be set to None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                    for multiple inputs, please give the specified inputs directly in the form of
                                    a list of arrays

        device (str, optional): 转换过程的执行设备 the execution device, defaults to 'cuda'.
    """

    # set model to eval mode, stabilize normalization weights.
    assert isinstance(model, torch.nn.Module), (
        f'Model must be instance of torch.nn.Module, however {type(model)} is given.')
    model.eval()

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    torch.onnx.export(
        model=model, args=dummy_input,
        verbose=False, f=onnx_export_file, opset_version=11,
    )

@ empty_ppq_cache
def quantize_onnx_model(
    onnx_import_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    platform: TargetPlatform,
    input_dtype: torch.dtype = torch.float,
    inputs: List[Any] = None,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    device: str = 'cuda',
    verbose: int = 0,
    do_quantize: bool = True,
) -> BaseGraph:
    """量化一个 onnx 原生的模型 输入一个 onnx 模型的文件路径 返回一个量化后的 PPQ.IR.BaseGraph quantize
    onnx model, input onnx model and return quantized ppq IR graph.

    Args:
        onnx_import_file (str): 被量化的 onnx 模型文件路径 onnx model location

        calib_dataloader (DataLoader): 校准数据集 calibration data loader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True.


        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.

        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息
                   The quantized IR, containing all information needed for backend execution
    """
    if do_quantize:
        if calib_dataloader is None or calib_steps is None:
            raise TypeError('Quantization needs a valid calib_dataloader and calib_steps setting.')

    if setting is None:
        setting = QuantizationSettingFactory.default_setting()

    ppq_ir = load_onnx_graph(onnx_import_file=onnx_import_file)
    ppq_ir = dispatch_graph(graph=ppq_ir, platform=platform, 
                            dispatcher=setting.dispatcher, 
                            dispatching_table=setting.dispatching_table)
    
    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    quantizer = PFL.Quantizer(platform, ppq_ir)
    executor = TorchExecutor(graph=quantizer._graph, device=device)
    if do_quantize:
        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            calib_steps=calib_steps,
            collate_fn=collate_fn
        )
        if verbose: quantizer.report()
        return quantizer._graph
    else:
        executor = TorchExecutor(graph=ppq_ir, device=device)
        executor.tracing_operation_meta(inputs=dummy_input)
        return quantizer._graph

@ empty_ppq_cache
def quantize_torch_model(
    model: torch.nn.Module,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    platform: TargetPlatform,
    input_dtype: torch.dtype = torch.float,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    inputs: List[Any] = None,
    do_quantize: bool = True,
    onnx_export_file: str = 'onnx.model',
    device: str = 'cuda',
    verbose: int = 0,
    ) -> BaseGraph:
    """量化一个 Pytorch 原生的模型 输入一个 torch.nn.Module 返回一个量化后的 PPQ.IR.BaseGraph.

        quantize a pytorch model, input pytorch model and return quantized ppq IR graph
    Args:
        model (torch.nn.Module): 被量化的 torch 模型(torch.nn.Module) the pytorch model

        calib_dataloader (DataLoader): 校准数据集 calibration dataloader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True, defaults to True.

        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.

        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息
                   The quantized IR, containing all information needed for backend execution
    """
    # dump pytorch model to onnx
    dump_torch_to_onnx(model=model, onnx_export_file=onnx_export_file,
        input_shape=input_shape, input_dtype=input_dtype,
        inputs=inputs, device=device)

    return quantize_onnx_model(onnx_import_file=onnx_export_file,
        calib_dataloader=calib_dataloader, calib_steps=calib_steps, collate_fn=collate_fn,
        input_shape=input_shape, input_dtype=input_dtype, inputs=inputs, setting=setting,
        platform=platform, device=device, verbose=verbose, do_quantize=do_quantize)

@ empty_ppq_cache
def quantize_caffe_model(
    caffe_proto_file: str,
    caffe_model_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    platform: TargetPlatform,
    input_dtype: torch.dtype = torch.float,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    inputs: List[Any] = None,
    do_quantize: bool = True,
    device: str = 'cuda',
    verbose: int = 0,
) -> BaseGraph:
    """
        量化一个 caffe 原生的模型
            输入一个 caffe 模型的文件路径和权重路径
            返回一个量化后的 PPQ.IR.BaseGraph
        quantize caffe model, input caffe prototxt and weight path, return a quantized ppq graph
    Args:
        caffe_proto_file (str): 被量化的 caffe 模型文件 .prototxt 路径
                                caffe prototxt location

        caffe_model_file (str): 被量化的 caffe 模型文件 .caffemodel 路径
                                caffe weight location

        calib_dataloader (DataLoader): 校准数据集 calibration data loader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True, defaults to True.

        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.

        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息
                   The quantized IR, containing all information needed for backend execution
    """
    if do_quantize:
        if calib_dataloader is None or calib_steps is None:
            raise TypeError('Quantization needs a valid calib_dataloader and calib_steps setting.')

    if setting is None:
        setting = QuantizationSettingFactory.default_setting()

    ppq_ir = load_graph(file_path=caffe_proto_file,
                        caffemodel_path=caffe_model_file,
                        from_framework=NetworkFramework.CAFFE)

    ppq_ir = format_graph(ppq_ir)
    ppq_ir = dispatch_graph(ppq_ir, platform, 
                            dispatcher=setting.dispatcher, 
                            dispatching_table=setting.dispatching_table)

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    quantizer = PFL.Quantizer(platform=platform, graph=ppq_ir)
    executor = TorchExecutor(graph=quantizer._graph, device=device)
    executor.tracing_operation_meta(inputs=dummy_input)

    if do_quantize:
        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            calib_steps=calib_steps,
            collate_fn=collate_fn
        )
        if verbose: quantizer.report()
        return quantizer._graph
    else:
        return quantizer._graph

@ empty_ppq_cache
def quantize_native_model(
    model: BaseGraph,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    platform: TargetPlatform,
    input_dtype: torch.dtype = torch.float,
    inputs: List[Any] = None,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    device: str = 'cuda',
    verbose: int = 0,
    do_quantize: bool = True,
) -> BaseGraph:
    """量化一个已经在内存中的 ppq 模型 输入一个量化前的 PPQ.IR.BaseGraph 返回一个量化后的 PPQ.IR.BaseGraph
    quantize ppq model, input ppq graph and return quantized ppq graph.

    Args:
        native (BaseGraph): 被量化的 ppq graph

        calib_dataloader (DataLoader): 校准数据集 calibration data loader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True.


        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.

        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息
                   The quantized IR, containing all information needed for backend execution
    """
    if do_quantize:
        if calib_dataloader is None or calib_steps is None:
            raise TypeError('Quantization needs a valid calib_dataloader and calib_steps setting.')

    if setting is None:
        setting = QuantizationSettingFactory.default_setting()
    ppq_ir = dispatch_graph(graph=model, platform=platform, 
                            dispatcher=setting.dispatcher, 
                            dispatching_table=setting.dispatching_table)

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else: dummy_input = inputs

    quantizer = PFL.Quantizer(platform=platform, graph=ppq_ir)
    executor = TorchExecutor(graph=quantizer._graph, device=device)
    
    if do_quantize:
        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            calib_steps=calib_steps,
            collate_fn=collate_fn
        )
        if verbose: quantizer.report()
        return quantizer._graph
    else:
        executor = TorchExecutor(graph=ppq_ir, device=device)
        executor.tracing_operation_meta(inputs=dummy_input)
        return quantizer._graph


def export_ppq_graph(
    graph: BaseGraph,
    platform: TargetPlatform,
    graph_save_to: str,
    config_save_to: str = None,
    copy_graph: bool = True,
    **kwargs) -> None:
    """使用这个函数将 PPQ ir 保存到文件，同时导出 PPQ 的量化配置信息。 该函数可以将 PPQ ir 保存为不同格式的模型文件。 this
    func dumps ppq IR to file, and exports quantization setting information
    simultaneously.

    详细的支持情况请参考: ppq.parser.__ini__.py
    for details please refer to ppq.parser.__ini__.py

    Args:
        graph (BaseGraph): 被保存的 ir
                           the ppq IR graph

        platform (TargetPlatform): 期望部署的目标平台
                           target backend platform

        graph_save_to (str): 模型保存文件名，不要写后缀名，ppq 会自己加后缀
                           filename to save, do not add postfix to this

        config_save_to (str): 量化配置信息保存文件名。
            注意部分平台导出时会将量化配置信息直接写入模型，在这种情况下设置此参数无效
            note that some of platforms requires to write quantization setting
            directly into the model file, this parameter won't have effect at
            this situation
            
        copy_graph (bool): 导出图的时候是否需要把图复制一份
            Whether to copy graph when export.
    """
    for save_path in [graph_save_to, config_save_to]:
        if save_path is None: continue
        if os.path.exists(save_path):
            if os.path.isfile(save_path):
                ppq_warning(f'File {save_path} is already existed, Exporter will overwrite it.')
            if os.path.isdir(save_path):
                raise FileExistsError(f'File {save_path} is already existed, and it is a directory, '
                                    'Exporter can not create file here.')

    exporter = PFL.Exporter(platform)
    if copy_graph: graph = graph.copy()
    exporter.export(file_path=graph_save_to, config_path=config_save_to, graph=graph, **kwargs)


def format_graph(graph: BaseGraph) -> BaseGraph:
    """这个函数对计算图进行预处理工作，其主要内容是将计算图的格式进行统一 这个函数将会统一 cast, slice, parameter,
    constant 算子的格式，并且执行有关 batchnorm 的合并工作.

    在 PPQ 中，我们不希望出现 Constant 算子，所有 Constant 输入将被当作 parameter variable 连接到下游算子上
    在 PPQ 中，我们不希望出现 Batchnorm 算子，所有 Batchnorm 将被合并
    在 PPQ 中，我们不希望出现权重共享的算子，所有被共享的权重将被复制分裂成多份
    在 PPQ 中，我们不希望出现孤立算子，所有孤立算子将被移除

    This function takes pre-processing procedure with your graph.
    This function will convert operations like cast, slice, parameter, constant to the format that supported by ppq.
    This function will merge batchnorm when possible.

    During quantization logic, we do not expect there is any constant operation in your network, so
        all of them will be converted as parameter input variable.

    We do not expect there is any shared parameter in your network, all of them will be copied and spilted.
    We do not expect any isolated operation in your network, all of them will be removed.
    """
    
    # do graph level optimization
    formatter = GraphReplacer(GraphFormatter(GraphMerger(graph)))
    if FORMATTER_FORMAT_CONSTANT_INPUT:
        formatter(GraphCommand(GraphCommandType.FORMAT_CONSTANT_INPUT))
    formatter(GraphCommand(GraphCommandType.CONVERT_TO_TENSOR))
    formatter(GraphCommand(GraphCommandType.FORMAT_PARAMETERS))
    
    if FORMATTER_FUSE_BIAS_ADD:
        formatter(GraphCommand(GraphCommandType.FUSE_BIAS_ADD))
    
    if FORMATTER_FUSE_BN:
        formatter(GraphCommand(GraphCommandType.FUSE_BN))
    
    if FORMATTER_REPLACE_BN_TO_CONV:
        formatter(GraphCommand(GraphCommandType.REPLACE_BATCHNORM_TO_CONV))

    if FORMATTER_FUSE_SWISH:
        formatter(GraphCommand(GraphCommandType.FUSE_SWISH))

    formatter(GraphCommand(GraphCommandType.FORMAT_CAST))
    formatter(GraphCommand(GraphCommandType.FORMAT_SLICE))
    formatter(GraphCommand(GraphCommandType.FORMAT_CLIP))
    formatter(GraphCommand(GraphCommandType.FORMAT_PAD))
    formatter(GraphCommand(GraphCommandType.FORMAT_RESIZE))
    
    if FORMATTER_REMOVE_IDENTITY:
        formatter(GraphCommand(GraphCommandType.REMOVE_IDENTITY))
    
    if FORMATTER_REMOVE_ISOLATED:
        formatter(GraphCommand(GraphCommandType.DELETE_ISOLATED))

    return graph


def dispatch_graph(
    graph: BaseGraph, platform: TargetPlatform, 
    dispatcher: Union[str, GraphDispatcher] = 'conservative', 
    dispatching_table: DispatchingTable = None) -> BaseGraph:
    """这个函数执行图切分与调度，你的计算图将被切分成一系列子图，并被调度到不同设备上。 
    调度的逻辑分为自动控制的部分以及手动覆盖的部分，你可以使用 QuantizationSetting 来向这个函数传递手动调度表 从而覆盖 PPQ 的调度逻辑。

    注意：这个函数依据调度器和 TargetPlatform 平台的不同而产生行为差异，生成不同的调度计划。

    This function will cut your graph into a series of subgraph and send them to different device.
    PPQ provides an automatic dispatcher which, will generate different dispatching scheme on your TargetPlatform.
    A dispatching table can be passed via QuantizationSetting to override
        the default dispatching logic of ppq dispatcher manually.
    """
    dispatching_override = dispatching_table
    quantizer = PFL.Quantizer(platform=platform, graph=graph) # 初始化一个 quantizer 没有很大代价...

    if isinstance(dispatcher, str):
        dispatcher = dispatcher.lower()
        if dispatcher not in DISPATCHER_TABLE:
            raise ValueError(f'Can not found dispatcher type "{dispatcher}", check your input again.')
        dispatcher = DISPATCHER_TABLE[dispatcher](graph)
    else:
        if not isinstance(dispatcher, GraphDispatcher):
            raise TypeError('Parameter "dispachter" of function ppq.api.dispatch_graph must be String or GraphDispatcher, '
                            f'however {type(dispatcher)} was given.')
        dispatcher = dispatcher

    assert isinstance(dispatcher, GraphDispatcher)
    assert isinstance(quantizer, BaseQuantizer)
    quant_types = quantizer.quant_operation_types
    dispatching_table = dispatcher.dispatch(
        graph=graph, quant_types=quant_types,
        quant_platform=TargetPlatform.UNSPECIFIED, # MUST BE UNSPECIFIED, 这里的意思是交由 Quantizer 决定是否量化这个算子
        fp32_platform=TargetPlatform.FP32,
        SOI_platform=TargetPlatform.SOI)

    # override dispatching result
    if dispatching_override is not None:
        if not isinstance(dispatching_override, DispatchingTable):
            raise TypeError('Parameter "dispatching_table" of function ppq.api.dispatch_graph must be DispatchingTable, '
                            f'however {type(dispatching_override)} was given.')

        for opname, platform in dispatching_override.dispatchings.items():
            if opname not in graph.operations: continue
            assert isinstance(platform, int), (
                f'Your dispatching table contains a invalid setting of operation {opname}, '
                'All platform setting given in dispatching table is expected given as int, '
                f'however {type(platform)} was given.')
            dispatching_table[opname] = TargetPlatform(platform)

    for operation in graph.operations.values():
        assert operation.name in dispatching_table, (
            f'Internal Error, Can not find operation {operation.name} in dispatching table.')
        operation.platform = dispatching_table[operation.name]
    return graph


class UnbelievableUserFriendlyQuantizationSetting:
    """
    量化配置文件 -- 入门版

    这个文件包含了最基本的量化配置。
    """

    def __init__(self, platform: TargetPlatform, finetune_steps: int = 500, finetune_lr: float = 3e-5,
                 interested_outputs: List[str] = None, calibration: str = 'percentile', equalization: bool = True,
                 non_quantable_op: List[str] = None) -> None:
        """
        量化配置文件 -- 入门版

        这个文件包含了最基本的量化配置。

        Args:
            platform (TargetPlatform): 目标量化平台
            finetune_steps (int, optional): 网络 finetune 步数. Defaults to 5000.
            finetune_lr (float, optional): 网络 finetune 学习率. Defaults to 3e-4.
            interested_outputs (List[str], optional): 用来finetune的variable名字，请注意对于静态图而言其总是由 op 和 variable 组成的，
                有时候你的网络输出并不是可导的，或者是一个softmax或者sigmoid的输出，这些时候finetune的结果不会很好，你可以通过这个属性
                来指定一个variable的名字，我们将用这个variable的输出结果来引导finetune流程，当然一个variable list也是可以的。
            equalization (bool, optional): 是否要拉平网络权重. Defaults to True.
            non_quantable_op (List[str], optional): 非量化算子集合，所有名字出现在该集合里的算子将不被量化. Defaults to None.
        """
        self.equalization     = equalization
        self.finetune_steps   = finetune_steps
        self.finetune_lr      = finetune_lr
        self.calibration      = calibration
        self.platform         = platform
        self.non_quantable_op = non_quantable_op
        self.interested_outputs = interested_outputs

        if isinstance(self.non_quantable_op, str): self.non_quantable_op = [self.non_quantable_op]
        if isinstance(self.interested_outputs, str): self.interested_outputs = [self.interested_outputs]

    def convert_to_daddy_setting(self) -> QuantizationSetting:
        # 将菜鸡版量化配置转换成高级版的
        daddy = QuantizationSettingFactory.default_setting()
        daddy.quantize_activation_setting.calib_algorithm = self.calibration

        if self.platform in {TargetPlatform.METAX_INT8_C, TargetPlatform.METAX_INT8_T}:
            daddy.fusion_setting.force_alignment_overlap = True

        if self.finetune_steps > 0:
            daddy.lsq_optimization               = True
            daddy.lsq_optimization_setting.steps = self.finetune_steps
            daddy.lsq_optimization_setting.lr    = self.finetune_lr

        if self.equalization == True:
            daddy.equalization                    = True
            daddy.equalization_setting.iterations = 3
            daddy.equalization_setting.opt_level  = 1
            daddy.equalization_setting.value_threshold = 0

        if self.non_quantable_op is not None:
            for op_name in self.non_quantable_op:
                assert isinstance(op_name, str), (
                    f'你尝试使用 non_quantable_op 来设定非量化算子，'
                    f'non_quantable_op 只应当包含算子的名字，而你传入的数据中包括了 {type(op_name)}')
                daddy.dispatching_table.append(op_name, TargetPlatform.FP32)

        return daddy

    def to_json(self, file_path: str) -> str:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                raise FileExistsError(f'文件 {file_path} 已经存在且是一个目录，无法将配置文件写入到该位置！')
            ppq_warning(f'文件 {file_path} 已经存在并将被覆盖')

        # TargetPlatform is not a native type, convert it to string.
        dump_dict = self.__dict__.copy()
        dump_dict['platform'] = self.platform.name

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(obj=dump_dict, fp=file, sort_keys=True, indent=4, ensure_ascii=False)

    @ staticmethod
    def from_file(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError('找不到你的配置文件，检查配置文件路径是否正确！')
        with open(file_path, 'r', encoding='utf-8') as file:
            loaded = json.load(file)
        assert isinstance(loaded, dict), 'Json文件无法解析，格式不正确'
        assert 'platform' in loaded, 'Json文件缺少必要项目 "platform"'

        platform = loaded['platform']
        if platform in TargetPlatform._member_names_:
            platform = TargetPlatform._member_map_[platform]
        else: raise KeyError('无法解析你的json配置文件，遇到了未知的platform属性。')

        setting = UnbelievableUserFriendlyQuantizationSetting(platform)
        for key, value in loaded.items():
            if key == 'platform': continue
            if key in setting.__dict__: setting.__dict__[key] = value
            if key not in setting.__dict__: ppq_warning(f'你的Json文件中包含无法解析的属性 {key} ，该属性已经被舍弃')
        assert isinstance(setting, UnbelievableUserFriendlyQuantizationSetting)
        return setting

    def __str__(self) -> str:
        return str(self.__dict__)


def quantize(working_directory: str, setting: QuantizationSetting, model_type: NetworkFramework,
             executing_device: str, input_shape: List[int], target_platform: TargetPlatform,
             dataloader: DataLoader, calib_steps: int = 32) -> BaseGraph:
    """Helper function for quantize your model within working directory, This
    function will do some check and redirect your requirement to:
    ppq.api.quantize_onnx_model ppq.api.quantize_caffe_model.

    see them for more information.

    Args:
        working_directory (str): A path that indicates working directory.
        setting (QuantizationSetting): Quantization setting
        model_type (NetworkFramework): Onnx or Caffe
        executing_device (str): 'cuda' or 'cpu'
        input_shape (List[int]): sample input's shape
        target_platform (TargetPlatform): Target deploy platform
        dataloader (DataLoader): calibraiton dataloader
        calib_steps (int, optional): Defaults to 32.

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_

    Returns:
        BaseGraph: _description_
    """
    if model_type == NetworkFramework.ONNX:
        if not os.path.exists(os.path.join(working_directory, 'model.onnx')):
            raise FileNotFoundError(f'无法找到你的模型: {os.path.join(working_directory, "model.onnx")},'
                                    '如果你使用caffe的模型, 请设置MODEL_TYPE为CAFFE')
        return quantize_onnx_model(
            onnx_import_file=os.path.join(working_directory, 'model.onnx'),
            calib_dataloader=dataloader, calib_steps=calib_steps, input_shape=input_shape, setting=setting,
            platform=target_platform, device=executing_device, collate_fn=lambda x: x.to(executing_device)
        )
    if model_type == NetworkFramework.CAFFE:
        if not os.path.exists(os.path.join(working_directory, 'model.caffemodel')):
            raise FileNotFoundError(f'无法找到你的模型: {os.path.join(working_directory, "model.caffemodel")},'
                                    '如果你使用ONNX的模型, 请设置MODEL_TYPE为ONNX')
        return quantize_caffe_model(
            caffe_proto_file=os.path.join(working_directory, 'model.prototxt'),
            caffe_model_file=os.path.join(working_directory, 'model.caffemodel'),
            calib_dataloader=dataloader, calib_steps=calib_steps, input_shape=input_shape, setting=setting,
            platform=target_platform, device=executing_device, collate_fn=lambda x: x.to(executing_device)
        )


def export(working_directory: str, quantized: BaseGraph, platform: TargetPlatform, **kwargs):
    """Helper function to export your graph to working directory, You should
    notice this function just redirect your invoking to export_ppq_graph. see
    export_ppq_graph for more information.

    Args:
        working_directory (str): _description_
        quantized (BaseGraph): _description_
        platform (TargetPlatform): _description_
    """
    export_ppq_graph(
        graph=quantized, platform=platform,
        graph_save_to=os.path.join(working_directory, 'quantized'),
        config_save_to=os.path.join(working_directory, 'quantized.json'),
        **kwargs
    )


def manop(graph: BaseGraph, list_of_passes: List[QuantizationOptimizationPass],
          calib_dataloader: Iterable, executor: BaseGraphExecutor,
          collate_fn: Callable = None, **kwargs) -> BaseGraph:
    """manop 是一个很方便的函数，你可以调用这个函数来手动地执行一些量化优化工作
    你可以在默认量化逻辑之前或之后调用这个函数来自定义量化处理流程，相比于直接实现 Quantizer来修改量化逻辑的方式, 使用 manop
    会更加灵活。

    MANOP (manually optimize) function is introduced since PPQ 0.6.4.
    This function allows you to manually invoke
        QuantizationOptimizationPass before or after default quantization logic.

    We do not use function name like apply, optim, do ...
    Because they are so widely-used in other python libraries,
        and can easily conflict with each other.

    Args:
        graph (BaseGraph): processing graph.
        list_of_passes (List[QuantizationOptimizationPass]): a collection of optimization logic.
        calib_dataloader (Iterable): _description_
        executor (BaseGraphExecutor): _description_
        collate_fn (Callable): _description_

    Raises:
        TypeError: _description_
        TypeError: _description_

    Returns:
        BaseGraph: processed graph
    """
    if isinstance(list_of_passes, QuantizationOptimizationPass):
        list_of_passes = [list_of_passes]

    if not (isinstance(list_of_passes, list) or isinstance(list_of_passes, tuple)):
        raise TypeError('Can not apply optimization on your graph, '
                        'expect a list of QuantizationOptimizationPass as input, '
                        f'while {type(list_of_passes)} was given.')

    for optim in list_of_passes:
        if not isinstance(optim, QuantizationOptimizationPass):
            raise TypeError('Invoking this function needs a list of QuantizationOptimizationPass, '
                            f'however there is a/an {type(optim)} in your list')
        optim.apply(graph, dataloader=calib_dataloader, executor=executor, collate_fn=collate_fn, **kwargs)
    return graph


class ENABLE_CUDA_KERNEL:
    """ Any code surrounded by 
    with ENABLE_CUDA_KERNEL():
    will invoke ppq's kernel functions for speed boost.
    
    This is a helper class for invoking highly-effcient custimized cuda
    kernel. PPQ developer team has implemented a series of quantization related
    cuda kernel, They are 5-100x faster than torch kernels, with less gpu
    memory cost.
    """
    def __init__(self) -> None:
        from ppq.core.ffi import CUDA_COMPLIER
        CUDA_COMPLIER.complie()
        self._state = False

    def __enter__(self):
        self._state = PPQ_CONFIG.USING_CUDA_KERNEL
        PPQ_CONFIG.USING_CUDA_KERNEL = True

    def __exit__(self, *args):
        PPQ_CONFIG.USING_CUDA_KERNEL = self._state


class DISABLE_CUDA_KERNEL:
    """ Any code surrounded by 
    with DISABLE_CUDA_KERNEL():
    will block ppq's kernel functions.
    
    This is a helper class for blocking ppq custimized cuda
    kernel. PPQ developer team has implemented a series of quantization related
    cuda kernel, They are 5-100x faster than torch kernels, with less gpu
    memory cost.
    """
    def __init__(self) -> None:
        pass
    
    def __enter__(self):
        self._state = PPQ_CONFIG.USING_CUDA_KERNEL
        PPQ_CONFIG.USING_CUDA_KERNEL = False
    
    def __exit__(self, *args):
        PPQ_CONFIG.USING_CUDA_KERNEL = self._state


class DEQUANTIZE_GRAPH:
    """ A short-cut function to dequantize given graph. """
    def __init__(self, graph: BaseGraph) -> None:
        self._graph          = graph
        self._state_recorder = {}
        self._quantable_ops  = [op for op in graph.operations.values() if isinstance(op, QuantableOperation)]

    def __enter__(self):
        for op in self._quantable_ops:
            op.dequantize()

    def __exit__(self, *args):
        for op in self._quantable_ops:
            op.restore_quantize_state()


__all__ = ['load_graph', 'load_onnx_graph', 'load_caffe_graph',
           'dispatch_graph', 'dump_torch_to_onnx', 'quantize_onnx_model',
           'quantize_torch_model', 'quantize_caffe_model', 'load_native_graph',
           'export_ppq_graph', 'format_graph', 'quantize', 'export', 'load_torch_model',
           'UnbelievableUserFriendlyQuantizationSetting', 'manop',
           'quantize_native_model', 'ENABLE_CUDA_KERNEL', 'DISABLE_CUDA_KERNEL', 
           'DEQUANTIZE_GRAPH']

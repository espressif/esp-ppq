from .onnxruntime_exporter import ONNXRUNTIMExporter

import onnx
import torch
from onnx import helper
from ppq.core import (GRAPH_OPSET_ATTRIB, PPQ_CONFIG,
                      QuantizationProperty, TensorQuantizationConfig)
from ppq.IR import (BaseGraph)
from .onnx_exporter import OP_CONVERTERS, OperationExporter


class EspressifExporter(ONNXRUNTIMExporter):
    def infer_qtype(self, config: TensorQuantizationConfig):
        offset_dtype, value_dtype = torch.int8, torch.int8
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            offset_dtype = torch.uint8
            value_dtype  = torch.uint8
        if config.num_of_bits == 16:
            offset_dtype = torch.int16
            value_dtype  = torch.int16
        elif config.num_of_bits > 16:
            offset_dtype = torch.int32
            value_dtype  = torch.int32
        return offset_dtype, value_dtype

    """去除原生 ONNXRUNTIMExporter 对非 8bit 量化算子会产生的 warning"""
    def export(self, file_path: str, graph: BaseGraph, 
            config_path: str = None, 
            quantized_param: bool = True,
            remove_activation: bool = True, 
            save_as_external_data: bool = False) -> None:
        """
        Export PPQ Graph to Onnx QDQ format.
            This function requires a set of parameters to configure onnx format.
        
        Args:
            file_path (str): Onnx file name.
            
            graph (BaseGraph): Exporting ppq graph.
            
            config_path (str, optional): config file is a json file that contains quant-related
                information, this file is require by TensorRT for initialize its quantization
                pipeline. If config_path = None, no json file will be created.

            export_QDQ_op (bool, optional): whether to export QDQ node in onnx model.

            quantized_param (bool, optional): export quantized parameter, if quantized_param = False,
                PPQ will export parameter in FP32 format.
            
            remove_activation (bool, optional): this option will remove activation op(Relu, Clip),
                requires ASYMMTRICAL quantizaiton.
            
            save_as_external_data (bool, optional): for model larger than 2GB, 
                this option will split model into external param files.
        """
        # In prepare stage, quant & dequant node are inserted into graph.
        graph = self.prepare_graph(
            graph, remove_activation_fn=remove_activation, 
            quant_parameter_to_int=quantized_param)

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            super().export_quantization_config(config_path, graph)

        # before we can export them, we firstly convert all ops to proper format.
        for op in [_ for _ in graph.topological_sort()]:
            if op.type in OP_CONVERTERS:
                exporter = OP_CONVERTERS[op.type]()
                assert isinstance(exporter, OperationExporter), (
                    f'Expected an OpExporter here, however {type(exporter)} was given.')
                op = exporter.export(op=op, graph=graph)

        name = graph._name
        if not name: name = 'PPL Quantization Tool - Onnx Export'

        # Ready to export onnx graph definition.
        _inputs, _outputs, _initilizers, _nodes, _value_info = [], [], [], [], []
        for operation in graph.topological_sort():
            _nodes.append(super().build_operator_proto(operation))

        for variable in graph.variables.values():
            tensor_proto = super().build_variable_proto(variable)
            if variable.name in graph.inputs: _inputs.append(tensor_proto)
            if variable.name in graph.outputs: _outputs.append(tensor_proto)
            if variable.is_parameter: _initilizers.append(tensor_proto)
            else: _value_info.append(tensor_proto)

        graph_def = helper.make_graph(
            name=name, nodes=_nodes, inputs=_inputs,
            outputs=_outputs, initializer=_initilizers, 
            value_info=_value_info)
        extra_opsets = self.required_opsets

        opsets = []
        if GRAPH_OPSET_ATTRIB in graph._detail:
            for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
                if opset['domain'] in extra_opsets or opset['domain'] == '':
                    continue
                op = onnx.OperatorSetIdProto()
                op.domain = opset['domain']
                op.version = opset['version']
                opsets.append(op)

        for key, value in extra_opsets.items():
            op = onnx.OperatorSetIdProto()
            op.domain = key
            op.version = value
            opsets.append(op)

        onnx_model = helper.make_model(
            graph_def, producer_name=PPQ_CONFIG.NAME, opset_imports=opsets)
        onnx_model.ir_version = 7
        # onnx.checker.check_model(onnx_model)
        size_threshold = 0 if save_as_external_data else 1024
        onnx.save(onnx_model, file_path, size_threshold=size_threshold,
                  save_as_external_data=save_as_external_data,
                  all_tensors_to_one_file=(not save_as_external_data))

        # # Check Graph
        # unsupportable_quant_op = set()
        # for op in graph.operations.values():
        #     if isinstance(op, QuantableOperation):
        #         for cfg, var in op.config_with_variable:
        #             if not QDQHelper.TQC_Exportable_Check(TQC=cfg, bounded_var=var): continue
        #             if cfg.num_of_bits != 8 or cfg.policy.has_property(QuantizationProperty.FLOATING):
        #                 unsupportable_quant_op.add(op)

        # if len(unsupportable_quant_op) != 0:
        #     ppq_warning('Exported Onnx Model is not executable, following Op has onnxruntime-unsupported quant policy:')
        #     for op in unsupportable_quant_op:
        #         ppq_warning(f'{op.name} (bitwidth != 8)')
        #     ppq_warning('For Generating onnxruntime-executable Model, use TargetPlatform = Onnxruntime or OnnxruntimeQuantizer instead.')

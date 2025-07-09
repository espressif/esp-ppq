import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from esp_ppq.core import (
    ONNX_VERSION,
    PPQ_CONFIG,
    DataType,
    QuantizationStates,
    convert_any_to_numpy,
    convert_any_to_torch_tensor,
)
from esp_ppq.IR import BaseGraph, GraphExporter, Operation, OperationExporter, Variable
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.log import NaiveLogger
from esp_ppq.quantization.qfunction.linear import PPQLinearQuant_toInt

from .espdl import helper
from .espdl.espdl_typedef import ExporterPatternInfo, LayoutAnnotation
from .espdl.export_patterns import (
    AddLUTPattern,
    FuseReluLikePattern,
    InsertDequantNodePattern,
    InsertPreNodeOfMatMulPattern,
    InsertQuantNodePattern,
    InsertQuantTypePattern,
    InsertRequantNodePattern,
    QuantVariableToIntPattern,
    ResetParamLayoutPattern,
)
from .espdl.layout_patterns import (
    reset_graph_layout,
    transpose_shape,
)
from .onnx_exporter import OP_CONVERTERS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

logger = NaiveLogger.get_logger("ESPDL")


def convert_value(value: Union[int, float, np.ndarray, torch.Tensor]) -> Any:
    if type(value) in {int, float}:
        return value
    else:
        value = convert_any_to_numpy(value, accept_none=True)
        if value is None:
            return value  # SOI config has Nona as its scale and
        return value.tolist()


class EspdlExporter(GraphExporter):
    """
    The EspdlExporter is used to export computational graphs into the esp-dl standard format.
    The export logic of any exporter is performed in-place, meaning it will modify the incoming computational graph object in-place.
    Therefore, you need to manually clone the computational graph before exporting.
    """

    def __init__(self) -> None:
        super().__init__()

    def export(
        self,
        file_path: str,
        graph: BaseGraph,
        config_path: str = None,
        model_version: int = 0,
        values_for_test: Dict[str, Dict[str, torch.Tensor]] = None,
        export_config: bool = True,
        encrypt_data: bool = False,
        int16_lut_step: int = 256,
        metadata_props: Dict[str, str] = None,
        **kwargs: Any,
    ):
        """Export model to flatbuffers file

        Args:
            file_path: flatbuffers file
            graph: ppq graph
            config_path: quantization config
            model_version (int): model version
            values_for_test (Dict[str, Dict[str, np.ndarray]]): the test values used to compare accuracy.
                                                            The input format is as follows:
                {
                    'inputs': {
                        'input_0_name': np.ndarray
                        ......
                        'input_n_name': np.ndarray
                    },
                    'outputs': {
                        'output_0_name': np.ndarray
                        ......
                        'output_n_name': np.ndarray
                    },
                }
            int16_lut_step (int): -1: do not add lut for int16, otherwise add lut for int16 with given step
            metadata_props (Dict[str, str]): metadata properties
        """

        # during export we will remove all boundary operations from graph.
        # we do not want to change the structure of original graph,
        # so there have to take a clone of it.
        graph = graph.copy()

        # In prepare stage, run all graph pattern
        # fuse Conv and Relu and insert quant node if necessary
        exporter_patterns = {
            "pre_patterns": [
                FuseReluLikePattern,
                InsertQuantNodePattern,
                InsertRequantNodePattern,
                InsertDequantNodePattern,
                InsertPreNodeOfMatMulPattern,
            ],
            "post_patterns": [
                InsertQuantTypePattern,
                QuantVariableToIntPattern,
                ResetParamLayoutPattern,
            ],
        }

        graph = self.prepare_graph(graph, exporter_patterns, int16_lut_step)
        file_base_name, _ = os.path.splitext(file_path)

        # if a valid config path is given, export quantization config to there.
        if export_config:
            if not config_path:
                config_path = file_base_name + ".json"
            self.export_quantization_config(config_path, graph)

        model = self.export_graph(
            graph=graph,
            model_version=model_version,
            values_for_test=values_for_test,
            metadata_props=metadata_props,
            **kwargs,
        )

        # Export the information of quantized espdl model.
        if export_config:
            info_file_path = file_base_name + ".info"
            with open(info_file_path, "w") as file_info:
                file_info.write(
                    helper.printable_graph(
                        model,
                        print_initializer_value=True,
                        print_value_info=True,
                        print_test_value=True,
                    )
                )

        key = helper.save(
            model,
            file_path,
            encrypt_data,  # if True, encrypt data for security
        )
        if key:
            logger.info(f"AES 128-bit secret key: {key}")

        ExporterPatternInfo().reset()

    def prepare_graph(
        self,
        graph: BaseGraph,
        exporter_patterns: Dict[str, List[OperationExporter]],
        int16_lut_step: int,
    ) -> BaseGraph:
        """Prepare your graph for exporting.

        There are many works to do with your graph:

            1. Insert Quant and Dequant operation within your graph.

            2. Remove all unnecessary activations.

            3. Quantize all parameters of your graph, convert them to int8.

        Args:
            graph (BaseGraph): Processing Graph

        Returns:
            BaseGraph: Processed Graph
        """
        self.convert_operation_from_opset11_to_opset13(graph)

        # before we can export them, we firstly convert all ops to proper format. Copy from onnx exporter
        for op in [_ for _ in graph.topological_sort()]:
            if op.type in OP_CONVERTERS:
                exporter = OP_CONVERTERS[op.type]()
                assert isinstance(exporter, OperationExporter), (
                    f"Expected an OpExporter here, however {type(exporter)} was given."
                )
                op = exporter.export(op=op, graph=graph)

        for pattern in exporter_patterns.get("pre_patterns", {}):
            exporter = pattern()
            for op in graph.topological_sort():
                exporter.export(op=op, graph=graph)

        # reset Conv layout from NCHW to NHWC and insert transpose node if necessary
        reset_graph_layout(graph)

        for pattern in exporter_patterns.get("post_patterns", {}):
            exporter = pattern()
            for op in graph.topological_sort():
                exporter.export(op=op, graph=graph)

        # prepare LUT
        exporter = AddLUTPattern(int16_step=int16_lut_step)
        for op in graph.topological_sort():
            exporter.export(op=op, graph=graph)

        info = ExporterPatternInfo()
        for variable in graph.variables.values():
            if variable.is_parameter or len(variable.name) == 0:
                continue

            perm = info.get_var_permute(variable.name)
            exponents = info.get_var_exponents(variable.name)
            layout = info.get_var_layout(variable.name)
            if exponents and perm:
                logger.debug(f"{variable.name} perm: {perm}, exponents: {exponents}, layout:{layout}")
            elif not perm:
                logger.warning(f"{variable.name} does not bind perm parameter")
            elif not exponents:
                logger.warning(f"{variable.name} does not bind exponents parameter")

        for op in graph.topological_sort():
            if op.type in ["Conv", "Gemm", "MatMul", "Mul"]:
                input0_exponent = info.get_var_exponents(op.inputs[0].name)
                input1_exponent = info.get_var_exponents(op.inputs[1].name)
                output_exponent = info.get_var_exponents(op.outputs[0].name)
                # By default, it is per-tensor. Currently, esp-dl does not support per-channel.
                if (output_exponent[0] - input0_exponent[0] - input1_exponent[0]) < 0:
                    logger.error(
                        f"When deploying with esp-dl, the calculation result of the {op.name}(type: {op.type}) "
                        "operator will cause an exception. Please adjust the model to ensure "
                        "that (output_exponent - input0_exponent - input1_exponent) >= 0."
                    )

        return graph

    def export_graph(
        self,
        graph: BaseGraph,
        model_version: int = 0,
        values_for_test: Dict[str, Dict[str, torch.Tensor]] = None,
        metadata_props: Dict[str, str] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Convert a PPQ IR to Onnx IR.
        This export will only convert PPQ Op and var to onnx, all quantization configs will be skipped.

        This function will try to keep the opset version of your graph unchanged.
        However if the opset is not given, ppq will convert it to with the global parameter esp_ppq.core.ONNX_EXPORT_OPSET.
        """
        name = graph._name
        if not name:
            name = f"{PPQ_CONFIG.NAME} - v({PPQ_CONFIG.VERSION})"

        # Ready to export espdl graph defination.
        _inputs, _outputs, _initilizers, _nodes, _value_info = [], [], [], [], []
        pattern_info = ExporterPatternInfo()

        # Add LUT to graph
        _initilizers += self.build_lut_proto(graph, pattern_info)

        for op in graph.topological_sort():
            _nodes.append(self.build_operator_proto(op))

        for variable in graph.variables.values():
            tensor_proto = self.build_variable_proto(
                variable,
                pattern_info.get_var_exponents(variable.name, [0]),
                pattern_info.get_var_layout(variable.name, LayoutAnnotation.UNKNOWN),
                pattern_info.get_var_permute(variable.name),
            )
            if variable.name in graph.inputs:
                _inputs.append(tensor_proto)
            if variable.name in graph.outputs:
                _outputs.append(tensor_proto)
            if variable.is_parameter:
                _initilizers.append(tensor_proto)
            else:
                _value_info.append(tensor_proto)

        test_inputs_value, test_outputs_value = self.build_test_value_proto(values_for_test)

        graph_def = helper.make_graph(
            name=name,
            nodes=_nodes,
            inputs=_inputs,
            outputs=_outputs,
            initializer=_initilizers,
            value_info=_value_info,
            test_inputs_value=test_inputs_value,
            test_outputs_value=test_outputs_value,
        )

        metadata_props = helper.make_metadata_props(metadata_props)

        espdl_model = helper.make_model(
            graph=graph_def,
            producerName=PPQ_CONFIG.NAME,
            model_version=model_version,
            ir_version=graph._detail.get("ir_version", ONNX_VERSION),
            metadata_props=metadata_props,
        )

        return espdl_model

    def build_lut_proto(self, graph: BaseGraph, info: ExporterPatternInfo) -> List[int]:
        """
        Convert torch.Tensor to flatbuffer.Tensor and add them to graph.
        """
        _lut = []
        for op in graph.topological_sort():
            lut_name = op.attributes.get("lut", None)
            if lut_name:
                lut = info.get_lut(lut_name)
                onnx_dtype = DataType.convert_from_torch(lut.dtype).value
                lut_value = convert_any_to_numpy(lut).flatten()
                lut_proto = helper.make_tensor(
                    name=lut_name,
                    data_type=onnx_dtype,
                    dims=lut.shape,
                    vals=lut_value,
                    raw=True,
                    exponents=info.get_var_exponents(lut_name),
                )

                _lut.append(lut_proto)
        return _lut

    def build_test_value_proto(
        self, values_for_test: Dict[str, Dict[str, torch.Tensor]] = None
    ) -> Tuple[Sequence[int], Sequence[int]]:
        def quantize_and_transpose(var_name, tensor, pattern_info):
            perm = pattern_info.get_var_permute(var_name)
            config = pattern_info.get_var_config(var_name)
            if perm:
                tensor = tensor.permute(perm)
            if config:
                tensor = PPQLinearQuant_toInt(tensor, config=config)

            return convert_any_to_numpy(tensor)

        pattern_info = ExporterPatternInfo()
        quant_values_for_test = {}
        if values_for_test is not None:
            if "inputs" not in quant_values_for_test:
                quant_values_for_test["inputs"] = {}
            if "outputs" not in quant_values_for_test:
                quant_values_for_test["outputs"] = {}

            for var_name in values_for_test.get("inputs", {}):
                tensor = quantize_and_transpose(var_name, values_for_test["inputs"][var_name], pattern_info)
                quant_values_for_test["inputs"][var_name] = tensor

            for var_name in values_for_test.get("outputs", {}):
                tensor = quantize_and_transpose(var_name, values_for_test["outputs"][var_name], pattern_info)
                quant_values_for_test["outputs"][var_name] = tensor

        return helper.make_graph_test_value(quant_values_for_test, pattern_info.var_exponents)

    def build_operator_proto(self, operation: Operation) -> int:
        """
        Convert PPQ Op to Onnx Operation
        An Op consumes zero or more Tensors, and produces zero or more Tensors.
        """
        attributes = operation.attributes
        for key in attributes:
            value = attributes[key]
            if isinstance(value, DataType):
                attributes[key] = value.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    attributes[key] = None
                elif value.numel() == 1:
                    attributes[key] = convert_any_to_numpy([value.item()])  # convert to 1d array
                else:
                    attributes[key] = convert_any_to_numpy(value)

        if PPQ_CONFIG.EXPORT_PPQ_INTERNAL_INFO:
            attributes["platform"] = operation.platform.name

        op_proto = helper.make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            **attributes,
        )

        return op_proto

    def build_variable_proto(
        self,
        variable: Variable,
        exponent: List[int],
        layout: str,
        perm: Union[None, List[int]] = None,
    ) -> int:
        """
        Convert PPQ Variable to Onnx TensorProto, There are 2 different types of Tensor in Onnx:
            Variable: Represents a Tensor whose value is not known until inference-time.
            Constant: Represents a Tensor whose value is known.
        """
        # Parameter Varaible in PPQ, Constant Variable in Onnx
        if variable.is_parameter:
            if variable.value is not None:
                var_shape = variable.value.shape
                pytorch_dtype = variable.value.dtype
                onnx_dtype = DataType.convert_from_torch(pytorch_dtype).value

        # Non Parameter
        else:
            var_shape = variable.shape
            onnx_dtype = variable.dtype.value
            if var_shape:
                if len(var_shape) != len(perm):
                    logger.error(f"{variable.name} permute do not match shape")
                var_shape = transpose_shape(var_shape, perm)
            elif var_shape is not None and len(var_shape) == 0:
                var_shape = torch.tensor([1]).shape

        # if variable not in exponents, set exponent to 0
        var_exponents = exponent

        if not variable.is_parameter:
            tensor_proto = helper.make_tensor_value_info(
                name=variable.name,
                elem_type=onnx_dtype,
                shape=var_shape,
                exponents=var_exponents,
            )
        else:
            value = variable.value
            is_raw_format = False
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    value = []
                elif value.ndim >= 1:
                    value = convert_any_to_numpy(variable.value).flatten()
                    is_raw_format = True
                elif value.ndim == 0:  # Pytorch Scalar Type
                    value = [
                        value.item(),
                    ]  # it is fine for onnx, shape for this value will be []
                    var_shape = torch.tensor(value).shape
                    is_raw_format = True
            else:
                value = value  # value is python primary type.
            tensor_proto = helper.make_tensor(
                name=variable.name,
                data_type=onnx_dtype,
                dims=var_shape,
                vals=value,
                raw=is_raw_format,
                exponents=var_exponents,
                doc_string="layout ==> " + layout,
            )
        return tensor_proto

    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        """Export Tensor Quantization Config to File(Json)."""

        render_buffer = {"configs": {}, "dispatchings": {}, "values": {}}

        # Render quantization config.
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                op_dict = {
                    var.name: {
                        "bit_width": config.num_of_bits,
                        "policy": config.policy.to_dict(),
                        "state": config.state.name,
                        "quant_min": config.quant_min,
                        "quant_max": config.quant_max,
                        "hash": config.__hash__(),
                        "dominator": config.dominated_by.__hash__(),
                    }
                    for config, var in operation.config_with_variable
                }

                for config, _ in operation.config_with_variable:
                    if config.dominated_by == config:
                        if config.state != QuantizationStates.FP32:
                            render_buffer["values"][config.__hash__()] = {
                                "scale": convert_value(config.scale),
                                "zero_point": convert_value(config.offset),
                            }

                render_buffer["configs"][operation.name] = op_dict
                render_buffer["dispatchings"][operation.name] = operation.platform.name

        with open(file=config_path, mode="w") as file:
            json.dump(render_buffer, file, indent=4)

    def convert_operation_from_opset11_to_opset13(self, graph: BaseGraph) -> None:
        """Convert your network from opset 11 standard towards opset 13 With
        Onnx definition, per-channel quant operation requires opset 13.

        Args:
            graph (BaseGraph): Processing graph.
        """
        # this func transform representation of certain op from opset 11 to 13
        for op in graph.operations.values():
            if op.type == "ReduceSum" or op.type == "Squeeze" or op.type == "Unsqueeze":
                if "axes" not in op.attributes:
                    continue  # is already v13
                axes = convert_any_to_torch_tensor(op.attributes.pop("axes"), dtype=torch.int64)
                graph.create_variable(name=None, value=axes, is_parameter=True, dest_ops=[op])

            elif op.type == "Split":
                if "split" not in op.attributes:
                    continue  # split is already v13
                split = convert_any_to_torch_tensor(op.attributes.pop("split"), dtype=torch.int64)
                graph.create_variable(name=None, value=split, is_parameter=True, dest_ops=[op])

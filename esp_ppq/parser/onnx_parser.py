from typing import Any, Dict, Iterable, List, Optional

import onnx
from onnx import helper, numpy_helper

from esp_ppq.core import (
    DEFAULT_OPSET_DOMAIN,
    DEFAULT_OPSET_VERSION,
    GRAPH_OPSET_ATTRIB,
    DataType,
    NetworkFramework,
    is_file_exist,
    ppq_warning,
)
from esp_ppq.IR import BaseGraph, GraphBuilder, Operation, Opset, Variable


def _strip_trailing_empty_names(names: List[str]) -> List[str]:
    end = len(names)
    while end > 0 and names[end - 1] == '':
        end -= 1
    return names[:end]


def _extract_onnx_tensor_shape(value_info: Any) -> Optional[List[int]]:
    if not value_info.HasField('type') or not value_info.type.HasField('tensor_type'):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField('dim_value') and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            return None
    return dims


def _extract_onnx_tensor_dtype(value_info: Any) -> Optional[DataType]:
    if not value_info.HasField('type') or not value_info.type.HasField('tensor_type'):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('elem_type'):
        return None
    elem_type = int(tensor_type.elem_type)
    if elem_type <= 0:
        return None

    # DataType values follow ONNX TensorProto enums for supported types.
    try:
        return DataType(elem_type)
    except ValueError:
        mapper = helper.tensor_dtype_to_np_dtype
        try:
            np_dtype = mapper(elem_type) if callable(mapper) else mapper[elem_type]
            return DataType.convert_from_numpy(np_dtype)
        except (KeyError, TypeError, ValueError):
            return None


class OnnxParser(GraphBuilder):
    def build_variables(
        self,
        graph: BaseGraph,
        graph_inputs: List[str],
        graph_outputs: List[str],
        op_inputs: Dict[str, list],
        op_outputs: Dict[str, list],
    ) -> BaseGraph:
        # Work on local copies so the caller's dicts are never mutated.
        op_inputs = {name: _strip_trailing_empty_names(list(names)) for name, names in op_inputs.items()}
        op_outputs = {name: _strip_trailing_empty_names(list(names)) for name, names in op_outputs.items()}

        var_list = []

        for op_name in graph.operations:
            for var_name in op_inputs[op_name]:
                var_list.append(var_name)
            for var_name in op_outputs[op_name]:
                if var_name != '':
                    var_list.append(var_name)

        # create all variable at once.
        for var_name in set(var_list):
            graph.variables[var_name] = Variable(name=var_name)

        # build graph's input, output variables.
        try:
            for var_name in graph_inputs:
                if var_name not in graph.variables:
                    continue
                graph.inputs[var_name] = graph.variables[var_name]
            for var_name in graph_outputs:
                graph.outputs[var_name] = graph.variables[var_name]
        except KeyError as e:
            raise KeyError('seems you got an input/output variable that is not linked to any operation.')

        # build operation inputs, outputs variables.
        for op in graph.operations.values():
            for var_name in op_inputs[op.name]:
                var = graph.variables[var_name]
                var.dest_ops.append(op)
                op.inputs.append(var)

            for var_name in op_outputs[op.name]:
                if var_name == '':
                    # Give omitted interior outputs (e.g. GRU ["", "Y_h"]) a
                    # unique sink variable so multiple ops do not share
                    # variables[""] as source_op and form a fake cycle.
                    # create_variable(source_op=op) appends to op.outputs itself.
                    graph.create_variable(source_op=op)
                else:
                    var = graph.variables[var_name]
                    var.source_op = op
                    op.outputs.append(var)
        return graph

    def initialize_params(self, graph: BaseGraph, initializer: Dict[str, Any]) -> BaseGraph:
        for var in graph.variables.values():
            if var.name in initializer:
                for dest_op in var.dest_ops:
                    assert isinstance(dest_op, Operation)
                    dest_op.parameters.append(var)
                var.value = initializer[var.name]
                var.is_parameter = True
        return graph

    def de_inplace(self, graph: BaseGraph) -> BaseGraph:
        """Remove inplace layer in netdef If the names of bottom and top are same,
        it means the computation of this layer is in place."""

        def new_name(_name):
            if _name == '':
                return ''
            elif _name not in total_write_times or current_write_times:
                return _name
            elif current_write_times[_name] == total_write_times[_name]:
                return _name
            else:
                return f'{_name}_ver{current_write_times[_name]}'

        total_write_times = {}
        for op in graph.operations.values():
            for top in op.outputs:
                total_write_times.setdefault(top._name, 0)
                total_write_times[top._name] += 1

        current_write_times = {}
        for name in graph.inputs.keys():
            total_write_times[name] = 0
            current_write_times[name] = 0

        for op in graph.operations.values():
            for bottom in op.inputs:
                if bottom.is_parameter:
                    continue
                bottom._name = new_name(bottom._name)
            for top in op.outputs:
                current_write_times.setdefault(top._name, 0)
                current_write_times[top._name] += 1
                top._name = new_name(top._name)

    def refine_graph(self, graph: BaseGraph) -> BaseGraph:
        for op in graph.operations.values():
            for key, value in op.attributes.items():
                if isinstance(value, bytes):
                    # Change bytes to string
                    value = value.decode('utf-8')
                if op.type == 'Constant' or op.type == 'ConstantOfShape':
                    # The attribute of 'Constant' node is a value, needs to convert to numpy array
                    value = numpy_helper.to_array(value).copy()
                if op.type == 'Cast' and key == 'to':
                    # The attribute of 'Cast' node is data type (represented in int), need to convert to numpy data type.
                    # onnx 1.19 made ``tensor_dtype_to_np_dtype`` a function; older releases keep a dict.
                    mapper = helper.tensor_dtype_to_np_dtype
                    value = mapper(value) if callable(mapper) else mapper[value]
                op.attributes[key] = value

        graph_initializers = []
        for input_var in graph.inputs.values():
            # remove initilizer from graph.inputs
            if input_var.value is not None:
                graph_initializers.append(input_var.name)
        for non_input_var in graph_initializers:
            graph.inputs.pop(non_input_var)
        return graph

    def convert_opsets_to_str(self, opsets: Iterable) -> List[Dict[str, str]]:
        results = []
        for opset in opsets:
            results.append({'domain': opset.domain, 'version': opset.version})
        return results

    def build(self, file_path: str) -> BaseGraph:
        _rand_seed = 0  # used for name generation.
        if not is_file_exist(file_path):
            raise FileNotFoundError(f'file {file_path} does not exist, or it is a directory.')
        model_pb = onnx.load(file_path)
        assert isinstance(model_pb, onnx.ModelProto), (
            f'onnx load failed, only ProtoBuffer object is expected here, while {type(model_pb)} is loaded.'
        )
        inferred_pb = None
        try:
            inferred_pb = onnx.shape_inference.infer_shapes(model_pb, strict_mode=True, data_prop=True)
        except Exception as e:
            ppq_warning(f"ONNX shape inference failed for {file_path}, fallback to value_info from the raw model: {e}")
        opsets = model_pb.opset_import
        graph_pb = model_pb.graph
        graph = BaseGraph(name=graph_pb.name, built_from=NetworkFramework.ONNX)
        graph._detail[GRAPH_OPSET_ATTRIB] = self.convert_opsets_to_str(opsets)
        graph._detail['ir_version'] = model_pb.ir_version

        onnx_import_opset = DEFAULT_OPSET_VERSION
        for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
            if opset['domain'] == DEFAULT_OPSET_DOMAIN or opset['domain'] == '':
                onnx_import_opset = opset['version']
                break

        # a temporary storage for operation's inputs and outputs
        op_inputs_dict, op_outputs_dict = {}, {}
        for node in graph_pb.node:
            op_name = node.name
            if len(op_name) == 0:  # some operation do not have a name, we just generate one.
                op_name = 'generated_name_' + str(_rand_seed)
                _rand_seed += 1

            if op_name in graph.operations:
                raise KeyError(f'Duplicated operation {op_name} was found.')

            graph.operations[op_name] = Operation(
                name=op_name,
                op_type=node.op_type,
                attributes={item.name: helper.get_attribute_value(item) for item in node.attribute},
                opset=Opset(domain=DEFAULT_OPSET_DOMAIN, version=onnx_import_opset),
            )
            op_inputs_dict[op_name] = [var_name for var_name in node.input]
            op_outputs_dict[op_name] = [var_name for var_name in node.output]

        initializer = {}
        for item in graph_pb.initializer:
            init_name = item.name
            value = numpy_helper.to_array(item)
            initializer[init_name] = value

        inputs = [item.name for item in graph_pb.input]
        outputs = [item.name for item in graph_pb.output]
        graph = self.build_variables(
            graph, graph_inputs=inputs, graph_outputs=outputs, op_inputs=op_inputs_dict, op_outputs=op_outputs_dict
        )
        graph = self.initialize_params(graph, initializer)
        self.de_inplace(graph)
        shape_infos = (
            (*inferred_pb.graph.input, *inferred_pb.graph.output, *inferred_pb.graph.value_info)
            if inferred_pb is not None
            else (*graph_pb.input, *graph_pb.output, *graph_pb.value_info)
        )
        for info in shape_infos:
            var = graph.variables.get(info.name)
            if var is None or var.is_parameter:
                continue
            shape = _extract_onnx_tensor_shape(info)
            if shape is not None:
                var.shape = shape
            dtype = _extract_onnx_tensor_dtype(info)
            if dtype is not None:
                var.dtype = dtype
        return self.refine_graph(graph)

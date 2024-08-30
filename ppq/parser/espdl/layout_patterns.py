from typing import List

from ppq.core import (
    OperationQuantizationConfig,
)
from ppq.IR import BaseGraph, Operation, OperationExporter, Variable
from ppq.IR.quantize import QuantableOperation
from ppq.log import NaiveLogger
from ppq.parser.espdl.espdl_typedef import (
    ExporterPatternInfo,
)
from ppq.parser.espdl.export_patterns import fuse_downstream_operation

logger = NaiveLogger.get_logger('ESPDL')
# logger.set_level("DEBUG")

ACTIVATION_OP_SET = {
    "Relu",
    "PRelu",
    "Sigmoid",
    "Tanh",
    "HardSwish",
    "Elu",
    "Gelu",
    "Softmax",
    "Clip",
    "Cast",
}
CONV_LAYOUT_OP_SET = {"Conv", "GlobalAveragePool", "AveragePool", "MaxPool"}
ADD_LIKE_OP_SET = {"Add", "Sub", "Mul", "Div"}
OTHER_OP_SET = {
    "Matmul",
    "Gemm",
    "Flatten",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Slice",
    "Pad",
    "Split",
    "Concat",
    "Constant",
    "Gather",
    "Shape",
    "ConstantOfShape",
    "Expand",
    "ReduceMean",
}


def transpose_shape(input_shape, perm: List[int]) -> List[int]:
    if not perm:
        return input_shape
    return [input_shape[i] for i in perm]


def get_inverse_transpose(perm: List[int]) -> List[int]:
    """
    tensor == transpose(transpose(tensor))
    perm = [perm.index(i) for i in range(len(perm))]
    """
    return perm

def get_default_perm(var: Variable) -> List[int]:
    """
    return the default permute for given variable, [0,1,2,3,...]
    """
    if not var or not var.shape:
        return []

    return [i for i in range(len(var.shape))]


def insert_transpose_node(
    graph: BaseGraph, var: Variable, op: Operation, perm: List[int]
) -> Operation:
    """
    Insert a Transpose Node on given variable, according to given perm.
    """
    info = ExporterPatternInfo()

    if perm != range(len(perm)):
        logger.debug(
            f"insert transpose node: op: {op.name}, var:{var.name}, perm:{perm}"
        )
        created = graph.create_operation(op_type="Transpose", attributes={"perm": perm})
        if var in op.inputs:
            var_index = op.inputs.index(var)
            if isinstance(op, QuantableOperation):
                config = op.config
                # For transpose op,  input_quantization_config == output_quantization_config
                new_config = OperationQuantizationConfig(
                    [config.input_quantization_config[var_index]],
                    [config.input_quantization_config[var_index]],
                )
                created = QuantableOperation(created, new_config, op.platform)
                graph.operations[created.name] = created

            graph.insert_op_before(A=created, B=op, input_idx=var_index)
            new_var = created.outputs[0]
            new_var.shape = var.shape
            new_var.is_parameter = var.is_parameter
            new_var.dtype = var.dtype
            perm = get_default_perm(created.outputs[0])
            info.add_var_permute(
                created.outputs[0].name, get_default_perm(created.outputs[0])
            )

        else:
            raise ValueError(f"Unexpected Error in Exporting Op {op.name}({op.type}).")

        return created


def restore_origin_shape(op: Operation, graph: BaseGraph):
    info = ExporterPatternInfo()
    for var in op.inputs:
        if var.is_parameter:
            continue

        var_perm = info.get_var_permute(var.name)
        if var_perm and var_perm != get_default_perm(var):
            # There is already a permute, but this op need keep origin shape
            # A transpose node needs to be inserted into the word.
            inverse_perm = get_inverse_transpose(var_perm)
            insert_transpose_node(graph, var, op, inverse_perm)
        else:
            info.add_var_permute(var.name, get_default_perm(var))

    for var in op.outputs:
        info.add_var_permute(var.name, get_default_perm(var))
    return op


class ResetConvLayoutPattern(OperationExporter):
    """
    Modify Conv inputs and outputs layout from NCHW to NHWC
    And Update all variable's shape
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        info = ExporterPatternInfo()
        if op.type in CONV_LAYOUT_OP_SET:
            for var in op.inputs:
                if var.is_parameter:
                    continue

                var_shape = var.shape
                if len(var_shape) == 4:  # conv2d, NCHW -> NHWC
                    perm = [0, 2, 3, 1]
                elif len(var_shape) == 3:  # conv1d, NCW -> NWC
                    perm = [0, 2, 1]

                var_perm = info.get_var_permute(var.name)

                if var_perm:
                    if perm != var_perm:
                        # There is already a permute, but it does not match the conv layout.
                        # A transpose node needs to be inserted into the graph.
                        inverse_perm = get_inverse_transpose(var_perm)
                        new_perm = transpose_shape(inverse_perm, perm)
                        insert_transpose_node(graph, var, op, new_perm)
                else:
                    info.add_var_permute(var.name, perm)

            for var in op.outputs:
                var_shape = var.shape
                if len(var_shape) == 4:  # conv2d, NCHW -> NHWC
                    perm = [0, 2, 3, 1]
                else:  # conv1d, NCW -> NWC
                    perm = [0, 2, 1]
                info.add_var_permute(var.name, perm)

        return op


class RestoreOriginLayoutPattern(OperationExporter):
    """
    Restore original layout
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in OTHER_OP_SET:
            return restore_origin_shape(op, graph)

        return op


class BypassActivationLayoutPattern(OperationExporter):
    """
    Activation Node inherit transpose from upstream
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ACTIVATION_OP_SET:
            info = ExporterPatternInfo()
            assert len(op.outputs) == 1
            input1 = op.inputs[0]
            output = op.outputs[0]
            assert input1.shape == output.shape

            var_perm = info.get_var_permute(input1.name)
            if not var_perm:
                var_perm = get_default_perm(input1)
                info.add_var_permute(input1.name, var_perm)

            info.add_var_permute(op.outputs[0].name, var_perm)
        return op


class BypassAddLikePattern(OperationExporter):
    """
    Add,Mul,Sub,Div:

    two input and one output,
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ADD_LIKE_OP_SET:
            info = ExporterPatternInfo()
            input1 = op.inputs[0]
            input2 = op.inputs[1]
            output = op.outputs[0]
            input1_perm = info.get_var_permute(input1.name)
            input2_perm = info.get_var_permute(input2.name)
            output_perm = None
            if not input1.is_parameter and not input2.is_parameter:
                if input1_perm == input2_perm:
                    if input1_perm:
                        # using upstream's perm
                        output_perm = input1_perm
                    else:
                        # input1_perm is None, add new perm
                        info.add_var_permute(input1.name, get_default_perm(input1))
                        info.add_var_permute(input2.name, get_default_perm(input2))
                        output_perm = get_default_perm(output)
                    # logger.debug(f"{info.get_var_permute(input1.name)}, {info.get_var_permute(input2.name)}, {output_perm}")
                    info.add_var_permute(output.name, output_perm)
                else:
                    # insert transpose node and restore origin shape
                    return restore_origin_shape(op, graph)
            elif input2.is_parameter or input1.is_parameter:
                return restore_origin_shape(op, graph)

        return op


class ResetConcatPattern(OperationExporter):
    """
    Concat pattern with two input and one output,
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ["Concat"]:
            perm_dict = {}
            info = ExporterPatternInfo()

            for var in op.inputs:
                var_perm = info.get_var_permute(var.name)
                perm_str = str(var_perm)
                if perm_str not in perm_dict:
                    perm_dict[perm_str] = var_perm

            output_var = op.outputs[0]

            if len(perm_dict) == 1:  # all input have same perm, output bypass
                var_perm = list(perm_dict.values())[0]
                if not var_perm:
                    restore_origin_shape(op, graph)
                else:
                    axis = op.attributes["axis"]
                    new_axis = var_perm.index(int(axis))
                    op.attributes["axis"] = new_axis
                    info.add_var_permute(output_var.name, var_perm)
                    logger.debug(f"{op.name} update axes from {axis} to {new_axis}")
            else:
                logger.debug(f"transpose perm {perm_dict}")
                restore_origin_shape(op, graph)
        return op


class ResetResizePattern(OperationExporter):
    """
    Reize Layout Pattern
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.type in ["Resize"]:
            info = ExporterPatternInfo()
            input_var = op.inputs[0]
            if len(op.inputs) > 1:
                roi = op.inputs[1]
            else:
                roi = None
            
            if len(op.inputs) > 2:
                scales = op.inputs[2]
            else:
                scales = None
            
            if len(op.inputs) > 3:
                sizes = op.inputs[3]
            else:
                sizes = None
            
            for var in op.inputs[1:]:
                perm = info.get_var_permute(var.name, get_default_perm(var))

            input_perm = info.get_var_permute(input_var.name)
            output_var = op.outputs[0]

            if input_perm:
                for var in op.inputs[1:]:
                    if var:
                        info.add_var_permute(var.name, get_default_perm(var))
                info.add_var_permute(output_var.name, input_perm)
                # axes = op.attributes["axis"]
                # if axes:
                #     new_axes = [input_perm.index(i) for i in axes]
                #     op.attributes["axis"] = new_axes
                #     logger.debug(f"resize axes from {axes} to {new_axes} ")
                # else:
                #     if scales:
                #         values = scales.tolist()
                #         new_values = [
                #             values[input_perm[int(i)]] for i in range(len(values))
                #         ]
                #         scales.value = torch.Tensor(new_values, dtype=torch.int32)
                #         logger.debug(
                #             f"{op.name} reset scales from {values} to {new_values} "
                #         )
                #     if sizes:
                #         values = sizes.tolist()
                #         new_values = [
                #             values[input_perm[int(i)]] for i in range(len(values))
                #         ]
                #         sizes.value = torch.Tensor(new_values, dtype=torch.int32)
                #         logger.debug(
                #             f"{op.name} reset sizes from {values} to {new_values} "
                #         )
                #     if roi:
                #         values = sizes.tolist()
                #         new_values = []
                #         for i in range(len(values) / 2):
                #             new_values.append(values[input_perm[int(i)] * 2])
                #             new_values.append(values[input_perm[int(i)] * 2 + 1])
                #         roi.value = torch.Tensor(new_values, dtype=torch.int32)
                #         logger.debug(
                #             f"{op.name} reset sizes from {values} to {new_values} "
                #         )
            else:
                if len(input_var.shape) == 4:  # conv2d, NCHW -> NHWC
                    perm = [0, 2, 3, 1]
                elif len(input_var.shape) == 3:  # conv1d, NCW -> NWC
                    perm = [0, 2, 1]
                else:
                    logger.error(f"Reize: do not support shape for {input_var.shape}")
                
                info.add_var_permute(input_var.name, perm)
                info.add_var_permute(output_var.name, perm)

        return op


class FuseTransposePattern(OperationExporter):
    """
    Fuse Transpose Pattern
    """

    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # The FUSE_OP_PATTERNS may remove some ops.
        if op.name not in graph.operations:
            return op

        if op.type == "Transpose":
            downstream_transpose_op = None
            perm = op.attributes["perm"]
            while True:
                downstream_op = graph.get_downstream_operations(op)
                # the downstream op have only one op and this op is Transpose
                if len(downstream_op) == 1 and downstream_op[0].type == "Transpose":
                    downstream_transpose_op = downstream_op[0]
                    perm = transpose_shape(perm, downstream_op[0].attributes["perm"])

                    if isinstance(op, QuantableOperation):
                        op_config = op.config
                        downstream_config = downstream_transpose_op.config
                        new_config = OperationQuantizationConfig(
                            op_config.input_quantization_config,
                            downstream_config.output_quantization_config,
                        )
                        op.config = new_config
                    graph = fuse_downstream_operation(
                        graph, downstream_transpose_op, keep_coherence=True
                    )
                    op.attributes["perm"] = perm
                else:
                    break

            perm = op.attributes["perm"]
            if perm == [i for i in range(len(perm))]:
                # Removed redundant transpose
                graph.remove_operation(op, keep_coherence=True)
        return op

def print_vars(op: Operation):
    logger.info(f"Op: {op.name}, {op.type}, {op.attributes}")
    for var in op.inputs:
        print("inputs:", var.name, var.shape)
    for var in op.outputs:
        print("outputs:", var.name, var.shape)
    
def reset_graph_layout(graph: BaseGraph):
    """
    Reset layout from NCHW -> NHWC
    """

    layout_patterns = [
        [CONV_LAYOUT_OP_SET, ResetConvLayoutPattern], 
        [ACTIVATION_OP_SET, BypassActivationLayoutPattern],
        [ADD_LIKE_OP_SET, BypassAddLikePattern],
        [["Concat"], ResetConcatPattern],
        [["Resize"], ResetResizePattern],
        [OTHER_OP_SET, RestoreOriginLayoutPattern]
    ]
        
    for op in graph.topological_sort():
        flag = 1
        for pattern in layout_patterns:
            if op.type in pattern[0]:
                pattern[1]().export(op, graph)
                flag = 0
                break
        if flag:
            logger.error(f"Can not reset {op.type}:{op.name} layout")
    
    # fuse transpose op
    pattern = FuseTransposePattern()
    for op in graph.topological_sort():
        pattern.export(op, graph)

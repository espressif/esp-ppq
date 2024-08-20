

from abc import abstractmethod
from typing import (
    Dict,
    List,
)

from ppq.core import SingletonMeta
from ppq.IR import BaseGraph, Operation


def transpose_shape(input_shape, perm):
    return [input_shape[i] for i in perm]

def get_inverse_permute(perm):
    return [perm.index(i) for i in range(len(perm))]

class EspShapeInfer(metaclass=SingletonMeta):
    @ abstractmethod
    def shape_infer(self, shape_trans_dict: Dict[str, Dict[str, List[int]]], operation:Operation, graph: BaseGraph, **kwargs) -> Operation: pass
    
class ConvShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        
        op_vars = op.inputs() +  op.outputs()
        for var in op_vars:
            if var.is_parameter():
                continue

            var_shape = var.shape()
            if len(var_shape) == 4:     # conv2d, NCHW -> NHWC
                perm = [0, 2, 3, 1]
            else:                       # conv1d, NCW -> NWC
                perm = [0, 2, 1]
            
            if var.name in shape_trans_dict:
                if perm == shape_trans_dict[var.name]["perm"]:
                    return op
                else:
                    print(f"ERROR: shape inference error for {op.name}, perm mismatch!")

            shape_trans_dict[var.name] = {
                'origin': var_shape, 
                "perm": perm, 
                "transposed": transpose_shape(var_shape, perm)
            }
        return op

class GemmShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        input_var = op.inputs()[0]
        output_var = op.outputs()[0]

        if input_var.name in shape_trans_dict:
            shape_trans_dict[output_var.name] = {
                'origin': output_var.shape, 
                "perm": [1], 
                "transposed": output_var.shape
            }
            output_var.value(shape_trans_dict[input_var.name]["transposed"])
        return op

class MatmulShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        input_var = op.inputs()[0]
        output_var = op.outputs()[0]

        if input_var.name in shape_trans_dict:
            shape_trans_dict[output_var.name] = {
                'origin': output_var.shape, 
                "perm": [1], 
                "transposed": output_var.shape
            }
            output_var.value(shape_trans_dict[input_var.name]["transposed"])
        return op

class BypassShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        perm = None
        for var in op.inputs():
            if var.is_parameter():
                continue

            if var.name in shape_trans_dict:
                perm = shape_trans_dict[var.name]['perm']
                break
        
        if not perm: # If this op is one of graph's inputs, do not modify shape of variable
            return op

        op_vars = op.inputs() +  op.outputs()
        for var in op_vars:
            if var.is_parameter():
                continue

            if var.name in shape_trans_dict:
                continue

            var_shape = var.shape()
            shape_trans_dict[var.name] = {
                'origin': var_shape, 
                "perm": perm, 
                "transposed": transpose_shape(var_shape, perm)
            }
        return op
    
class TransposeShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        perm = None
        input_var = op.inputs()[0]
        output_var = op.outputs()[0]

        if input_var.name in shape_trans_dict:
            inverse_perm = get_inverse_permute(shape_trans_dict[input_var.name]['perm'])
            perm = transpose_shape(inverse_perm, op.attributes['perm'])
        else:
            return op
        
        output_shape = transpose_shape(output_var.shape(), perm)
        if output_var.shape == output_shape:  
            # remove this op
            graph.remove_operation(op, keep_coherence=True)
            return None
        else:
            shape_trans_dict[output_var.name] = {
                'origin': output_var.shape, 
                "perm": perm, 
                "transposed": output_shape
            }
        return op

class ShapeShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        input_var = op.inputs()[0]
        output_var = op.outputs()[0]

        if input_var.name in shape_trans_dict:
            shape_trans_dict[output_var.name] = {
                'origin': output_var.shape, 
                "perm": [1], 
                "transposed": output_var.shape
            }
            output_var.value(shape_trans_dict[input_var.name]["transposed"])
        return op

class ReshapeShapeInfer(EspShapeInfer):
    def shape_infer(self, 
                    shape_trans_dict: Dict[str, Dict[str, List[int]]],
                    op: Operation, 
                    graph: BaseGraph, 
                    **kwargs) -> Operation:
        input_var = op.inputs()[0]
        shape = op.inputs()[1]
        output_var = op.outputs()[0]
        input_shape = input_var.shape

        if input_var.name in shape_trans_dict:
            for dim in shape:

            reshape_dims = op.attributes['shape']

        print("ERROR: do not support this operation yet for shape inference")


GRAPH_SHAPE_INFERPATTERN = {
    "Conv": ConvShapeInfer,
    "GlobalAveragePool": ConvShapeInfer,
    "AveragePool": ConvShapeInfer,
    "MaxPool": ConvShapeInfer,
    "Transpose": TransposeShapeInfer,
    "Shape": ShapeShapeInfer,
    "Reshape": ReshapeShapeInfer
}
import torch

from esp_ppq import *
from esp_ppq.api import *
from esp_ppq.IR.morph import GraphMerger

graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = graph.create_operation(
    op_type='Matmul',
    name='matmul',
    platform=TargetPlatform.UNSPECIFIED,
    inputs=[graph.create_variable(), graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10]))],
    outputs=[graph.create_variable()],
)
graph.create_operation(
    op_type='Relu',
    name='relu',
    platform=TargetPlatform.UNSPECIFIED,
    inputs=[
        matmul.outputs[0],
        graph.create_variable(
            is_parameter=True,
            value=torch.zeros(
                size=[
                    10,
                ]
            ),
        ),
    ],
    outputs=[graph.create_variable()],
)
processor = QuantableGraph(graph)
processor.quantize_operation('matmul', target_platform=TargetPlatform.PPL_CUDA_INT8)
processor.quantize_operation('relu', target_platform=TargetPlatform.PPL_CUDA_INT8)

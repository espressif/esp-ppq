## PPQ Quantize Fusion Pass(通用量化图融合过程)

Operation fusion (or kernel/layer fusion) is key optimization in many state-of-the-art execution frameworks.

Graph fusion can combine operations into a single op to obtain higher accuracy and performance,
    Pattern like: Conv + Relu can be reduced to ConvRelu. This fusion will reduce memory accesses,
    and the quantization point after conv can also be removed.

Technically we can fuse those layers before quantization, while fused layers are not supported by onnx standard.
    So to say ConvRelu is not a valid onnx operation, no execution framework can parse it.

Therefore, PPQ will simulate the graph fusion by adjusting quantization config: if PPQ finds their is a
    pattern like Conv + Relu, the output quantization of Conv will be forbidden, pretending that the Conv + Relu
    fusion has happened.

This Pass is designed for 2 types fusion:

* activation fusion

    For activation fusion, PPQ will identify the pattern: Computing op + Activation Op from your network. The output
        quantization of computing op will be disabled with their state being set to QuantizationState.OVERLAPPED.

    Activation fusion here supports only simple activation patterns,
        for complex activation functions like mish, swish,
        will be represented as mish = tanh + mul + softplus, swish = sigmoid + mul in onnx,
        cause onnx does not have a op defination for them.
        Identifying those complex patterns requires pattern matching, which is implemented in esp_ppq.IR.search.py

    Complex quantization fusions must be invoked manually, PPQ implemented softplus & swish fusion functions in
        esp_ppq.quantization.optim.refine.MishFusionPass
        esp_ppq.quantization.optim.refine.SwishFusionPass

* passive operation fusion

    For passive operation fusion, PPQ will keep the input and the output variable share a same scale for passive operations.
        An operation is identified as passive op only if its attribute "is_active_quant_op" = False, this
        attribute is initilized by quantizer.

    If there is a passive operation having multiple input and output, the fusion procedure will make its
    FIRST input variable and ALL output variables share the same scale(the same scale as its first input).
    The quantization states of all output variables will be set to QuantizationState.OVERLAPPED.

### Graph-output boundary safety

#### Issue

ONNX permits an intermediate value to be both a public graph output and an input of a later operation:

```text
X -> Conv -> pre_activation -> Relu -> post_activation
                |
                +--------------------> graph output
```

In this graph, `pre_activation` and `post_activation` have different semantics and may both be public outputs.
The former can contain negative values, while the latter cannot.

The old fusion checks counted only downstream Operations. A public graph output is stored in
`graph.outputs`; it is not represented as another Operation in `Variable.dest_ops`. Consequently,
`pre_activation` appeared to have only one consumer (`Relu`) and was incorrectly considered an exclusive
fusion edge.

This caused two independent problems:

1. `QuantizeFusionPass` made the Conv output TQC depend on the Relu output TQC. The public
   pre-activation output therefore lost its independent calibration range.
2. `FuseReluLikePattern` physically removed Relu and replaced the Conv output Variable with
   `post_activation`. `fuse_downstream_operation()` then removed `pre_activation` from both
   `graph.variables` and `graph.outputs`, silently changing the model output interface.

The trigger is narrow, but the graph is valid ONNX and the failure is silent. It is therefore treated as a
low-frequency, high-impact graph-boundary issue. A graph output on `post_activation` is normal and does not
block fusion; only a public value on the edge being absorbed must block it.

At present, ESP-DL physically fuses `Conv`/`Gemm`/`MatMul -> Relu`. Quantization-level Relu/Clip fusion uses
the same boundary rule. Clip is not physically removed by the current ESP-DL exporter.

#### Resolution

Before either quantization-config fusion or physical exporter fusion, PPQ now checks the actual edge between
the upstream operation and the activation:

```python
downstream in variable.dest_ops and graph.outputs.get(variable.name) is variable
```

If that edge is a graph output, fusion is skipped. Its TQC, Variable identity, output name and output position
remain unchanged. Ordinary Conv-like-to-Relu graphs whose pre-activation edge is not public continue to fuse.

The protection is applied at both fusion entry points instead of changing the generic
`fuse_downstream_operation()` helper, because that helper is also used by layout rewrites with different
graph-output replacement rules.

Regression coverage verifies both quantization-fusion paths, physical ESP-DL fusion, preservation of both
public outputs, and the normal non-boundary fusion path.

### Parameters:

* activation_type(Set[str]):

        A collection contains all activation types.

        The pattern will be recognized as [Computing Op -> Activation Op],

        By graph fusion, the output quantization of the Computing Op and
            the input quantization of the activation op will be disabled.

* fuse_activation(bool)

        Whether to fuse activation op with computing op.

* fuse_passive_op(bool)

        Whether to fuse passive op so that the input variable and output variable will share a same scale.

### Usage
This pass is included in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.fusion = True

    # calling esp_ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE,
    collate_fn=collate_fn)

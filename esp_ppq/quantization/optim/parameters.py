from typing import Dict, Iterable

import torch

from esp_ppq.core import (
    ESPDL_QUANT_PLATFORMS,
    GRU_QUANT_EXPONENT,
    LSTM_QUANT_EXPONENT,
    QuantizationProperty,
    QuantizationStates,
    QuantizationVisibility,
    TargetPlatform,
    ppq_warning,
)
from esp_ppq.executor import BaseGraphExecutor, TorchExecutor
from esp_ppq.IR import BaseGraph, QuantableOperation
from esp_ppq.parser.espdl.espdl_typedef import QUANT_EXCLUDE_OP_SET, QUANT_OP_SET
from esp_ppq.parser.espdl.export_patterns import EspdlQuantHelper
from esp_ppq.quantization.observer import OperationObserver

from .base import QuantizationOptimizationPass


class PassiveParameterQuantizePass(QuantizationOptimizationPass):
    """
    ## PPQ Passive Parameter Quantization Pass(通用被动量化过程)

    Passive Parameters are those parameters that must share a same scale and offset with
    other tensors. This pass process 4 types of passive parameter by default, namely:

        Bias value in Gemm, it must has a scale = input scale * weight scale.

        Bias value in Conv, it must has a scale = input scale * weight scale.

        Clip min & Clip max in Clip, must has a scale = input scale

            Every platform still aligns a Clip bound with ``master_by``. The checks
            below apply only when ``op.platform`` is an ESP-DL quant platform,
            because that integer Clip compares raw codes:

                - static and uniquely-consumed constants can be remapped to input scale
                  by setting master_by, with bit width and clamp range aligned to the
                  Clip output so simulation and export clamp at the same integer code.
                - a runtime bound whose producer already writes the Clip input grid is
                  kept. Export inserts no conversion, so this stays allowed when the
                  bound visibility is INTERNAL.
                - a runtime bound on another grid is kept only when export will insert
                  QuantizeLinear or RequantizeLinear. This pass writes master_by, the
                  output bit range, and its visibility into the bound config, then calls
                  EspdlQuantHelper.TQC_Exportable_Check. INTERNAL visibility, an
                  activation grid with no scale/offset, or an illegal aligned INT8 range
                  fails that check and is rejected. The producer must also be one the
                  export patterns rewrite: not a graph input, and not an op in
                  QUANT_OP_SET or QUANT_EXCLUDE_OP_SET. A graph input has no producer
                  to rewrite and is rejected.
                - shared constants (dest_ops > 1) are always rejected. esp-dl export
                  requires each parameter to have one consumer, so a shared bound
                  fails there even when its scale already matches the input.
                  Integer Clip compares raw codes and can not requantize bounds online.

        Pading Value, must has a scale = input scale

    ### Parameters:

    * process_clip(Set[str]):

            Whether to process clip min, max

            If not processed, clip min, max will has their state = QuantizationState.FP32.
            esp-dl export then rejects an integer Clip kernel: the simulator clamps the
            float bound, while the device reads it as a code. Keep
            quantize_passive_parameter enabled for that export.

    * process_bias(bool)

            Whether to process bias

            If not processed, bias will has their state = QuantizationState.ACTIVED

    * process_pad(bool)

            Whether to process clip min, max

            If not processed, pad value will has their state = QuantizationState.SOI

    * clip_visiblity(bool)

            Whether to export quant info of clip min, max

    * pad_visiblity(bool)

            Whether to export quant info of pad value

    ### Usage
    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.parameter_setting.quantize_passive_parameter = True = True
        # calling esp_ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn)
    """

    def __init__(
        self,
        process_clip: bool = True,
        process_bias: bool = True,
        process_pad: bool = True,
        clip_visiblity: QuantizationVisibility = QuantizationVisibility.INTERNAL,
        pad_visiblity: QuantizationVisibility = QuantizationVisibility.INTERNAL,
    ):
        self.process_clip = process_clip
        self.process_bias = process_bias
        self.process_pad = process_pad
        self.clip_visiblity = clip_visiblity
        self.pad_visiblity = pad_visiblity
        super().__init__(name='PPQ Passive Parameter Quantization')

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        def check_state(state: QuantizationStates):
            return state in {
                QuantizationStates.PASSIVE,
                QuantizationStates.ACTIVATED,
                QuantizationStates.BAKED,
                QuantizationStates.OVERLAPPED,
            }

        def share_scale(left: torch.Tensor, right: torch.Tensor) -> bool:
            # atol=0 keeps distant power-of-two scales apart; the default 1e-8
            # would call 2**-27 and 2**-28 the same grid.
            if left is None or right is None:
                return False
            left_scale = torch.as_tensor(left)
            right_scale = torch.as_tensor(right)
            return torch.allclose(left_scale, right_scale, atol=0.0)

        def espdl_clip_compares_raw_codes(op) -> bool:
            return getattr(op, 'platform', None) in ESPDL_QUANT_PLATFORMS

        runtime_integer_states = {
            QuantizationStates.ACTIVATED,
            QuantizationStates.PASSIVE,
            QuantizationStates.BAKED,
            QuantizationStates.PASSIVE_BAKED,
            QuantizationStates.OVERLAPPED,
        }

        def runtime_bound_can_be_aligned_during_export(
            var, config, master, aligned_bits, aligned_min, aligned_max
        ) -> bool:
            """Whether export will insert QuantizeLinear or RequantizeLinear for this bound.

            Exportability is EspdlQuantHelper.TQC_Exportable_Check on the config this
            pass is about to commit. The bound is restored afterwards; the caller
            writes it only when the whole Clip is accepted. Producer exclusion is
            QUANT_OP_SET | QUANT_EXCLUDE_OP_SET, the same sets InsertQuantNodePattern
            and InsertRequantNodePattern consult.
            """
            producer = var.source_op
            if producer is None:
                return False
            if getattr(producer, "type", None) in (QUANT_OP_SET | QUANT_EXCLUDE_OP_SET):
                return False

            saved = (
                config._dominator,
                config.state,
                config.num_of_bits,
                config.quant_min,
                config.quant_max,
                config.visibility,
            )
            try:
                config.master_by = master
                config.num_of_bits = aligned_bits
                config.quant_min = aligned_min
                config.quant_max = aligned_max
                config.visibility = self.clip_visiblity
                if not EspdlQuantHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                    return False
                if not isinstance(producer, QuantableOperation):
                    # InsertQuantNodePattern quantizes regular FP32 operation outputs.
                    return True
                if var not in producer.outputs:
                    return False

                produced_config = producer.output_quant_config[producer.outputs.index(var)]
                # InsertQuantNodePattern recognizes this exact state and can add a
                # QuantizeLinear node using the rewritten Clip-side config.
                if produced_config.state == QuantizationStates.FP32:
                    return True

                # InsertRequantNodePattern reads an existing integer grid. These states
                # are the ones that still hold those codes; this is not a second
                # exportability check. An INTERNAL producer can still be requantized.
                produced = produced_config.dominated_by
                return (
                    produced_config.state in runtime_integer_states
                    and produced.state in runtime_integer_states
                    and isinstance(produced.scale, torch.Tensor)
                    and isinstance(produced.offset, torch.Tensor)
                    and produced.policy.has_property(QuantizationProperty.LINEAR)
                    and int(produced.num_of_bits) in (8, 16)
                )
            finally:
                (
                    config._dominator,
                    config.state,
                    config.num_of_bits,
                    config.quant_min,
                    config.quant_max,
                    config.visibility,
                ) = saved

        def runtime_bound_grid_mismatch_reason(var, i_cfg):
            # esp-dl Clip compares integer codes already stored in the runtime bound.
            # Those codes come from the producer output, not this Clip input config.
            producer = var.source_op
            if not isinstance(producer, QuantableOperation) or var not in producer.outputs:
                return "has no quantized producer output"
            produced_config = producer.output_quant_config[producer.outputs.index(var)]
            produced = produced_config.dominated_by
            if produced_config.state == QuantizationStates.FP32 or produced.state == QuantizationStates.FP32:
                return "producer output is FP32 and has no quantized codes"
            input_grid = i_cfg.dominated_by
            if produced.scale is None or produced.offset is None:
                return "producer output has no quantized scale/offset"
            if produced_config.state not in runtime_integer_states or produced.state not in runtime_integer_states:
                return f"producer output state {produced_config.state.name} has no quantized codes"
            if not share_scale(produced.scale, input_grid.scale) or not share_scale(produced.offset, input_grid.offset):
                return "producer output does not share the input scale/offset"
            produced_bits = int(produced.num_of_bits)
            input_bits = int(input_grid.num_of_bits)
            if produced_bits != input_bits:
                return f"producer output bit width {produced_bits} differs from Clip input bit width {input_bits}"
            return None

        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation):
                continue

            if op.type in {'Conv', 'ConvTranspose', 'Gemm'} and self.process_bias:
                # inputs are [input value, weight, bias(optional)]
                if op.num_of_input == 3:
                    i_cfg, w_cfg, b_cfg = op.config.input_quantization_config
                    if b_cfg.state not in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT}:
                        continue

                    # PATCH 2022.07.29 有的时候 bias 是个多维的东西，此时要求前面的维度都是1
                    bias = op.inputs[-1].value
                    if bias is None:
                        raise ValueError(
                            f'Bias Varaible {op.inputs[-1].name} must be a constant. Please check it again.'
                        )

                    assert bias.numel() == bias.shape[-1], (
                        f'For op {op.name}, expect Bias shape to be {[bias.numel()]}, however {bias.shape} was given'
                    )
                    op.inputs[-1].value = bias.squeeze()
                    # PATCH 2022.08.02 只有一个数的 bias 经过 squeeze 会变成零维的, 再给它多加一维补回来
                    if op.inputs[-1].value.ndim == 0 and op.inputs[-1].value.numel() == 1:
                        op.inputs[-1].value = op.inputs[-1].value.unsqueeze(0)

                    if not check_state(i_cfg.state):
                        raise PermissionError(
                            f'Can not quantize bias of layer {op.name}, cause input has not been correctly quantized.'
                        )

                    b_cfg.scale = w_cfg.scale * i_cfg.scale
                    b_cfg.state = QuantizationStates.PASSIVE
                    b_cfg.offset = torch.zeros_like(b_cfg.scale)
                    assert not b_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                        'Passive parameter does not support ASYMMETRICAL quantization'
                    )

            if op.type in {'GRU', 'LSTM'} and self.process_bias:
                # inputs are [input value, weight, bias(optional)]
                reset_variables = [3]  # 3:bias for GRU
                if op.type == "LSTM":
                    reset_variables = [3, 6]  # 3:bias, 6: initial_c for LSTM

                i_cfg = op.config.input_quantization_config[0]
                w_cfg = op.config.input_quantization_config[1]
                if not check_state(i_cfg.state):
                    raise PermissionError(
                        f'Can not quantize bias of layer {op.name}, cause input has not been correctly quantized.'
                    )

                for idx in reset_variables:
                    if op.num_of_input > idx:
                        var_cfg = op.config.input_quantization_config[idx]
                        if var_cfg.state not in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT}:
                            continue

                        var_cfg.scale = w_cfg.scale * 0 + pow(
                            2, LSTM_QUANT_EXPONENT if op.type == 'LSTM' else GRU_QUANT_EXPONENT
                        )
                        var_cfg.state = QuantizationStates.PASSIVE
                        var_cfg.offset = torch.zeros_like(var_cfg.scale)
                        assert not var_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                            'Passive parameter does not support ASYMMETRICAL quantization'
                        )

                if op.num_of_output == 3 and op.type == 'LSTM':
                    cell_cfg = op.config.output_quantization_config[2]
                    if cell_cfg.state not in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT}:
                        continue

                    cell_cfg.scale = w_cfg.scale * 0 + pow(2, LSTM_QUANT_EXPONENT)
                    cell_cfg.state = QuantizationStates.PASSIVE
                    cell_cfg.offset = torch.zeros_like(cell_cfg.scale)
                    assert not cell_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                        'Passive parameter does not support ASYMMETRICAL quantization'
                    )

            if op.type in {'Clip'} and self.process_clip:
                # inputs are [input value, min[optional], max[optional]]
                i_cfg = op.config.input_quantization_config[0]
                o_cfg = op.config.output_quantization_config[0] if op.config.output_quantization_config else i_cfg

                if not check_state(i_cfg.state):
                    raise PermissionError(
                        f'Can not quantize clip value of layer {op.name}, cause input has not been correctly quantized.'
                    )

                in_dom = i_cfg.dominated_by
                out_dom = o_cfg.dominated_by
                if espdl_clip_compares_raw_codes(op) and out_dom.state == QuantizationStates.FP32:
                    raise ValueError(
                        f'Clip {op.name} on esp-dl requires a quantized activation output (8 or 16-bit), '
                        'but output is FP32. Keep this op on FP32 platform, or calibrate and activate output '
                        'quantization first.'
                    )
                if espdl_clip_compares_raw_codes(op) and int(in_dom.num_of_bits) != int(out_dom.num_of_bits):
                    raise ValueError(
                        f'Clip {op.name} on esp-dl requires input and output activations to share one bit width '
                        f'(8 or 16), got input {int(in_dom.num_of_bits)} and output {int(out_dom.num_of_bits)}.'
                    )

                bound_vars = list(op.inputs)[1:]
                for index, config in enumerate(op.config.input_quantization_config[1:]):
                    var = bound_vars[index] if index < len(bound_vars) else None
                    if var is not None and var.name != '' and espdl_clip_compares_raw_codes(op):
                        if var.value is None:
                            reason = runtime_bound_grid_mismatch_reason(var, i_cfg)
                            if reason is not None and not runtime_bound_can_be_aligned_during_export(
                                var,
                                config,
                                i_cfg,
                                out_dom.num_of_bits,
                                out_dom.quant_min,
                                out_dom.quant_max,
                            ):
                                raise PermissionError(
                                    f'Can not quantize clip value of layer {op.name}, '
                                    f'cause {var.name} (runtime tensor) {reason}.'
                                )
                        elif len(var.dest_ops) > 1:
                            raise PermissionError(
                                f'Can not quantize clip value of layer {op.name}, '
                                f'cause {var.name} (shared constant) is used by more than one operation.'
                            )
                    config.master_by = i_cfg
                    if espdl_clip_compares_raw_codes(op):
                        config.num_of_bits = out_dom.num_of_bits
                        config.quant_min = out_dom.quant_min
                        config.quant_max = out_dom.quant_max
                    config.visibility = self.clip_visiblity

            if op.type in {'Pad'} and self.process_pad:
                # inputs are [input value, pad[shape-related], pad value[optional]]
                if op.num_of_input != 3:
                    continue
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(
                        f'Can not quantize pad value of layer {op.name}, cause input has not been correctly quantized.'
                    )

                if len(op.config.input_quantization_config) > 1:
                    pad_config = op.config.input_quantization_config[-1]
                    pad_config.master_by = i_cfg
                    pad_config.visibility = self.pad_visiblity

        # final check
        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation):
                continue
            for cfg, var in op.config_with_variable:
                if cfg.state == QuantizationStates.PASSIVE_INIT:
                    ppq_warning(
                        f'Unexpected quantization state of variable {var.name} at op {op.name}, '
                        'The configuration state has been initialized as PASSIVE_INIT, '
                        'however PassiveParameterQuantizePass do not kown how to deal with it.'
                    )


class ParameterQuantizePass(QuantizationOptimizationPass):
    """PPQ PassiveParameterQuantizePass completes the quantization of positive
    parameters. By default, all parameters with initial state will be quantized
    during this optimization, all non-parameter tensors will be excluded from
    this pass by temporary dequantization.

    Then, operation observers will be established automatically to record necessary statistics,
        observers are also responsible for rendering quantization configuration (computing scale and offset).

    This pass needs no data, however it uses fake data to finish a dummy forward process.
    see also: TorchExecutor.dummy_forward function
    """

    def __init__(self, method: str = None, method_per_layer: Dict[str, str] = None):
        self._method = method
        self._method_per_layer = method_per_layer or {}
        super().__init__(name='PPQ Parameter Quantization Pass')

    def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: TorchExecutor, **kwargs) -> None:
        # build observer and hook for each quantable operation
        hooks, observers, state_records = {}, {}, {}
        for op_name, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue

            # per-layer method takes precedence over the global method.
            layer_method = self._method_per_layer.get(op_name, self._method)
            for config, var in operation.config_with_variable:
                # deactivate non-parameter variable quantization just for now
                if not var.is_parameter:
                    state_records[config] = config.state
                    config.state = QuantizationStates.FP32
                elif layer_method is not None:
                    # override quantizer's setting if necessary
                    config.observer_algorithm = layer_method

            observer = OperationObserver(
                operation=executor._graph.operations[op_name], monitor_outputs=False, monitor_inputs=False
            )
            observers[op_name] = observer
            hooks[op_name] = observer.hook

        # dummy forward, quant all parameter.
        assert isinstance(executor, TorchExecutor), 'ParameterQuantizePass Only support TorchExecutor now.'
        executor.dummy_forward(hooks=hooks)

        # render quantization config, restore non-parameter quantization state
        for op_name, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue

            for cfg, var in operation.config_with_variable:
                if not var.is_parameter:
                    cfg.state = state_records[cfg]

            observer = observers[op_name]
            assert isinstance(observer, OperationObserver)
            observer.render_quantization_config()
            observer.report()

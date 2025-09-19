from .default import DEFAULT_BACKEND_TABLE

ESPDL_QUANT_BACKEND_TABLE = DEFAULT_BACKEND_TABLE.copy()

from esp_ppq.core import (
    GRU_QUANT_BITS,
    GRU_QUANT_EXPONENT,
    LSTM_QUANT_BITS,
    LSTM_QUANT_EXPONENT,
    RoundingPolicy,
    TargetPlatform,
    TensorQuantizationConfig,
)
from esp_ppq.quantization.qfunction import PPQuantFunction
from esp_ppq.utils.round import ppq_tensor_round

from .base import *


def GRU_float_forward(
    op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs
) -> torch.Tensor:
    """Computes an one-layer GRU using basic PyTorch operations."""
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    # Extract inputs
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)

    # Get attributes
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)

    # Configuration
    bidirectional = direction == 'bidirectional'
    num_directions = 2 if bidirectional else 1
    batch_first = layout == 1
    has_bias = b is not None

    # Reshape input if batch_first
    if batch_first:
        x = x.transpose(0, 1)  # [batch, seq, input] -> [seq, batch, input]

    seq_length, batch_size, input_size = x.shape

    # Initialize hidden state
    if initial_h is None:
        h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device, dtype=x.dtype)
    else:
        h = initial_h

    # Prepare biases according to GRU formula
    def prepare_biases(bias, hidden_size, direction_idx=0):
        if bias is None:
            return (None, None, None, None, None, None)

        if bias.dim() == 2:  # [num_directions, ...]
            b_dir = bias[direction_idx]
        else:
            b_dir = bias

        # Split into Wb (input bias) and Rb (recurrent bias) components
        # Bias layout: [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
        wb_z = b_dir[:hidden_size]  # Wbz
        wb_r = b_dir[hidden_size : 2 * hidden_size]  # Wbr
        wb_h = b_dir[2 * hidden_size : 3 * hidden_size]  # Wbh
        rb_z = b_dir[3 * hidden_size : 4 * hidden_size]  # Rbz
        rb_r = b_dir[4 * hidden_size : 5 * hidden_size]  # Rbr
        rb_h = b_dir[5 * hidden_size : 6 * hidden_size]  # Rbh

        return wb_z, wb_r, wb_h, rb_z, rb_r, rb_h

    # Process single direction
    def process_direction(x, w, r, b, initial_h, direction_idx=0, reverse=False):
        # Get weights for this direction
        if w.dim() == 3:  # [num_directions, 3*hidden_size, input_size]
            w_dir = w[direction_idx]  # [3*hidden_size, input_size]
            r_dir = r[direction_idx]  # [3*hidden_size, hidden_size]
        else:
            w_dir = w  # [3*hidden_size, input_size]
            r_dir = r  # [3*hidden_size, hidden_size]

        # Get biases for this direction
        if has_bias:
            wb_z, wb_r, wb_h, rb_z, rb_r, rb_h = prepare_biases(b, hidden_size, direction_idx)
        else:
            wb_z, wb_r, wb_h, rb_z, rb_r, rb_h = None, None, None, None, None, None

        # Initialize hidden state for this direction
        h_t = initial_h[direction_idx] if initial_h.dim() == 3 else initial_h

        # Reverse sequence if needed
        if reverse:
            x = x.flip(0)

        # Process sequence
        outputs = []
        for t in range(seq_length):
            x_t = x[t]  # [batch_size, input_size]

            # Compute all gates at once: x_t @ W^T -> [batch_size, 3*hidden_size]
            xw = torch.matmul(x_t, w_dir.t())  # [batch_size, 3*hidden_size]

            # Split into z, r, h components
            xw_z, xw_r, xw_h = torch.split(xw, hidden_size, dim=1)

            # Compute all recurrent connections at once: h_t @ R^T -> [batch_size, 3*hidden_size]
            hr = torch.matmul(h_t, r_dir.t())  # [batch_size, 3*hidden_size]

            # Split into z, r, h components
            hr_z, hr_r, hr_h = torch.split(hr, hidden_size, dim=1)

            # Update gate (z): zt = σ(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
            z_input = xw_z + hr_z
            if wb_z is not None:
                z_input = z_input + wb_z + rb_z  # Wbz
            z_t = torch.sigmoid(z_input)

            # Reset gate (r): rt = σ(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
            r_input = xw_r + hr_r
            if wb_r is not None:
                r_input = r_input + wb_r + rb_r  # Wbr
            r_t = torch.sigmoid(r_input)

            # Hidden gate (h): ht = tanh(Xt*(Wh^T) + (rt ⊙ (Ht-1*(Rh^T) + Rbh)) + Wbh)
            # linear_before_reset != 0 的情况

            # 计算 Ht-1*(Rh^T) + Rbh
            hr_h_with_bias = hr_h
            if rb_h is not None:
                hr_h_with_bias = hr_h_with_bias + rb_h  # Rbh

            # 计算 rt ⊙ (Ht-1*(Rh^T) + Rbh)
            reset_gated_hidden = r_t * hr_h_with_bias

            # 计算 Xt*(Wh^T) + Wbh
            xw_h_with_bias = xw_h
            if wb_h is not None:
                xw_h_with_bias = xw_h_with_bias + wb_h  # Wbh

            # 最终隐藏门计算
            h_input = xw_h_with_bias + reset_gated_hidden
            h_tilde = torch.tanh(h_input)

            # New hidden state: Ht = (1 - zt) ⊙ ht + zt ⊙ Ht-1
            h_t = (1 - z_t) * h_tilde + z_t * h_t
            outputs.append(h_t)

        outputs = torch.stack(outputs)  # [seq_length, batch_size, hidden_size]

        if reverse:
            outputs = outputs.flip(0)

        return outputs, h_t

    # Process forward direction
    forward_output, forward_final = process_direction(x, w, r, b, h, 0, False)

    if bidirectional:
        # Process reverse direction
        reverse_output, reverse_final = process_direction(x, w, r, b, h, 1, True)

        # Concatenate outputs
        outputs = torch.cat([forward_output.unsqueeze(1), reverse_output.unsqueeze(1)], dim=1)
        final_hidden = torch.stack([forward_final, reverse_final])
    else:
        outputs = forward_output.unsqueeze(1)  # [seq_length, 1, batch_size, hidden_size]
        final_hidden = forward_final.unsqueeze(0)  # [1, batch_size, hidden_size]

    return outputs, final_hidden


def GRU_quant_forward(
    op: Operation,
    values: List[torch.Tensor],
    ctx: TorchBackendContext = None,
    **kwargs,
) -> torch.Tensor:
    """Computes an one-layer GRU using basic PyTorch operations."""
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    # Extract inputs
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)

    # Get attributes
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)

    # Configuration
    bidirectional = direction == 'bidirectional'
    num_directions = 2 if bidirectional else 1
    batch_first = layout == 1
    has_bias = b is not None

    # Reshape input if batch_first
    if batch_first:
        x = x.transpose(0, 1)  # [batch, seq, input] -> [seq, batch, input]

    seq_length, batch_size, input_size = x.shape

    # Initialize hidden state
    if initial_h is None:
        h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device, dtype=x.dtype)
    else:
        h = initial_h

    # Prepare biases according to GRU formula
    def prepare_biases(bias, hidden_size, direction_idx=0):
        if bias is None:
            return (None, None, None, None, None, None)

        if bias.dim() == 2:  # [num_directions, ...]
            b_dir = bias[direction_idx]
        else:
            b_dir = bias

        # Split into Wb (input bias) and Rb (recurrent bias) components
        # Bias layout: [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
        wb_z = b_dir[:hidden_size]  # Wbz
        wb_r = b_dir[hidden_size : 2 * hidden_size]  # Wbr
        wb_h = b_dir[2 * hidden_size : 3 * hidden_size]  # Wbh
        rb_z = b_dir[3 * hidden_size : 4 * hidden_size]  # Rbz
        rb_r = b_dir[4 * hidden_size : 5 * hidden_size]  # Rbr
        rb_h = b_dir[5 * hidden_size : 6 * hidden_size]  # Rbh

        z_bias = wb_z + rb_z
        # z_bias = fake_quantize(z_bias)
        r_bias = wb_r + rb_r
        # r_bias = fake_quantize(r_bias)

        return z_bias, r_bias, wb_h, rb_h

    def fake_quantize(
        tensor,
        exponent=GRU_QUANT_EXPONENT,
        bits=GRU_QUANT_BITS,
        rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
    ):
        scale = pow(2, exponent)
        quant_min = -pow(2, bits - 1)
        quant_max = -quant_min - 1
        tensor = ppq_tensor_round((tensor / scale), rounding)
        tensor = torch.clamp(tensor, quant_min, quant_max)
        tensor = tensor * scale
        return tensor

    # Process single direction
    def process_direction(
        x, w, r, b, initial_h, direction_idx=0, reverse=False, quant_config: TensorQuantizationConfig = None
    ):
        # Get weights for this direction
        if w.dim() == 3:  # [num_directions, 3*hidden_size, input_size]
            w_dir = w[direction_idx]  # [3*hidden_size, input_size]
            r_dir = r[direction_idx]  # [3*hidden_size, hidden_size]
        else:
            w_dir = w  # [3*hidden_size, input_size]
            r_dir = r  # [3*hidden_size, hidden_size]

        if quant_config is not None:
            rounding = quant_config.rounding
        else:
            rounding = RoundingPolicy.ROUND_HALF_EVEN

        # Get biases for this direction
        if has_bias:
            z_bias, r_bias, wb_h, rb_h = prepare_biases(b, hidden_size, direction_idx)
        else:
            z_bias, r_bias, wb_h, rb_h = None, None, None, None

        # Initialize hidden state for this direction
        h_t = initial_h[direction_idx] if initial_h.dim() == 3 else initial_h

        # Reverse sequence if needed
        if reverse:
            x = x.flip(0)

        # Process sequence
        outputs = []
        for t in range(seq_length):
            x_t = x[t]  # [batch_size, input_size]

            # Compute all gates at once: x_t @ W^T -> [batch_size, 3*hidden_size]
            xw = torch.matmul(x_t, w_dir.t())  # [batch_size, 3*hidden_size]
            xw = fake_quantize(xw, rounding=rounding)

            # Split into z, r, h components
            xw_z, xw_r, xw_h = torch.split(xw, hidden_size, dim=1)

            # Compute all recurrent connections at once: h_t @ R^T -> [batch_size, 3*hidden_size]
            hr = torch.matmul(h_t, r_dir.t())  # [batch_size, 3*hidden_size]
            hr = fake_quantize(hr, rounding=rounding)

            # Split into z, r, h components
            hr_z, hr_r, hr_h = torch.split(hr, hidden_size, dim=1)

            # Update gate (z): zt = σ(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
            z_input = xw_z + hr_z
            if z_bias is not None:
                z_input = z_input + z_bias  # Wbz
            z_t = torch.sigmoid(z_input)

            # Reset gate (r): rt = σ(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
            r_input = xw_r + hr_r
            if r_bias is not None:
                r_input = r_input + r_bias  # Wbr
            r_t = torch.sigmoid(r_input)

            # Hidden gate (h): ht = tanh(Xt*(Wh^T) + (rt ⊙ (Ht-1*(Rh^T) + Rbh)) + Wbh)
            # linear_before_reset != 0 的情况

            # 计算 Ht-1*(Rh^T) + Rbh
            hr_h_with_bias = hr_h
            if rb_h is not None:
                hr_h_with_bias = hr_h_with_bias + rb_h  # Rbh

            # 计算 rt ⊙ (Ht-1*(Rh^T) + Rbh)
            reset_gated_hidden = r_t * hr_h_with_bias

            # 计算 Xt*(Wh^T) + Wbh
            xw_h_with_bias = xw_h
            if wb_h is not None:
                xw_h_with_bias = xw_h_with_bias + wb_h  # Wbh

            # 最终隐藏门计算
            h_input = fake_quantize(xw_h_with_bias + reset_gated_hidden)
            h_tilde = torch.tanh(h_input)

            # New hidden state: Ht = (1 - zt) ⊙ ht + zt ⊙ Ht-1
            h_t = (1 - z_t) * h_tilde + z_t * h_t

            # scale is not None when model is quantized
            if output_config.scale is not None:
                h_t = PPQuantFunction(h_t, quant_config)
            outputs.append(h_t)

        outputs = torch.stack(outputs)  # [seq_length, batch_size, hidden_size]

        if reverse:
            outputs = outputs.flip(0)

        return outputs, h_t

    # Create quantization config for gates and hidden state

    if len(op.config.output_quantization_config) > 0:
        output_config = op.config.output_quantization_config[0]
    else:
        raise TypeError('GRU_quant_forward except a TensorQuantizationConfig instance.')

    # Process forward direction
    forward_output, forward_final = process_direction(x, w, r, b, h, 0, False, output_config)

    if bidirectional:
        # Process reverse direction
        reverse_output, reverse_final = process_direction(x, w, r, b, h, 1, True, output_config)

        # Concatenate outputs
        outputs = torch.cat([forward_output.unsqueeze(1), reverse_output.unsqueeze(1)], dim=1)
        final_hidden = torch.stack([forward_final, reverse_final])
    else:
        outputs = forward_output.unsqueeze(1)  # [seq_length, 1, batch_size, hidden_size]
        final_hidden = forward_final.unsqueeze(0)  # [1, batch_size, hidden_size]

    return outputs, final_hidden


ESPDL_QUANT_BACKEND_TABLE['GRU'] = GRU_quant_forward

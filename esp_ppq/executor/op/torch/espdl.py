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
            xw = fake_quantize(xw, GRU_QUANT_EXPONENT, GRU_QUANT_BITS, rounding=rounding)

            # Split into z, r, h components
            xw_z, xw_r, xw_h = torch.split(xw, hidden_size, dim=1)

            # Compute all recurrent connections at once: h_t @ R^T -> [batch_size, 3*hidden_size]
            hr = torch.matmul(h_t, r_dir.t())  # [batch_size, 3*hidden_size]
            hr = fake_quantize(hr, GRU_QUANT_EXPONENT, GRU_QUANT_BITS, rounding=rounding)

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
            h_input = fake_quantize(
                xw_h_with_bias + reset_gated_hidden, GRU_QUANT_EXPONENT, GRU_QUANT_BITS, rounding=rounding
            )
            h_tilde = torch.tanh(h_input)

            # New hidden state: Ht = (1 - zt) ⊙ ht + zt ⊙ Ht-1
            h_t = (1 - z_t) * h_tilde + z_t * h_t

            # scale is not None when model is quantized
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


# -------------- LSTM --------------


def LSTM_float_forward(
    op: Operation,
    values: List[torch.Tensor],
    ctx=None,
    **kwargs,
) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=8)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    x, w, r = values[:3]  # x: [seq, batch, input]  (layout=0)
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    initial_c = GET_VALUE_FROM_INPUTS(values, 6)
    p = GET_VALUE_FROM_INPUTS(values, 7)  # peephole weight, 这里先留接口，暂不使用

    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    bidirectional = direction == 'bidirectional'
    num_directions = 2 if bidirectional else 1
    batch_first = layout == 1
    has_bias = b is not None

    if batch_first:
        x = x.transpose(0, 1)  # -> [seq, batch, input]

    seq_length, batch_size, input_size = x.shape

    # --- 初始化 h0/c0 ---
    if initial_h is None:
        initial_h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device, dtype=x.dtype)
    if initial_c is None:
        initial_c = torch.zeros_like(initial_h)

    # --- 拆 bias：LSTM 有 8 个 bias 向量 (Wb_i,f,g,o + Rb_i,f,g,o) ---
    def prepare_lstm_biases(bias, hidden_size, direction_idx=0):
        if bias is None:
            return (None,) * 8
        if bias.dim() == 2:
            b_dir = bias[direction_idx]  # [8*hidden]
        else:
            b_dir = bias
        splits = torch.split(b_dir, hidden_size, dim=0)  # 8 份
        return splits  # (Wb_i, Wb_f, Wb_g, Wb_o, Rb_i, Rb_f, Rb_g, Rb_o)

    # --- 单方向计算 ---
    def process_direction(x, w, r, b, h0, c0, direction_idx=0, reverse=False):
        # 取该方向权重
        if w.dim() == 3:
            w_dir = w[direction_idx]  # [4*hidden, input]
            r_dir = r[direction_idx]  # [4*hidden, hidden]
        else:
            w_dir, r_dir = w, r

        # - W/R/B parameter weight matrix for input, output, forget, and cell gates
        # 处理 bias
        if has_bias:
            Wb_i, Wb_o, Wb_f, Wb_g, Rb_i, Rb_o, Rb_f, Rb_g = prepare_lstm_biases(b, hidden_size, direction_idx)
            # 合并成 gate_bias = Wb + Rb
            i_bias = Wb_i + Rb_i
            f_bias = Wb_f + Rb_f
            g_bias = Wb_g + Rb_g
            o_bias = Wb_o + Rb_o
        else:
            i_bias = f_bias = g_bias = o_bias = None

        h_t = h0[direction_idx] if h0.dim() == 3 else h0
        c_t = c0[direction_idx] if c0.dim() == 3 else c0

        if reverse:
            x = x.flip(0)

        outputs = []
        for t in range(seq_length):
            xt = x[t]  # [batch, input]

            # 一次 matmul 得到 4 个门
            xw = torch.matmul(xt, w_dir.t())  # [batch, 4*hidden]
            x_i, x_o, x_f, x_g = torch.split(xw, hidden_size, dim=1)

            hr = torch.matmul(h_t, r_dir.t())  # [batch, 4*hidden]
            h_i, h_o, h_f, h_g = torch.split(hr, hidden_size, dim=1)

            # 加 bias
            if has_bias:
                x_i, x_f, x_g, x_o = x_i + i_bias, x_f + f_bias, x_g + g_bias, x_o + o_bias

            i_t = torch.sigmoid(x_i + h_i)
            f_t = torch.sigmoid(x_f + h_f)
            g_t = torch.tanh(x_g + h_g)
            o_t = torch.sigmoid(x_o + h_o)

            # 更新 cell
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            # 对输出 h_t 再做一次量化（与 GRU 侧保持一致）
            outputs.append(h_t)

        outputs = torch.stack(outputs)  # [seq, batch, hidden]
        if reverse:
            outputs = outputs.flip(0)
        return outputs, h_t, c_t

    # --- 前向 ---
    fo, fh, fc = process_direction(x, w, r, b, initial_h, initial_c, 0, False)
    if bidirectional:
        ro, rh, rc = process_direction(x, w, r, b, initial_h, initial_c, 1, True)
        outputs = torch.cat([fo.unsqueeze(1), ro.unsqueeze(1)], dim=1)  # [seq, 2, batch, hidden]
        last_h = torch.stack([fh, rh])  # [2, batch, hidden]
        last_c = torch.stack([fc, rc])
    else:
        outputs = fo.unsqueeze(1)  # [seq, 1, batch, hidden]
        last_h = fh.unsqueeze(0)
        last_c = fc.unsqueeze(0)

    return outputs, last_h, last_c


def LSTM_quant_forward(
    op: Operation,
    values: List[torch.Tensor],
    ctx=None,
    **kwargs,
) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=8)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    x, w, r = values[:3]  # x: [seq, batch, input]  (layout=0)
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    initial_c = GET_VALUE_FROM_INPUTS(values, 6)
    p = GET_VALUE_FROM_INPUTS(values, 7)  # peephole weight, 这里先留接口，暂不使用

    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    bidirectional = direction == 'bidirectional'
    num_directions = 2 if bidirectional else 1
    batch_first = layout == 1
    has_bias = b is not None

    if batch_first:
        x = x.transpose(0, 1)  # -> [seq, batch, input]

    seq_length, batch_size, input_size = x.shape

    # --- 初始化 h0/c0 ---
    if initial_h is None:
        initial_h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device, dtype=x.dtype)
    if initial_c is None:
        initial_c = torch.zeros_like(initial_h)

    # --- 拆 bias：LSTM 有 8 个 bias 向量 (Wb_i,f,g,o + Rb_i,f,g,o) ---
    def prepare_lstm_biases(bias, hidden_size, direction_idx=0):
        if bias is None:
            return (None,) * 8
        if bias.dim() == 2:
            b_dir = bias[direction_idx]  # [8*hidden]
        else:
            b_dir = bias
        splits = torch.split(b_dir, hidden_size, dim=0)  # 8 份
        return splits  # (Wb_i, Wb_f, Wb_g, Wb_o, Rb_i, Rb_f, Rb_g, Rb_o)

    # --- 单方向计算 ---
    def process_direction(
        x, w, r, b, h0, c0, direction_idx=0, reverse=False, quant_config: TensorQuantizationConfig = None
    ):
        # 取该方向权重
        if w.dim() == 3:
            w_dir = w[direction_idx]  # [4*hidden, input]
            r_dir = r[direction_idx]  # [4*hidden, hidden]
        else:
            w_dir, r_dir = w, r

        rounding = quant_config.rounding if quant_config else RoundingPolicy.ROUND_HALF_EVEN

        # 处理 bias
        if has_bias:
            Wb_i, Wb_o, Wb_f, Wb_g, Rb_i, Rb_o, Rb_f, Rb_g = prepare_lstm_biases(b, hidden_size, direction_idx)
            # 合并成 gate_bias = Wb + Rb
            i_bias = Wb_i + Rb_i
            f_bias = Wb_f + Rb_f
            g_bias = Wb_g + Rb_g
            o_bias = Wb_o + Rb_o
        else:
            i_bias = f_bias = g_bias = o_bias = None

        h_t = h0[direction_idx] if h0.dim() == 3 else h0
        c_t = c0[direction_idx] if c0.dim() == 3 else c0

        if reverse:
            x = x.flip(0)

        outputs = []
        for t in range(seq_length):
            xt = x[t]  # [batch, input]

            # 一次 matmul 得到 4 个门
            xw = torch.matmul(xt, w_dir.t())  # [batch, 4*hidden]
            xw = fake_quantize(xw, LSTM_QUANT_EXPONENT, LSTM_QUANT_BITS, rounding=rounding)
            x_i, x_o, x_f, x_g = torch.split(xw, hidden_size, dim=1)

            hr = torch.matmul(h_t, r_dir.t())  # [batch, 4*hidden]
            hr = fake_quantize(hr, LSTM_QUANT_EXPONENT, LSTM_QUANT_BITS, rounding=rounding)
            h_i, h_o, h_f, h_g = torch.split(hr, hidden_size, dim=1)

            # 加 bias
            if has_bias:
                x_i, x_f, x_g, x_o = x_i + i_bias, x_f + f_bias, x_g + g_bias, x_o + o_bias

            i_t = torch.sigmoid(x_i + h_i)
            f_t = torch.sigmoid(x_f + h_f)
            g_t = torch.tanh(x_g + h_g)
            o_t = torch.sigmoid(x_o + h_o)

            # 更新 cell
            c_t = f_t * c_t + i_t * g_t
            c_t = fake_quantize(c_t, LSTM_QUANT_EXPONENT, LSTM_QUANT_BITS, rounding=rounding)  # 很多框架把 c_t 也量化
            h_t = o_t * torch.tanh(c_t)

            # 对输出 h_t 再做一次量化（与 GRU 侧保持一致）
            h_t = PPQuantFunction(h_t, quant_config)

            outputs.append(h_t)

        outputs = torch.stack(outputs)  # [seq, batch, hidden]
        if reverse:
            outputs = outputs.flip(0)
        return outputs, h_t, c_t

    # --- 取输出量化配置 ---
    if len(op.config.output_quantization_config) > 0:
        output_config = op.config.output_quantization_config[0]
    else:
        raise TypeError('LSTM_quant_forward expects a TensorQuantizationConfig instance.')

    # --- 前向 ---
    fo, fh, fc = process_direction(x, w, r, b, initial_h, initial_c, 0, False, output_config)
    if bidirectional:
        ro, rh, rc = process_direction(x, w, r, b, initial_h, initial_c, 1, True, output_config)
        outputs = torch.cat([fo.unsqueeze(1), ro.unsqueeze(1)], dim=1)  # [seq, 2, batch, hidden]
        last_h = torch.stack([fh, rh])  # [2, batch, hidden]
        last_c = torch.stack([fc, rc])
    else:
        outputs = fo.unsqueeze(1)  # [seq, 1, batch, hidden]
        last_h = fh.unsqueeze(0)
        last_c = fc.unsqueeze(0)

    return outputs, last_h, last_c


ESPDL_QUANT_BACKEND_TABLE['GRU'] = GRU_quant_forward
ESPDL_QUANT_BACKEND_TABLE['LSTM'] = LSTM_quant_forward

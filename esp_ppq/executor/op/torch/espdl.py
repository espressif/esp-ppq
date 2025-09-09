from .default import DEFAULT_BACKEND_TABLE

ESPDL_QUANT_BACKEND_TABLE = DEFAULT_BACKEND_TABLE.copy()

from .base import *


def GRU_float_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.
    只支持 pytorch 导出来的 GRU 啊亲; 必须要 6 个输入 Variable
    onnx: GRU - 22
    """
     
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    # first 3 are mandatory input
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)

    # sequence length will be dropped without warrning.
    # if seq_len is not None: raise NotImplementedError('PPQ do not support LSTM with explicite length.')

    # check attributes
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    linear_before_reset = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='linear_before_reset', default=0)
    if linear_before_reset == 0:
        #torch.GRU only support linear_before_reset != 0
        raise NotImplementedError('PPQ do not support GRU with linear_before_reset == 1.')
    if activation_alpha is not None:
        raise NotImplementedError('PPQ do not support GRU with cutimized activation.')
    if activation_beta is not None:
        raise NotImplementedError('PPQ do not support GRU with cutimized activation.')
    if activations is not None:
        raise NotImplementedError('PPQ do not support GRU with cutimized activation.')
    if direction == 'bidirectional':
        raise NotImplementedError('PPQ do not support bidirectional GRU.')

    # flag
    has_bias = b is not None
    batch_first = layout == 1
    
    if batch_first:
        _, seq_len, _ = x.size()
    else:
        seq_len, _, _ = x.size()

    initial_h_prev = initial_h.reshape([-1, hidden_size])  # [num_directions * num_layers, hidden_size] -> [num_directions * num_layers, hidden_size]
    
    output_seq = []
    for t in range(seq_len):
        # 取当前时间步输入
        x_t = x[t] if not batch_first else x[:, t, :]          # [batch, input_size]

        # 当前 hidden 状态：第一次用 initial，后面用上一步的输出
        h_prev = output_seq[-1] if output_seq else initial_h_prev # [batch, hidden_size]

        # --- GRU cell 计算 ---
        gi = torch.mm(x_t, w[0].t())            # [batch, 3*hidden]
        if has_bias:
            gi += b[0, :3*hidden_size]

        gh = torch.mm(h_prev, r[0].t())         # 这里 h_prev 已经是 2-D
        if has_bias:
            gh += b[0, 3*hidden_size:]

        i_i, i_r, i_n = gi.chunk(3, 1)
        h_i, h_r, h_n = gh.chunk(3, 1)

        inputgate = torch.sigmoid(i_i + h_i)
        resetgate = torch.sigmoid(i_r + h_r)
        newgate   = torch.tanh(i_n + resetgate * h_n)
        h_next    = newgate + inputgate * (h_prev - newgate)

        output_seq.append(h_next)
    layer_output = torch.stack(output_seq).unsqueeze(1)

    # if batch_first:
    #     hidden = hidden.unsqueeze(0)
    # else:
    #     hidden = hidden.unsqueeze(1)

    return layer_output, layer_output[-1] 



def GRU_quant_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.
    只支持 pytorch 导出来的 GRU 啊亲; 必须要 6 个输入 Variable
    onnx: GRU - 22
    """
     
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    # first 3 are mandatory input
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)

    # sequence length will be dropped without warrning.
    # if seq_len is not None: raise NotImplementedError('PPQ do not support LSTM with explicite length.')

    # check attributes
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    linear_before_reset = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='linear_before_reset', default=0)
    if linear_before_reset == 0:
        #torch.GRU only support linear_before_reset != 0
        raise NotImplementedError('PPQ do not support GRU with linear_before_reset == 1.')
    if activation_alpha is not None:
        raise NotImplementedError('PPQ do not support GRU with cutimized activation.')
    if activation_beta is not None:
        raise NotImplementedError('PPQ do not support GRU with cutimized activation.')
    if activations is not None:
        raise NotImplementedError('PPQ do not support GRU with cutimized activation.')
    if direction == 'bidirectional':
        raise NotImplementedError('PPQ do not support bidirectional GRU.')


    # flag
    has_bias = b is not None
    batch_first = layout == 1
    
    if batch_first:
        batch_size, seq_len, _ = x.size()
    else:
        seq_len, batch_size, _ = x.size()

    if batch_first:
        hidden = torch.zeros(size=[1, batch_size, hidden_size], device=x.device, dtype=torch.float32)
    else:
        hidden = torch.zeros(size=[1, batch_size, hidden_size], device=x.device, dtype=torch.float32)
    
    if initial_h is not None:
        hidden[:] = initial_h[:]
    
    output_seq = []
    for t in range(seq_len):
        # 获取当前时间步的输入
        x_t = x[t] if not batch_first else x[:, t, :]# (batch, input_size)
        
        # 使用GRU cell计算
        gi = torch.mm(x_t, w[0].t())
        if has_bias:
            gi += b[0, :3*hidden_size]

        gh = torch.mm(hidden, r[0].t())
        if has_bias:
            gh += b[0, 3*hidden_size:]
        
        gi = torch.fake_quantize_per_tensor_affine(gi, scale=1.0/(2**8), zero_point=0, quant_min=-128, quant_max=127)
        gh = torch.fake_quantize_per_tensor_affine(gh, scale=1.0/(2**8), zero_point=0, quant_min=-128, quant_max=127)
        i_i, i_r, i_n = gi.chunk(3, 1)
        h_i, h_r, h_n = gh.chunk(3, 1)
        
        inputgate = torch.sigmoid(i_i + h_i)
        resetgate = torch.sigmoid(i_r + h_r)
        newgate =torch.tanh(i_n + resetgate * h_n)
        hidden = newgate + inputgate * (hidden - newgate)
        
        output_seq.append(hidden)
    layer_output = torch.stack(output_seq)

    if batch_first:
        hidden = hidden.unsqueeze(0)
    else:
        hidden = hidden.unsqueeze(1)

    return layer_output.unsqueeze(1), hidden 



# ESPDL_QUANT_BACKEND_TABLE['GRU'] = GRU_float_forward
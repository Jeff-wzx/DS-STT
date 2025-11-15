import torch
import torch.nn as nn

from multi_attention_forward import *
from utilies import  _get_activation_fn, conv_init, bn_init
from math import sqrt

class STLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.s_attn = STAttention(nhead, d_model, mode = 'spatial', dropout=dropout)
        
        kernel_size = 7
        padding = ((kernel_size - 1) // 2, 0)

        self.conv = nn.Conv2d(d_model, d_model, (kernel_size, 1), (1, 1), padding,)
        self.bn = nn.BatchNorm2d(d_model)

        self.tcn = nn.Sequential( self.conv, self.bn)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, query, att_mask = None):

        src2, attn1 = self.s_attn(query, query, query, att_mask=att_mask)
        
        src2 = src2.permute(0,3,1,2)
        src2 = self.tcn(src2)
        src2 = src2.permute(0,2,3,1)
        
        src = self.norm1(query + self.dropout1(src2))

        return src
    
# 该部分主要实现空时注意力。
# nhead多头注意力的头数量，d_model模型维度；mode模式用以指定是时间还是空间，注意类型？
class STAttention(nn.Module):
    def __init__(self, nhead, d_model, mode = 'spatial', dropout = 0.1, attn_type = 'norm'):
        super().__init__()
        # 2空间 1时间
        assert mode in ['spatial', 'temporal']
        assert attn_type in ['prob', 'norm']
        self.mode = mode
        self.attn_type = attn_type
        self.d_model = d_model
        self.attn = MultiHeadedAttention(nhead, d_model, dropout=dropout)

    def forward(self, query, key, value, key_padding_mask = None, att_mask = None):
        batch_size, seq_len, obj_len, d_model = query.shape
        attn_dim = 2 if self.mode == 'spatial' else 1
        target_len = query.shape[attn_dim]
        source_len = key.shape[attn_dim]
        # 获取注意力机制当中的Q、K、V三个向量
        q = query if attn_dim == 2 else query.permute(0, 2, 1, 3)
        k = key if attn_dim == 2 else key.permute(0, 2, 1, 3)
        v = value if attn_dim == 2 else value.permute(0, 2, 1, 3)
        # 重塑改变三者的格式形状
        q = q.reshape(-1, target_len, self.d_model)
        k = k.reshape(-1, source_len, self.d_model)
        v = v.reshape(-1, source_len, self.d_model)
        
        if self.attn_type == 'norm':
            output, attn = self.attn(
                q, k, v, key_padding_mask=key_padding_mask, att_mask=att_mask)
        else:
            # print(q.shape,k.shape,v.shape)
            output, attn = self.attn(q, k, v, attn_mask=att_mask)

        if attn_dim == 1:
            output = output.reshape(batch_size, obj_len, seq_len, d_model).transpose(1, 2)
        else:
            output = output.reshape(batch_size, seq_len, obj_len, d_model)

        return output, attn



# 转换函数，线性层-->relu激活函数--->线性层，在每个时间步
class TransitionFunction(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """
    def __init__(self, input_depth, filter_size, output_depth, 
                    layer_config='ll', padding='left', dropout=0.0):

        super(TransitionFunction, self).__init__()
        # 层的创建，输入-->中间隐藏-->输出（全部都是32）[(32,32),(32,32)]
        layers = []
        sizes = ([(input_depth, filter_size)] + 
                 [(filter_size, filter_size)]*(len(layer_config)-2) + 
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))
        # 在前面搭建的层上使用激活函数过后应用在dropout上
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # batch_size, timesteps, input_depth（vocal）, channel
    # 然后维度变换成为batch_size, input_depth, timesteps, channel ,
    # 再变成维度batch_size*input_depth, timesteps, channel ，将变换后的输入储存到x当中
    def forward(self, inputs):
        b,t,v,c = inputs.shape
        inputs = inputs.permute(0,2,1,3).reshape(-1,t,c)
        
        x = inputs
        # 把input输入进模型层里面，不是最后一层的时候会经过relu和dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        # 恢复为原来一开始的形状。
        x = x.reshape(b,v,t,c).permute(0,2,1,3)

        return x
# 主要是对输入执行填充和卷积操作
class Conv(nn.Module):
    """
    卷积，对输入数据进行填充和卷积。
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides（左右两侧）
        """
        super(Conv, self).__init__()
        # 零填充张量、卷积
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size//2, (kernel_size - 1)//2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs

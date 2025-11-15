from utilies import _get_clones, _get_activation_fn
import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
import torch.nn.functional as F

from sttf_layer import STAttention,TransitionFunction
# 表示transformer模型的每个单层layer；feedforward_ratio：隐藏大小与 输入维度之间的比率
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 feedforward_ratio=1, dropout=0.1, module='encoder',activation="relu"):
        super().__init__()
        
        # Implementation of Feedforward model，这里的ff我认为是隐藏层的维度。
        # 其中的前馈网络部分用两个线性层表示
        # 将输入映射到隐藏维度的线性变换层--->用于正则化的丢弃率--->从隐藏维度映射回原始输入维度的线性变换
        ff = int(d_model*feedforward_ratio)
        self.linear1 = nn.Linear(d_model, ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff, d_model)
        # 归一化，norm1是在自注意力操作之前的归一化-->norm2是自注意力和前馈之后的归一化-->norm3是多注意力和前馈之后的归一化，用在下面译码器部分，在这部分还没有使用
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 看激活函数是relu还是gelu
        self.activation = _get_activation_fn(activation)

        self.module=module
        # 解码器，自注意力、多头注意力、
        # 自注意： self.s_attn：编码器中应用的自注意层。它关注输入序列本身。
        # 多头注意：self.self_attn：解码器中应用的自注意层。它关注解码器的输入序列。 self.mh_attn：解码器中应用的交叉注意层。它关注解码器的输入序列和编码器的输出序列。
        if module == 'decoder': #is
            self.self_attn = STAttention(
                nhead, d_model, mode = 'temporal', dropout=dropout, attn_type = 'norm')
            self.mh_attn = STAttention(
                nhead, d_model, mode = 'temporal', dropout=dropout, attn_type = 'norm')
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.positionwise_feed_forward = TransitionFunction(
                d_model, ff, d_model,
                layer_config='cc', padding='left', dropout=dropout)
        # 编码器
        else:

            self.positionwise_feed_forward = TransitionFunction(
                d_model, ff, d_model,
                layer_config='cc', padding = 'both', dropout=dropout)            
            self.s_attn = STAttention(nhead, d_model, mode = 'temporal', dropout=dropout)

    def forward(self, query, key, att_mask=None, key_padding_mask=None):
        # 编码器,与模型图相对应。
        if self.module == 'encoder': #is

            src2, attn1 = self.s_attn(query, query, query)            
            src = self.norm1(query + self.dropout1(src2))
            src2 = self.positionwise_feed_forward(src)
            src = self.norm2(src + self.dropout2(src2))
        # 译码器，与模型图相对应。就是最基本的模型部分，对应于论文的最下面。
        else:

            src2, attn1 = self.self_attn(
                query, query, query, att_mask=att_mask,key_padding_mask=key_padding_mask)
            query = query + self.dropout1(src2)
            src = self.norm1(query) 
            src3, attn2 = self.mh_attn(src, key, key)
            src = src + self.dropout3(src3)
            src = self.norm3(src)
            src2 = self.positionwise_feed_forward(src)

            src = self.norm2(src + self.dropout2(src2))

        return src, None, None
            
# transformer中的一些个模块。
class TransformerModel(nn.Module):
    def __init__(self, layer, num_layers, module, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.module = module

    def forward(self, query, key, att_mask=None, key_padding_mask=None):
        
        output = query

        atts1 = []
        atts2 = []
        # 遍历每一层，
        for i in range(self.num_layers):
            if self.module == 'encoder': #is
                key = output
            
            output, attn1, attn2 = self.layers[i](
                output, key, att_mask=att_mask, key_padding_mask=key_padding_mask)

            atts1.append(attn1)
            atts2.append(attn2)
        
        if self.norm:
            output = self.norm(output)

        return output, atts1, atts2
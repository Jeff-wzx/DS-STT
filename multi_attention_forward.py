import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

# 该模块用于使用正弦和余弦函数执行时间嵌入。时间嵌入通常用于序列建模任务，以将时态信息合并到模型中。
# 这个是在计算位置编码的时候一并使用的。
# 生成的张量包含计算的时间嵌入因子。每个因子表示输入张量中相应时间索引的权重或尺度。
# 然后使用这些因素逐个单元划分输入张量，根据时间维度有效地缩放输入。
# d_model为32。
# 将时间维度dim_t除以这些缩放因子，可以引入不同时间尺度的变化，以增强位置编码的表达能力。
# 将输入张量中的时间维度进行嵌入。它使用正弦函数和余弦函数对时间维度进行变换，以引入时间信息和周期性。
# 该模块可用于将时间特征引入到模型中，并提供时间的连续表示。
class TimeEmbeddingSine(nn.Module):
    def __init__(self,
                 d_model = 64,
                 temperature = 10000,
                 scale = None,
                 requires_grad = False):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        # scale=none则默认为2pi
        self.scale = 2 * math.pi if scale is None else scale
        # 一个布尔标志，指示时间嵌入参数是否需要梯度计算
        self.requires_grad = requires_grad
    # 克隆输入，避免后面会对原始张量进行修改、初始化时间嵌入维度
    def forward(self, inputs):
        x = inputs.clone()
        d_embed = self.d_model
        # 创建表示间时维度索引的张量，可以为每个维度生成一个不同的缩放因子，用于将输入除以该缩放因
        dim_t = torch.arange(d_embed, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_embed)
        x = x / dim_t
        #  将x拆分为奇数索引和偶数索引的部分，并分别对它们应用正弦函数和余弦函数。
        #  然后，将两部分重新堆叠在一起，并通过 flatten 操作压缩最后两个维度。
        x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
        return x if self.requires_grad else x.detach()
# 注意力
def attention(query, key, value, att_mask=None, dropout=None,key_padding_mask=None):
    "Compute 'Scaled Dot Product Attention'"
    # 首先取query的最后一维的大小，对应嵌入维度
    d_k = query.size(-1)
    # 按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 判断是否使用掩码张量
    if att_mask is not None:
        # print(scores.shape,att_mask.shape)
        # 使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较，
        # 如果使用掩码张量，则将对应的scores张量用-1e9这个置来替换
        scores = scores.masked_fill(att_mask == 0, -1e9)
    if key_padding_mask is not None:
        # print(key_padding_mask.shape)
        scores = scores.masked_fill(key_padding_mask == 0, -1e9)
    # softmax
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 相乘并返回注意力分数
    return torch.matmul(p_attn, value), p_attn
#多头注意力层
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h头数，d_model维度，丢弃置零的比率0.1
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # 判断h是否能被d_model整除，这是因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head个
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # 得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // h
        self.h = h
        # 4个矩阵：K Q V还需要一个整合最后进行拼接
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, att_mask=None,key_padding_mask=None):
        "Implements Figure 2"
        if att_mask is not None:
            # Same mask applied to all h heads. unsqueeze扩展维度，代表多头中的第几头
            att_mask = att_mask.unsqueeze(1)
        
        if key_padding_mask is not None:           
            b, t, v, c = key_padding_mask.shape 
            key_padding_mask = key_padding_mask.permute(0,2,1,3).reshape(b*v,t,c)
            key_padding_mask = key_padding_mask.unsqueeze(1)

        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(
            query, key, value, att_mask=att_mask, dropout=self.dropout,key_padding_mask=key_padding_mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), self.attn
# 输出
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)
# 在多头注意力中有很多相似的东西，直接调用克隆函数方便简洁。
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 残差连接：有助于将信息从一层传播到下一层，同时缓解梯度消失问题并改善训练期间梯度的流动
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# 此函数生成一个掩码，用于mask注意力机制中的后续位置。用于防止注意力关注未来位置。
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# 构建前馈全连接网络
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 将输入进行映射到神经网络模型中的相应嵌入。vocab(vocabulary词表大小，unique token 的数量,)，d_model嵌入向量维数
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        # 调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut
        # 这里使用的是一个线形层，实际上就是一个线性变换，y=wx+b(v输入特征数，d输出特征数)(w是要学习的参数，b是偏置)
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # return self.lut(x)
        # 将处理后的输出嵌入乘以的平方根。这种缩放通常用于transformer的模型中，以减轻训练期间大嵌入维度对梯度的影响。
        return self.lut(x) * math.sqrt(self.d_model)

# 为模型提供当前时间步的前后出现顺序的信息
# 使用不同频率的sin和cos函数来进行位置编码，奇数位和偶数位分别使用sin和cos.
# 任意两个相距k个时间步的位置编码向量的内积都是相同的，这就相当于蕴含了两个时间步之间相对位置关系的信息。
# 而每个时间步的位置编码又是唯一的
# 该模块用于向输入张量添加位置编码，以便将位置信息合并到模型中。
# 使用super的方式指明继承nn.module的初始化函数
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # 维度、丢弃率（用于在训练期间将输入的某些元素随机归零，以防止过拟合）、最大长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # 初始化一个位置编码矩阵，大小是max_len * d_model，计算位置编码并将其存储在张量中。
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，大小为max_len*1，表示每个位置的索引
        # 在张量（0，max_len-1）上在维度 1 处添加新维度。这有效地将张量重塑为max_len*1，添加的维度表示序列中每个位置的位置索引。
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算除法因子（16，1）（由0到32，每次的步长选择为2）
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 从0开始选择步长为2的元素：偶数，即0，2，4，6....
        pe[:, 1::2] = torch.cos(position * div_term) # 从1开始选择步长为2的元素：奇数，即1，3，5，7....
        pe = pe.unsqueeze(0) # 沿0维度缩张量以添加批量维度，使其成形pe(1, max_len, d_model)
        # 将计算出的编码位置输入进缓冲区内，这允许将位置编码移动到与模型的参数和状态相同的设备，但它们不会在训练期间更新
        self.register_buffer('pe', pe)
    # 将输入张量和位置编码进行对应结合。
    # 将位置编码添加到输入张量中。位置编码被切片以匹配输入张量的长度并逐个元素添加
    # x代表的是Embedding层的输出
    # x_size(0)表示的是行的数量大小，x_size(1)表示的是列的数量大小。确保位置编码与输入序列的长度匹配
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 自注意机制中的概率掩码,创建布尔掩码(ture,false)
# B, H, L, index, scores分别表示批大小（batch），注意力头的数量（head），序列长度(length)，当前位置的索引(index)，表示注意力分数(score)
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
    # 用于生成一个掩码，该掩码自注意计算期间屏蔽每个头部和批次的注意力机制中的后续位置。
    # 然后将应用于注意力分数，以防止关注未来的位置。
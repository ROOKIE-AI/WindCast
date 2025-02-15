import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    """
    标准的全注意力机制实现
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        初始化函数
        mask_flag: 是否使用掩码
        factor: 缩放因子
        scale: 缩放值
        attention_dropout: 注意力dropout率
        output_attention: 是否输出注意力权重
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数
        queries: 查询张量 [B, L, H, E]
        keys: 键张量 [B, S, H, D]
        values: 值张量 [B, S, H, D]
        attn_mask: 注意力掩码
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        # 计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 应用softmax和dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 计算输出值
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    """
    概率注意力机制实现，用于减少计算复杂度
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        初始化函数
        mask_flag: 是否使用掩码
        factor: 采样因子
        scale: 缩放值
        attention_dropout: 注意力dropout率
        output_attention: 是否输出注意力权重
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        计算稀疏注意力的核心函数
        Q: 查询矩阵
        K: 键矩阵
        sample_k: 采样的键的数量
        n_top: 选择的顶部查询数量
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 采样计算Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 找到最重要的查询
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 使用筛选后的查询计算注意力分数
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        获取初始上下文向量
        V: 值矩阵
        L_Q: 查询序列长度
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        更新上下文向量
        context_in: 输入上下文
        V: 值矩阵
        scores: 注意力分数
        index: 选中的索引
        L_Q: 查询序列长度
        attn_mask: 注意力掩码
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数
        queries: 查询张量
        keys: 键张量
        values: 值张量
        attn_mask: 注意力掩码
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # 转置维度顺序
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # 计算采样参数
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        # 计算注意力分数
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # 应用缩放因子
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
            
        # 获取并更新上下文
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    """
    注意力层的封装，包含多头注意力机制
    """
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        """
        初始化函数
        attention: 注意力机制类型
        d_model: 模型维度
        n_heads: 注意力头数
        d_keys: 键的维度
        d_values: 值的维度
        mix: 是否混合注意力头
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数
        queries: 查询张量
        keys: 键张量
        values: 值张量
        attn_mask: 注意力掩码
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 投影并重塑维度
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 应用注意力机制
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

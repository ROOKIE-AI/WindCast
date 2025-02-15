import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """
    解码器层
    
    包含自注意力机制、交叉注意力机制和前馈神经网络
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        初始化函数
        
        Args:
            self_attention: 自注意力层
            cross_attention: 交叉注意力层
            d_model: 模型维度
            d_ff: 前馈网络维度,默认为4*d_model
            dropout: Dropout比率
            activation: 激活函数类型
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention  # 自注意力层
        self.cross_attention = cross_attention  # 交叉注意力层
        # 两个一维卷积构成前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 三个层归一化
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后的归一化
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后的归一化
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络后的归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu  # 选择激活函数

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量
            cross: 编码器输出的交叉注意力张量
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            
        Returns:
            输出张量
        """
        # 自注意力子层
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        # 交叉注意力子层
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        # 前馈网络子层
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)  # 残差连接和层归一化

class Decoder(nn.Module):
    """
    解码器
    
    由多个解码器层组成,用于生成预测序列
    """
    def __init__(self, layers, norm_layer=None):
        """
        初始化函数
        
        Args:
            layers: 解码器层列表
            norm_layer: 归一化层(可选)
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)  # 解码器层列表
        self.norm = norm_layer  # 最终的归一化层

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量
            cross: 编码器输出的交叉注意力张量
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            
        Returns:
            解码器输出张量
        """
        # 依次通过所有解码器层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)  # 最终的层归一化

        return x
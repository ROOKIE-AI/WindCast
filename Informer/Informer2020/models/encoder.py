import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    """
    下采样卷积层
    
    使用一维卷积、批归一化、激活函数和最大池化进行下采样
    """
    def __init__(self, c_in):
        """
        Args:
            c_in: 输入通道数
        """
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        # 一维卷积,保持通道数不变
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in, 
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)  # 批归一化
        self.activation = nn.ELU()  # ELU激活函数
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化,步长为2进行下采样

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量 [Batch, Length, Channel]
            
        Returns:
            x: 下采样后的张量 [Batch, Length//2, Channel]
        """
        x = self.downConv(x.permute(0, 2, 1))  # 调整维度顺序以适应卷积
        x = self.norm(x)  # 批归一化
        x = self.activation(x)  # 激活函数
        x = self.maxPool(x)  # 最大池化下采样
        x = x.transpose(1,2)  # 恢复维度顺序
        return x

class EncoderLayer(nn.Module):
    """编码器层
    
    包含多头自注意力机制和前馈神经网络
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        Args:
            attention: 注意力层
            d_model: 模型维度
            d_ff: 前馈网络维度,默认为4*d_model
            dropout: Dropout比率
            activation: 激活函数类型
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        # 两个一维卷积构成前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu  # 选择激活函数

    def forward(self, x, attn_mask=None):
        """前向传播
        
        Args:
            x: 输入张量 [Batch, Length, Dimension]
            attn_mask: 注意力掩码
            
        Returns:
            x: 输出张量 [Batch, Length, Dimension]
            attn: 注意力权重
        """
        # 自注意力层
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)  # 残差连接

        y = x = self.norm1(x)  # 层归一化
        # 前馈网络
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn  # 残差连接和层归一化

class Encoder(nn.Module):
    """
    编码器
    
    由多个编码器层和可选的卷积层组成
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        Args:
            attn_layers: 注意力层列表
            conv_layers: 卷积层列表(可选)
            norm_layer: 归一化层(可选)
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """前向传播
        
        Args:
            x: 输入张量 [Batch, Length, Dimension]
            attn_mask: 注意力掩码
            
        Returns:
            x: 输出张量
            attns: 各层的注意力权重列表
        """
        attns = []
        if self.conv_layers is not None:
            # 交替使用注意力层和卷积层
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            # 只使用注意力层
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)  # 最终的层归一化

        return x, attns

class EncoderStack(nn.Module):
    """
    编码器堆叠
    
    将多个编码器堆叠在一起,每个编码器处理不同长度的输入
    """
    def __init__(self, encoders, inp_lens):
        """
        Args:
            encoders: 编码器列表
            inp_lens: 每个编码器的输入长度列表
        """
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        """前向传播
        
        Args:
            x: 输入张量 [Batch, Length, Dimension]
            attn_mask: 注意力掩码
            
        Returns:
            x_stack: 堆叠后的输出张量
            attns: 各编码器的注意力权重列表
        """
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)  # 计算每个编码器的输入长度
            x_s, attn = encoder(x[:, -inp_len:, :])  # 编码最后一段序列
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)  # 在长度维度上拼接
        
        return x_stack, attns

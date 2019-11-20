import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """ Linear => GLU => (Light/Dynamic) Conv => Linear """
    def __init__(self, conv_type, n_channels, kernel_size, stride=1, padding=(0, 0),
                 n_heads=1, bias=True, dropconnect=0.):
        super().__init__()
        if conv_type == 'light':
            self.conv = LightConv1d(n_channels, kernel_size, stride, padding, n_heads, bias,
                                    dropconnect)
        elif conv_type == 'dynamic':
            self.conv = DynamicConv1d(n_channels, kernel_size, stride, padding, n_heads, bias,
                                      dropconnect)

        self.linear1 = nn.Linear(n_channels, n_channels*2)
        self.linear2 = nn.Linear(n_channels, n_channels)

    def forward(self, x, mask=None):
        """
        x: [B, T, C] (= [B, T, d_model])
        mask: [B, 1, T]
        """
        x = self.linear1(x)
        x = F.glu(x, dim=-1)
        x = x.transpose(1,2).contiguous() # [B, C, T]
        if mask is not None:
            x.masked_fill_(mask == 0, 0.)
        x = self.conv(x) # [B, C, T]
        x = self.linear2(x.transpose(1,2)) # [B, T, C]
        return x


class LightConv1d(nn.Module):
    """ Lightweight convolution
    Ref: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/lightweight_convolution.py

    Implements weight-sharing of LightConv through reshaping batch dim and channel dim.
    for H = n_heads,
        N = h_dim // H  (# of shared kernel applied):
    input x: [B, T, h_dim] (h_dim % H == 0),
    x_reshaped: [B*N, T, H]
    Then Conv1d(x_reshaped) sharing there kernels.
    """
    def __init__(self, n_channels, kernel_size, stride=1, padding=(0, 0), n_heads=1, bias=True,
                 dropconnect=0.):
        super().__init__()
        assert n_channels % n_heads == 0
        assert isinstance(padding, tuple) or isinstance(padding, list)
        assert len(padding) == 2
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_heads = n_heads
        self.dropconnect = dropconnect

        self.weight = nn.Parameter(torch.empty(n_heads, 1, kernel_size)) # [H, 1, K]
        if bias:
            self.bias = nn.Parameter(torch.empty(n_heads)) # [H]
        else:
            self.bias = None

        # init
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        """
        x: [B, C, T] (= [B, h_dim, L])
        """
        B, C, T = x.shape
        H = self.n_heads

        # softmax normalization in kernel
        weight = F.softmax(self.weight, dim=-1)

        # dropconnect
        weight = F.dropout(weight, p=self.dropconnect, training=self.training)

        # LightConv
        # [B, C, T] => [B*(C/H), H, T] => conv => [B, C, T]
        x = x.view(-1, H, T)
        x = F.pad(x, self.padding)
        x = F.conv1d(x, weight, self.bias, stride=self.stride, padding=0, groups=H)
        return x.view(B, C, T)

    def extra_repr(self):
        s = f"C={self.n_channels}, H={self.n_heads}, kernel_size={self.kernel_size}" \
            f", padding={self.padding}, dropconnect={self.dropconnect}"

        if self.stride > 1:
            s += f", stride={self.stride}"
        if self.bias is None:
            s += ', bias=False'

        return s


class DynamicConv1d(nn.Module):
    """ Dynamic convolution
    Ref: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/dynamic_convolution.py
    for C=h_dim, T=seq_len, H=n_heads, K=kernel_width:
    x: [B, C, T].
    W: [H, K, C].
    (dynamic) kernel: x @ W => [B, T, H, K]. (= Linear(C, H*K))
    That is, each (shared) kernel has different weights in each timestep, for each datapoint.

    Since this dynamic conv method cannot exploit the Conv1d like in LightConv,
    we implement conv operation using matrix multiplication (MM).
    There is two methods for this:
    1) input unfolding
        input 을 conv 연산에 맞도록 window 간에 겹치는 부분을 카피해서
        MM이 가능하도록 해주는 방식.
    2) weight expanding (padding)
        conv 가 MM 으로 계산이 안되는 이유는 local MM 이기 때문이므로,
        나머지 부분을 그냥 0으로 채워서 band matrix 형태로 바꾸어
        MM 으로 연산이 가능하도록 변경.

    unfold 방식이 conventional 이라고 하여 여기서는 unfold 방식으로 구현.
    expand 방식이 short sequence 에서는 더 빠르되, memory inefficient 라서 decoding 시에는
    expand 방식이 더 좋다고 함.
    """
    def __init__(self, n_channels, kernel_size, stride=1, padding=(0, 0), n_heads=1, bias=True,
                 dropconnect=0.):
        super().__init__()
        assert n_channels % n_heads == 0
        assert isinstance(padding, tuple) or isinstance(padding, list)
        assert len(padding) == 2
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_heads = n_heads
        self.dropconnect = dropconnect

        self.weight_gen = nn.Linear(n_channels, n_heads*kernel_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(n_heads)) # [H]
        else:
            self.bias = None

    def forward(self, x):
        """
        x: [B, C, T] (= [B, h_dim, L])
        """
        B, C, T = x.shape
        H = self.n_heads
        R = C // H
        K = self.kernel_size

        # weight generation
        weight = self.weight_gen(x.transpose(1,2)).view(B, T, H, K) # [B, T, H*K]

        # softmax normalization in kernel
        weight = F.softmax(weight, dim=-1)

        # dropconnect
        weight = F.dropout(weight, p=self.dropconnect, training=self.training)

        ## LightConv
        # x: [B, C, T]
        # weight: [B, T, H, K]
        x = F.pad(x, self.padding) # [B, C, T+(K-1)]
        x = x.unfold(-1, K, 1) # [B, C, T, K]
        x = x.view(B, H, R, T, K) # C == H*R

        # final einsum
        # [B, T, H, K] @ [B, H, R, T, K] => [B, H, R, T]
        x = torch.einsum('bthk,bhrtk->bhrt', weight, x)
        x = x + self.bias.view(1, H, 1, 1)
        x = x.reshape(B, C, T)

        return x

    def extra_repr(self):
        s = f"C={self.n_channels}, H={self.n_heads}, kernel_size={self.kernel_size}" \
            f", padding={self.padding}, dropconnect={self.dropconnect}"

        if self.stride > 1:
            s += f", stride={self.stride}"
        if self.bias is None:
            s += ', bias=False'

        return s


if __name__ == "__main__":
    n_channels = 12
    n_heads = 4
    kernel_size = 3
    stride = 1
    padding = (2, 0)
    #conv = LightConv1d(n_channels, kernel_size, stride, padding, n_heads, dropconnect=0.1)
    conv = DynamicConv1d(n_channels, kernel_size, stride, padding, n_heads, dropconnect=0.1)

    # [B, C, T]
    B = 4
    C = n_channels
    T = 3
    x = torch.rand(B, C, T)
    r = conv(x)
    print(x.shape)
    print(r.shape)

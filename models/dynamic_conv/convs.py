import torch
import torch.nn as nn
import torch.nn.functional as F


class LightConv1d(nn.Module):
    """ Lightweight convolution
    Ref: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/lightweight_convolution.py

    Conv1d 모듈을 활용하여 parameter sharing 구현.
    H 개의 heads 가 있다고 하면: depthwise conv1d = [H, K] kernel weights.
    데이터가 K width 라고 가정하자: [B, K, h_dim] (h_dim % H == 0)
    이 때 각 헤드에 들어가는 shared kernel 의 개수는 N = h_dim // H.
    이 때 [B*N, K, H] 로 변환하여 depthwise conv1d 를 태우면 batch dim 에서는 kernel 이 공유되므로
    자연스럽게 parameter sharing 을 구현할 수 있다.
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
            # ref code 에서는 bias 는 sharing 하지 않고 C 개만큼 생성하여 사용하나,
            # 여기서는 그냥 weight 와 함께 sharing 하여 H 개만 생성.
            # 어차피 코드를 보니 안 쓰는 듯.
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
    x: [B, C, T].
    W: [H, K, C].
    kernel: x @ W => [B, T, H, K]. (= Linear(C, H*K))
    즉, data point 별로 time-dependent 한 kernel 을 구할 수 있음.

    그러면 weight sharing 이 없기 때문에 기존의 conv1d 를 사용할 수가 없으므로,
    conv 연산을 직접 구현한다.
    unfold 방식이 conventional 이라고 함.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from const import *


#  class SkipConnection(nn.Module):
#      """ Skip-connection with dropout & layer norm """
#      def __init__(self, d_model, dropout):
#          super().__init__()
#          self.dropout = nn.Dropout(dropout)
#          self.layer_norm = nn.LayerNorm(d_model)

#      def forward(self, x, skip_con):
#          x = self.dropout(x)
#          return self.layer_norm(skip_con + x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_qk = self.d_v = d_model // n_heads
        self.n_heads = n_heads
        """ Multi-head projection """
        self.q_proj = nn.Linear(d_model, n_heads * self.d_qk)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_qk)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_v)
        self.out = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1. / (self.d_qk ** 0.5)

    def forward(self, q, k, v, mask=None):
        """
        q: [B, Q, d_model]
        k: [B, N, d_model]
        v: [B, N, d_model]
        mask: [B, Q, N]
        N = src_len
        """
        B, Q = q.shape[0:2]
        N = k.shape[1]
        q_proj = self.q_proj(q).view(B, Q, self.n_heads, self.d_qk) # [B, Q, H, d_qk]
        k_proj = self.k_proj(k).view(B, N, self.n_heads, self.d_qk) # [B, N, H, d_qk]
        v_proj = self.v_proj(v).view(B, N, self.n_heads, self.d_v) # [B, N, H, d_v]

        attn_score = torch.einsum('bqhi,bnhi->bhqn', q_proj, k_proj) # [B, H, Q, N]
        attn_score *= self.scale
        if mask is not None:
            mask = mask.unsqueeze(1) # unsqueeze on head dim
            attn_score.masked_fill_(mask == 0, -1e10)
        attn_w = F.softmax(attn_score, dim=-1) # [B, H, Q, N]
        attn_w = self.dropout(attn_w)
        attn_out = torch.einsum('bhqn,bnhi->bqhi', attn_w, v_proj) # [B, Q, H, d_v]
        attn_out = attn_out.reshape(B, Q, self.n_heads * self.d_v) # [B, Q, H*d_v]
        attn_out = self.out(attn_out) # [B, Q, d_model]

        return attn_out, attn_w


class PositionWiseFFN(nn.Module):
    """ Two-layer word-wise FC """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """ [B, Q, d_model] => [B, Q, d_ff] => [B, Q, d_model]
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        # SOS, EOS buffer +2.
        pe = torch.empty(max_len+2, d_model) # [L, D]
        pos = torch.arange(0, max_len+2, dtype=torch.float32).unsqueeze(1) # [L, 1]
        indices = torch.arange(0, d_model, 2, dtype=torch.float32) # [D/2]
        div = 10000. ** (indices / d_model) # [D/2]
        pos_value = pos / div # [L, D/2]
        pe[:, 0::2] = torch.sin(pos_value)
        pe[:, 1::2] = torch.cos(pos_value)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pe = self.pe[:x.size(1)].unsqueeze(0) # [1, x_len, D]
        return self.dropout(x + pe)


class ScaledEmbedding(nn.Module):
    def __init__(self, n_words, d_model, padding_idx=PAD_idx):
        super().__init__()
        self.lut = nn.Embedding(n_words, d_model, padding_idx=padding_idx)
        self.scale = d_model ** 0.5

    def forward(self, x):
        return self.lut(x) * self.scale


# std/var has `unbiased=True` parameter.
# nn.LayerNorm 은 unbiased=False 로 하면 결과가 같게 나옴.
#  class LayerNorm(nn.Module):
#      # nn.LayerNorm 이랑 결과가 조금 달라서 한번 써봄
#      "Construct a layernorm module (See citation for details)."
#      def __init__(self, features, eps=1e-6):
#          super(LayerNorm, self).__init__()
#          self.a_2 = nn.Parameter(torch.ones(features))
#          self.b_2 = nn.Parameter(torch.zeros(features))
#          self.eps = eps

#      def forward(self, x):
#          mean = x.mean(-1, keepdim=True)
#          std = x.std(-1, keepdim=True)
#          return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

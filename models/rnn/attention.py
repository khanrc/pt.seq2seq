import torch
import torch.nn as nn
import torch.nn.functional as F


""" Attentions (h: encoder hidden states, s: decoder hidden states)
h: [enc_len, h_dim] (key/value)
s: [1, h_dim] (query)
- Dot product
    score = s*h
- Multiplicative (Luong)
    score = s*W*h
- Additive (Bahdanau)
    score = v*tanh(W1*h + W2*s)
- KV
    q = Wq*s
    k = Wk*h
    v = Wv*h
    score = sum(q * k)
"""

class KeyValueAttention(nn.Module):
    def __init__(self, q_in_dim, qk_dim, kv_in_dim, v_dim, out_dim):
        """ qkv attention.
        Assume that key source == value source.
        params:
            q_in_dim: query in dim. = decoder hidden dim.
            qk_dim: query & key dim. it should be same.
            kv_in_dim: key value in dim. = encoder hidden dim.
            v_dim: value dim.
            out_dim: final attention out dim.
        """
        super().__init__()
        self.q_proj = nn.Linear(q_in_dim, qk_dim)
        self.k_proj = nn.Linear(kv_in_dim, qk_dim)
        self.v_proj = nn.Linear(kv_in_dim, v_dim)
        self.scale = qk_dim ** 0.5

        self.out = nn.Linear(v_dim, out_dim)

    def forward(self, q, s, mask):
        """
        q: [B, q_len, q_in_dim]
        s: [B, src_len, kv_in_dim]
        mask: [B, 1, src_len]
        """
        q = self.q_proj(q) # [B, q_len, qk_dim]
        k = self.k_proj(s) # [B, src_len, qk_dim]
        v = self.v_proj(s) # [B, src_len, v_dim]

        attn_score = torch.einsum('bqh,bsh->bqs', q, k) # [B, q_len, src_len]
        # scale
        attn_score /= self.scale
        attn_score.masked_fill_(mask == 0, -1e10)
        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = torch.einsum('bqs,bsh->bqh', attn_w, v) # [B, q_len, v_dim]
        attn_out = self.out(attn_out)

        return attn_w, attn_out


class AdditiveAttention(nn.Module):
    """ Bahdanau attention
        score = v*tanh(W1*q + W2*s)
    """
    def __init__(self, q_dim, s_dim, h_dim, out_dim):
        """
        params:
            q_dim: query dim
            s_dim: source dim
            h_dim: attn hidden dim
            out_dim: final out dim
        """
        super().__init__()
        self.q_proj = nn.Linear(q_dim, h_dim)
        self.s_proj = nn.Linear(s_dim, h_dim)
        self.linear = nn.Linear(h_dim, 1)
        self.out = nn.Linear(s_dim, out_dim)

    def forward(self, q, s, mask):
        """
        q: [B, q_len, q_dim]
        s: [B, s_len, s_dim]
        mask: [B, 1, s_len]
        """
        q_proj = self.q_proj(q).unsqueeze(2) # [B, q_len, 1, h_dim]
        s_proj = self.s_proj(s).unsqueeze(1) # [B, 1, s_len, h_dim]
        out = torch.tanh(q_proj + s_proj) # by broadcasting: [B, q_len, s_len, h_dim]
        attn_score = self.linear(out).squeeze(-1) # [B, q_len, s_len]
        attn_score.masked_fill_(mask == 0, -1e10)
        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = attn_w @ s # [B, q_len, s_dim]
        # dimension matching
        attn_out = self.out(attn_out)

        return attn_w, attn_out


class MultiplicativeAttention(nn.Module):
    """ Luong attention
        score = q*W*s
        [q_len, q_dim] * [q_dim, s_dim] * [s_dim, s_len] = [q_len, s_len]
    """
    def __init__(self, q_dim, s_dim, h_dim, out_dim):
        """
        params:
            q_dim: query dim
            s_dim: source dim
            h_dim: attn hidden dim
            out_dim: final out dim
        """
        super().__init__()
        self.w = nn.Parameter(torch.randn(q_dim, s_dim) * 1e-3)
        self.out = nn.Linear(s_dim, out_dim)

    def forward(self, q, s, mask):
        """
        q: [B, q_len, q_dim]
        s: [B, s_len, s_dim]
        mask: [B, 1, s_len]
        """
        attn_score = torch.einsum('bqi,ij,bsj->bqs', q, self.w, s) # [B, q_len, s_len]
        attn_score.masked_fill_(mask == 0, -1e10)
        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = attn_w @ s # [B, q_len, s_len] @ [B, s_len, s_dim] = [B, q_len, s_dim]
        # dim matching
        attn_out = self.out(attn_out)

        return attn_w, attn_out

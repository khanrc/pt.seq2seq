import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ConvS2SAttention(nn.Module):
    def __init__(self, emb_dim, h_dim):
        super().__init__()
        """
        ConvS2S attention 은 residual 을 계산함
        """
        self.q_proj = weight_norm(nn.Linear(h_dim, emb_dim))
        self.out = weight_norm(nn.Linear(emb_dim, h_dim))
        self.scale = 0.5 ** 0.5

    def forward(self, x, emb, key, value, kv_mask):
        """
        x: conv output. [B, h_dim, tgt_len]
        emb: embeddings. [B, tgt_len, emb_dim]
        key: encoder output z. [B, src_len, emb_dim]
        value: encoder combined output (z+e). [B, src_len, emb_dim]
        kv_mask: encoder mask. [B, src_len]
        """
        x = x.transpose(1, 2) # [B, tgt_len, h_dim]
        query = (self.q_proj(x) + emb) * self.scale # [B, tgt_len, emb_dim]
        attn_score = torch.einsum('bte,bse->bts', query, key) # [B, tgt_len, src_len]
        attn_score = attn_score.masked_fill(kv_mask.unsqueeze(1) == 0, float('-inf'))
        attn_w = F.softmax(attn_score, dim=-1)
        # [B, tgt_len, src_len] * [B, src_len, emb_dim] => [B, tgt_len, emb_dim]
        attn_out = torch.bmm(attn_w, value)
        # attention scaling
        attn_scale = kv_mask.sum(dim=1).float().sqrt()
        attn_out = torch.einsum('bte,b->bte', attn_out, attn_scale) # [B, tgt_len, emb_dim]
        # projection: emb => h
        attn_out = self.out(attn_out) # [B, tgt_len, h_dim]
        attn_out = attn_out.transpose(1,2) # [B, h_dim, tgt_len]

        return attn_out, attn_w

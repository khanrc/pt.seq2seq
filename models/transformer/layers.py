import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import ops


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, norm_pos='after'):
        super().__init__()
        # [!] caution: mask for multihead attention should be pad-mask (pad=1, value=0).
        # self.self_attn = nn.MultiHeadAttention(d_model, n_heads, dropout=dropout, need_weights=True)
        self.slf_attn = ops.MultiHeadAttention(d_model, n_heads, dropout)
        self.slf_attn_ln = nn.LayerNorm(d_model)

        self.ffn = ops.PositionWiseFFN(d_model, d_ff, dropout)
        self.ffn_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm_pos = norm_pos

    def forward(self, x, mask):
        """
        x: [B, N, d_model]
        mask: [B, N]
        """
        # MultiHeadAttention
        skip_con = x
        if self.norm_pos == 'before':
            x = self.slf_attn_ln(x)
        attn_out, attn_w = self.slf_attn(x, x, x, mask)
        x = skip_con + self.dropout(attn_out)
        if self.norm_pos == 'after':
            x = self.slf_attn_ln(x)

        # Position-wise FFN
        skip_con = x
        if self.norm_pos == 'before':
            x = self.ffn_ln(x)
        x = self.ffn(x)
        x = skip_con + self.dropout(x)
        if self.norm_pos == 'after':
            x = self.ffn_ln(x)

        return x, attn_w


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, norm_pos='after'):
        super().__init__()
        self.slf_attn = ops.MultiHeadAttention(d_model, n_heads, dropout)
        self.slf_attn_ln = nn.LayerNorm(d_model)

        self.enc_attn = ops.MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_attn_ln = nn.LayerNorm(d_model)

        self.ffn = ops.PositionWiseFFN(d_model, d_ff, dropout)
        self.ffn_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm_pos = norm_pos

    def forward(self, enc_out, x, enc_mask, dec_mask):
        """
        enc_out: [B, src_len, d_model]
        x: [B, N, d_model]
        enc_mask: [B, src_len]
        dec_mask: [B, N]
        """
        # Self MultiHeadAttention
        skip_con = x
        if self.norm_pos == 'before':
            x = self.slf_attn_ln(x)
        slf_attn_out, slf_attn_w = self.slf_attn(x, x, x, dec_mask)
        x = skip_con + self.dropout(slf_attn_out)
        if self.norm_pos == 'after':
            x = self.slf_attn_ln(x)

        # Encoder MultiHeadAttention
        skip_con = x
        if self.norm_pos == 'before':
            x = self.enc_attn_ln(x)
        enc_attn_out, enc_attn_w = self.enc_attn(x, enc_out, enc_out, enc_mask)
        x = skip_con + self.dropout(enc_attn_out)
        if self.norm_pos == 'after':
            x = self.enc_attn_ln(x)

        # Position-wise FFN
        skip_con = x
        if self.norm_pos == 'before':
            x = self.ffn_ln(x)
        x = self.ffn(x)
        x = skip_con + self.dropout(x)
        if self.norm_pos == 'after':
            x = self.ffn_ln(x)

        return x, slf_attn_w, enc_attn_w

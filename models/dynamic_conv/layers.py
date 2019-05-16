import torch
import torch.nn as nn
import torch.nn.functional as F
from .sublayers import ConvBlock, MultiHeadAttention, PositionWiseFFN


class EncoderLayer(nn.Module):
    def __init__(self, conv_type, kernel_size, d_model, d_ff, n_heads, dropout, norm_pos='after'):
        super().__init__()
        padding = (kernel_size//2, kernel_size//2)
        self.slf_conv = ConvBlock(conv_type, d_model, kernel_size, padding=padding,
                                  n_heads=n_heads, dropconnect=dropout)
        self.slf_attn_ln = nn.LayerNorm(d_model)

        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.ffn_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm_pos = norm_pos

    def forward(self, x, mask):
        """
        x: [B, T, d_model]
        mask: [B, T]
        """
        # MultiHeadAttention
        skip_con = x
        if self.norm_pos == 'before':
            x = self.slf_attn_ln(x)
        x = self.slf_conv(x, mask)
        x = skip_con + self.dropout(x)
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

        return x


class DecoderLayer(nn.Module):
    def __init__(self, conv_type, kernel_size, d_model, d_ff, n_heads, dropout, norm_pos='after'):
        super().__init__()
        padding = (kernel_size-1, 0) # left-only padding
        self.slf_conv = ConvBlock(conv_type, d_model, kernel_size, padding=padding,
                                  n_heads=n_heads, dropconnect=dropout)
        self.slf_attn_ln = nn.LayerNorm(d_model)

        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_attn_ln = nn.LayerNorm(d_model)

        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.ffn_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm_pos = norm_pos

    def forward(self, enc_out, x, enc_mask):
        """
        enc_out: [B, src_len, d_model]
        x: [B, T, d_model]
        enc_mask: [B, src_len]
        """
        # Self MultiHeadAttention
        skip_con = x
        if self.norm_pos == 'before':
            x = self.slf_attn_ln(x)
        x = self.slf_conv(x)
        x = skip_con + self.dropout(x)
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

        return x, enc_attn_w

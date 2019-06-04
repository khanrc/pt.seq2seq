import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import ops
from . import layers


class Encoder(nn.Module):
    def __init__(self, src_n_words, max_len, conv_type, kernel_sizes, d_model=512, d_ff=2048,
                 n_heads=8, dropout=0.1, norm_pos='after'):
        super().__init__()
        self.norm_pos = norm_pos
        self.w_emb = ops.ScaledEmbedding(src_n_words, d_model)
        self.pos_encoder = ops.PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            layers.EncoderLayer(conv_type, kernel_size, d_model, d_ff, n_heads, dropout, norm_pos)
            for kernel_size in kernel_sizes
        ])
        if norm_pos == 'before':
            self.ln = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        emb = self.w_emb(x)
        emb = self.pos_encoder(emb)

        x = emb # [B, T, d_model]
        for layer in self.layers:
            # [B, T, d_model]
            x = layer(x, mask)

        if self.norm_pos == 'before':
            x = self.ln(x)

        return x


class Decoder(nn.Module):
    def __init__(self, tgt_n_words, max_len, conv_type, kernel_sizes, d_model=512, d_ff=2048,
                 n_heads=8, dropout=0.1, norm_pos='after'):
        super().__init__()
        self.norm_pos = norm_pos
        self.w_emb = ops.ScaledEmbedding(tgt_n_words, d_model)
        self.pos_encoder = ops.PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            layers.DecoderLayer(conv_type, kernel_size, d_model, d_ff, n_heads, dropout, norm_pos)
            for kernel_size in kernel_sizes
        ])
        if norm_pos == 'before':
            self.ln = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, tgt_n_words)

    def forward(self, enc_out, x, enc_mask):
        emb = self.w_emb(x)
        emb = self.pos_encoder(emb)

        x = emb # [B, T, d_model]
        enc_attn_ws = []
        for layer in self.layers:
            # [B, T, d_model], [B, H, Q, T]
            x, enc_attn_w = layer(enc_out, x, enc_mask)
            enc_attn_ws.append(enc_attn_w)

        if self.norm_pos == 'before':
            x = self.ln(x)

        avg_attn_ws = sum(enc_attn_ws) / len(enc_attn_ws) # [B, Q, H, T]

        return self.readout(x), avg_attn_ws # [B, T, tgt_n_words], [B, H, Q, T]

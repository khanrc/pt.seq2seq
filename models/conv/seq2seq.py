import torch
import torch.nn as nn
from const import *
from .encdec import ConvEncoder, ConvDecoder


class ConvS2S(nn.Module):
    def __init__(self, in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers, kernel_size,
                 dropout, max_len, cache_mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = ConvEncoder(in_dim, emb_dim, h_dim, n_layers=enc_layers,
                                   kernel_size=kernel_size, dropout=dropout, max_len=max_len)
        self.decoder = ConvDecoder(emb_dim, h_dim, out_dim, n_layers=dec_layers,
                                   kernel_size=kernel_size, dropout=dropout, max_len=max_len,
                                   cache_mode=cache_mode)
        self.max_len = max_len

    def forward(self, src, src_lens, tgt, tgt_lens, teacher_forcing):
        """
        src_lens and tgt_lens are not required
        """
        B = src.size(0)
        # encoder
        enc_out, attn_value, enc_mask = self.encoder(src)
        # decoder
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing
        if use_teacher_forcing:
            dec_in = tgt[:, :-1]
            # [B, tgt_len, out_dim], [B, tgt_len, src_len]
            dec_outs, attn_ws, _ = self.decoder(dec_in, enc_out, attn_value, enc_mask)
        else:
            dec_in = torch.full([B, 1], SOS_idx, dtype=torch.long, device='cuda')
            dec_outs = []
            attn_ws = []
            dec_max_len = tgt.size(1)-1 if tgt is not None else self.max_len+1
            # cache is similar to hidden state of RNN.
            caches = [torch.empty([B, self.decoder.h_dim, 0], dtype=torch.float32, device='cuda')
                      for _ in range(self.decoder.n_layers)]
            for i in range(dec_max_len):
                # [B, cur_len, out_dim], [B, cur_len, src_len]
                dec_out, attn_w, caches = self.decoder(
                    dec_in, enc_out, attn_value, enc_mask, caches, timestep=i)
                _, topi = dec_out[:, -1].topk(1) # [B, 1]
                dec_in = topi.detach() # [B, 1]

                dec_outs.append(dec_out)
                attn_ws.append(attn_w)

            dec_outs = torch.cat(dec_outs, dim=1) # [B, max_len, out_dim]
            attn_ws = torch.cat(attn_ws, dim=1) # [B, max_len, src_len]

        return dec_outs, attn_ws

    def generate(self, src, src_lens):
        # src_lens is not required
        return self.forward(src, src_lens, None, None, 0.)

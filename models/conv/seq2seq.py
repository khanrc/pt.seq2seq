import torch
import torch.nn as nn
from const import *


class ConvS2S(nn.Module):
    def __init__(self, encoder, decoder, max_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len

    # 일단 귀찮아서 이렇게 해놨는데 수정해야함
    #  def forward(self, src, tgt, teacher_forcing):
    def forward(self, src, src_lens, tgt, tgt_lens, teacher_forcing):
        B = src.size(0)
        # encoder
        enc_out, attn_value, enc_mask = self.encoder(src)
        # decoder
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing
        dec_in = torch.full([B, 1], SOS_idx, dtype=torch.long, device='cuda')
        if use_teacher_forcing:
            # [B, tgt_len, out_dim], [B, tgt_len, src_len]
            # Convert tgt to dec_in: SOS 추가 / EOS 제거.
            # EOS 를 제거하지 않는다고 문제가 생기는 것은 아님. 단지 할 필요가 없을 뿐.
            # 어차피 해도 gt 가 pad 이므로 로스를 먹지 않음.
            dec_in = torch.cat([dec_in, tgt[:, :-1]], dim=1)
            dec_outs, attn_ws = self.decoder(dec_in, enc_out, attn_value, enc_mask)
        else:
            dec_outs = []
            attn_ws = []
            for i in range(self.max_len):
                # [B, cur_len, out_dim], [B, cur_len, src_len]
                dec_out, attn_w = self.decoder(dec_in, enc_out, attn_value, enc_mask)
                _, top1 = dec_out[:, -1].topk(1) # [B, 1]
                top1 = top1.detach() # [B, 1]
                dec_in = torch.cat([dec_in, top1], dim=1)

                #dec_outs.append(dec_out)
                #attn_ws.append(attn_w)

            #  dec_outs = torch.cat(dec_outs, dim=1) # [B, max_len, out_dim]
            #  attn_ws = torch.cat(attn_ws, dim=1) # [B, max_len, src_len]
            dec_outs = dec_out
            attn_ws = attn_w

        return dec_outs, attn_ws

    def generate(self, src, src_lens):
        return self.forward(src, src_lens, None, None, 0.)

    #  def generate(self, src):
    #      return self.forward(src, None, 0.)

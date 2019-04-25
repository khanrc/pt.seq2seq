import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import data_prepare as dp
import random
from attention import KeyValueAttention, AdditiveAttention, MultiplicativeAttention


device = torch.device('cuda')


class Encoder(nn.Module):
    """ Simple encoder without attention """
    def __init__(self, in_dim, h_dim, bidirect=False):
        super().__init__()
        self.in_dim = in_dim # in_lang.n_words
        self.h_dim = h_dim
        self.embedding = nn.Embedding(in_dim, h_dim, padding_idx=dp.PAD_token)
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True, bidirectional=bidirect)
        self.n_direct = 2 if bidirect else 1
        self.linear = nn.Linear(h_dim * self.n_direct, h_dim)

    def forward(self, x, x_lens, h=None):
        """
        x: [B, seq_len, token_ids]
        """
        emb = self.embedding(x) # [B, seq_len, h_dim]
        # hidden 은 last state 만 나오고, out 은 각 타임스텝 다 나옴.
        # hidden 은 pack sequence 에 대해 각 길이에 맞게 마지막 타임스텝을 뽑아주고,
        # bidirectional case 에 대해서도 대응을 해 줌.
        emb = nn.utils.rnn.pack_padded_sequence(emb, x_lens, batch_first=True)
        out, h = self.gru(emb, h) # [B, seq_len, h_dim], [n_layers * n_direction, B, h_dim]
        out, x_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.n_direct == 2:
            h = torch.cat([h[0], h[1]], dim=-1).unsqueeze(0) # [1, B, h_dim*n_direct]

        h = torch.tanh(self.linear(h))

        return out, h # attention 을 위한 out 도 리턴.


class Decoder(nn.Module):
    """ Simple decoder without attention """
    def __init__(self, h_dim, out_dim):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim # out_lang.n_words
        # decoder 도 인풋은 word.
        # 이전 timestep 의 output 이거나,
        # 이전 timestep 의 true target (teacher-forcing)
        self.embedding = nn.Embedding(out_dim, h_dim, padding_idx=dp.PAD_token)
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.readout = nn.Linear(h_dim, out_dim)

    def forward(self, x, h=None):
        """
        x: [B, seq_len, token_ids]
        """
        emb = self.embedding(x) # [B, seq_len, h_dim]
        emb = F.relu(emb) # why?
        # out: [B, seq_len, h_dim]
        # h: [1, B, h_dim] (1 <= n_layers * n_direction)
        out, h = self.gru(emb, h)
        logits = self.readout(out)
        return logits, h


class AttnDecoder(nn.Module):
    """ Key-value attention decoder """
    def __init__(self, h_dim, out_dim, enc_h_dim, dropout=0.1, attention=None):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(out_dim, h_dim, padding_idx=dp.PAD_token)
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)

        if attention.startswith('kv'):
            self.attention = KeyValueAttention(q_in_dim=h_dim,
                                               qk_dim=h_dim,
                                               kv_in_dim=enc_h_dim,
                                               v_dim=h_dim)
        elif attention.startswith('add'):
            self.attention = AdditiveAttention(h_dim, enc_h_dim, h_dim, h_dim)
        elif attention.startswith('mul'):
            self.attention = MultiplicativeAttention(h_dim, enc_h_dim, h_dim, h_dim)
        else:
            self.attention = None

        #self.attn_combine = nn.Linear(h_dim * 2, h_dim)
        self.readout = nn.Linear(h_dim*2, out_dim)

    def forward(self, x, h, enc_h, mask):
        """
        x: [B, dec_len]
        h: [1, B, h_dim]
        enc_h: [B, enc_len, enc_h_dim]
        """
        #import pdb; pdb.set_trace()
        enc_len = enc_h.size(1) # enc_len
        dec_len = x.size(1)
        emb = self.embedding(x)
        out = self.dropout(emb) # [B, dec_len, h_dim]

        ### attention by embedding vector => gru inputs
        #  if self.attention is not None:
        #      attn_w, attn_out = self.attention(out, enc_h, mask)

        #      # combine emb out & attention out
        #      out = torch.cat([out, attn_out], dim=-1) # [B, dec_len, h_dim*2]
        #      out = self.attn_combine(out) # [B, dec_len, h_dim]
        #      out = F.relu(out)
        #  else:
        #      B = x.size(0)
        #      attn_w = torch.zeros([B, dec_len, enc_len], dtype=torch.float32)

        # final GRU
        # out: [B, dec_len, h_dim]
        # h: [1, B, h_dim]
        out, h = self.gru(out, h)

        # attention for last readout (better)
        if self.attention is not None:
            # [B, dec_len, seq_len], [B, dec_len, out_dim] (out dim == v dim)
            attn_w, attn_out = self.attention(out, enc_h, mask)

            # combine emb out & attention out
            out = torch.cat([out, attn_out], dim=-1) # [B, dec_len, h_dim*2]
            #  out = self.attn_combine(out) # [B, dec_len, h_dim]
            #  out = F.relu(out)
        else:
            B = x.size(0)
            attn_w = torch.zeros([B, dec_len, enc_len], dtype=torch.float32)

        out = self.readout(out)

        return out, h, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio, max_len=None):
        """
        params:
            max_len: generation mode 에서는 이 값이 주어져서 tgt로부터 max_len 을 구하지 않고
                     이 값을 사용. 이 경우, tgt 는 아예 사용되지 않음.

        Variable length
        Variable length 때문에 pad 가 생기는데 이에 대한 처리에 대한 생각을 해볼 필요가 있다.
        1. encoder: pack_sequence 를 통해 패킹을 해서 집어넣으면 알아서 처리해줌.
        이를 위해 dataloader 에서 길이 기준으로 소팅이 필요 & src_lens 필요.
        2. decoder: 디코더에서는 어차피 길이 기준 소팅이 안 되므로 (인풋 기준으로 해버리니까),
        pack_sequence 를 사용할 수는 없다. 따라서 그냥 하고, ignore_index 를 활용해서 padding loss
        를 무시하게끔 한다 (아니면 직접 EOS 에서 짤라서 로스 계산해줄수도 있다).
        """
        if max_len:
            assert teacher_forcing_ratio == 0., "Teacher forcing should be turn off in eval mode"

        B = src.size(0)
        # encoder
        enc_out, enc_h = self.encoder(src, src_lens)

        # decoder
        dec_in = torch.full([B, 1], dp.SOS_token, dtype=torch.long, device=device)
        dec_h = enc_h
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        # [B, enc_len]
        attn_mask = (src != dp.PAD_token).unsqueeze_(1) # [B, 1, src_len]
        attn_ws = []

        #print("use teacher forcing = {}".format(use_teacher_forcing))

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            dec_in = torch.cat([dec_in, tgt[:, :-1]], dim=1)
            # attn_w: [B, dec_len, enc_len]
            dec_outs, dec_h, attn_ws = self.decoder(dec_in, dec_h, enc_out, attn_mask)
        else:
            # Without Teacher forcing: use its own predictions as the next input
            #  alive = np.ones(B)
            if max_len is None:
                max_len = tgt.size(1)
            dec_outs = []
            for i in range(max_len):
                # [B, 1, out_lang.n_words], [1, B, h_dim], [B, 1, enc_len]
                dec_out, dec_h, attn_w = self.decoder(dec_in, dec_h, enc_out, attn_mask)
                topv, topi = dec_out.topk(1) # [B, 1, 1]
                dec_in = topi.squeeze(2).detach() # [B, 1]

                #  eos = dec_in == dp.EOS_token
                #  alive = alive & ~eos
                #dec_outs[:, i] = dec_out.squeeze(1)
                dec_outs.append(dec_out)
                attn_ws.append(attn_w)
                # 여기서 alive 를 사용해서 죽은 애들은 잘라줄 수도 있지만, 그냥 놔두자.
                # => 차이: (pred) EOS 가 나오고 난 이후에도 loss 를 먹일 것인가 말 것인가

            dec_outs = torch.cat(dec_outs, dim=1)
            attn_ws = torch.cat(attn_ws, dim=1)

            # may need continguous
            dec_outs = dec_outs.contiguous()

        return dec_outs, attn_ws

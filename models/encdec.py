import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import KeyValueAttention, AdditiveAttention, MultiplicativeAttention
from const import *


class Encoder(nn.Module):
    """ Simple encoder without attention """
    def __init__(self, in_dim, emb_dim, h_dim, bidirect=False):
        super().__init__()
        self.in_dim = in_dim # in_lang.n_words
        self.h_dim = h_dim
        self.embedding = nn.Embedding(in_dim, emb_dim, padding_idx=PAD_idx)
        self.gru = nn.GRU(emb_dim, h_dim, batch_first=True, bidirectional=bidirect)
        self.n_direct = 2 if bidirect else 1
        self.linear = nn.Linear(h_dim * self.n_direct, h_dim)

    def forward(self, x, x_lens, h=None):
        """
        x: [B, seq_len]
        """
        emb = self.embedding(x) # [B, seq_len, h_dim]
        # hidden 은 last state 만 나오고, out 은 각 타임스텝 다 나옴.
        # hidden 은 pack sequence 에 대해 각 길이에 맞게 마지막 타임스텝을 뽑아주고,
        # bidirectional case 에 대해서도 대응을 해 줌.
        emb = nn.utils.rnn.pack_padded_sequence(emb, x_lens, batch_first=True)
        # [B, seq_len, h_dim * n_direction], [n_layers * n_direction, B, h_dim]
        out, h = self.gru(emb, h)
        out, x_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.n_direct == 2:
            h = torch.cat([h[0], h[1]], dim=-1).unsqueeze(0) # [1, B, h_dim*n_direct]

        # dimension matching (h_dim*n_direct => h_dim)
        h = torch.tanh(self.linear(h))

        return out, h # attention 을 위한 out 도 리턴.


# AttnDecoder 에서 Non-attention 까지 커버
#  class Decoder(nn.Module):
#      """ Simple decoder without attention """
#      def __init__(self, emb_dim, h_dim, out_dim):
#          super().__init__()
#          self.h_dim = h_dim
#          self.out_dim = out_dim # out_lang.n_words
#          # decoder 도 인풋은 word.
#          # 이전 timestep 의 output 이거나,
#          # 이전 timestep 의 true target (teacher-forcing)
#          self.embedding = nn.Embedding(out_dim, emb_dim, padding_idx=PAD_idx)
#          self.gru = nn.GRU(emb_dim, h_dim, batch_first=True)
#          self.readout = nn.Linear(h_dim, out_dim)

#      def forward(self, x, h=None):
#          """
#          x: [B, seq_len, token_ids]

#          enc_hs and mask are not used here - just for protocol compatibility.
#          """
#          emb = self.embedding(x) # [B, seq_len, h_dim]
#          emb = F.relu(emb) # why?
#          # out: [B, seq_len, h_dim]
#          # h: [1, B, h_dim] (1 <= n_layers * n_direction)
#          out, h = self.gru(emb, h)
#          logits = self.readout(out)
#          return logits, h


class AttnDecoder(nn.Module):
    """ Key-value attention decoder """
    def __init__(self, emb_dim, h_dim, out_dim, enc_h_dim, dropout=0.1, attention=None):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(out_dim, emb_dim, padding_idx=PAD_idx)
        self.gru = nn.GRU(emb_dim, h_dim, batch_first=True)

        if attention.startswith('kv'):
            self.attention = KeyValueAttention(q_in_dim=h_dim,
                                               qk_dim=h_dim,
                                               kv_in_dim=enc_h_dim,
                                               v_dim=h_dim,
                                               out_dim=h_dim)
        elif attention.startswith('add'):
            self.attention = AdditiveAttention(q_dim=h_dim,
                                               s_dim=enc_h_dim,
                                               h_dim=h_dim,
                                               out_dim=h_dim)
        elif attention.startswith('mul'):
            self.attention = MultiplicativeAttention(q_dim=h_dim,
                                                     s_dim=enc_h_dim,
                                                     h_dim=h_dim,
                                                     out_dim=h_dim)
        else:
            self.attention = None

        #self.attn_combine = nn.Linear(h_dim * 2, h_dim)
        ro_dim = h_dim * (2 if self.attention is not None else 1)
        self.readout = nn.Linear(ro_dim, out_dim)

    def forward(self, x, h, enc_hs, mask):
        """
        x: [B, dec_len]
        h: [1, B, h_dim]
        enc_hs: [B, enc_len, enc_h_dim]
        """
        #import pdb; pdb.set_trace()
        enc_len = enc_hs.size(1) # enc_len
        dec_len = x.size(1)
        emb = self.embedding(x)
        out = self.dropout(emb) # [B, dec_len, h_dim]

        ### attention by embedding vector => gru inputs
        #  if self.attention is not None:
        #      attn_w, attn_out = self.attention(out, enc_hs, mask)

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
            attn_w, attn_out = self.attention(out, enc_hs, mask)

            # combine emb out & attention out
            out = torch.cat([out, attn_out], dim=-1) # [B, dec_len, h_dim*2]
            #  out = self.attn_combine(out) # [B, dec_len, h_dim]
            #  out = F.relu(out)
        else:
            B = x.size(0)
            attn_w = torch.zeros([B, dec_len, enc_len], dtype=torch.float32)

        out = self.readout(out)

        return out, h, attn_w

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import KeyValueAttention, AdditiveAttention, MultiplicativeAttention
from const import *


class Encoder(nn.Module):
    """ Simple encoder without attention """
    def __init__(self, in_dim, emb_dim, h_dim, n_layers=1, bidirect=False, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim # in_lang.n_words
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(in_dim, emb_dim, padding_idx=PAD_idx)
        self.gru = nn.GRU(emb_dim, h_dim, num_layers=n_layers, batch_first=True,
                          bidirectional=bidirect)
        self.n_layers = n_layers
        self.n_direct = 2 if bidirect else 1
        self.linear = nn.Linear(h_dim * self.n_direct, h_dim)

    def forward(self, x, x_lens, h=None):
        """
        x: [B, seq_len]
        """
        B = x.size(0)
        emb = self.embedding(x) # [B, seq_len, h_dim]
        emb = self.dropout(emb)
        emb = nn.utils.rnn.pack_padded_sequence(emb, x_lens, batch_first=True)
        # out: [B, seq_len, h_dim * n_direction]; every hidden states in each timestep
        # h: [n_layers * n_direction, B, h_dim]; last hidden state
        # h also consider each sequence length and bidirectionality.
        out, h = self.gru(emb, h)
        out, x_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # additional computation to hidden state for context vector (not for attention)
        h = h.view(self.n_layers, self.n_direct, B, self.h_dim)
        # concat bidirect hidden states in last layer
        # => [n_layers, B, h_dim*n_direct]
        h = torch.cat([h[:, i] for i in range(self.n_direct)], dim=-1)

        # dimension matching (h_dim*n_direct => h_dim)
        h = torch.tanh(self.linear(h))

        return out, h  # out for attention


class AttnDecoder(nn.Module):
    """ Key-value attention decoder """
    def __init__(self, emb_dim, h_dim, out_dim, enc_h_dim, n_layers=1, dropout=0.1,
                 attention=None):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(out_dim, emb_dim, padding_idx=PAD_idx)
        self.gru = nn.GRU(emb_dim, h_dim, num_layers=n_layers, batch_first=True)

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

        ro_dim = h_dim * (2 if self.attention is not None else 1)
        self.readout = nn.Linear(ro_dim, out_dim)

    def forward(self, x, h, enc_hs, mask):
        """
        x: [B, dec_len]
        h: [n_layers*n_direction, B, h_dim]
        enc_hs: [B, enc_len, enc_h_dim]
        """
        enc_len = enc_hs.size(1) # enc_len
        dec_len = x.size(1)
        emb = self.embedding(x)
        emb = self.dropout(emb) # [B, dec_len, emb_dim]

        ## outputs of GRU:
        # out: [B, dec_len, h_dim]
        # h: [n_layers*n_direction, B, h_dim]
        out, h = self.gru(emb, h)

        # attention for last readout (better)
        if self.attention is not None:
            # [B, dec_len, seq_len], [B, dec_len, out_dim] (out dim == v dim)
            attn_w, attn_out = self.attention(out, enc_hs, mask)

            # combine emb out & attention out
            out = torch.cat([out, attn_out], dim=-1) # [B, dec_len, h_dim*2]
        else:
            B = x.size(0)
            attn_w = torch.zeros([B, dec_len, enc_len], dtype=torch.float32)

        out = self.readout(out)

        return out, h, attn_w

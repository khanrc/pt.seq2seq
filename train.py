import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import numpy as np
import data_prepare as dp
import utils
from dataset import TranslationDataset, collate_data
from model import Encoder, Decoder, AttnDecoder, Seq2Seq
import random
from evaluate import random_eval, evaluateAndShowAttentions


def train(loader, seq2seq, optimizer, criterion, teacher_forcing_ratio=0.5):
    losses = utils.AverageMeter()
    seq2seq.train()

    for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
        B = src.size(0)
        src = src.cuda()
        src_lens = src_lens.cuda()
        tgt = tgt.cuda()
        tgt_lens = tgt_lens.cuda()

        dec_outs, attn_ws = seq2seq(src, src_lens, tgt, teacher_forcing_ratio)

        optimizer.zero_grad()
        loss = criterion(dec_outs, tgt)
        loss.backward()
        optimizer.step()

        losses.update(loss, B)

    return losses.avg


def evaluate(loader, seq2seq, criterion, max_len=dp.MAX_LENGTH):
    losses = utils.AverageMeter()
    seq2seq.eval()

    with torch.no_grad():
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
            B = src.size(0)
            src = src.cuda()
            src_lens = src_lens.cuda()
            tgt = tgt.cuda()
            tgt_lens = tgt_lens.cuda()

            dec_outs, attn_ws = seq2seq(src, src_lens, tgt, teacher_forcing_ratio=0.)
            loss = criterion(dec_outs, tgt)
            losses.update(loss, B)

    return losses.avg


def criterion(logits, targets):
    """
    배치 내에서는 summation 을 하고, 배치끼리는 mean 을 하고 싶다.
    (TODO) 클래스로 따로 빼는게 낫겠는데...
    logits: [B, max_len, out_lang.n_words]
    targets: [B, max_len]
    """
    losses = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), reduction='none',
                             ignore_index=dp.PAD_token)
    losses = losses.view(targets.shape)

    return losses.sum(1).mean()


if __name__ == "__main__":
    batch_size = 64
    epochs = 15
    h_dim = 256
    bidirect = False
    attention_type = 'kv'
    lr = 0.01

    dset = TranslationDataset()
    valid_indices = np.load("valid_indices.npy")
    train_indices = list(set(range(len(dset))) - set(valid_indices))
    train_dset = Subset(dset, train_indices)
    valid_dset = Subset(dset, valid_indices)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_data)
    valid_loader = DataLoader(valid_dset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_data)

    # build model
    encoder = Encoder(dset.in_lang.n_words, h_dim, bidirect=bidirect)
    decoder = AttnDecoder(h_dim, dset.out_lang.n_words, enc_h_dim=encoder.h_dim*encoder.n_direct,
                          attention=attention_type)
    seq2seq = Seq2Seq(encoder, decoder)
    seq2seq.cuda()

    # batch size 1, epoch 8, lr 0.01 => val_loss 11.63
    #optimizer = optim.SGD(seq2seq.parameters(), lr=lr)
    # batch size 64, epoch 40 => val_loss 13.95 (min=13.5)
    optimizer = optim.Adamax(seq2seq.parameters())
    #optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

    #  print("Random eval:")
    #  random_eval(valid_dset, seq2seq)
    evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, 0)
    #  print("")

    for epoch in range(epochs):
        print("Epoch {}".format(epoch+1))
        loss = train(train_loader, seq2seq, optimizer, criterion, teacher_forcing_ratio=0.5)
        print("\ttrain: {}".format(loss))
        loss = evaluate(valid_loader, seq2seq, criterion)
        print("\tvalid: {}".format(loss))
        print("Random eval:")
        random_eval(valid_dset, seq2seq)
        evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch+1)
        print("")

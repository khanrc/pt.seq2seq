import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import numpy as np

import utils
from dataset import TranslationDataset, collate_data
from model import Encoder, Decoder, AttnDecoder, Seq2Seq
from evaluate import random_eval, evaluateAndShowAttentions
import data_prepare as dp
from lang import Lang


def train(loader, seq2seq, optimizer, criterion, teacher_forcing_ratio=0.5):
    losses = utils.AverageMeter()
    seq2seq.train()

    for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
        B = src.size(0)
        src = src.cuda()
        tgt = tgt.cuda()

        dec_outs, attn_ws = seq2seq(src, src_lens, tgt, tgt_lens, teacher_forcing_ratio)

        optimizer.zero_grad()
        loss = criterion(dec_outs, tgt)
        loss.backward()
        optimizer.step()

        losses.update(loss, B)

    return losses.avg


def evaluate(loader, seq2seq, criterion):
    losses = utils.AverageMeter()
    seq2seq.eval()

    with torch.no_grad():
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
            B = src.size(0)
            src = src.cuda()
            tgt = tgt.cuda()

            dec_outs, attn_ws = seq2seq(src, src_lens, tgt, tgt_lens, teacher_forcing_ratio=0.)
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
    if logits.size(1) < targets.size(1):
        # logits 는 뒤에 패딩들을 잘라내고 난 max_len 이고,
        # targets 는 패딩이 포함된 MAX_LENGTH.
        # 어차피 무시할거니까 여기서 잘라주자.
        targets = targets[:, :logits.size(1)]
    losses = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), reduction='none',
                             ignore_index=dp.PAD_idx)
    losses = losses.view(targets.shape)

    return losses.sum(1).mean()


if __name__ == "__main__":
    batch_size = 64
    epochs = 15
    h_dim = 1024 # encoder / decoder hidden dims
    emb_dim = 300
    bidirect = True
    attention_type = 'kv'
    VIZ_ATTN = False
    N_eval = 3
    MAX_LEN = dp.MAX_LENGTH

    # load dataset
    print("Load dataset")
    N = 127357 # MAX_LEN = 14
    input_lang = Lang.load_from_file("fra", f"cache/in-fra-{N}.pkl")
    output_lang = Lang.load_from_file("eng", f"cache/out-eng-{N}.pkl")
    pairs = pickle.load(open(f"cache/fra2eng-{N}.pkl", "rb"))
    print("\t# of input_lang = {}".format(input_lang.n_words))
    print("\t# of output_lang = {}".format(output_lang.n_words))
    print("\t# of pairs = {}".format(len(pairs)))
    dset = TranslationDataset(input_lang, output_lang, pairs)
    print(random.choice(pairs))

    # split dset by valid indices
    valid_indices = np.load("valid_indices.npy")
    train_indices = list(set(range(len(dset))) - set(valid_indices))
    train_dset = Subset(dset, train_indices)
    valid_dset = Subset(dset, valid_indices)

    # loader
    print("Load loader")
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_data)
    valid_loader = DataLoader(valid_dset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_data)

    # build model
    print("Build model")
    encoder = Encoder(dset.in_lang.n_words, emb_dim, h_dim, bidirect=bidirect)
    decoder = AttnDecoder(emb_dim, h_dim, dset.out_lang.n_words, enc_h_dim=encoder.h_dim*encoder.n_direct,
                          attention=attention_type)
    seq2seq = Seq2Seq(encoder, decoder, MAX_LEN)
    seq2seq.cuda()
    print(seq2seq)

    # batch size 1, epoch 8, lr 0.01 => val_loss 11.63
    #optimizer = optim.SGD(seq2seq.parameters(), lr=0.01)

    # batch size 64, epoch 40 => val_loss 13.95 (min=13.5)
    optimizer = optim.Adamax(seq2seq.parameters())

    #optimizer = optim.Adam(seq2seq.parameters(), lr=3e-4)

    #  print("Random eval:")
    #  random_eval(valid_dset, seq2seq)
    if VIZ_ATTN:
        utils.makedirs('evals')
        evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, 0)
    #  print("")

    for epoch in range(epochs):
        print("Epoch {}".format(epoch+1))
        loss = train(train_loader, seq2seq, optimizer, criterion, teacher_forcing_ratio=0.5)
        print("\ttrain: {}".format(loss))
        loss = evaluate(valid_loader, seq2seq, criterion)
        print("\tvalid: {}".format(loss))
        print("Random eval:")
        random_eval(valid_dset, seq2seq, N=N_eval)
        if VIZ_ATTN:
            evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch+1, print_attn=False)
        print("")

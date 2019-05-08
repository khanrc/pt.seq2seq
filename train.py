import pickle
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import yaml

import utils
from dataset import TranslationDataset, collate_data
from models.rnn import Encoder, AttnDecoder, Seq2Seq
from models.conv import ConvEncoder, ConvDecoder, ConvS2S
from evaluate import random_eval, evaluateAndShowAttentions
import data_prepare
from lang import Lang
from logger import Logger
from const import *


logger = Logger.get()
writer = SummaryWriter()


def train(loader, seq2seq, optimizer, criterion, teacher_forcing, grad_clip, epoch):
    losses = utils.AverageMeter()
    seq2seq.train()
    N = len(loader)

    for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
        B = src.size(0)
        src = src.cuda()
        tgt = tgt.cuda()

        #  if i == 0 and epoch == 0:
        #      writer.add_graph(seq2seq, input_to_model=(src, src_lens, tgt, tgt_lens, 0.0), verbose=True)

        dec_outs, attn_ws = seq2seq(src, src_lens, tgt, tgt_lens, teacher_forcing)

        optimizer.zero_grad()
        loss = criterion(dec_outs, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), grad_clip)
        optimizer.step()

        losses.update(loss, B)
        cur_step = N*epoch + i
        writer.add_scalar('train/loss', loss, cur_step)

    return losses.avg


def evaluate(loader, seq2seq, criterion):
    losses = utils.AverageMeter()
    seq2seq.eval()

    with torch.no_grad():
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
            B = src.size(0)
            src = src.cuda()
            tgt = tgt.cuda()

            dec_outs, attn_ws = seq2seq(src, src_lens, tgt, tgt_lens, teacher_forcing=0.)
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
                             ignore_index=PAD_idx)
    losses = losses.view(targets.shape)

    return losses.sum(1).mean()


if __name__ == "__main__":
    ## configuration
    logger.info("### Configuration ###")
    r = open("config.yaml").read()
    logger.nofmt(r)
    cfg = yaml.load(r, Loader=yaml.Loader)
    # train
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']
    teacher_forcing = cfg['train']['teacher_forcing']
    grad_clip = cfg['train']['grad_clip']
    # model
    model_type = cfg['model']['type']
    h_dim = cfg['model']['args']['h_dim']
    emb_dim = cfg['model']['args']['emb_dim']
    # eval
    N_eval = cfg['eval']['N']
    VIZ_ATTN = cfg['eval']['viz_attn']
    # data (preproc)
    max_len = cfg['data']['max_len']
    min_freq = cfg['data']['min_freq']

    # load dataset
    logger.info("### Load dataset ###")
    in_lang_path = f"cache/in-fra-{max_len}-{min_freq}.pkl"
    out_lang_path = f"cache/out-eng-{max_len}-{min_freq}.pkl"
    pair_path = f"cache/fra2eng-{max_len}.pkl"
    exist_all = all(os.path.exists(path) for path in [in_lang_path, out_lang_path, pair_path])
    if not exist_all:
        data_prepare.prepare(max_len, min_freq)

    input_lang = Lang.load_from_file("fra", in_lang_path)
    output_lang = Lang.load_from_file("eng", out_lang_path)
    pairs = pickle.load(open(pair_path, "rb"))
    logger.info("\tinput_lang.n_words = {}".format(input_lang.n_words))
    logger.info("\toutput_lang.n_words = {}".format(output_lang.n_words))
    logger.info("\t# of pairs = {}".format(len(pairs)))
    dset = TranslationDataset(input_lang, output_lang, pairs, max_len)
    logger.info(random.choice(pairs))

    # split dset by valid indices
    N_pairs = len(pairs)
    val_indices_path = f"cache/valid_indices-{N_pairs}.npy"
    if not os.path.exists(val_indices_path):
        data_prepare.gen_valid_indices(N_pairs, 0.1, val_indices_path)
    valid_indices = np.load(val_indices_path)
    train_indices = list(set(range(len(dset))) - set(valid_indices))
    train_dset = Subset(dset, train_indices)
    valid_dset = Subset(dset, valid_indices)

    # loader
    logger.info("Load loader")
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_data)
    valid_loader = DataLoader(valid_dset, batch_size, shuffle=False, num_workers=4,
                              collate_fn=collate_data)

    # build model
    logger.info("### Build model ###")
    in_dim = dset.in_lang.n_words
    out_dim = dset.out_lang.n_words
    if model_type == 'rnn':
        bidirect = cfg['model']['args']['bidirect']
        attention_type = cfg['model']['args']['attention_type']

        encoder = Encoder(in_dim, emb_dim, h_dim, bidirect=bidirect)
        decoder = AttnDecoder(emb_dim, h_dim, out_dim, enc_h_dim=encoder.h_dim*encoder.n_direct,
                              attention=attention_type)
        seq2seq = Seq2Seq(encoder, decoder, max_len)
    elif model_type == 'conv':
        enc_layers = cfg['model']['args']['enc_layers']
        dec_layers = cfg['model']['args']['dec_layers']
        kernel_size = cfg['model']['args']['kernel_size']
        dropout = cfg['model']['args']['dropout']

        encoder = ConvEncoder(in_dim, emb_dim, h_dim, n_layers=enc_layers, kernel_size=kernel_size,
                              dropout=dropout, max_len=max_len)
        decoder = ConvDecoder(emb_dim, h_dim, out_dim, n_layers=dec_layers, kernel_size=kernel_size,
                              dropout=dropout, max_len=max_len)
        seq2seq = ConvS2S(encoder, decoder, max_len)

    seq2seq.cuda()
    logger.nofmt(seq2seq)

    #  optimizer = optim.SGD(seq2seq.parameters(), lr=0.25)
    #  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-4,
    #                                                      verbose=True)

    #optimizer = optim.Adamax(seq2seq.parameters())

    optimizer = optim.Adam(seq2seq.parameters(), lr=3e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    #  logger.info("Random eval:")
    #  random_eval(valid_dset, seq2seq)
    if VIZ_ATTN:
        utils.makedirs('evals')
        evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch=0, print_attn=True)
    #  logger.info("")

    best = 999.
    for epoch in range(epochs):
        logger.info("Epoch {}, LR = {}".format(epoch+1, optimizer.param_groups[0]["lr"]))

        # train
        trn_loss = train(train_loader, seq2seq, optimizer, criterion, teacher_forcing=teacher_forcing,
                     grad_clip=grad_clip, epoch=epoch)
        logger.info("\ttrain: {}".format(trn_loss))

        # validation
        val_loss = evaluate(valid_loader, seq2seq, criterion)
        logger.info("\tvalid: {}".format(val_loss))
        cur_step = len(train_loader) * (epoch+1)
        writer.add_scalar('val/loss', val_loss, cur_step)

        if val_loss < best:
            best = val_loss

        # step lr scheduler
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        # evaluation & attention visualization
        logger.info("Random eval:")
        random_eval(valid_dset, seq2seq, N=N_eval)
        if VIZ_ATTN:
            evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch=epoch+1, print_attn=True)
        logger.info("")

    logger.info("Best loss = {}".format(best))

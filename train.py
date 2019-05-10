import pickle
import random
import os
import sys
import argparse

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
from models import Seq2Seq, ConvS2S
from evaluate import random_eval, evaluateAndShowAttentions
import data_prepare
from lang import Lang
from logger import Logger
from const import *


### Load config, logger, and tb writer
parser = argparse.ArgumentParser("ConvS2S")
parser.add_argument("config_path")
parser.add_argument("name")
args = parser.parse_args()
if not args.config_path.endswith(".yaml"):
    args.config_path += ".yaml"

# logger
logger = Logger.get(comment=args.name)
# tb
tb_path = utils.tb_name(args.name)
writer = SummaryWriter(log_dir=tb_path)
# config
cfg = yaml.load(open(args.config_path), Loader=yaml.Loader)


def train(loader, seq2seq, optimizer, criterion, teacher_forcing, grad_clip, epoch):
    losses = utils.AverageMeter()
    ppls = utils.AverageMeter()
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
        loss, ppl = criterion(dec_outs, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), grad_clip)
        optimizer.step()

        losses.update(loss, B)
        ppls.update(ppl, B)
        cur_step = N*epoch + i
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/ppl', ppl, cur_step)

    return losses.avg, ppls.avg


def evaluate(loader, seq2seq, criterion):
    losses = utils.AverageMeter()
    ppls = utils.AverageMeter()
    seq2seq.eval()

    with torch.no_grad():
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
            B = src.size(0)
            src = src.cuda()
            tgt = tgt.cuda()

            dec_outs, attn_ws = seq2seq(src, src_lens, tgt, tgt_lens, teacher_forcing=0.)
            loss, ppl = criterion(dec_outs, tgt)
            losses.update(loss, B)
            ppls.update(ppl, B)

    return losses.avg, ppls.avg


def criterion(logits, targets):
    """
    배치 내에서는 summation 을 하고, 배치끼리는 mean 을 하고 싶다.
    (TODO) 클래스로 따로 빼는게 낫겠는데...
    logits: [B, max_len, out_lang.n_words]
    targets: [B, max_len]
    """
    if logits.size(1) < targets.size(1):
        # logits 는 뒤에 공통 패딩들을 잘라내고 난 tgt_lens.max() 이고,
        # targets 는 패딩이 포함된 MAX_LEN.
        # 어차피 무시할거니까 여기서 잘라주자.
        targets = targets[:, :logits.size(1)]
    losses = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), reduction='none',
                             ignore_index=PAD_idx)
    losses = losses.view(targets.shape)

    sum_loss = losses.sum(1).mean()
    avg_loss = losses.sum() / (targets != PAD_idx).sum()
    perplexity = torch.exp(avg_loss).item()

    return sum_loss, perplexity


if __name__ == "__main__":
    ## configuration
    logger.info("### Configuration ###")
    cfg_str = yaml.dump(cfg, sort_keys=False)
    logger.nofmt(cfg_str)

    cfg_str_mk = cfg_str.replace(' ', '&nbsp;').replace('\n', '  \n')
    writer.add_text("config", cfg_str_mk)
    # train
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']
    teacher_forcing = cfg['train']['teacher_forcing']
    grad_clip = cfg['train']['grad_clip']
    # model
    model_type = cfg['model']['type']
    h_dim = cfg['model']['h_dim']
    emb_dim = cfg['model']['emb_dim']
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
        bidirect = cfg['model']['bidirect']
        attention_type = cfg['model']['attention_type']
        dropout = cfg['model']['dropout']
        enc_layers = cfg['model']['enc_layers']
        dec_layers = cfg['model']['dec_layers']
        seq2seq = Seq2Seq(in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers,
                          enc_bidirect=bidirect, dropout=dropout,
                          attention=attention_type, max_len=max_len)
    elif model_type == 'conv':
        enc_layers = cfg['model']['enc_layers']
        dec_layers = cfg['model']['dec_layers']
        kernel_size = cfg['model']['kernel_size']
        dropout = cfg['model']['dropout']
        cache_mode = cfg['model']['cache_mode']
        seq2seq = ConvS2S(in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers,
                          kernel_size=kernel_size, dropout=dropout, max_len=max_len,
                          cache_mode=cache_mode)

    seq2seq.cuda()
    logger.nofmt(seq2seq)

    #  optimizer = optim.SGD(seq2seq.parameters(), lr=0.25)
    #  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-4,
    #                                                      verbose=True)

    #optimizer = optim.Adamax(seq2seq.parameters())

    optimizer = optim.Adam(seq2seq.parameters(), lr=3e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-6)

    #  logger.info("Random eval:")
    #  random_eval(valid_dset, seq2seq)
    if VIZ_ATTN:
        utils.makedirs('evals')
        evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch=0, print_attn=True,
                                  writer=writer)
    #  logger.info("")

    best_ppl = 999.
    best_loss = 999.
    for epoch in range(epochs):
        logger.info("Epoch {}, LR = {}".format(epoch+1, optimizer.param_groups[0]["lr"]))

        # train
        trn_loss, trn_ppl = train(train_loader, seq2seq, optimizer, criterion,
                                  teacher_forcing=teacher_forcing, grad_clip=grad_clip, epoch=epoch)
        logger.info("\ttrain: Loss {:7.3f}  PPL {:7.3f}".format(trn_loss, trn_ppl))

        # validation
        val_loss, val_ppl = evaluate(valid_loader, seq2seq, criterion)
        logger.info("\tvalid: Loss {:7.3f}  PPL {:7.3f}".format(val_loss, val_ppl))
        cur_step = len(train_loader) * (epoch+1)
        writer.add_scalar('val/loss', val_loss, cur_step)
        writer.add_scalar('val/ppl', val_ppl, cur_step)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_loss = val_loss

        # step lr scheduler
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        # evaluation & attention visualization
        logger.info("Random eval:")
        random_eval(valid_dset, seq2seq, N=N_eval)
        if VIZ_ATTN:
            evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch=epoch+1,
                                      print_attn=True, writer=writer)
        logger.info("")

    logger.info("Name: {}".format(args.name))
    logger.info("Best: Loss {:7.3f}  PPL {:7.3f}".format(best_loss, best_ppl))

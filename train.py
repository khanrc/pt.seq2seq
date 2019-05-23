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
from yaml_config import YAMLConfig
from warmup import WarmupLR

import utils
from dataset import TranslationDataset, src_sort
from models import Seq2Seq, ConvS2S, Transformer, DynamicConvS2S
from evaluate import random_eval, evaluateAndShowAttentions
import data_prepare
from lang import Lang
from logger import Logger
from const import *
from bleu import BLEU


### Setup: load config, logger, and tb writer ###
# arg parser
parser = argparse.ArgumentParser("Seq2Seq")
parser.add_argument("config_path")
parser.add_argument("name")
parser.add_argument("--param_tracing", action="store_true", default=False)
parser.add_argument("--log_lv", default="info")
args, left_argv = parser.parse_known_args()
if not args.config_path.endswith(".yaml"):
    args.config_path += ".yaml"

# prepare
timestamp = utils.timestamp()
utils.makedirs('logs')
utils.makedirs('runs')
# config
cfg = YAMLConfig(args.config_path, left_argv)
# logger
logger_path = os.path.join('logs', "{}_{}.log".format(timestamp, args.name))
logger = Logger.get(file_path=logger_path, level=args.log_lv)
# tb
tb_path = os.path.join('runs', "{}_{}".format(timestamp, args.name))
writer = SummaryWriter(log_dir=tb_path)


def train(loader, seq2seq, optimizer, lr_scheduler, criterion, teacher_forcing, epoch,
          grad_clip=0.):
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
        if grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), grad_clip)
        optimizer.step()

        losses.update(loss, B)
        ppls.update(ppl, B)
        cur_step = N*epoch + i
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/ppl', ppl, cur_step)

        # step lr scheduler
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    return losses.avg, ppls.avg


def evaluate(loader, seq2seq, criterion, max_len):
    import time
    losses = utils.AverageMeter()
    ppls = utils.AverageMeter()
    seq2seq.eval()
    bleu = BLEU()

    tot_st = time.time()
    bleu_time = 0.
    # BLEU time: 13k 개에 대해서 약 4s. multi-cpu parallelization 은 가능함.

    def get_lens(tensor, max_len=max_len):
        """ get first position (index) of EOS_idx in tensor
            = length of each sentence
        tensor: [B, T]
        """
        # assume that former idx coming earlier in nonzero().
        # tensor 가 [B, T] 이므로 nonzero 함수도 [i, j] 형태의 tuple 을 결과로 내놓는데,
        # 이 결과가 i => j 순으로 sorting 되어 있다고 가정.
        # e.g) nonzero() => [[1,1], [1,2], [2,1], [2,3], [2,5], ...]
        nz = (tensor == EOS_idx).nonzero()
        is_first = nz[:-1, 0] != nz[1:, 0]
        is_first = torch.cat([torch.cuda.ByteTensor([1]), is_first]) # first mask

        # convert is_first from mask to indice by nonzero()
        first_nz = nz[is_first.nonzero().flatten()]
        lens = torch.full([tensor.size(0)], max_len, dtype=torch.long).cuda()
        lens[first_nz[:, 0]] = first_nz[:, 1]
        return lens

    with torch.no_grad():
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
            B = src.size(0)
            src = src.cuda()
            tgt = tgt.cuda()

            dec_outs, attn_ws = seq2seq(src, src_lens, tgt, tgt_lens, teacher_forcing=0.)
            loss, ppl = criterion(dec_outs, tgt)
            losses.update(loss, B)
            ppls.update(ppl, B)

            # BLEU
            bleu_st = time.time()
            # convert logits to preds
            preds = dec_outs.max(-1)[1]
            # get pred lens by finding EOS token
            pred_lens = get_lens(preds)

            for pred, target, pred_len, target_len in zip(preds, tgt, pred_lens, tgt_lens):
                # target_len include EOS token => -1.
                bleu.add_sentence(pred[:pred_len].cpu().numpy(), target[:target_len-1].cpu().numpy())

            bleu_time += time.time() - bleu_st
    total_time = time.time() - tot_st

    logger.debug("TIME: tot = {:.3f}\t bleu = {:.3f}".format(total_time, bleu_time))

    return losses.avg, ppls.avg, bleu.score()


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
    logger.nofmt(cfg.str())
    writer.add_text("config", cfg.markdown())
    # train
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']
    teacher_forcing = cfg['train']['teacher_forcing']
    grad_clip = cfg['train']['grad_clip']
    # model
    model_type = cfg['model']['type']
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
    collate_fn = src_sort if model_type == 'rnn' else torch.utils.data.dataloader.default_collate
    logger.info("Load loader")
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dset, batch_size, shuffle=False, num_workers=4,
                              collate_fn=collate_fn)

    # build model
    logger.info("### Build model ###")
    in_dim = dset.in_lang.n_words
    out_dim = dset.out_lang.n_words
    mcfg = cfg['model']
    if model_type == 'rnn':
        h_dim = mcfg['h_dim']
        emb_dim = mcfg['emb_dim']
        bidirect = mcfg['bidirect']
        attention_type = mcfg['attention_type']
        dropout = mcfg['dropout']
        enc_layers = mcfg['enc_layers']
        dec_layers = mcfg['dec_layers']
        seq2seq = Seq2Seq(in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers,
                          enc_bidirect=bidirect, dropout=dropout,
                          attention=attention_type, max_len=max_len)
    elif model_type == 'conv':
        h_dim = mcfg['h_dim']
        emb_dim = mcfg['emb_dim']
        enc_layers = mcfg['enc_layers']
        dec_layers = mcfg['dec_layers']
        kernel_size = mcfg['kernel_size']
        dropout = mcfg['dropout']
        cache_mode = mcfg['cache_mode']
        seq2seq = ConvS2S(in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers,
                          kernel_size=kernel_size, dropout=dropout, max_len=max_len,
                          cache_mode=cache_mode)
    elif model_type == 'transformer':
        d_model = mcfg['d_model']
        d_ff = mcfg['d_ff']
        n_layers = mcfg['n_layers']
        n_heads = mcfg['n_heads']
        dropout = mcfg['dropout']
        norm_pos = mcfg['norm_pos']
        seq2seq = Transformer(in_dim, out_dim, max_len, d_model, d_ff, n_layers, n_heads, dropout,
                              norm_pos)
    elif model_type == 'dynamic_conv':
        conv_type = mcfg['conv_type']
        kernel_sizes = mcfg['kernel_sizes']
        d_model = mcfg['d_model']
        d_ff = mcfg['d_ff']
        n_heads = mcfg['n_heads']
        dropout = mcfg['dropout']
        norm_pos = mcfg['norm_pos']
        seq2seq = DynamicConvS2S(in_dim, out_dim, max_len, conv_type, kernel_sizes, d_model, d_ff,
                                 n_heads, dropout, norm_pos)

    seq2seq.cuda()
    K = 1024
    n_params = utils.num_params(seq2seq) / K / K
    logger.nofmt(seq2seq)
    logger.info("# of params = {:.1f} M".format(n_params))

    # parameter size tracing
    if args.param_tracing:
        # sequential tracing
        #  for name, p in seq2seq.named_parameters():
        #      numel = p.numel()
        #      unit = 'M'
        #      numel /= 1024*1024
        #      fmt = "10.3f" if numel < 1.0 else "10.1f"

        #      print("{:50s}\t{:{fmt}}{}".format(name, numel, unit, fmt=fmt))

        # recursive tracing
        def param_trace(name, module, depth, max_depth=999, threshold=0):
            if depth > max_depth:
                return
            prefix = "  " * depth
            n_params = utils.num_params(module)
            if n_params > threshold:
                print("{:60s}\t{:10.2f}M".format(prefix + name, n_params / K / K))
            for n, m in module.named_children():
                if depth == 0:
                    child_name = n
                else:
                    child_name = "{}.{}".format(name, n)
                param_trace(child_name, m, depth+1, max_depth, threshold)

        param_trace('seq2seq', seq2seq, 0, max_depth=5, threshold=K*100)

        exit()

    #  optimizer = optim.SGD(seq2seq.parameters(), lr=0.25)
    #  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-4,
    #                                                      verbose=True)

    #optimizer = optim.Adamax(seq2seq.parameters())

    T_ep = len(train_loader)
    #optimizer = optim.Adam(seq2seq.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adam(seq2seq.parameters(), lr=3e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_ep*epochs, eta_min=3e-6)
    if 'warmup' in cfg['train']:
        warmup_ep = cfg['train']['warmup']
        lr_scheduler = WarmupLR(optimizer, init_scale=1e-3, T_max=T_ep*warmup_ep,
                                after=lr_scheduler)

    if VIZ_ATTN:
        utils.makedirs('evals')
        evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch=0, print_attn=True,
                                  writer=writer)

    best_ppl = utils.BestTracker('min')
    best_loss = utils.BestTracker('min')
    best_bleu = utils.BestTracker('max')
    for epoch in range(epochs):
        logger.info("Epoch {}/{}, LR = {}".format(epoch+1, epochs, optimizer.param_groups[0]["lr"]))

        # train
        trn_loss, trn_ppl = train(train_loader, seq2seq, optimizer, lr_scheduler, criterion,
                                  teacher_forcing=teacher_forcing, epoch=epoch, grad_clip=grad_clip)
        logger.info("\ttrain: Loss {:7.3f}  PPL {:7.3f}".format(trn_loss, trn_ppl))

        # validation
        val_loss, val_ppl, val_bleu = evaluate(valid_loader, seq2seq, criterion, max_len)
        logger.info("\tvalid: Loss {:7.3f}  PPL {:7.3f}  BLEU {:7.3f}".format(
            val_loss, val_ppl, val_bleu))

        cur_step = len(train_loader) * (epoch+1)
        writer.add_scalar('val/loss', val_loss, cur_step)
        writer.add_scalar('val/ppl', val_ppl, cur_step)
        writer.add_scalar('val/bleu', val_bleu, cur_step)

        best_ppl.check(val_ppl, epoch+1)
        best_loss.check(val_loss, epoch+1)
        best_bleu.check(val_bleu, epoch+1)

        # evaluation & attention visualization
        logger.info("Random eval:")
        random_eval(valid_dset, seq2seq, N=N_eval)
        if VIZ_ATTN:
            evaluateAndShowAttentions(seq2seq, dset.in_lang, dset.out_lang, epoch=epoch+1,
                                      print_attn=True, writer=writer)
        logger.info("")

    logger.info("Name: {}".format(args.name))
    logger.info("Best: Loss {loss.val:7.3f} ({loss.ep})  PPL {ppl.val:7.3f} ({ppl.ep})  "
                "BLEU {bleu.val:7.3f} ({bleu.ep})".format(
                    loss=best_loss, ppl=best_ppl, bleu=best_bleu))

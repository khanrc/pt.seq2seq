import random
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from const import *
from logger import Logger
from torchtext.data import Batch


logger = Logger.get()


def idx2words(indice, vocab):
    words = []
    for idx in indice:
        words.append(vocab.itos[idx])
        if idx == EOS_idx:
            break
    return words


def random_eval(dset, seq2seq, N=3):
    seq2seq.eval()
    src_vocab = dset.fields['src'].vocab
    trg_vocab = dset.fields['trg'].vocab

    examples = np.random.choice(dset.examples, replace=False, size=N).tolist()
    examples = sorted(examples, key=lambda ex: len(ex.src), reverse=True)
    x = Batch(examples, dset, 'cuda')

    # [B, T], [B]
    src, src_lens = x.src
    tgt, tgt_lens = x.trg

    dec_outs, attn_ws = seq2seq.generate(src, src_lens)
    topi = dec_outs.topk(1)[1].squeeze() # [B, max_len, 1]

    for src_idx, tgt_idx, out_idx in zip(src, tgt, topi):
        src_sentence = " ".join(idx2words(src_idx[1:], src_vocab))
        tgt_sentence = " ".join(idx2words(tgt_idx[1:], trg_vocab))
        out_sentence = " ".join(idx2words(out_idx, trg_vocab))

        logger.info("> {}".format(src_sentence))
        logger.info("= {}".format(tgt_sentence))
        logger.info("< {}".format(out_sentence))
        logger.info("")


def showAttention(input_sentence, output_words, attentions, file_path=None):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['EOS'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if file_path:
        plt.savefig(file_path)
        plt.close()
    else:
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer._renderer)
        plt.close()
        return X


def evaluateAndShowAttention(in_s, out_s, seq2seq, in_proc, out_vocab, out_file):
    seq2seq.eval()
    words = in_s.split(' ')
    src, src_len = in_proc([words], 'cuda')

    dec_outs, attn_ws = seq2seq.generate(src, src_len)
    topi = dec_outs.topk(1)[1] # [1, max_len, 1]
    out_words = idx2words(topi.squeeze(), out_vocab)

    logger.info("input  = {}".format(in_s))
    logger.info("answer = {}".format(out_s))
    logger.info("output = {}".format(' '.join(out_words)))
    attn_ws = attn_ws.squeeze().detach().cpu()[:len(out_words)]
    image = showAttention(in_s, out_words, attn_ws, out_file)
    return attn_ws, image


def evaluateAndShowAttentions(batch, seq2seq, dset, epoch, print_attn, writer):
    #  sens = [
    #      "elle a cinq ans de moins que moi .",
    #      "elle est trop petit .",
    #      "je ne crains pas de mourir .",
    #      "c est un jeune directeur plein de talent ."
    #  ]
    in_proc = dset.fields['src'].process
    out_vocab = dset.fields['trg'].vocab
    for i, ex in enumerate(batch):
        src = " ".join(ex.src)
        trg = " ".join(ex.trg)
        #file_path = "evals/{:02d}-{}.png".format(epoch, i)
        attn_ws, image = evaluateAndShowAttention(src, trg, seq2seq, in_proc, out_vocab,
                                                  out_file=None)
        """ tag regex: [^-\w\.] substitute to _
        => [-_.] + alphanumerics only.
        """
        tag = "{}--{}".format(i, src)
        writer.add_image(tag, image, global_step=epoch, dataformats='HWC')
        if print_attn:
            logger.nofmt(attn_ws.numpy().round(1))

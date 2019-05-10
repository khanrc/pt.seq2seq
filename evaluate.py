import random
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from dataset import TranslationDataset
from const import *
from logger import Logger


logger = Logger.get()


def idx2words(indices, lang):
    words = []
    for idx in indices:
        words.append(lang.idx2word[idx])
        if idx == EOS_idx:
            break
    return words


def random_eval(dset, seq2seq, N=3):
    seq2seq.eval()
    in_lang = dset.dataset.in_lang
    out_lang = dset.dataset.out_lang

    for i in range(N):
        src, src_len, tgt, tgt_len = random.choice(dset)
        src_sentence = ' '.join(idx2words(src, in_lang))
        tgt_sentence = ' '.join(idx2words(tgt, out_lang))

        src = torch.LongTensor(src).view(1, -1)
        src_len = torch.LongTensor([src_len]).view(1)
        # [1, max_len, out_lang.n_words]
        src = src.cuda()
        src_len = src_len.cuda()
        dec_outs, attn_ws = seq2seq.generate(src, src_len)
        topi = dec_outs.topk(1)[1] # [1, max_len, 1]
        out_words = idx2words(topi.squeeze(), out_lang)
        out_sentence = ' '.join(out_words)

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


def evaluateAndShowAttention(in_s, seq2seq, in_lang, out_lang, out_file):
    seq2seq.eval()
    src = TranslationDataset.to_ids(in_s, in_lang) + [EOS_idx]
    src_len = len(src)
    src = torch.LongTensor(src).view(1, -1).cuda()
    src_len = torch.tensor([src_len])
    dec_outs, attn_ws = seq2seq.generate(src, src_len)
    topi = dec_outs.topk(1)[1] # [1, max_len, 1]
    out_words = idx2words(topi.squeeze(), out_lang)

    logger.info("input = {}".format(in_s))
    logger.info("output = {}".format(' '.join(out_words)))
    attn_ws = attn_ws.squeeze().detach().cpu()[:len(out_words)]
    image = showAttention(in_s, out_words, attn_ws, out_file)
    return attn_ws, image


def evaluateAndShowAttentions(seq2seq, in_lang, out_lang, epoch, print_attn, writer):
    sens = [
        "elle a cinq ans de moins que moi .",
        "elle est trop petit .",
        "je ne crains pas de mourir .",
        "c est un jeune directeur plein de talent ."
    ]
    for i, s in enumerate(sens):
        #file_path = "evals/{:02d}-{}.png".format(epoch, i)
        attn_ws, image = evaluateAndShowAttention(s, seq2seq, in_lang, out_lang, out_file=None)
        """ tag regex: [^-\w\.] substitute to _
        즉, -_. + alphanumeric 만 가능.
        """
        tag = "{}--{}".format(i, s)
        writer.add_image(tag, image, global_step=epoch, dataformats='HWC')
        if print_attn:
            logger.nofmt(attn_ws.numpy().round(1))

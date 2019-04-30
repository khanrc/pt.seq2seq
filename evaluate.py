import torch
import data_prepare as dp
import matplotlib.pyplot as plt
from functools import partial
import random
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def idx2words(indices, lang):
    words = []
    for idx in indices:
        words.append(lang.idx2word[idx])
        if idx == dp.EOS_idx:
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

        print("> {}".format(src_sentence))
        print("= {}".format(tgt_sentence))
        print("< {}".format(out_sentence))
        print("")


def showAttention(input_sentence, output_words, attentions, file_path):
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

    #plt.show()
    plt.savefig(file_path)
    plt.close()


def evaluateAndShowAttention(in_s, seq2seq, in_lang, out_lang, out_file):
    seq2seq.eval()
    src = [in_lang.word2idx[word] for word in in_s.split(' ')] + [dp.EOS_idx]
    src_len = len(src)
    src = torch.LongTensor(src).view(1, -1).cuda()
    src_len = torch.LongTensor([src_len]).view(1).cuda()
    dec_outs, attn_ws = seq2seq.generate(src, src_len)
    topi = dec_outs.topk(1)[1] # [1, max_len, 1]
    out_words = idx2words(topi.squeeze(), out_lang)

    print('input =', in_s)
    print('output =', ' '.join(out_words))
    attn_ws = attn_ws.squeeze().detach().cpu()[:len(out_words)]
    showAttention(in_s, out_words, attn_ws, out_file)
    return attn_ws


def evaluateAndShowAttentions(seq2seq, in_lang, out_lang, epoch, print_attn=False):
    esa = partial(evaluateAndShowAttention, seq2seq=seq2seq, in_lang=in_lang,
                  out_lang=out_lang)
    sens = [
        "elle a cinq ans de moins que moi .",
        "elle est trop petit .",
        "je ne crains pas de mourir .",
        "c est un jeune directeur plein de talent ."
    ]
    for i, s in enumerate(sens):
        file_path = "evals/{:02d}-{}.png".format(epoch, i)
        attn_ws = esa(s, out_file=file_path)
        if print_attn:
            print(attn_ws.numpy().round(1))

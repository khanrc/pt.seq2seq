import os
import re
import pickle
import random
from torchtext.datasets import Multi30k, IWSLT, WMT14
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from functools import partial
import spacy
from const import *
from logger import Logger


logger = Logger.get()


def get_data(name, max_len, min_freq, batch_size, batch_sort, device='cuda'):
    # commons
    field_kwargs = {
        "init_token": SOS_token,
        "eos_token": EOS_token,
        "lower": True,
        "batch_first": True,
        "include_lengths": True
    }
    iterator_kwargs = {
        "batch_size": batch_size,
        "sort_within_batch": batch_sort,
        "sort_key": lambda x: len(x.src),
        "device": device
    }
    def length_filter(x):
        return len(x.src) <= max_len and len(x.trg) <= max_len

    if name == "org":
        # tokenizer
        import unicodedata

        # unicode => ascii
        # C 위에 점찍혀있다거나 그런 문자들 그냥 알파벳으로 바꿔줌
        def unicode2ascii(s):
            asc = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
            return ''.join(asc)

        # lowercase, trim, remove non-letters
        def normalize(s):
            s = unicode2ascii(s.lower().strip())
            # .!? 에 대해 띄어쓰기를 해줌. hi! => hi !
            s = re.sub(r"([.!?])", r" \1", s)
            # alphabet, .!? 이 아니면 공백으로 변경. hey11 ! => hey   !
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            return s.split()

        SRC = Field(tokenize=normalize, **field_kwargs)
        TRG = Field(tokenize=normalize, **field_kwargs)

        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )

        # 10/0 => smallest dataset. apply prefix filter
        if max_len == 10 and min_freq == 0:
            filter_pred = lambda x: length_filter(x) and " ".join(x.trg).startswith(eng_prefixes)
        else:
            filter_pred = length_filter

        dset = TabularDataset("./data/eng-fra.txt", format="tsv",
                              fields=[('trg', TRG), ('src', SRC)],
                              filter_pred=filter_pred)
        # information leakage ...
        SRC.build_vocab(dset, min_freq=min_freq)
        TRG.build_vocab(dset, min_freq=min_freq)

        random.seed(42)
        train_data, valid_data = dset.split(0.9, random_state=random.getstate())
        logger.debug("first train:", vars(train_data[0]))
        logger.info(f"# of total pairs = {len(dset)}")

        # iterator
        train_loader, valid_loader = BucketIterator.splits(
            (train_data, valid_data), **iterator_kwargs)
        test_loader = None

    else:
        datasets = {
            "multi30k": Multi30k,
            "iwslt": IWSLT,
            "wmt14": WMT14
        }[name]

        # tokenizer
        spacy_en = spacy.load('en')
        spacy_de = spacy.load('de')
        def tokenize_trg(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]
        def tokenize_src(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        # field (max_len+1 for SOS/EOS token)
        SRC = Field(tokenize=tokenize_src, **field_kwargs)
        TRG = Field(tokenize=tokenize_trg, **field_kwargs)

        # data
        train_data, valid_data, test_data = datasets.splits(
            exts=('.de', '.en'), fields=(SRC, TRG),
            filter_pred=length_filter)

        # vocab
        SRC.build_vocab(train_data, min_freq=min_freq)
        TRG.build_vocab(train_data, min_freq=min_freq)

        # iterator
        train_loader, valid_loader, test_loader = BucketIterator.splits(
            (train_data, valid_data, test_data), **iterator_kwargs)

    # log data
    logger.info(f"# of train pairs = {len(train_data)}")
    logger.info(f"# of valid pairs = {len(valid_data)}")
    if test_loader is not None:
        logger.info(f"# of test pairs = {len(test_data)}")

    in_dim = len(SRC.vocab)
    out_dim = len(TRG.vocab)
    logger.info(f"# of source vocab = {in_dim}")
    logger.info(f"# of target vocab = {out_dim}")
    SRC.init_token = None # remove <sos> in front of source words

    # special tokens test
    indices = [PAD_idx, UNK_idx, SOS_idx, EOS_idx]
    tokens = [PAD_token, UNK_token, SOS_token, EOS_token]
    for idx, token in zip(indices, tokens):
        assert SRC.vocab.itos[idx] == token
        assert SRC.vocab.stoi[token] == idx
        assert TRG.vocab.itos[idx] == token
        assert TRG.vocab.stoi[token] == idx

    return train_loader, valid_loader, test_loader


def get_max_len(name):
    import torch
    logger.info("Load {} dataset ...".format(name))
    train_loader, valid_loader, test_loader = get_data(name, 9999, 0, 1024, False, 'cpu')
    logger.info("Calc max len ...")
    srcs = torch.empty([0], dtype=torch.long)
    trgs = torch.empty([0], dtype=torch.long)

    for loader in [train_loader, valid_loader, test_loader]:
        if loader is None:
            continue
        for ex in loader:
            src, src_lens = ex.src
            trg, trg_lens = ex.trg

            srcs = torch.cat([srcs, src_lens])
            trgs = torch.cat([trgs, trg_lens])

            #  cur_src_max = src_lens.max().item()
            #  cur_trg_max = trg_lens.max().item()

            #  src_mlen = max(src_mlen, cur_src_max)
            #  trg_mlen = max(trg_mlen, cur_trg_max)

    n_src = srcs.shape[0]
    n_trg = trgs.shape[0]
    logger.info("# of lens = {}, {}".format(n_src, n_trg))

    srcs = srcs.sort()[0]
    trgs = trgs.sort()[0]
    collect = {}

    for ratio in [0., 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        p_src = int((n_src-1) * ratio)
        p_trg = int((n_trg-1) * ratio)
        ls = srcs[p_src]
        lt = trgs[p_trg]

        collect[ratio] = (ls, lt)

        logger.info("{:4.0%}: {}, {}".format(ratio, ls, lt))

    rec_ratio = 0.95
    rsrc, rtrg = collect[rec_ratio]
    rmax = max(rsrc, rtrg)
    logger.info("Recommend max_len is {}, for {:4.0%}: {}, {}".format(rmax, rec_ratio, rsrc, rtrg))

    print()
    return rmax


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    #  import logging
    #  logger.setLevel(logging.WARNING)

    if args.name == "all":
        # recommend max_len: [15, 23, 48]
        for name in ["org", "multi30k", "iwslt"]: #, "wmt14"]:
            get_max_len(name)
    else:
        get_max_len(args.name)


""" Dataset information
INFO::05/29 20:25:17 | Load org dataset ...
INFO::05/29 20:25:28 | # of total pairs = 135842
INFO::05/29 20:25:28 | # of train pairs = 122258
INFO::05/29 20:25:28 | # of valid pairs = 13584
INFO::05/29 20:25:28 | # of source vocab = 21335
INFO::05/29 20:25:28 | # of target vocab = 13044
INFO::05/29 20:25:28 | Calc max len ...
INFO::05/29 20:25:31 | # of lens = 135842, 135842
INFO::05/29 20:25:31 |   0%: 3, 4
INFO::05/29 20:25:31 |  50%: 9, 9
INFO::05/29 20:25:31 |  75%: 11, 11
INFO::05/29 20:25:31 |  90%: 13, 13
INFO::05/29 20:25:31 |  95%: 15, 14
INFO::05/29 20:25:31 |  99%: 19, 18
INFO::05/29 20:25:31 | 100%: 62, 52

INFO::05/29 20:25:31 | Load multi30k dataset ...
INFO::05/29 20:25:45 | # of train pairs = 29000
INFO::05/29 20:25:45 | # of valid pairs = 1014
INFO::05/29 20:25:45 | # of test pairs = 1000
INFO::05/29 20:25:45 | # of source vocab = 18660
INFO::05/29 20:25:45 | # of target vocab = 9799
INFO::05/29 20:25:45 | Calc max len ...
INFO::05/29 20:25:45 | # of lens = 31014, 31014
INFO::05/29 20:25:45 |   0%: 2, 6
INFO::05/29 20:25:45 |  50%: 13, 14
INFO::05/29 20:25:45 |  75%: 16, 17
INFO::05/29 20:25:45 |  90%: 19, 21
INFO::05/29 20:25:45 |  95%: 21, 23
INFO::05/29 20:25:45 |  99%: 26, 28
INFO::05/29 20:25:45 | 100%: 45, 43

INFO::05/29 20:25:46 | Load iwslt dataset ...
INFO::05/29 20:27:45 | # of train pairs = 196884
INFO::05/29 20:27:45 | # of valid pairs = 993
INFO::05/29 20:27:45 | # of test pairs = 1305
INFO::05/29 20:27:45 | # of source vocab = 125718
INFO::05/29 20:27:45 | # of target vocab = 50839
INFO::05/29 20:27:45 | Calc max len ...
INFO::05/29 20:28:01 | # of lens = 199182, 199182
INFO::05/29 20:28:01 |   0%: 2, 3
INFO::05/29 20:28:01 |  50%: 16, 19
INFO::05/29 20:28:01 |  75%: 25, 28
INFO::05/29 20:28:01 |  90%: 36, 39
INFO::05/29 20:28:01 |  95%: 44, 48
INFO::05/29 20:28:01 |  99%: 65, 70
INFO::05/29 20:28:01 | 100%: 764, 755

INFO::05/29 20:30:25 | Load wmt14 dataset ...
INFO::05/29 21:26:25 | # of train pairs = 4500966
INFO::05/29 21:26:25 | # of valid pairs = 3000
INFO::05/29 21:26:25 | # of test pairs = 3003
INFO::05/29 21:26:25 | # of source vocab = 29791
INFO::05/29 21:26:25 | # of target vocab = 29170
INFO::05/29 21:26:25 | Calc max len ...
INFO::05/29 21:34:39 | # of lens = 4506969, 4506969
INFO::05/29 21:34:40 |   0%: 2, 3
INFO::05/29 21:34:40 |  50%: 27, 28
INFO::05/29 21:34:40 |  75%: 39, 39
INFO::05/29 21:34:40 |  90%: 53, 53
INFO::05/29 21:34:40 |  95%: 63, 62
INFO::05/29 21:34:40 |  99%: 84, 82
INFO::05/29 21:34:40 | 100%: 338, 486
INFO::05/29 21:34:40 | Recommend max_len is 63, for  95%: 63, 62
"""

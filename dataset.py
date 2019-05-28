import os
import pickle
from torchtext.datasets import Multi30k, IWSLT, WMT14
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from functools import partial
import spacy
from const import *
from logger import Logger


logger = Logger.get()


def get_data(name, max_len, min_freq, batch_size, batch_sort):
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
        "device": "cuda"
    }
    def length_filter(x):
        return len(x.src) <= max_len and len(x.trg) <= max_len

    if name == "org":
        # tokenizer
        from data_prepare import normalize
        def tokenize(text):
            return normalize(text).split(' ')

        SRC = Field(tokenize=tokenize, **field_kwargs)
        TRG = Field(tokenize=tokenize, **field_kwargs)

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

        random_state_cache = "cache/org-random-state.pkl"
        if os.path.exists(random_state_cache):
            random_state = pickle.load(open(random_state_cache, 'rb'))
        else:
            random_state = random.getstate()
            pickle.dump(random_state, open(random_state_cache, 'wb'))

        train_data, valid_data = dset.split(0.9, random_state=random_state)
        logger.debug("first train:", vars(train_data[0]))
        logger.info(f"# of total pairs = {len(dset)}")

        # iterator
        train_loader, valid_loader = BucketIterator.splits(
            (train_data, valid_data), **iterator_kwargs)
        test_loader = None

    else:
        #if name == "multi30k":
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

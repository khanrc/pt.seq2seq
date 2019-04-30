import unicodedata
import re
import random
import numpy as np
from lang import Lang
import utils
import pickle
from functools import partial
from collections import Counter


PAD_idx = 0
SOS_idx = 1
EOS_idx = 2


# 데이터셋에는 13만개의 eng-fra pair 가 있는데,
# 이 튜토리얼에서는 아래의 prefix 로 시작하고, 최대 길이가 10 이하인 페어들만 사용한다.
MAX_LENGTH = 14
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


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
    return s


def read_langs(lang1, lang2, reverse=False):
    """
    reverse == False, input [tap] output pair.
    reverse == True, output [tap] input pair.
    """
    print("Reading lines ...")

    file_path = 'data/{}-{}.txt'.format(lang1, lang2)
    #lines = open(file_path).readlines()
    pairs = []
    for line in open(file_path):
        pair = [normalize(s) for s in line.strip().split('\t')]
        if reverse:
            pair = list(reversed(pair))
        pairs.append(pair)

    if reverse:
        lang1, lang2 = lang2, lang1

    return Lang(lang1), Lang(lang2), pairs


def filter_len(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH


def filter_eng_prefix(pair):
    # assume that pair[1] is eng.
    return pair[1].startswith(eng_prefixes)


def prepare_data():
    """
    reverse: if true, lang2 => input lang, lang1 => output lang.

    1. read sentence pairs
    2. filter the pairs
    3. construct langs from the pairs
    """
    # ready pairs
    input_lang, output_lang, pairs = read_langs("eng", "fra", True)
    print(f"Read {len(pairs)} sentence pairs")

    # filter pairs
    pairs = filter(filter_len, pairs)
    #pairs = filter(filter_eng_prefix, pairs)
    pairs = list(pairs)
    print(f"Trimmed to {len(pairs)} sentece pairs")

    # counting words
    print("Counting words ...")
    counter1 = Counter()
    counter2 = Counter()
    for pair in pairs:
        counter1.update(pair[0].split(' '))
        counter2.update(pair[1].split(' '))
    print("Lang1 most common: ", counter1.most_common(10))
    print("Lang2 most common: ", counter2.most_common(10))

    c1 = list(filter(lambda x: x[1] >= 2, counter1.items()))
    c2 = list(filter(lambda x: x[1] >= 2, counter2.items()))

    print(f"#Lang1: {len(counter1)} => {len(c1)}")
    print(f"#Lang2: {len(counter2)} => {len(c2)}")

    print("Making word dictionary ...")
    max_len = 0
    for pair in pairs:

        l1 = len(pair[0].split(' '))
        l2 = len(pair[1].split(' '))
        if max_len < l1:
            max_len = l1
        if max_len < l2:
            max_len = l2
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print(f"Max len = {max_len}")

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print("Caching ... ")
    N = len(pairs)
    utils.makedirs("cache")
    input_lang.dump_to_file("cache/in-{}-{}.pkl".format(input_lang.name, N))
    output_lang.dump_to_file("cache/out-{}-{}.pkl".format(output_lang.name, N))
    path = "cache/{}2{}-{}.pkl".format(input_lang.name, output_lang.name, N)
    pickle.dump(pairs, open(path, "wb"))
    print(f"pairs dumped to {path}")

    return input_lang, output_lang, pairs


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepare_data()
    print(random.choice(pairs))

    # gen validation indices
    N = len(pairs)
    N_valid = int(N * 0.1)
    valid_indices = np.random.choice(N, N_valid)
    path = "valid_indices.npy"
    np.save(path, valid_indices)
    print(f"Valid indices dumped to {path} -- {N_valid}/{N}")

import unicodedata
import re
import random
import numpy as np
from lang import Lang
import utils
import pickle
from functools import partial
from collections import Counter


# 데이터셋에는 13만개의 eng-fra pair 가 있는데,
# 이 튜토리얼에서는 아래의 prefix 로 시작하고, 최대 길이가 10 이하인 페어들만 사용한다.
#MAX_LENGTH = 14
#  eng_prefixes = (
#      "i am ", "i m ",
#      "he is", "he s ",
#      "she is", "she s ",
#      "you are", "you re ",
#      "we are", "we re ",
#      "they are", "they re "
#  )


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


def filter_len(pair, max_len):
    return len(pair[0].split(' ')) <= max_len and len(pair[1].split(' ')) <= max_len 


def filter_eng_prefix(pair):
    # assume that pair[1] is eng.
    return pair[1].startswith(eng_prefixes)


def prepare(max_len, min_freq):
    """ Prepare dataset for ENG to FRA.
    1. read sentence pairs
    2. filter the pairs
    3. construct langs from the pairs
    """
    print(f"Generate cache for max_len={max_len}, min_freq={min_freq}")
    # ready pairs
    input_lang, output_lang, pairs = read_langs("eng", "fra", True)
    print(f"Read {len(pairs)} sentence pairs")

    # filter pairs
    pairs = filter(lambda p: filter_len(p, max_len), pairs)
    #pairs = filter(filter_eng_prefix, pairs)
    pairs = list(pairs)
    print(f"Trimmed to {len(pairs)} sentece pairs")

    print("Making word dictionary ...")
    mlen = 0
    for pair in pairs:
        l1 = len(pair[0].split(' '))
        l2 = len(pair[1].split(' '))
        if mlen < max(l1, l2):
            mlen = max(l1, l2)
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print(f"Real max len = {mlen}")

    print("Counted words: {}={}, {}={}", format(
        input_lang.name, input_lang.n_words, output_lang.name, output_lang.n_words))

    print(f"Cut by min_freq = {min_freq}:")
    input_lang.cut_freq(min_freq)
    output_lang.cut_freq(min_freq)
    print("Counted words: {}={}, {}={}", format(
        input_lang.name, input_lang.n_words, output_lang.name, output_lang.n_words))

    print("Caching ... ")
    utils.makedirs("cache")
    input_lang.dump_to_file("cache/in-{}-{}-{}.pkl".format(input_lang.name, max_len, min_freq))
    output_lang.dump_to_file("cache/out-{}-{}-{}.pkl".format(output_lang.name, max_len, min_freq))
    path = "cache/{}2{}-{}.pkl".format(input_lang.name, output_lang.name, max_len)
    pickle.dump(pairs, open(path, "wb"))
    print(f"pairs dumped to {path}")

    return input_lang, output_lang, pairs


def gen_valid_indices(N, ratio, path):
    N_valid = int(N * ratio)
    valid_indices = np.random.choice(N, N_valid)
    np.save(path, valid_indices)
    print(f"Valid indices dumped to {path} -- {N_valid}/{N}")


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepare(max_len=14, min_freq=2)
    print(random.choice(pairs))

    # gen validation indices
    path = "valid_indices.npy"
    if not os.path.exists(path):
        N = len(pairs)
        gen_valid_indices(N, 0.1, path)

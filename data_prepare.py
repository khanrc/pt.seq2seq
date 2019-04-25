import unicodedata
import re
import random


PAD_token = 0
SOS_token = 1
EOS_token = 2

# 데이터셋에는 13만개의 eng-fra pair 가 있는데,
# 이 튜토리얼에서는 아래의 prefix 로 시작하고, 최대 길이가 10 이하인 페어들만 사용한다.
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.word2cnt = {} # count word frequency
        self.idx2word = ["PAD", "SOS", "EOS"]
        self.n_words = 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2cnt[word] = 1
            self.idx2word.append(word)
            self.n_words += 1
        else:
            self.word2cnt[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


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


def filter_pair(pair):
    # assume that pair[1] is eng.
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and \
            pair[1].startswith(eng_prefixes)


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = list(filter(filter_pair, pairs))
    print(f"Trimmed to {len(pairs)} sentece pairs")
    print("Counting words ...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    print(random.choice(pairs))

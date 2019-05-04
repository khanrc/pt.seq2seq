import pickle


default_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]


class Lang:
    """ Language dictionary """
    def __init__(self, name, word2idx=None, wordcounter=None, idx2word=None):
        self.name = name
        self.word2idx = {} if word2idx is None else word2idx
        self.wordcounter = {} if wordcounter is None else wordcounter
        self.idx2word = [] if idx2word is None else idx2word
        self.n_words = len(self.idx2word)

        if not self.word2idx and not self.idx2word:
            for token in default_tokens:
                self.add_word(token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.wordcounter[word] = 1
            self.idx2word.append(word)
            self.n_words += 1
        else:
            self.wordcounter[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def cut_freq(self, min_freq):
        for word in list(self.word2idx.keys()):
            if word in default_tokens:
                continue
            if self.wordcounter[word] < min_freq:
                self.wordcounter.pop(word)
                self.idx2word.remove(word)
        # re-set word2idx
        self.word2idx = {}
        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx
        self.n_words = len(self.idx2word)


    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.wordcounter, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, name, path):
        print('loading dictionary from %s' % path)
        word2idx, wordcounter, idx2word = pickle.load(open(path, 'rb'))
        wd = cls(name, word2idx, wordcounter, idx2word)
        return wd

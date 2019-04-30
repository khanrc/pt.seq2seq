import pickle


class Lang:
    """ Language dictionary """
    def __init__(self, name, word2idx=None, idx2word=None):
        self.name = name
        self.word2idx = {} if word2idx is None else word2idx
        self.idx2word = [] if idx2word is None else idx2word
        self.n_words = len(self.idx2word)

        if not self.word2idx and not self.idx2word:
            self.add_sentence("[PAD] [SOS] [EOS]")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word.append(word)
            self.n_words += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, name, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        wd = cls(name, word2idx, idx2word)
        return wd

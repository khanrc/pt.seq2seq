import numpy as np
from collections import Counter


class BLEU:
    """ Calc BLEU score module.
    Ref: https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py
    """
    def __init__(self, N=4, weights=None):
        self.N = N
        self.weights = weights

        # stats[0] = (hyp_nwords, ref_nwords)
        # stats[n] = (hyp_n-gram, ref_n-gram)
        self.stats = np.zeros([N+1, 2], dtype=np.int)

    def add_sentence(self, hypothesis, reference):
        self.stats += self.compute_stats(hypothesis, reference)

    def add_corpus(self, hypotheses, references):
        for hyp, ref in zip(hypotheses, references):
            self.add_sentence(hyp, ref)

    def score(self, N=None):
        """ Get BLEU score.
        if N is given, calc BLEU-N. (default: self.N)
        """
        if N is None:
            N = self.N
        assert 1 <= N <= self.N
        return self.compute_bleu(self.stats[:N+1], weights=self.weights) * 100.

    @staticmethod
    def compute_stats(hypothesis, reference, N=4):
        """
        hypothesis: list of words. ['this', 'is', 'a', 'book']
        reference: list of words. ['this', 'is', 'a', 'desk']

        return: stats; [N+1, 2]
        """
        stats = np.zeros([N+1, 2], dtype=np.int)
        n_hyp = len(hypothesis)
        n_ref = len(reference)
        stats[0] = (n_hyp, n_ref)
        max_n = min(N, n_hyp)
        for n in range(1, max_n+1):
            hyp_ngram = Counter(tuple(hypothesis[i:i+n]) for i in range(n_hyp+1-n))
            ref_ngram = Counter(tuple(reference[i:i+n]) for i in range(n_ref+1-n))
            matches = hyp_ngram & ref_ngram
            n_matches = sum(matches.values()) # total match counts
            stats[n] = (n_matches, n_hyp+1-n)

        return stats

    @staticmethod
    def compute_bleu(stats, weights=None):
        """
        from paper,
        log BLEU = min(1-r/c, 0) + sum(w[n]*log(p[n]))
        where:
            r: n_ref
            c: n_hyp
            p[n]: n-gram n_matches
            w[n]: n-gram weights
        """
        if (stats == 0.).any():
            return 0.

        log_prec = np.log(stats[1:, 0] / stats[1:, 1])
        log_prec = np.average(log_prec, weights=weights)
        # brevity penalty
        c, r = stats[0]
        BP = min(1-r/c, 0)

        return np.exp(BP + log_prec)

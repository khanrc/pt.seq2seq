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


if __name__ == "__main__":
    # org codes for test
    import math
    def bleu_stats(hypothesis, reference):
        """Compute statistics for BLEU."""
        stats = []
        stats.append(len(hypothesis))
        stats.append(len(reference))
        for n in range(1, 5):
            s_ngrams = Counter(
                [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
            )
            r_ngrams = Counter(
                [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
            )
            stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
            stats.append(max([len(hypothesis) + 1 - n, 0]))
        return stats

    def bleu(stats):
        """Compute BLEU given n-gram statistics."""
        if len(list(filter(lambda x: x == 0, stats))) > 0:
            return 0
        (c, r) = stats[:2]
        log_bleu_prec = sum(
            [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
        ) / 4.
        return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

    def get_bleu(hypotheses, reference):
        """Get validation BLEU score for dev set."""
        stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for hyp, ref in zip(hypotheses, reference):
            stats += np.array(bleu_stats(hyp, ref))
        return 100 * bleu(stats)
    ##############################################

    hyps = [
        ['i', 'am', 'a', 'boy', 'and', 'test'],
        ['this', 'is', 'the', 'game', 'of', 'the', 'throne'],
        ['hyp', 'is', 'long', 'and', 'this', 'is', 'the', 'game', 'of', 'the', 'throne'],
        ['what', 'the', 'fuck', 'this', 'hmm'],
        ['the', 'short']
    ]
    refses = [
        ['i', 'am', 'a', 'boy', 'and', 'girl', 'and', 'long'],
        ['i', 'like', 'this', 'is', 'the', 'game', 'of', 'the', 'throne'],
        ['this', 'is', 'the', 'game', 'of', 'the', 'throne'],
        ['what', 'a', 'fucking', 'serious', '?'],
        ['too', 'short', 'lang']
    ]

    for hyp, refs in zip(hyps, refses):
        # stats
        mine_stats = BLEU.compute_stats(hyp, refs)
        org_stats = bleu_stats(hyp, refs)
        assert (mine_stats.flatten().astype(np.int) == org_stats).all()

        # bleu
        mine_bleu = BLEU.compute_bleu(mine_stats)
        org_bleu = bleu(org_stats)

        #print(mine_bleu, org_bleu)
        assert mine_bleu == org_bleu

    # total bleu score
    org = get_bleu(hyps, refses)
    bleu = BLEU()
    bleu.add_corpus(hyps, refses)
    print("org:", org)
    print("mine:", bleu.score())
    assert org == bleu.score()

    print("All test passed !")

    print("Multi BLEU:")
    print("BLEU-1:", bleu.score(1))
    print("BLEU-2:", bleu.score(2))
    print("BLEU-3:", bleu.score(3))
    print("BLEU-4:", bleu.score(4))

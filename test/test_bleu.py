import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from bleu import BLEU
import math
import re
import subprocess
import tempfile
import numpy as np
from six.moves import urllib
from collections import Counter


## Ref funcs
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


## moses BLEU
def get_moses_multi_bleu(hypotheses, references, lowercase=False):
    """Get the BLEU score using the moses `multi-bleu.perl` script.

    **Script:**
    https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

    Args:
      hypotheses (list of str): List of predicted values
      references (list of str): List of target values
      lowercase (bool): If true, pass the "-lc" flag to the `multi-bleu.perl` script

    Returns:
      (:class:`np.float32`) The BLEU score as a float32 value.

    Example:

      >>> hypotheses = [
      ...   "The brown fox jumps over the dog 笑",
      ...   "The brown fox jumps over the dog 2 笑"
      ... ]
      >>> references = [
      ...   "The quick brown fox jumps over the lazy dog 笑",
      ...   "The quick brown fox jumps over the lazy dog 笑"
      ... ]
      >>> get_moses_multi_bleu(hypotheses, references, lowercase=True)
      46.51
    """
    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses)
    if isinstance(references, list):
        references = np.array(references)

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")
        os.chmod(multi_bleu_path, 0o755)
    except:
        print("Unable to fetch multi-bleu.perl script")
        return None

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
            bleu_score = np.float32(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
            bleu_score = None

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return bleu_score


if __name__ == "__main__":
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

    m_hyps = [" ".join(hyp) for hyp in hyps]
    m_refs = [" ".join(ref) for ref in refses]
    bleu_score = get_moses_multi_bleu(m_hyps, m_refs)
    print("moses:", bleu_score)
    assert round(float(bleu_score), 2) == round(bleu.score(), 2)

    print("All test passed !")

    print("Multi BLEU:")
    print("BLEU-1:", bleu.score(1))
    print("BLEU-2:", bleu.score(2))
    print("BLEU-3:", bleu.score(3))
    print("BLEU-4:", bleu.score(4))

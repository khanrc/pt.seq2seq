# Seq2seq

PyTorch implementations of seq2seq models for Neural Machine Translation (NMT) task:

- seq2seq (RNN)
- seq2seq with attention (RNN + attention)
- ConvS2S
- Transformer
- DynamicConv (+ LightConv)

## No-torchtext version

Please refer to [no-torchtext](https://github.com/khanrc/seq2seq/tree/no-torchtext) tag.
In this version, the `dataset.py`, `lang.py` and `data_prepare.py` structuralize low-level text to
make it easier to use in the training code.

## Supporting datasets

Supporting datasets include pytorch tutorial ENG to FRA translation dataset and torchtext NMT datasets.

- `org`: ENG to FRA translation from pytorch tutorial
    - To use this data, please download dataset from https://download.pytorch.org/tutorial/data.zip first.
- `multi30k`
- `iwslt`
- `wmt14`

## Dependencies

- python3
- pyyaml
- pytorch >= 1.10
- tensorboard >= 1.14
- torchtext
- spacy
    - `python -m spacy download en`
    - `python -m spacy download de`

## Results

Hparams:

- Task & data: ENG to FRA translation task, max\_len=14, min\_freq=2.

Models:

| Model | Loss (sum) | PPL | BLEU\* | Note |
| - | - | - | - | - |
| Seq2Seq                       | 15.11 | 6.320 | | |
| Seq2Seq + KV attn             | 13.57 | 5.244 | 64.10 | |
| Seq2Seq + Additive attn       | 13.28 | 5.054 | 64.48 | |
| Seq2Seq + Multiplicative attn | 14.01 | 5.526 | | |
| ConvS2S                       | 13.06 | 4.931 | 61.62 | |
| ConvS2S + out-caching         | 12.44 | 4.572 | 60.90 | |
| Transformer-init              | 12.73 | 4.675 | 66.38 | |
| LightConv                     | 12.29 | 4.493 | | K=[3,3,5,5,7,7] |
| DynamicConv                   | 11.81 | 4.237 | 68.35 | K=[3,3,5,5,7,7] |

- [!] BLEU is recorded in different run
- PPL and BLEU does not match
- about the Transformer
    - after-norm does not work; should use before-norm.
    - LR warmup and xavier init is important for the performance

## ToDo

- Beam search
- Word tokenization
    - BPE
    - Word piece model

## References

- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/pytorch/fairseq

# Seq2Seq with Attention

## Run

1. Download dataset from https://download.pytorch.org/tutorial/data.zip
2. Run: `python train.py`

Etc:

- Hyperparams are in `config.yaml`.

## Dependencies

- python3
- pyyaml
- pytorch >= 1.10
- tensorboard >= 1.14
    - `pip install tb-nightly future`

## Results

Hparams:

- Task & data: ENG to FRA translation task, max\_len=14, min\_freq=2.
- RNN: BiGRU, B=256, h\_dim=1024, emb\_dim=300, L=1, dropout=0.1.
    - L=1, dropout=0.1 is best in variations.

Models:

| Model | Loss (sum) | PPL | BLEU\* | Note |
| - | - | - | - |
| Seq2Seq                       | 15.11 | 6.320 | | |
| Seq2Seq + KV attn             | 13.57 | 5.244 | 64.10 | |
| Seq2Seq + Additive attn       | 13.28 | 5.054 | 64.48 | |
| Seq2Seq + Multiplicative attn | 14.01 | 5.526 | | |
| ConvS2S                       | 13.06 | 4.931 | | |
| ConvS2S + out-caching         | 12.44 | 4.572 | | |
| Transformer                   | 13.79 | 5.400 | 64.87 | something wrong? |
| Transformer-Big               | 16.14 | 7.203 | | |
| LightConv                     | 12.29 | 4.493 | | K=[3,3,5,5,7,7] |
| DynamicConv                   | 11.81 | 4.237 | 68.35 | K=[3,3,5,5,7,7] |
| AAN (Averaged Attention)      | | | | |

- Gradual Warmup does not helps
- [!] BLEU is recorded in different run

## ToDo

- torchtext (or torchnlp)
    - Other dataset
- Beam search
- Add pre-trained word embeddings?
- Word tokenization
    - BPE
    - Word piece model
- AAN?

## References

- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/pytorch/fairseq

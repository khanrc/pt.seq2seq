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

| Model | Loss (sum) | PPL | Note |
| - | - | - | - |
| Seq2Seq                       | 15.11 | 6.320 | |
| Seq2Seq + KV attn             | 13.57 | 5.244 | |
| Seq2Seq + Additive attn       | 13.28 | 5.054 | |
| Seq2Seq + Multiplicative attn | 14.01 | 5.526 | |
| SelfAttnS2S                   | | | Skip |
| ConvS2S                       | 13.09 | 4.951 | |
| ConvS2S + out-caching         | 12.53 | 4.629 | |
| Transformer                   | | | |
| LightConv                     | | | |
| DynamicConv                   | | | |

## ToDo

- Torchtext?
- Self-attention
- Transformer
- Add pre-trained word embeddings
- Other dataset
- ETC
    - TB model graph
    - LightConv & DynamicConv

## References

- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- https://github.com/bentrevett/pytorch-seq2seq

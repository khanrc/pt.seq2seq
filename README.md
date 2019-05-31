# Seq2seq

Neural Machine Translation (NMT) models:

- seq2seq (RNN)
- seq2seq with attention (RNN + attention)
- ConvS2S
- Transformer
- DynamicConv (+ LightConv)

## Run

1. Download dataset from https://download.pytorch.org/tutorial/data.zip
2. Run: `python train.py config.yaml name`

You can use multiple config.yaml (following config file overwrite if duplicate key exists) and direct config update in command line.

```
# direct config update supports leaf-key only update
python train.py cfgs/model/transformer.yaml cfgs/data/iwslt.yaml transformer-iwslt --train.warmup 4 --epochs 20
```

## Dependencies

- python3
- pyyaml
- pytorch >= 1.10
- tensorboard >= 1.14
    - `pip install tb-nightly future`
- torchtext
- spacy
    - `python -m spacy download en`
    - `python -m spacy download de`

Optionals: fire (for tb cleanup)

```
fire==0.1.3
numpy==1.16.2
PyYAML==5.1
spacy==2.1.4
tb-nightly==1.14.0a20190528
torch==1.1.0
torchtext==0.4.0
```

## Results

Hparams:

- Task & data: ENG to FRA translation task, max\_len=14, min\_freq=2.
- RNN: BiGRU, B=256, h\_dim=1024, emb\_dim=300, L=1, dropout=0.1.
    - L=1, dropout=0.1 is best in variations.

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
| AAN (Averaged Attention)      | | | | |

- [!] BLEU is recorded in different run
- PPL and BLEU does not match ...
- Transformer
    - after-norm does not work; should use before-norm.
    - LR warmup and xavier init is important for performance

## ToDo

- Beam search
- Add pre-trained word embeddings?
- Word tokenization
    - BPE
    - Word piece model
- Models
    - AAN
    - Levenshtein

## References

- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/pytorch/fairseq

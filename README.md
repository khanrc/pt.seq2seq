# Seq2Seq with Attention

## Run

1. Download dataset from https://download.pytorch.org/tutorial/data.zip
2. Prepare validation indices by `python data_prepare.py`
3. Run: `python train.py`

Etc:

- Hyperparams are in `train.py`.

## Results

Hparams:

eng to fra task, max\_len=14, min\_freq=2.
BiGRU-encoder, B=256, h\_dim=1024, emb\_dim=300.

Models:

- Seq2Seq: 14.8
- Seq2Seq + KVAttention: 13.5
    - AdditiveAttention: 13.1
    - MultiplicativeAttention: 14.1

## ToDo

- Torchtext
- Self-attention
- Transformer
- ConvS2S
- Config
- Add pre-trained word embeddings
- Other dataset

## References

- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- https://github.com/bentrevett/pytorch-seq2seq

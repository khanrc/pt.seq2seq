import torch
from torch.utils.data import Dataset
import numpy as np
import data_prepare as dp


class TranslationDataset(Dataset):
    def __init__(self):
        input_lang, output_lang, pairs = dp.prepare_data('eng', 'fra', True)
        self.in_lang = input_lang
        self.out_lang = output_lang
        self.pairs = pairs

    def to_ids(self, s, lang):
        return [lang.word2idx[token] for token in s.split(' ')]

    def __getitem__(self, index):
        pair = self.pairs[index]

        ids1 = self.to_ids(pair[0], self.in_lang) + [dp.EOS_token]
        ids2 = self.to_ids(pair[1], self.out_lang) + [dp.EOS_token]

        return ids1, len(ids1), ids2, len(ids2)

    def __len__(self):
        return len(self.pairs)


def collate_data(batch):
    B = len(batch)
    #  lens = [(len(src), len(tgt)) for src, tgt in batch]
    #  src_lens, tgt_lens = list(zip(*lens))
    batch = sorted(batch, key=lambda x: x[1], reverse=True) # sort by len(src)
    _, src_lens, _, tgt_lens = zip(*batch)

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_batch = np.full((B, max_src), dp.PAD_token, dtype=np.long)
    tgt_batch = np.full((B, max_tgt), dp.PAD_token, dtype=np.long)

    for i, b in enumerate(batch):
        src_batch[i, :b[1]] = b[0]
        tgt_batch[i, :b[3]] = b[2]

    return (
        torch.from_numpy(src_batch),
        torch.LongTensor(src_lens),
        torch.from_numpy(tgt_batch),
        torch.LongTensor(tgt_lens)
    )


if __name__ == "__main__":
    # test
    dset = TranslationDataset()
    print("dset:")
    for i in range(4):
        print(dset[i])

    from torch.utils.data import DataLoader
    loader = DataLoader(dset, batch_size=32, collate_fn=collate_data, shuffle=True)
    print("Loader (B=4):")
    for i, b in enumerate(loader):
        print(i)
        print(b)
        if i == 5:
            break

    print("Generate validation indices ...")
    N = len(dset)
    valid_ratio = 0.1
    N_valid = int(N * valid_ratio)
    print(f"{N_valid} data points will be used for validation")
    valid_indices = np.random.choice(N, N_valid)
    np.save("valid_indices.npy", valid_indices)
    import pdb; pdb.set_trace()

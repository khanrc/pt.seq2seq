import torch
from torch.utils.data import Dataset
import numpy as np
import data_prepare as dp


class TranslationDataset(Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        #input_lang, output_lang, pairs = dp.prepare_data('eng', 'fra', True)
        self.in_lang = input_lang
        self.out_lang = output_lang
        self.pairs = pairs

    def to_ids(self, s, lang):
        return [lang.word2idx[token] for token in s.split(' ')]

    def __getitem__(self, index):
        pair = self.pairs[index]

        ids1 = self.to_ids(pair[0], self.in_lang) + [dp.EOS_idx]
        ids2 = self.to_ids(pair[1], self.out_lang) + [dp.EOS_idx]
        # assume that PAD_idx == 0
        ids1_np = np.zeros(dp.MAX_LENGTH, dtype=np.long)
        ids1_np[:len(ids1)] = ids1
        ids2_np = np.zeros(dp.MAX_LENGTH, dtype=np.long)
        ids2_np[:len(ids2)] = ids2

        return (
            torch.from_numpy(ids1_np), torch.tensor(len(ids1)),
            torch.from_numpy(ids2_np), torch.tensor(len(ids2))
        )

    def __len__(self):
        return len(self.pairs)


def collate_data(batch):
    batch.sort(key=lambda b: b[1], reverse=True)
    return [torch.stack(b) for b in zip(*batch)]


if __name__ == "__main__":
    pass
    # test
    #  dset = TranslationDataset()
    #  print("dset:")
    #  for i in range(4):
    #      print(dset[i])

    #  from torch.utils.data import DataLoader
    #  loader = DataLoader(dset, batch_size=32, collate_fn=collate_data, shuffle=True)
    #  print("Loader (B=4):")
    #  for i, b in enumerate(loader):
    #      print(i)
    #      print(b)
    #      if i == 5:
    #          break

    #  print("Generate validation indices ...")
    #  N = len(dset)
    #  valid_ratio = 0.1
    #  N_valid = int(N * valid_ratio)
    #  print(f"{N_valid} data points will be used for validation")
    #  valid_indices = np.random.choice(N, N_valid)
    #  np.save("valid_indices.npy", valid_indices)
    #  import pdb; pdb.set_trace()

import torch
import torch.utils.data

import numpy as np


class CelebALandmarks(torch.utils.data.Dataset):
    def __init__(self, filename='datasets/landmarks_in_bytes.txt'):
        with open(filename, 'r', encoding='latin1') as f:
            data_in_bytes = f.read().encode('latin1')
            data = np.frombuffer(data_in_bytes, dtype=np.float32)
            self.data = data.reshape((-1, 478 * 2))
        del data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def __bytes__(self):
        return self.data.tobytes().decode('latin1')

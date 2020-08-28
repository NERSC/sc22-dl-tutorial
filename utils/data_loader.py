import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))


def get_data_loader_distributed(params, world_rank):
    dataset = RandomJunkDataset(params)
    train_loader = DataLoader(dataset,
                              batch_size=params.batch_size,
                              num_workers=params.num_data_workers,
                              worker_init_fn=worker_init,
                              pin_memory=torch.cuda.is_available())
    return train_loader



class RandomJunkDataset(Dataset):
    """Random crops"""
    def __init__(self, params):
        self.length = params.box_size
        self.size = params.data_size
        self.RandInp = np.random.normal(size=(4, self.length, self.length, self.length)).astype(np.float32)
        self.RandTar = np.random.normal(size=(5, self.length, self.length, self.length)).astype(np.float32)
        self.Nsamples = params.Nsamples
        self.rotate = RandomRotator()

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        x = np.random.randint(low=0, high=self.length-self.size)
        y = np.random.randint(low=0, high=self.length-self.size)
        z = np.random.randint(low=0, high=self.length-self.size)
        inp = self.RandInp[:, x:x+self.size, y:y+self.size, z:z+self.size]
        tar = self.RandTar[:, x:x+self.size, y:y+self.size, z:z+self.size]
        rand = np.random.randint(low=1, high=25)
        inp = np.copy(self.rotate(inp, rand))
        tar = np.copy(self.rotate(tar, rand))
        return torch.as_tensor(inp), torch.as_tensor(tar)


class RandomRotator(object):
    """Composable transform that applies random 3D rotations by right angles.
       Adapted from tf code:
       https://github.com/doogesh/halo_painting/blob/master/wasserstein_halo_mapping_network.ipynb"""

    def __init__(self):
        self.rot = {1:  lambda x: x[:, ::-1, ::-1, :],
                    2:  lambda x: x[:, ::-1, :, ::-1],
                    3:  lambda x: x[:, :, ::-1, ::-1],
                    4:  lambda x: x.transpose([0, 2, 1, 3])[:, ::-1, :, :],
                    5:  lambda x: x.transpose([0, 2, 1, 3])[:, ::-1, :, ::-1],
                    6:  lambda x: x.transpose([0, 2, 1, 3])[:, :, ::-1, :],
                    7:  lambda x: x.transpose([0, 2, 1, 3])[:, :, ::-1, ::-1],
                    8:  lambda x: x.transpose([0, 3, 2, 1])[:, ::-1, :, :],
                    9:  lambda x: x.transpose([0, 3, 2, 1])[:, ::-1, ::-1, :],
                    10: lambda x: x.transpose([0, 3, 2, 1])[:, :, :, ::-1],
                    11: lambda x: x.transpose([0, 3, 2, 1])[:, :, ::-1, ::-1],
                    12: lambda x: x.transpose([0, 1, 3, 2])[:, :, ::-1, :],
                    13: lambda x: x.transpose([0, 1, 3, 2])[:, ::-1, ::-1, :],
                    14: lambda x: x.transpose([0, 1, 3, 2])[:, :, :, ::-1],
                    15: lambda x: x.transpose([0, 1, 3, 2])[:, ::-1, :, ::-1],
                    16: lambda x: x.transpose([0, 2, 3, 1])[:, ::-1, ::-1, :],
                    17: lambda x: x.transpose([0, 2, 3, 1])[:, :, ::-1, ::-1],
                    18: lambda x: x.transpose([0, 2, 3, 1])[:, ::-1, :, ::-1],
                    19: lambda x: x.transpose([0, 2, 3, 1])[:, ::-1, ::-1, ::-1],
                    20: lambda x: x.transpose([0, 3, 1, 2])[:, ::-1, ::-1, :],
                    21: lambda x: x.transpose([0, 3, 1, 2])[:, ::-1, :, ::-1],
                    22: lambda x: x.transpose([0, 3, 1, 2])[:, :, ::-1, ::-1],
                    23: lambda x: x.transpose([0, 3, 1, 2])[:, ::-1, ::-1, ::-1],
                    24: lambda x: x}

    def __call__(self, x, rand):
        return self.rot[rand](x)

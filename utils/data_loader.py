import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py
import os


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))


def get_data_loader_distributed(params, world_rank):

    if params.data_loader_config == 'synthetic':
        train_data =  RandomJunkDataset(params)
        val_data = train_data
    else:
        train_data, val_data = RandomCropDataset(params, validation=False), RandomCropDataset(params, validation=True)

    train_loader = DataLoader(train_data,
                              batch_size=params.batch_size,
                              num_workers=params.num_data_workers,
                              worker_init_fn=worker_init,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_data,
                              batch_size=params.batch_size,
                              num_workers=params.num_data_workers,
                              worker_init_fn=worker_init,
                              pin_memory=torch.cuda.is_available())

    return train_loader, val_loader


class RandomJunkDataset(Dataset):
    """
    Random crops
    Synthetic data is just random numbers, but runtime I/O pattern of in-memory data loading is matched exactly
    """
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


class RandomCropDataset(Dataset):
    """
    Random crops
    "inmem" config: Load entire dataset into memory, randomly crop samples for training
    "lowmem" config: Crop samples from disk
    """
    def __init__(self, params, validation=False):
        
        self.fname = params.train_path if not validation else params.val_path
        if params.use_cache:
            self.fname = os.path.join(params.use_cache, os.path.basename(self.fname))
        self.length = params.box_size
        self.size = params.data_size
        self.Nsamples = params.Nsamples if not validation else params.Nsamples_val
        self.rotate = RandomRotator()
        self.inmem = params.data_loader_config == 'inmem'
        
        self.inp_buff = np.zeros((4, self.size, self.size, self.size), dtype=np.float32)
        self.tar_buff = np.zeros((5, self.size, self.size, self.size), dtype=np.float32)
        if self.inmem:
            with h5py.File(fname, 'r') as f:
                self.Hydro = f['Hydro'][...]
                self.Nbody = f['Nbody'][...]
        else:
            self.file = None

    def _open_file(self):
        self.file = h5py.File(self.fname, 'r')

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        if not self.file and not self.inmem:
            self._open_file()
        x = np.random.randint(low=0, high=self.length-self.size)
        y = np.random.randint(low=0, high=self.length-self.size)
        z = np.random.randint(low=0, high=self.length-self.size)
        if self.inmem:
            self.inp_buff = self.Nbody[:, x:x+self.size, y:y+self.size, z:z+self.size]
            self.tar_buff = self.Hydro[:, x:x+self.size, y:y+self.size, z:z+self.size]
        else:
            self.file['Nbody'].read_direct(self.inp_buff,
                                       np.s_[0:4, x:x+self.size, y:y+self.size, z:z+self.size],
                                       np.s_[0:4, 0:self.size, 0:self.size, 0:self.size])
            self.file['Hydro'].read_direct(self.tar_buff,
                                       np.s_[0:5, x:x+self.size, y:y+self.size, z:z+self.size],
                                       np.s_[0:5, 0:self.size, 0:self.size, 0:self.size])

        rand = np.random.randint(low=1, high=25)
        inp = np.copy(self.rotate(self.inp_buff, rand))
        tar = np.copy(self.rotate(self.tar_buff, rand))
        return torch.as_tensor(inp), torch.as_tensor(tar)


class RandomRotator(object):
    """
    Composable transform that applies random 3D rotations by right angles.
    Adapted from tf code:
    https://github.com/doogesh/halo_painting/blob/master/wasserstein_halo_mapping_network.ipynb
    """

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

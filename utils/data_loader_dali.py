import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor

#concurrent futures
import concurrent.futures as cf

#dali stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# O(8) transformations
from .symmetry import get_isomorphism_axes_angle


def get_data_loader_distributed(params, world_rank, device_id = 0):
    train_loader = DaliDataLoader(params, params.train_path_npy_data,  params.train_path_npy_label, params.Nsamples,
                                  num_workers=params.num_data_workers, device_id=device_id, validation=False)
    if params.enable_benchy:
        from benchy.torch import BenchmarkGenericIteratorWrapper
        train_loader = BenchmarkGenericIteratorWrapper(train_loader, params.batch_size)
    validation_loader = DaliDataLoader(params, params.val_path_npy_data, params.val_path_npy_label, params.Nsamples_val,
                                       num_workers=params.num_data_workers, device_id=device_id, validation=True)
    return train_loader, validation_loader


    
class DaliDataLoader(object):
    """Random crops"""
    def get_pipeline(self, params, data_file, label_file, num_samples, num_workers, device_id, validation):

        # construct master object
        pipeline = Pipeline(batch_size = params.batch_size,
                            num_threads = num_workers,
                            device_id = device_id)

        # helper function for retrieving the crop window
        def get_crop_coords(rng, length, size, batch_size):
            rstart = rng.randint(low=0, high=length-size, size=(batch_size, 3), dtype=np.int32)
            rend = rstart + size
            
            return rstart, rend

        
        length = params.box_size[0] if not validation else params.box_size[1]

        with pipeline:
            rstart, rend = fn.external_source(source = lambda x: get_crop_coords(self.rng, length, params.data_size, params.batch_size),
                                              num_outputs = 2,
                                              no_copy = False)
            
            data = fn.readers.numpy(device = 'cpu',
                                    name = "data_input",
                                    files = [data_file] * num_samples,
                                    cache_header_information = True,
                                    roi_start = rstart,
                                    roi_end = rend,
                                    roi_axes = [0, 1, 2])

            label = fn.readers.numpy(device = 'cpu',
                                     name = "label_input",
                                     files = [label_file] * num_samples,
                                     cache_header_information = True,
                                     roi_start = rstart,
                                     roi_end = rend,
                                     roi_axes = [0, 1, 2])

            # upload to gpu
            data, label = data.gpu(), label.gpu()

            # get random numbers
            axes, angles = fn.external_source(source = lambda x: get_isomorphism_axes_angle(self.rng, params.batch_size),
                                              device = "cpu",
                                              num_outputs = 2,
                                              no_copy = False,
                                              parallel = False)
            
            flip = fn.random.coin_flip(device = 'cpu',
                                       shape=(3))
            
            # copy to gpu: not necessary
            data_rot = fn.rotate(data,
                                 device = "gpu",
                                 angle = angles,
                                 axis = axes)

            label_rot = fn.rotate(label,
                                  device = "gpu",
                                  angle = angles,
                                  axis = axes)

            # flip
            data_rot = fn.flip(data_rot,
                               device = 'gpu',
                               depthwise = flip[0],
                               horizontal = flip[1],
                               vertical = flip[2])

            label_rot = fn.flip(label_rot,
                                device = 'gpu',
                                depthwise = flip[0],
                                horizontal = flip[1],
                                vertical = flip[2])
            
            # a final transposition to ncdhw layout
            data_out = fn.transpose(data_rot,
                                    device = "gpu",
                                    perm = [3, 0, 1, 2])
            
            label_out = fn.transpose(label_rot,
                                     device = "gpu",
                                     perm = [3, 0, 1, 2]) 
        
            pipeline.set_outputs(data_out, label_out)

        return pipeline

    
    def __init__(self, params, data_file, label_file, num_samples, num_workers=1, device_id=0, validation=False):

        # extract relevant parameters
        self.batch_size = params.batch_size
        self.size = params.data_size
        self.Nsamples = num_samples

        # RNG
        self.rng = np.random.RandomState(seed=12345)

        # shape gymnastics
        N, D, H, W = self.batch_size, self.size, self.size, self.size
        self.inp_shape = [N, 4, D, H, W]
        self.tar_shape = [N, 5, D, H, W]
        self.inp_strides = [ D*H*W*4, 1, H*W*4, W*4, 4]
        self.tar_strides = [ D*H*W*5, 1, H*W*5, W*5, 5]
        
        # construct pipeline
        self.pipe = self.get_pipeline(params, data_file, label_file, num_samples, num_workers, device_id, validation)
        self.pipe.build()
        
        self.iterator = DALIGenericIterator([self.pipe], ['inp', 'tar'],
                                            reader_name = "data_input",
                                            last_batch_policy = LastBatchPolicy.PARTIAL,
                                            auto_reset = True,
                                            prepare_first_batch = True)

        self.Nsamples = num_samples//params.batch_size

        
    def __len__(self):
        return self.Nsamples
        
    def __iter__(self):
        for token in self.iterator:
            inp = token[0]['inp']
            tar = token[0]['tar']

            yield inp, tar

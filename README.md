# SC21 Deep Learning at Scale Tutorial

This repository contains the example code material for the SC21 tutorial:
*Deep Learning at Scale*.

**Contents**

## Links

## Installation

The code can be run using the `romerojosh/containers:sc21_tutorial` docker container (on Perlmutter, docker containers are run via [shifter](https://docs.nersc.gov/development/shifter/), and this container is already available so no download is required). This container is based on the [Nvidia ngc 20.10 pytorch container](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-10.html#rel_21-10), with a few additional packages added. See the dockerfile in [`docker/Dockerfile`](docker/Dockerfile) for details.

For running slurm jobs on Perlmutter, we will use training accounts which are provided under the `ntrain4` project. The slurm script `submit_pm.sh` included in the repository is configured to work automatically(???) as is, but if you submit your own custom jobs via `salloc` or `sbatch` yout must include the following flags for slurm:
* `-A ntrain4_g` is required for training accounts
* `--reservation=ntrain4???` is required to access the set of GPU nodes we have reserved for the duration of the tutorial.

## Model, data, and training code overview

The model in this repository is adapted from a cosmological application of deep learning ([Harrington et al. 2021](https://arxiv.org/abs/2106.12662)), which aims to augment computationally expensive simulations by using a [U-Net](https://arxiv.org/abs/1505.04597) model to reconstruct physical fields of interest (namely, hydrodynamic quantities associated with diffuse gas in the universe):

![n-body to hydro schematic](tutorial_images/nbody2hydro.png)

The U-Net model architecture used in these examples can be found in [`networks/UNet.py`](networks/UNet.py). U-Nets are a popular and capable architecture, as they can extract long-range features through sequential downsampling convolutions, while fine-grained details can be propagated to the upsampling path via skip connections. This particular U-Net is relatively lightweight, to better accommodate our 3D data samples.

The basic data loading pipeline is defined in [`utils/data_loader.py`](utils/data_loader.py), whose primary components are:
* The `RandomCropDataset` which accesses the simulation data stored on disk, and randomly crops sub-volumes of the physical fields to serve for training and validation. For this repository, we will be using a crop size of 64^3
* The `RandomRotator` transform, which applies random rotations and reflections to the samples as data augmentations
* The above components are assembled to feed a PyTorch `DataLoader` which takes the augmented samples and combines them into a batch for each training step.

As we will see in the [Single GPU performance profiling and optimization](#Single-GPU-performance-profiling-and-optimization) section, the random rotations add considerable overhead to data loading during training, and we can achieve performance gains by doing these preprocessing steps on the GPU instead using Nvidia's DALI library. Code for this is found in [`utils/data_loader_dali.py`](utils/data_loader_dali.py).

The script to train the model is [`train.py`](train.py), which uses the following arguments to load the desired training setup:
```
  --yaml_config YAML_CONFIG   path to yaml file containing training configs
  --config CONFIG             name of desired config in yaml file
```

Based on the selected configuration, the train script will then:
1.  Set up the data loaders and construct our U-Net model, the Adam optimizer, and our L1 loss function.
2.  Loop over training epochs to run the training. See if you can identify the following key components: 
    * Looping over data batches from our data loader.
    * Applying the forward pass of the model and computing the loss function.
    * Calling `backward()` on the loss value to backpropagate gradients. Note the use of the `grad_scaler` will be explained below when enabling mixed precision.
    * Applying the model to the validation dataset and logging training and validation metrics to visualize in TensorBoard (see if you can find where we construct the TensorBoard `SummaryWriter` and where our specific metrics are logged via the `add_scalar` call).

Besides the `train.py` script, we have a slightly more complex [train_graph.py](train_graph.py)
script, which implements the same functionality with added capability for using the Cuda Graphs APIs introduced in PyTorch 1.10. This topic will be covered in the [Single GPU performance profiling and optimization](#Single-GPU-performance-profiling-and-optimization) section.

More info on the model and data can be found in the [?? slides link](link_to_gdrive_slides).


## Single GPU training

To launch a single GPU job for the baseline training script on Perlmutter, use the following command:
```
sbatch -n 1 submit_pm.sh
```
This will set up our run directories and launch the training in our container environment via shifter. The actual python command used by `submit_pm.sh` to launch training is
```
python train.py --config=A100_crop64_sqrt
```
This will run the training on a single GPU using a batch size of 64 (see `config/UNet.yaml` for specific configuration details). Note that any arguments for `train.py`, such as the desired config (`--config`), can be added after `submit_pm.sh` when submitting, and they will be passed to `train.py` properly.


In the baseline configuration, the model converges to about ??% accuracy on
the validation dataset in about ?? epochs:

## Single GPU performance profiling and optimization

This is the performance of the baseline script using the ??? container for the first two epochs on a 40GB A100 card with batch size ???:
```
```

### Profiling with Nsight Systems
Before generating a profile with Nsight, we can add NVTX ranges to the script to add context to the produced timeline. First, we can enable PyTorch's built-in NVTX annotations by using the `torch.autograd.profiler.emit_nvtx` context manager.
We can also manually add some manually defined NVTX ranges to the code using `torch.cuda.nvtx.range_push` and `torch.cuda.nvtx.range_pop`. Search `train.py` for comments labeled `# PROF` to see where we've added code.

To generate a timeline, run the following:
```
$ nsys profile -o baseline --trace=cuda,nvtx -c cudaProfilerApi ...
```
This command will run two shortened epochs of ?? iterations of the training script and produce a file `baseline.qdrep` that can be opened in the Nsight System's program. The arg `--trace=cuda,nvtx` is optional and is used here to disable OS Runtime tracing for speed.

Loading this profile in Nsight Systems will look like this:

With our NVTX ranges, we can easily zoom into a single iteration and get an idea of where compute time is being spent:


### Enabling Mixed Precision Training
As a first step to improve the compute performance of this training script, we can enable automatic mixed precision (AMP) in PyTorch. AMP provides a simple way for users to convert existing FP32 training scripts to mixed FP32/FP16 precision, unlocking
faster computation with Tensor Cores on NVIDIA GPUs. The AMP module in torch is composed of two main parts: `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast`. `torch.cuda.amp.GradScaler` handles automatic loss scaling to control the range of FP16 gradients.
The `torch.cuda.amp.autocast` context manager handles converting model operations to FP16 where appropriate. Search `train.py` for comments labeled `# AMP:` to see where we've added code to enable AMP in this script.

To run the script on a single GPU with AMP enabled, use the following command:
```
$ python ...
```
With AMP enabled, this is the performance of the baseline using the ??? container for the first two epochs on a 40GB A100 card:
```
...
```

You can run another profile (using `--config=???`) with Nsight Systems. Loading this profile and zooming into a single iteration, this is what we see:

With AMP enabled, we see that the `forward/loss/backward` time is significatly reduced. As this is a CNN, the forward and backward convolution ops are well-suited to benefit from acceleration with tensor cores.

### Data loading optimizations with DALI

### Just-in-time (JIT) compiliation 

### Using CUDA Graphs (advanced)

### Applying additional PyTorch optimizations
With the forward and backward pass accelerated with AMP and NHWC memory layout, the remaining NVTX ranges we added to the profile stand out, namely the `zero_grad` marker and `optimizer.step`.

To speed up the `zero_grad`, we can add the following argument to the `zero_grad` call:
```
self.model.zero_grad(set_to_none=True)
```
This optional argument allows PyTorch to skip memset operations to zero out gradients and also allows PyTorch to set gradients with a single write (`=` operator) instead of a read/write (`+=` operator).


If we look closely at the `optimizer.step` range in the profile, we see that there are many indivdual pointwise operation kernels launched. To make this more efficient, we can replace the native PyTorch SGD optimizer with the `FusedSGD` optimizer from the `Apex` package, which fuses many of these pointwise
operations.

Finally, as a general optimization, we add the line `torch.backends.cudnn.benchmark = True` to the start of training to enable cuDNN autotuning. This will allow cuDNN to test and select algorithms that run fastest on your system/model.

Search `train.py` for comments labeled `# EXTRA` to see where we've added changes for these additional optimizations.


To run the script on a single GPU with these additional optimizations, use the following command:
```
$ python ...
```
With all these features enabled, this is the performance of the script using the ??? container for the first two epochs on a 40GB A100 card:
```
...
```

We can run a final profile with all the optimizations enabled (using `--config=???`) with Nsight Systems. Loading this profile and zooming into a single iteration, this is what we see now:
With these additional optimizations enabled in PyTorch, we see the length of the `zero_grad` and `optimizer.step` ranges are greatly reduced, as well as a small improvement in the `forward/loss/backward` time.

## Distributed GPU training

Now that we have model training code that is optimized for training on a single GPU,
we are ready to utilize multiple GPUs and multiple nodes to accelerate the workflow
with *distributed training*. We will use the recommended `DistributedDataParallel`
wrapper in PyTorch with the NCCL backend for optimized communication operations on
systems with NVIDIA GPUs. Refer to the PyTorch documentation for additional details 
on the distributed package: https://pytorch.org/docs/stable/distributed.html

### Code basics

To submit a multi-GPU job, use the `submit_pm.sh` with the `-n` option set to the desired number of GPUs. For example, to launch a training with 4 GPUs (which is all available GPUs on each Perlmutter GPU node), do
```
sbatch -n 4 submit_pm.sh
```
This script automatically uses the slurm flags `--ntasks-per-node 4`, `--cpus-per-task 32`, `--gpus-per-task 1`, so slurm will allocate one process for each GPU we request, and give each process 1/4th of the CPU resources available on a Perlmutter GPU node. This way, multi-node trainings can easily be launched simply by setting `-n` greater than 4.

PyTorch `DistributedDataParallel`, or DDP for short, is flexible and can initialize process groups with a variety of methods. For this code, we will use the standard approach of initializing via environment variables, which can be easily read from the slurm environment. Take a look at the `export_DDP_vars.sh` helper script, which is used by our job script to expose for PyTorch DDP the global rank and node-local rank of each process, along with the total number of ranks and the address and port to use for network communication. In the [train.py](train.py) script, near the bottom in the main script execution, we set up the distributed backend using these environment variables via `torch.distributed.init_proces_group`.

Note that ordinarily `get_data_loader` function in
[utils/cifar100\_data\_loader.py](utils/cifar100_data_loader.py), we use the
DistributedSampler from PyTorch which takes care of partitioning the dataset
so that each training process sees a unique subset.

In our Trainer's `__init__` method, after our ResNet50 model is constructed,
we convert it to a distributed data parallel model by wrapping it as:

    self.model = DistributedDataParallel(self.model, ...)

The DistributedDataParallel (DDP) model wrapper takes care of broadcasting
initial model weights to all workers and performing all-reduce on the gradients
in the training backward pass to properly synchronize and update the model
weights in the distributed setting.

### Large batch convergence

To speed up training, we try to use larger batch sizes, spread across more GPUs,
with larger learning rates. In particular, we try increasing from ??? to ???,
and scale the batch size similarly to ???.
The first thing we demonstrate here is increasing
the learning rate according to the square-root scaling rule. The settings for
batch size ??? are in ???, respectively.
We view the accuracy plots in TensorBoard and notice that the convergence
performs worse with larger batch size, i.e. we see a generalization gap:


Next, as suggested in the presentation previously, we apply a linear learning rate
warmup for these batch sizes. You can see where we compute the learning rate
in the warmup phase in our Trainer's `train` method in the `train.py` script.
Look for the comment, "Apply learning rate warmup".
As shown in configs `???` and `???` in our
`config/???.yaml` file, we use ??? and ??? epochs for the warmup,
respectively.

Now we can see the generalization gap closes and
the higher batch size results are as good as the original batch size 128:


Next, we can now look at the wallclock time to see that, indeed, using
these tricks together result in a much faster convergence:


In particular, our batch size ??? run on 1 gpu takes about ??? min to converge,
while our batch size ??? run on ??? gpus takes around ??? min.

Finally, we look at the throughput (images/second) of our training runs as
we do this weak scaling of the batch size and GPUs:


These plots show ???% scaling efficiency with respect to ideal scaling at ??? GPUs.

## Multi-GPU performance profiling and optimization
### Profiling with Nsight Systems
### Adjusting DistributedDataParallel options

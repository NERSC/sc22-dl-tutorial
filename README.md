# SC21 Deep Learning at Scale Tutorial

This repository contains the example code material for the SC21 tutorial:
*Deep Learning at Scale*.

**Contents**

## Links

## Installation

## 3D U-Net for Cosmological Simulations

U-Net model adapted from https://arxiv.org/abs/2106.12662

### Configuring
Configs are stored in `config/UNet.yaml`. Adjust paths as needed to point to your data copies and scratch directory to store experiment results. Data can be downloaded from https://portal.nersc.gov/project/dasrepo/pharring/

All configs tested with the `nvcr.io/nvidia/pytorch:21.03-py3` image but should work with others as well. Code uses `h5py` and `ruamel.yaml` in addition to standard libs.

### Running
Scaling studies use crop sizes of either `64^3` or `96^3` for training (faster than full-scale problem). On Perlmutter, to submit tests for multi-GPU scaling, simply do 
```
bash launch_scaling.sh
```
This launches runs for training with 1,4,8,32, and 128 GPUs, using the square root scaling rule for learning rate. Multi-GPU runs warm up the learning rate over 128 iterations, and all configs use cosine annealing to decrease the learning rate throughout training. Each run will create its own experiment directory as specified in `config.yaml`.

## Model, data, and training code overview



## Single GPU training

To run single GPU training of the baseline training script, use the following command:
```
$ python ...
```
This will run the training on a single GPU using batch size of ???
(see `config/??.yaml` for specific configuration details).
Note we will use batch size ??? for the optimization work in the next section
and will push beyond to larger batch sizes in the distributed training section.

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

We use the `torch.distributed.launch` utility for launching training processes
on one node, one per GPU. The [submit\_multinode.slr](submit_multinode.slr)
script shows how we use the utility with SLURM to launch the tasks on each node
in our system allocation.

In the [train.py](train.py) script, near the bottom in the main script execution,
we set up the distributed backend. We use the environment variable initialization
method, automatically configured for us when we use the `torch.distributed.launch` utility.

In the `get_data_loader` function in
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

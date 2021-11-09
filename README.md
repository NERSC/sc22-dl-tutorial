# SC21 Deep Learning at Scale Tutorial

This repository contains the example code material for the SC21 tutorial:
*Deep Learning at Scale*.

**Contents**

## Links

## Installation

### Installing Nsight Systems
In this tutorial, we will be generating profile files using NVIDIA Nsight Systems on the remote systems. In order to open and view these
files on your local computer, you will need to install the Nsight Systems program, which you can download [here](https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2021-4-1-73). Select the download option required for your system (e.g. Mac OS host for MacOS, Window Host for Windows, or Linux Host .rpm/.deb/.run for Linux). You may need to sign up and create a login to NVIDIA's developer program if you do not
already have an account to access the download. Proceed to run and install the program using your selected installation method.

## 3D U-Net for Cosmological Simulations
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

Besides the `train.py` script, we have a slightly more complex [`train_graph.py`](train_graph.py)
script, which implements the same functionality with added capability for using the Cuda Graphs APIs introduced in PyTorch 1.10. This topic will be covered in the [Single GPU performance profiling and optimization](#Single-GPU-performance-profiling-and-optimization) section.

More info on the model and data can be found in the [?? slides link](link_to_gdrive_slides).


## Single GPU training

To run a single GPU training of the baseline training script without optimizations, use the following command if running interactively:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --num_epochs 3
```
* If running on a 16GB V100 card:
```
$ python train.py --config=V100_crop64_sqrt --num_epochs 3
```
This will run 3 epochs of training on a single GPU using a default batch size of 64 (A100) or 32 (V100).
(see `config/UNet.yaml` for specific configuration details).
Note we will use the default batch size for the optimization work in the next section
and will push beyond to larger batch sizes in the distributed training section.

On Perlmutter for the tutorial, we will be submitting jobs to the batch queue. To do this, use the following command:
```
$ sbatch -n 1 ./submit_pm.sh --num_epochs 3
```
`submit_pm.sh` is a batch submission script that defines resources to be requested by SLURM as well as the command to run.
Note that any arguments for `train.py`, such as the desired config (`--config`), can be added after `submit_pm.sh` when submitting, and they will be passed to `train.py` properly.
When using batch submission, you can see the job output by viewing the file `pm-crop64-<jobid>.out` in the submission
directory. You can find the job id of your job using the command `squeue --me` and looking at the first column of the output.


In the baseline configuration, the model converges to about ??% accuracy on
the validation dataset in about 80 epochs:

## Single GPU performance profiling and optimization

This is the performance of the baseline script for the first three epochs on a 40GB A100 card with batch size 64:
```
2021-11-09 00:19:04,091 - root - INFO - Time taken for epoch 1 is 110.217036485672 sec, avg 37.1630387697139 samples/sec
2021-11-09 00:19:04,092 - root - INFO -   Avg train loss=0.065003
2021-11-09 00:19:14,226 - root - INFO -   Avg val loss=0.040343
2021-11-09 00:19:14,227 - root - INFO -   Total validation time: 10.133511781692505 sec
2021-11-09 00:20:03,014 - root - INFO - Time taken for epoch 2 is 48.785075426101685 sec, avg 83.96010386833387 samples/sec
2021-11-09 00:20:03,049 - root - INFO -   Avg train loss=0.027986
2021-11-09 00:20:07,986 - root - INFO -   Avg val loss=0.025327
2021-11-09 00:20:07,987 - root - INFO -   Total validation time: 4.936376571655273 sec
2021-11-09 00:20:55,329 - root - INFO - Time taken for epoch 3 is 47.339499711990356 sec, avg 86.52393930902795 samples/sec
2021-11-09 00:20:55,329 - root - INFO -   Avg train loss=0.020926
2021-11-09 00:21:00,246 - root - INFO -   Avg val loss=0.024092
2021-11-09 00:21:00,269 - root - INFO -   Total validation time: 4.917020082473755 sec
```
After the first epoch, we see that the throughput achieved is about 85 samples/s.

### Profiling with Nsight Systems
#### Adding NVTX ranges and profiler controls
Before generating a profile with Nsight, we can add NVTX ranges to the script to add context to the produced timeline.
We can add some manually defined NVTX ranges to the code using `torch.cuda.nvtx.range_push` and `torch.cuda.nvtx.range_pop`.
We can also add calls to `torch.cuda.profiler.start()` and `torch.cuda.profiler.stop()` to control the duration of the profiling
(e.g., limit profiling to single epoch).

To generate a profile, use the following command if running interactively:
* If running on a 40GB A100 card:
```
$ nsys profile -o baseline --trace=cuda,nvtx -c cudaProfilerApi --kill none -f true python train.py --config=A100_crop64_sqrt --num_epochs 2 --enable_manual_profiling
```

* If running on a 80GB A100 card:
```
$ nsys profile -o baseline --trace=cuda,nvtx -c cudaProfilerApi --kill none -f true python train.py --config=A100_crop64_sqrt --num_epochs 2 --enable_manual_profiling
```

This command will run two epochs of the training script, profiling only 30 steps of the second epoch. It will produce a file `baseline.qdrep` that can be opened in the Nsight System's program. The arg `--trace=cuda,nvtx` is optional and is used here to disable OS Runtime tracing for speed.

If running on Perlmutter, the equivalent batch submission command is:
```
$ ENABLE_PROFILING=1 PROFILE_OUTPUT=baseline sbatch -n1 submit_pm.sh --num_epochs 2 --enable_manual_profiling
```

Loading this profile in Nsight Systems will look like this:
![NSYS Baseline](tutorial_images/nsys_baseline.png)

From this zoomed out view, we can see a lot idle gaps between iterations. These gaps are due to the data loading, which we will address in the next section.

Beyond this, we can zoom into a single iteration and get an idea of where compute time is being spent:
![NSYS Baseline zoomed](tutorial_images/nsys_baseline_zoomed.png)


#### Using the benchy profiling tool
As an alternative to manually specifying NVTX ranges, we've included the use of a simple profiling tool `benchy` that overrides the PyTorch dataloader in the script to produce throughput information to the terminal, as well as add NVTX ranges/profiler start and stop calls. This tool also runs a sequence of tests to measure and report the throughput of the dataloader in isolation, the model running with synthetic/cached data, and the throughput of the model running normally with real data.

To run using benchy, use the following command if running interactively:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --enable_benchy
```

* If running on a 80GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --enable_benchy
```

If running on Perlmutter, the equivalent batch submission command is:
```
$  sbatch -n1 submit_pm.sh --enable_benchy
```

benchy uses epoch boundaries to separate the test trials it runs, so in these cases we are not limiting the number of epochs to 2.

benchy will report throughput measurements directly to the terminal, including a simple summary of averages at the end of the job. For this case on Perlmutter, the summary output from benchy is:
```
BENCHY::SUMMARY::IO average trial throughput: 89.177 +/- 1.011
BENCHY::SUMMARY:: SYNTHETIC average trial throughput: 376.537 +/- 0.604
BENCHY::SUMMARY::FULL average trial throughput: 89.971 +/- 0.621
```
From these throughput values, we can see that the `SYNTHETIC` (i.e. compute) throughput is greater than the `IO` (i.e. data loading) throughput.
The `FULL` (i.e. real) throughput is bounded by the slower of these two values, which is `IO` in this case. What these throughput
values indicate is the GPU can achieve much greater training throughput for this model, but is being limited by the data loading
speed.

### Data loading optimizations
#### Improving the native PyTorch dataloader performance
The PyTorch dataloader has several knobs we can adjust to improve performance. If you look at the `DataLoader` initialization in
`utils/data_loader.py`, you'll see we've already set several useful options, like `pin_memory` and `persistent_workers`.
`pin_memory` has the data loader read input data into pinned host memory, which typically yields better host-to-device and device-to-host
memcopy bandwidth. `persistent_workers` allows PyTorch to reuse workers between epochs, instead of the default behavior which is to
respawn them. One knob we've left to adjust is the `num_workers` argument, which we can control via the `--num_data_workers` command
line arg to our script. The default in our config is two workers, but we can experiment with this value to see if increasing the number
of workers improves performance.

We can experiment by launching the script as follows:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --num_epochs 3 --num_data_workers <value of your choice>
```
* If running on a 16GB V100 card:
```
$ python train.py --config=V100_crop64_sqrt --num_epochs 3 --num_data_workers <value of your choice>
```
* If running on Perlmutter in the batch queue:
```
$ sbatch -n 1 ./submit_pm.sh --num_epochs 3 --num_data_workers <value of your choice>
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64 and 4 data workers:
```
2021-11-09 00:21:17,371 - root - INFO - Time taken for epoch 1 is 79.13155698776245 sec, avg 51.761903290155644 samples/sec
2021-11-09 00:21:17,372 - root - INFO -   Avg train loss=0.065546
2021-11-09 00:21:23,152 - root - INFO -   Avg val loss=0.044859
2021-11-09 00:21:23,185 - root - INFO -   Total validation time: 5.7792582511901855 sec
2021-11-09 00:21:48,916 - root - INFO - Time taken for epoch 2 is 25.728514432907104 sec, avg 159.20079686999583 samples/sec
2021-11-09 00:21:48,941 - root - INFO -   Avg train loss=0.028024
2021-11-09 00:21:52,277 - root - INFO -   Avg val loss=0.025949
2021-11-09 00:21:52,277 - root - INFO -   Total validation time: 3.3348052501678467 sec
2021-11-09 00:22:17,380 - root - INFO - Time taken for epoch 3 is 25.10083317756653 sec, avg 163.18183428511588 samples/sec
2021-11-09 00:22:17,387 - root - INFO -   Avg train loss=0.021308
2021-11-09 00:22:20,662 - root - INFO -   Avg val loss=0.024352
2021-11-09 00:22:20,662 - root - INFO -   Total validation time: 3.2743005752563477 sec
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64 and 8 data workers:
```
2021-11-09 00:32:48,064 - root - INFO - Time taken for epoch 1 is 62.2959144115448 sec, avg 65.75070032587757 samples/sec
2021-11-09 00:32:48,064 - root - INFO -   Avg train loss=0.073569
2021-11-09 00:32:52,265 - root - INFO -   Avg val loss=0.048459
2021-11-09 00:32:52,265 - root - INFO -   Total validation time: 4.200311183929443 sec
2021-11-09 00:33:07,551 - root - INFO - Time taken for epoch 2 is 15.283130884170532 sec, avg 268.00791219045453 samples/sec
2021-11-09 00:33:07,551 - root - INFO -   Avg train loss=0.032871
2021-11-09 00:33:10,462 - root - INFO -   Avg val loss=0.030250
2021-11-09 00:33:10,462 - root - INFO -   Total validation time: 2.910416841506958 sec
2021-11-09 00:33:25,404 - root - INFO - Time taken for epoch 3 is 14.93994927406311 sec, avg 274.16425081917566 samples/sec
2021-11-09 00:33:25,405 - root - INFO -   Avg train loss=0.024557
2021-11-09 00:33:28,357 - root - INFO -   Avg val loss=0.027871
2021-11-09 00:33:28,357 - root - INFO -   Total validation time: 2.9516751766204834 sec
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64 and 16 data workers:
```
2021-11-09 00:21:01,556 - root - INFO - Time taken for epoch 1 is 62.40265655517578 sec, avg 65.63823122463319 samples/sec
2021-11-09 00:21:01,565 - root - INFO -   Avg train loss=0.069824
2021-11-09 00:21:06,210 - root - INFO -   Avg val loss=0.043009
2021-11-09 00:21:06,225 - root - INFO -   Total validation time: 4.645080804824829 sec
2021-11-09 00:21:22,464 - root - INFO - Time taken for epoch 2 is 16.23646593093872 sec, avg 252.27164688560939 samples/sec
2021-11-09 00:21:22,479 - root - INFO -   Avg train loss=0.029511
2021-11-09 00:21:25,424 - root - INFO -   Avg val loss=0.028309
2021-11-09 00:21:25,444 - root - INFO -   Total validation time: 2.943828582763672 sec
2021-11-09 00:21:41,607 - root - INFO - Time taken for epoch 3 is 16.159828186035156 sec, avg 253.46804142012112 samples/sec
2021-11-09 00:21:41,608 - root - INFO -   Avg train loss=0.022431
2021-11-09 00:21:44,875 - root - INFO -   Avg val loss=0.026001
2021-11-09 00:21:44,897 - root - INFO -   Total validation time: 3.266282796859741 sec
```

Increasing the number of workers to 8 improves performance to around 270 samples per second, while increasing to 16 workers causes a slight reduction from this.

We can run the 8 worker configuration through profiler using the instructions in the previous section with the added `--num_data_workers`
argument and load that profile in Nsight Systems. This is what this profile looks like:
![NSYS Native Data](tutorial_images/nsys_nativedata_8workers.png)

and zoomed in:
![NSYS Native Data Zoomed](tutorial_images/nsys_nativedata_8workers_zoomed.png)

With 8 data workers, the large gaps between steps are mostly alleviated, improving the throughput. Looking at the zoomed in profile, we
still see that the H2D copy in of the input data takes some time and could be improved. One option here is to implement a prefetching
mechanism in PyTorch directly using CUDA streams to concurrently load and copy in the next batch of input during the current batch, however
this is left as an exercise outside of this tutorial. A good example of this can be found in [here](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L347).

Using benchy, we can also check how the various throughputs compare using 8 data workers. Running this configuration on Perlmutter
using the tool yields the following:
```
BENCHY::SUMMARY::IO average trial throughput: 303.304 +/- 1.468
BENCHY::SUMMARY:: SYNTHETIC average trial throughput: 359.426 +/- 3.380
BENCHY::SUMMARY::FULL average trial throughput: 252.044 +/- 0.253
```
`IO` is faster as expected, and the `FULL` throughput increases correspondingly. However, `IO` is still lower than `SYNTHETIC`, meaning we
should still address data loading before focusing on compute improvements.

#### Using NVIDIA DALI
While we were able to get more performance out of the PyTorch native DataLoader, there are several overheads we cannot overcome in
PyTorch alone:
1. The PyTorch DataLoader will use CPU operations for all I/O operations as well as data augmentations
2. The PyTorch DataLoader uses multi-processing to spawn data workers, which has performance overheads compared to true threads

The NVIDIA DALI library is a data loading library that can address both of these points:
1. DALI can perform a wide array of data augmentation operations on the GPU, benefitting from acceleration relative to the CPU.
2. DALI maintains its own worker threads in the C++ backend, enabling much more performant threading and concurrent operation.

For this tutorial, we've provided an alternative data loader using DALI to accelerate the data augementations used in this training script (e.g. 3D cropping, rotations, and flips) that can be found in `utils/data_loader_dali.py`. This data loader is enabled via the command line
argument `--data_loader_config=dali-lowmem` to the training script.

We can experiment by with the DALI dataloader launching the script as follows:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --num_epochs 3 --data_loader_config=dali-lowmem
```
* If running on a 16GB V100 card:
```
$ python train.py --config=V100_crop64_sqrt --num_epochs 3 --data_loader_config=dali-lowmem

```
* If running on Perlmutter in the batch queue:
```
$ sbatch -n 1 ./submit_pm.sh --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64 and DALI:
```
2021-11-09 01:00:49,434 - root - INFO - Time taken for epoch 1 is 174.0535707473755 sec, avg 23.532984600155135 samples/sec
2021-11-09 01:00:49,435 - root - INFO -   Avg train loss=0.071265
2021-11-09 01:00:53,839 - root - INFO -   Avg val loss=0.044646
2021-11-09 01:00:53,839 - root - INFO -   Total validation time: 4.403615236282349 sec
2021-11-09 01:01:04,155 - root - INFO - Time taken for epoch 2 is 10.313512086868286 sec, avg 397.1489018968859 samples/sec
2021-11-09 01:01:04,155 - root - INFO -   Avg train loss=0.030431
2021-11-09 01:01:05,138 - root - INFO -   Avg val loss=0.028278
2021-11-09 01:01:05,138 - root - INFO -   Total validation time: 0.9822394847869873 sec
2021-11-09 01:01:15,434 - root - INFO - Time taken for epoch 3 is 10.293895483016968 sec, avg 397.90573031926016 samples/sec
2021-11-09 01:01:15,434 - root - INFO -   Avg train loss=0.022368
2021-11-09 01:01:16,188 - root - INFO -   Avg val loss=0.026419
2021-11-09 01:01:16,188 - root - INFO -   Total validation time: 0.753748893737793 sec
```

We can run the DALI case through profiler using the instructions in the earlier section with the added `--data_loader_config=dali-lowmem`
argument and load that profile in Nsight Systems. This is what this profile looks like:
![NSYS DALI](tutorial_images/nsys_dali.png)

and zoomed in to a single iteration:
![NSYS DALI Zoomed](tutorial_images/nsys_dali_zoomed.png)

With DALI, you will see that there are now multiple CUDA stream rows in the timeline view, corresponding to internal streams DALI uses
to run data augmentation kernels and any memory movement concurrently with the existing PyTorch compute kernels. Stream 13 in this view, in particular, shows concurrent H2D memory copies of the batch input data, which is an improvement over the native dataloader.

Running this case using benchy on Perlmutter results in the following throughput measurements:
```
BENCHY::SUMMARY::IO average trial throughput: 917.607 +/- 0.156
BENCHY::SUMMARY:: SYNTHETIC average trial throughput: 420.991 +/- 0.006
BENCHY::SUMMARY::FULL average trial throughput: 395.771 +/- 0.162
```
One thing we can notice here is that the `SYNTHETIC` speed is increased from previous cases. This is because the synthetic data sample that
is cached and reused from the DALI data loader is already resident on the GPU, in contrast to the case using the PyTorch dataloader where
the cached sample is in CPU memory. As a result, the `SYNTHETIC` result here is improved due to no longer requiring a H2D memory copy.
In general, we now see that the `IO` throughput is greater than the `SYNTHETIC`, meaning the data loader can keep up with the compute
throughput with additional headroom for compute speed improvements. 

### Enabling Mixed Precision Training
Now that the data loading performance is faster than the synthetic compute throughput, we can start looking at improving compute performance. As a first step to improve the compute performance of this training script, we can enable automatic mixed precision (AMP) in PyTorch. AMP provides a simple way for users to convert existing FP32 training scripts to mixed FP32/FP16 precision, unlocking
faster computation with Tensor Cores on NVIDIA GPUs.

The AMP module in torch is composed of two main parts: `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast`. `torch.cuda.amp.GradScaler` handles automatic loss scaling to control the range of FP16 gradients.
The `torch.cuda.amp.autocast` context manager handles converting model operations to FP16 where appropriate.

As a quick note, the A100 GPUs we've been using to report results thus far have been able to benefit from Tensor Core compute via the use of TF32 precision operations, enabled by default for CUDNN and CUBLAS in PyTorch. We can measure the benefit of TF32 precision usage on the A100 GPU by temporarily disabling it via setting the environment variable `NVIDIA_TF32_OVERRIDE=0`.  
Running this experiment on Perlmutter using the following command:
```
$ NVIDIA_TF32_OVERRIDE=0 sbatch -n 1 ./submit_pm.sh --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem
```
yields the following result for 3 epochs:
```
2021-11-09 01:23:59,717 - root - INFO - Time taken for epoch 1 is 175.16923069953918 sec, avg 23.383102064458487 samples/sec
2021-11-09 01:23:59,717 - root - INFO -   Avg train loss=0.067822
2021-11-09 01:24:05,165 - root - INFO -   Avg val loss=0.045555
2021-11-09 01:24:05,165 - root - INFO -   Total validation time: 5.447303533554077 sec
2021-11-09 01:24:35,980 - root - INFO - Time taken for epoch 2 is 30.808427572250366 sec, avg 132.9506347052042 samples/sec
2021-11-09 01:24:35,981 - root - INFO -   Avg train loss=0.030442
2021-11-09 01:24:37,799 - root - INFO -   Avg val loss=0.028225
2021-11-09 01:24:37,799 - root - INFO -   Total validation time: 1.8185265064239502 sec
2021-11-09 01:25:08,618 - root - INFO - Time taken for epoch 3 is 30.81597113609314 sec, avg 132.9180891918272 samples/sec
2021-11-09 01:25:08,618 - root - INFO -   Avg train loss=0.022830
2021-11-09 01:25:10,511 - root - INFO -   Avg val loss=0.025803
2021-11-09 01:25:10,511 - root - INFO -   Total validation time: 1.8923368453979492 sec
```
From here, we can see that running in FP32 without TF32 acceleration is much slower and we are already seeing great performance from
TF32 Tensor Core operations without any code changes to add AMP. With that said, AMP can still be a useful improvement for A100 GPUs,
as TF32 is a compute type only, leaving all data in full precision FP32. FP16 precision has the compute benefits of Tensor Cores combined with a reduction in storage and memory bandwidth requirements. 

We can experiment with AMP by launching the script as follows:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp
```
* If running on a 16GB V100 card:
```
$ python train.py --config=V100_crop64_sqrt --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp

```
* If running on Perlmutter in the batch queue:
```
$ sbatch -n 1 ./submit_pm.sh --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64, DALI, and AMP:
```
2021-11-09 01:52:10,145 - root - INFO - Time taken for epoch 1 is 173.51554536819458 sec, avg 23.6059541023164 samples/sec
2021-11-09 01:52:10,145 - root - INFO -   Avg train loss=0.068746
2021-11-09 01:52:14,449 - root - INFO -   Avg val loss=0.042795
2021-11-09 01:52:14,449 - root - INFO -   Total validation time: 4.303220987319946 sec
2021-11-09 01:52:22,773 - root - INFO - Time taken for epoch 2 is 8.321907758712769 sec, avg 492.1948330551514 samples/sec
2021-11-09 01:52:22,773 - root - INFO -   Avg train loss=0.032682
2021-11-09 01:52:23,759 - root - INFO -   Avg val loss=0.031000
2021-11-09 01:52:23,759 - root - INFO -   Total validation time: 0.9850668907165527 sec
2021-11-09 01:52:31,742 - root - INFO - Time taken for epoch 3 is 7.98108172416687 sec, avg 513.2136396495268 samples/sec
2021-11-09 01:52:31,742 - root - INFO -   Avg train loss=0.025240
2021-11-09 01:52:32,306 - root - INFO -   Avg val loss=0.028672
2021-11-09 01:52:32,306 - root - INFO -   Total validation time: 0.5628025531768799 sec
```

We can run the case with AMP enabled through profiler using the instructions in the earlier section with the added `--data_loader_config=enable_amp`
argument and load that profile in Nsight Systems. This is what this profile looks like:
![NSYS DALI AMP](tutorial_images/nsys_dali_amp.png)

and zoomed in to a single iteration:
![NSYS DALI AMP Zoomed](tutorial_images/nsys_dali_amp_zoomed.png)

With AMP enabled, we see that the `forward` (and, correspondingly the backward) time is significatly reduced. As this is a CNN, the forward and backward convolution ops are well-suited to benefit from acceleration with tensor cores and that is where we see the most benefit.

Running this case using benchy on Perlmutter results in the following throughput measurements:
```
BENCHY::SUMMARY::IO average trial throughput: 877.734 +/- 2.838
BENCHY::SUMMARY:: SYNTHETIC average trial throughput: 661.466 +/- 0.046
BENCHY::SUMMARY::FULL average trial throughput: 528.898 +/- 0.619
```
From these results, we can see a big improvement in the `SYNTHETIC` and `FULL` throughput from using mixed-precision training over
TF32 alone.

### Just-in-time (JIT) compiliation and APEX fused optimizers
While AMP provided a large increase in compute speed already, there are a few other optimizations available for PyTorch to improve
compute throughput. A first (and simple change) is to replace the Adam optimizer from `torch.optim.Adam` with a fused version from
[APEX](https://github.com/NVIDIA/apex), `apex.optimizers.FusedAdam`. This fused optimizer uses fewer kernels to perform the weight
update than the standard PyTorch optimizer, reducing latency and making more efficient use of GPU bandwidth by increasing register
reuse. We can enabled the use of the `FusedAdam` optimizer in our training script by adding the flag `--enable_apex`. 

We can see the impact of adding the APEX fused optimizer by launching the script as follows:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp --enable_apex
```
* If running on a 16GB V100 card:
```
$ python train.py --config=V100_crop64_sqrt --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp --enable_apex

```
* If running on Perlmutter in the batch queue:
```
$ sbatch -n 1 ./submit_pm.sh --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp --enable_apex
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64, DALI, and AMP, and APEX:
```
2021-11-09 02:12:09,940 - root - INFO - Time taken for epoch 1 is 173.45775055885315 sec, avg 23.613819427516745 samples/sec
2021-11-09 02:12:09,949 - root - INFO -   Avg train loss=0.066497
2021-11-09 02:12:14,938 - root - INFO -   Avg val loss=0.044828
2021-11-09 02:12:14,968 - root - INFO -   Total validation time: 4.987832546234131 sec
2021-11-09 02:12:23,027 - root - INFO - Time taken for epoch 2 is 8.056141376495361 sec, avg 508.4319910212239 samples/sec
2021-11-09 02:12:23,034 - root - INFO -   Avg train loss=0.029358
2021-11-09 02:12:23,909 - root - INFO -   Avg val loss=0.026867
2021-11-09 02:12:23,910 - root - INFO -   Total validation time: 0.8748016357421875 sec
2021-11-09 02:12:31,403 - root - INFO - Time taken for epoch 3 is 7.491202116012573 sec, avg 546.7747280833245 samples/sec
2021-11-09 02:12:31,422 - root - INFO -   Avg train loss=0.021609
2021-11-09 02:12:32,071 - root - INFO -   Avg val loss=0.024649
2021-11-09 02:12:32,073 - root - INFO -   Total validation time: 0.6492981910705566 sec
```

While APEX provides some already fused kernels, for more general fusion of eligible pointwise operations in PyTorch, we can enable
JIT compilation, done in our training script via the flag `--enable_jit`. 

We can see the impact of enabling JIt compilation  by launching the script as follows:
* If running on a 40GB A100 card:
```
$ python train.py --config=A100_crop64_sqrt --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp --enable_apex --enable_jit
```
* If running on a 16GB V100 card:
```
$ python train.py --config=V100_crop64_sqrt --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp --enable_apex --enable_jit

```
* If running on Perlmutter in the batch queue:
```
$ sbatch -n 1 ./submit_pm.sh --num_epochs 3 --num_data_workers 8 --data_loader_config=dali-lowmem --enable_amp --enable_apex --enable_jit
```

This is the performance of the training script for the first three epochs on a 40GB A100 card with batch size 64, DALI, and AMP, APEX and JIT:
```
2021-11-09 02:12:10,479 - root - INFO - Time taken for epoch 1 is 173.40251851081848 sec, avg 23.621340884644955 samples/sec
2021-11-09 02:12:10,506 - root - INFO -   Avg train loss=0.070950
2021-11-09 02:12:14,683 - root - INFO -   Avg val loss=0.046055
2021-11-09 02:12:14,683 - root - INFO -   Total validation time: 4.176471710205078 sec
2021-11-09 02:12:22,879 - root - INFO - Time taken for epoch 2 is 8.19310736656189 sec, avg 499.9324208439884 samples/sec
2021-11-09 02:12:22,888 - root - INFO -   Avg train loss=0.030209
2021-11-09 02:12:23,721 - root - INFO -   Avg val loss=0.027962
2021-11-09 02:12:23,738 - root - INFO -   Total validation time: 0.8332135677337646 sec
2021-11-09 02:12:31,311 - root - INFO - Time taken for epoch 3 is 7.570817232131958 sec, avg 541.024816002137 samples/sec
2021-11-09 02:12:31,319 - root - INFO -   Avg train loss=0.021943
2021-11-09 02:12:31,887 - root - INFO -   Avg val loss=0.026240
2021-11-09 02:12:31,901 - root - INFO -   Total validation time: 0.5677013397216797 sec
```

Running a profile using these new options and loading in Nsight Systems looks like this:
![NSYS DALI AMP APEX JIT](tutorial_images/nsys_dali_amp_apex_jit.png)

and zoomed in to a single iteration:
![NSYS DALI AMP APEX JIT Zoomed](tutorial_images/nsys_dali_amp_apex_jit_zoomed.png)

Running this case with APEX and JIT enabled using benchy on Perlmutter results in the following throughput measurements:
```
BENCHY::SUMMARY::IO average trial throughput: 908.191 +/- 29.696
BENCHY::SUMMARY:: SYNTHETIC average trial throughput: 712.045 +/- 0.168
BENCHY::SUMMARY::FULL average trial throughput: 543.673 +/- 4.639
```
We see a modest gain in the `SYNTHETIC` throughput, resuling in a slight increase in the `FULL` throughput.

### Using CUDA Graphs (advanced)

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

PyTorch `DistributedDataParallel`, or DDP for short, is flexible and can initialize process groups with a variety of methods. For this code, we will use the standard approach of initializing via environment variables, which can be easily read from the slurm environment. Take a look at the `export_DDP_vars.sh` helper script, which is used by our job script to expose for PyTorch DDP the global rank and node-local rank of each process, along with the total number of ranks and the address and port to use for network communication. In the [`train.py`](train.py) script, near the bottom in the main script execution, we set up the distributed backend using these environment variables via `torch.distributed.init_proces_group`.

When distributing a batch of samples in DDP training, we must make sure each rank gets a properly-sized subset of the full batch. See if you can find where we use the `DistributedSampler` from PyTorch to properly partition the data in [`utils/data_loader.py`](utils/data_loader.py). Note that in this particular example, we are already cropping samples randomly form a large simulation volume, so the partitioning does not ensure each rank gets unique data, but simply shortens the number of steps needed to complete an "epoch". For datasets with a fixed number of unique samples, `DistributedSampler` will also ensure each rank sees a unique minibatch.

In `train.py`, after our U-Net model is constructed,
we convert it to a distributed data parallel model by wrapping it as:
```
model = DistributedDataParallel(model, device_ids=[local_rank])
```

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

# SC22 Deep Learning at Scale Tutorial (Summit Commands)

Refear to main `README.md` for details

Data location on Summit: `/gpfs/alpine/stf011/world-shared/atsaris/SC22_tutorial_data`

## Installation and Setup

### Software environment

For running jobs on Summit, we will use training accounts which are provided under the `TRN001` project. The script `submit_summit.sh` included in the repository is configured to work automatically as is.
* `-P TRN001` is required for training accounts

To begin, start a terminal and login to Summit:
```bash
mkdir -p $WORLDWORK/trn001/$USER/
cd $WORLDWORK/trn001/$USER/
git clone https://github.com/tsaris/sc22-dl-tutorial.git
cd sc22-dl-tutorial
mkdir logs
```

## Single GPU training

```
bsub -P stf218 -W 0:30 -J sc22.tut -o logs/sc22.tut.o%J -nnodes 1 -alloc_flags "gpumps smt4" -q debug "./submit_summit.sh -g 1 --config=short_sm --num_epochs 3 --enable_benchy"

bsub -P stf218 -W 0:30 -J sc22.tut -o logs/sc22.tut.o%J -nnodes 1 -alloc_flags "gpumps smt4" -q debug "./submit_summit.sh -g 1 --config=short_opt_sm --num_epochs 10 --enable_benchy"
```

## Single GPU performance profiling and optimization

### Profiling with Nsight Systems
#### Adding NVTX ranges and profiler controls

#### Using the benchy profiling tool

### Data loading optimizations
#### Improving the native PyTorch dataloader performance

#### Using NVIDIA DALI

### Enabling Mixed Precision Training

### Just-in-time (JIT) compiliation and APEX fused optimizers

### Using CUDA Graphs (optional)

### Full training with optimizations

## Distributed GPU training

### Code basics

### Large batch convergence

## Multi-GPU performance profiling and optimization

### Weak and Strong Throughput Scaling

### Profiling with Nsight Systems

### Adjusting DistributedDataParallel options

## Putting it all together

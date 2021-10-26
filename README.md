# SC21 Deep Learning at Scale Tutorial

This repository contains the example code material for the SC20 tutorial:
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


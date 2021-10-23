#!/bin/bash 
#SBATCH --nodes=1  --time=4:00:00  
#SBATCH -C gpu 
#SBATCH --account m1759
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --gpus-per-task 4
#SBATCH -J singlenode
#SBATCH -o %x-%j.out


ngpu=$1

hostname

~/dummy

if [[ "$ngpu" -eq 1 ]]; then
  # Single GPU
  shifter --image=nvcr.io/nvidia/pytorch:21.03-py3 --module=gpu \
      --env PYTHONUSERBASE=$HOME/.local/perlmutter/nvcr_pytorch_21.03-py3 \
      bash -c 'python train.py --config=A100_crop64_1GPU'
fi

if [[ "$ngpu" -eq 4 ]]; then
  # multi-GPU
  shifter --image=nvcr.io/nvidia/pytorch:21.03-py3 --module=gpu \
      --env PYTHONUSERBASE=$HOME/.local/perlmutter/nvcr_pytorch_21.03-py3 \
      bash -c 'python -m torch.distributed.launch --nproc_per_node=4 train.py --config=A100_crop64_4GPU'
fi


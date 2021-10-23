#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=4:00:00
#SBATCH -C gpu 
#SBATCH --account m1759
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --gpus-per-task 4
#SBATCH -J multinode
#SBATCH -o %x-%j.out

gpupernode=4
ngpu=$(expr $SLURM_JOB_NUM_NODES \* $gpupernode)

bash nodelist.sh $gpupernode A100_crop64_${ngpu}GPU


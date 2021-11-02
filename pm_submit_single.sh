#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --time=0:10:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -J crop64-single
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/j/joshr/nbody2hydro/datacopies
LOGDIR=${SCRATCH}/ampUNet/logs

mkdir -p ${LOGDIR}

hostname

#~/dummy

srun -u shifter --module=gpu \
    -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c '
    source export_DDP_vars.sh
    python train.py --config=A100_crop64_sqrt --data_loader_config=dali-lowmem
    '


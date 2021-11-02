#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --nodes=2
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --time=0:10:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -J crop64-multi
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/j/joshr/nbody2hydro/datacopies
LOGDIR=${SCRATCH}/ampUNet/logs

mkdir -p ${LOGDIR}

hostname

#~/dummy

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    PROFILE_OUTPUT=/logs/$SLURM_JOB_ID
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $PROFILE_OUTPUT"
fi

set -x
srun -u shifter --module=gpu \
    -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train.py \
        --config=A100_crop64_sqrt \
        --data_loader_config=dali-lowmem \
        --enable_benchy
    "


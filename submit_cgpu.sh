#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 10
#SBATCH --gpus-per-task 1
#SBATCH --time=0:30:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -J pm-crop64
#SBATCH -o %x-%j.out

DATADIR=/global/cscratch1/sd/sfarrell/sc21-dl-tutorial/data
LOGDIR=${SCRATCH}/sc21-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

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
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train.py --config=V100_crop64_sqrt ${args}
    "

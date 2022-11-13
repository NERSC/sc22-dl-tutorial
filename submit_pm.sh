#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A ntrain4_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=0:15:00
#SBATCH --image=nersc/sc22-dl-tutorial:latest
#SBATCH --reservation=sc22_tutorial
#SBATCH -J pm-crop64
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/j/joshr/nbody2hydro/datacopies
LOGDIR=${SCRATCH}/sc22-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

hostname

export NCCL_NET_GDR_LEVEL=PHB

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

BENCHY_CONFIG=benchy-conf.yaml
BENCHY_OUTPUT=${BENCHY_OUTPUT:-"benchy_output"}
sed "s/.*output_filename.*/        output_filename: ${BENCHY_OUTPUT}.json/" ${BENCHY_CONFIG} > benchy-run-${SLURM_JOBID}.yaml
export BENCHY_CONFIG_FILE=benchy-run-${SLURM_JOBID}.yaml
export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train.py ${args}
    "
rm benchy-run-${SLURM_JOBID}.yaml

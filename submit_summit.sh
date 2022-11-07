#!/bin/bash

nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
module use /sw/aaims/summit/modulefiles
module load open-ce-pyt

DATADIR=/gpfs/alpine/stf011/world-shared/atsaris/SC22_tutorial_data
LOGDIR=$WORLDWORK/stf011/$USER/sc22-dl-tutorial/logs
mkdir -p ${LOGDIR}

if [ "$1" != "-g" ]; then
    echo "You need to specify -g for gpus per node as a first argument" ; exit 1;
fi

re='^[0-9]+$'
if ! [[ $2 =~ $re ]] ; then
   echo "error: gpus per node must be a number" >&2; exit 1
fi
args="${@:3}"

hostname

#~/dummy

export NCCL_NET_GDR_LEVEL=PHB

BENCHY_CONFIG=benchy-conf.yaml
BENCHY_OUTPUT=${BENCHY_OUTPUT:-"benchy_output"}
sed "s/.*output_filename.*/        output_filename: ${BENCHY_OUTPUT}.json/" ${BENCHY_CONFIG} > benchy-run-${SLURM_JOBID}.yaml
export BENCHY_CONFIG_FILE=benchy-run-${SLURM_JOBID}.yaml
export MASTER_ADDR=$(hostname)

set -x

if [ -z "$ENABLE_PROFILING" ]
then
    ENABLE_PROFILING=0
fi

if [ -z "$PROFILE_OUTPUT" ]
then
    PROFILE_OUTPUT=0
fi

time jsrun -n${nnodes} -a"$(($2))" -c42 -g"$(($2))" -r1 \
     --smpiargs="-disable_gpu_hooks" \
     --bind=proportional-packed:7 \
     --launch_distribution=packed stdbuf -o0 \
     ./launch_summit.sh \
     "./run_summit.sh ${ENABLE_PROFILING} ${PROFILE_OUTPUT} ${args}"

rm benchy-run-${SLURM_JOBID}.yaml

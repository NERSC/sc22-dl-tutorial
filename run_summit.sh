#!/bin/bash

ENABLE_PROFILING=${1}
PROFILE_OUTPUT=${2}

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ] && [ "$PMIX_RANK" -eq 0 ] ; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,nvtx,osrt --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

echo ${PROFILE_CMD}
  
source export_DDP_vars_summit.sh && \
    ${PROFILE_CMD} python train.py ${@:3}


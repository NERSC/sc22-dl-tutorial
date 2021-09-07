#!/bin/bash

nproc_per_node=$1
config=$2
image="nvcr.io/nvidia/pytorch:21.03-py3"

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master_node=${nodes_array[0]}
#master_addr=$(srun --nodes=1 --ntasks=1 -w $master_node bash -c 'echo $SLURM_LAUNCH_NODE_IPADDR')
master_addr=$(hostname)
worker_num=$(($SLURM_JOB_NUM_NODES))

# Loop over nodes and submit training tasks
for ((  node_rank=0; node_rank<$worker_num; node_rank++ ))
do
  node=${nodes_array[$node_rank]}
  launch="python -m torch.distributed.launch \
    --nproc_per_node=$nproc_per_node --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$node_rank --master_addr=$master_addr \
    train.py --config=$config"
  echo "Submitting node # $node_rank, $node, with cmd $launch"

  # Launch one SLURM task per node, and use torch distributed launch utility
  # to spawn training worker processes; one per GPU
  srun -u -N 1 -n 1 -w $node ~/dummy # silly fix for CUDA unknown errors
  srun -u -N 1 -n 1 -w $node \
    shifter --image=$image --module=gpu \
    --env PYTHONUSERBASE=$HOME/.local/perlmutter/nvcr_pytorch_21.03-py3 \
    bash -c "$launch" &
  pids[${node_rank}]=$!
  echo $pids
done

# Wait for completion
for pid in ${pids[*]}; do
    wait $pid
done

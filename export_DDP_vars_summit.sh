export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | grep -v login | head -1)
export MASTER_PORT=29500 # default from torch launcher

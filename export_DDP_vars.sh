export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher

# Special NCCL libfabric setup on Perlmutter
NCCL_HOME=/pscratch/sd/s/sfarrell/nccl-ofi/nccl-2.15.5-plugin-rel3/install
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/deps/lib:$LD_LIBRARY_PATH
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"

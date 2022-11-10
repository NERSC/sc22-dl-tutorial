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
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1

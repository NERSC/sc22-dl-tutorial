#!/bin/bash

tag=crop64_warm128sqrt_L1only_endLR10

sbatch --job-name=crop64_1gpu_L1only pm_submit_singlenode.slr 1

sbatch --job-name=${tag}_4gpu pm_submit_singlenode.slr 4

sbatch --job-name=${tag}_8gpu --nodes=2 pm_submit_multinode.slr

sbatch --job-name=${tag}_32gpu --nodes=8 pm_submit_multinode.slr

sbatch --job-name=${tag}_128gpu --nodes=32 pm_submit_multinode.slr


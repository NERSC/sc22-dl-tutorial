
import numpy as np

def get_data_loader_distributed(params, world_rank, device_id=0):
    if params.data_loader_config.startswith("dali"):
        if params.data_loader_config == "dali-lowmem":
            from .data_loader_dali import get_data_loader_distributed
        else:
            raise NotImplementedError(f"Error, data loader config {params.data_loader_config} not supported!")
    else:
        from .data_loader import get_data_loader_distributed

    return get_data_loader_distributed(params, world_rank, device_id)

def lr_schedule(optimizer, iternum, nGPU=1, scaling='none', start_lr=1e-4, tot_steps=1000, end_lr=0., warmup_steps=0):
    if scaling=='sqrt':
        init_lr = np.sqrt(nGPU)*start_lr
    elif scaling=='linear':
        init_lr = nGPU*start_lr
    elif scaling=='none':
        init_lr = start_lr

    if nGPU > 1 and scaling != 'none':
        # warm-up lr rate
        if iternum<warmup_steps:
            lr = (iternum/warmup_steps)*init_lr
        else:
            lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + np.cos(np.pi * (iternum - warmup_steps)/tot_steps))
    else:
        lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + np.cos(np.pi * iternum/tot_steps))
    optimizer.param_groups[0]['lr'] = lr


import sys
import os
import gc
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed, lr_schedule
from networks import UNet

import apex.optimizers as aoptim


def capture_model(params, model, loss_func, lambda_rho, scaler, capture_stream, device, num_warmup=20):
  print("Capturing Model")
  inp_shape = (params.batch_size, 4, params.data_size, params.data_size, params.data_size)
  tar_shape = (params.batch_size, 5, params.data_size, params.data_size, params.data_size)
  static_input = torch.zeros(inp_shape, dtype=torch.float32, device=device)
  static_label = torch.zeros(tar_shape, dtype=torch.float32, device=device)

  capture_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(capture_stream):
    for _ in range(num_warmup):
      model.zero_grad(set_to_none=True)
      with autocast(params.enable_amp):
        static_output = model(static_input)
        static_loss = loss_func(static_output, static_label, lambda_rho)

      if params.enable_amp:
        scaler.scale(static_loss).backward()
      else:
        static_loss.backward()

    # sync here
    capture_stream.synchronize()
    
    gc.collect()
    torch.cuda.empty_cache()

    # create graph
    graph = torch.cuda.CUDAGraph()

    # zero grads before capture:
    model.zero_grad(set_to_none=True)

    # start capture
    graph.capture_begin()

    with autocast(params.enable_amp):
      static_output = model(static_input)
      static_loss = loss_func(static_output, static_label, lambda_rho)

      if params.enable_amp:
        scaler.scale(static_loss).backward()
      else:
        static_loss.backward()

    graph.capture_end()

  torch.cuda.current_stream().wait_stream(capture_stream)

  return graph, static_input, static_output, static_label, static_loss
  

def train(params, args, local_rank, world_rank, world_size):
  # set device and benchmark mode
  torch.backends.cudnn.benchmark = True
  torch.cuda.set_device(local_rank)
  device = torch.device('cuda:%d'%local_rank)

  # get data loader
  logging.info('rank %d, begin data loader init'%world_rank)
  train_data_loader, val_data_loader = get_data_loader_distributed(params, world_rank, device.index)
  logging.info('rank %d, data loader initialized with config %s'%(world_rank, params.data_loader_config))

  # create model
  model = UNet.UNet(params).to(device)
  model.apply(model.get_weights_function(params.weight_init))
  
  if params.enable_amp:
    scaler = GradScaler()
  else:
    scaler = None

  capture_stream = torch.cuda.Stream()
  with torch.cuda.stream(capture_stream):
    if params.distributed:
      model = DistributedDataParallel(model, device_ids=[local_rank])
  capture_stream.synchronize()

  if params.enable_apex:
    optimizer = aoptim.FusedAdam(model.parameters(), lr = params.lr_schedule['start_lr'],
                                 adam_w_mode=False, set_grad_none=True)
  else:
    optimizer = optim.Adam(model.parameters(), lr = params.lr_schedule['start_lr'])

  if params.enable_jit:
    torch._C._jit_set_nvfuser_enabled(True)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    #torch._C._jit_set_profiling_executor(True)
    #torch._C._jit_set_profiling_mode(True)
    #torch._C._jit_set_bailout_depth(20)
    model_handle = model.module if params.distributed else model
    model_handle = torch.jit.script(model_handle)  

  # select loss function
  if params.enable_jit:
    loss_func = UNet.loss_func_opt_final
    lambda_rho = torch.zeros((1,5,1,1,1), dtype=torch.float32).to(device)
    lambda_rho[:,0,:,:,:] = params.lambda_rho
  else:
    loss_func = UNet.loss_func
    lambda_rho = params.lambda_rho

  # capture the model
  graph, static_input, static_output, static_label, static_loss = capture_model(params, model, loss_func, lambda_rho, scaler,
                                                                                capture_stream, device, num_warmup=20)
  
  # start training
  iters = 0
  startEpoch = 0
  params.lr_schedule['tot_steps'] = params.num_epochs*len(train_data_loader)

  if args.no_val:
    if world_rank==0:
      logging.info(model)
      logging.info("Warming up 20 iters ...")
    wstart = time.time()
    for i, data in enumerate(train_data_loader, 0):
      iters += 1
      if iters>20:
          break
      inp, tar = map(lambda x: x.to(device), data)
      tr_start = time.time()
      b_size = inp.size(0)
      
      static_input.copy_(inp)
      static_label.copy_(tar)
      graph.replay()

      if params.enable_amp:
        scaler.step(optimizer)
        scaler.update()
      else:
        optimizer.step()
    wend = time.time()

    if world_rank==0:
      logging.info("Warmup took {} seconds, avg {} iters/sec".format(wend-wstart, 20/(wend-wstart)))
    
  if world_rank==0: 
    logging.info("Starting Training Loop...")

  iters = 0
  t1 = time.time()
  for epoch in range(startEpoch, startEpoch+params.num_epochs):
    start = time.time()
    tr_loss = []
    tr_time = 0.
    dat_time = 0.
    log_time = 0.

    model.train()
    step_count = 0
    for i, data in enumerate(train_data_loader, 0):
      iters += 1
      dat_start = time.time()
      inp, tar = map(lambda x: x.to(device), data)
      tr_start = time.time()
      b_size = inp.size(0)
      
      lr_schedule(optimizer, iters, nGPU=world_size, **params.lr_schedule)

      static_input.copy_(inp)
      static_label.copy_(tar)
      graph.replay()

      if params.enable_amp:
        scaler.step(optimizer)
        scaler.update()
      else:
        optimizer.step()

      tr_loss.append(static_loss.item())

      tr_end = time.time()
      tr_time += tr_end - tr_start
      dat_time += tr_start - dat_start
      step_count += 1

    end = time.time()
    if world_rank==0:
      logging.info('Time taken for epoch {} is {} sec, avg {} iters/sec'.format(epoch + 1, end-start, step_count/(end-start)))
      logging.info('  Step breakdown:')
      logging.info('  Data to GPU: %.2f ms, U-Net fwd/back/optim: %.2f ms'%(1e3*dat_time/step_count, 1e3*tr_time/step_count))
      logging.info('  Avg train loss=%f'%np.mean(tr_loss))
      args.tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)
      args.tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters)
      args.tboard_writer.add_scalar('Avg iters per sec', step_count/(end-start), iters)

    val_start = time.time()
    val_loss = []
    model.eval()
    if not args.no_val and world_rank==0:
      with torch.no_grad():
        for i, data in enumerate(val_data_loader, 0):
          with autocast(params.enable_amp):
            inp, tar = map(lambda x: x.to(device), data)
            gen = model(inp)
            loss = loss_func(gen, tar, lambda_rho)
            val_loss.append(loss.item())
      val_end = time.time()
      logging.info('  Avg val loss=%f'%np.mean(val_loss))
      logging.info('  Total validation time: {} sec'.format(val_end - val_start)) 
      args.tboard_writer.add_scalar('Loss/valid', np.mean(val_loss), iters)
      args.tboard_writer.flush()

  t2 = time.time()
  tottime = t2 - t1



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
  parser.add_argument("--config", default='base', type=str)
  parser.add_argument("--no_val", action='store_true', help='skip validation steps (for profiling train only)')
  parser.add_argument("--enable_amp", action='store_true', help='enable automatic mixed precision')
  parser.add_argument("--enable_apex", action='store_true', help='enable apex fused Adam optimizer')
  parser.add_argument("--enable_jit", action='store_true', help='enable JIT compilation')
  parser.add_argument("--enable_benchy", action='store_true', help='enable benchy tool usage')
  parser.add_argument("--data_loader_config", default=None, type=str,
                      choices=['synthetic', 'inmem', 'lowmem', 'dali-lowmem'],
                      help="dataloader configuration. choices: 'synthetic', 'inmem', 'lowmem', 'dali-lowmem'")
  args = parser.parse_args()
  
  run_num = args.run_num

  params = YParams(os.path.abspath(args.yaml_config), args.config)

  # Update config with modified args
  params.update({"enable_amp" : args.enable_amp,
                 "enable_apex" : args.enable_apex,
                 "enable_jit" : args.enable_jit,
                 "enable_benchy" : args.enable_benchy})

  if args.data_loader_config:
      params.update({"data_loader_config" : args.data_loader_config})

  params.distributed = False
  if 'WORLD_SIZE' in os.environ:
    params.distributed = int(os.environ['WORLD_SIZE']) > 1
    world_size = int(os.environ['WORLD_SIZE'])
  else:
    world_size = 1

  world_rank = 0
  local_rank = 0
  if params.distributed:
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    world_rank = torch.distributed.get_rank() 
    local_rank = int(os.environ['LOCAL_RANK'])

  # Set up directory
  baseDir = params.expdir
  expDir = os.path.join(baseDir, args.config+'/'+str(run_num)+'/')
  if world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()
    args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

  params.experiment_dir = os.path.abspath(expDir)

  train(params, args, local_rank, world_rank, world_size)
  if params.distributed:
    torch.distributed.barrier()
  logging.info('DONE ---- rank %d'%world_rank)


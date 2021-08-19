import sys
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader import get_data_loader_distributed
from networks import UNet


def train(params, args, world_rank):
  logging.info('rank %d, begin data loader init'%world_rank)
  train_data_loader, val_data_loader = get_data_loader_distributed(params, world_rank)
  logging.info('rank %d, data loader initialized with config %s'%(world_rank, params.data_loader_config))

  model = UNet.UNet(params).cuda()
  model.apply(model.get_weights_function(params.weight_init))

  optimizer = optim.Adam(model.parameters(), lr = params.lr)
  if params.enable_amp:
    scaler = GradScaler()

  if params.distributed:
    model = DistributedDataParallel(model) 

  iters = 0
  startEpoch = 0
  device = torch.cuda.current_device()
  
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
      
      optimizer.zero_grad()
      with autocast(params.enable_amp):
        gen = model(inp)
        loss = UNet.loss_func(gen, tar, params)

      if params.enable_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()
    wend = time.time()

    if world_rank==0:
      logging.info("Warmup took {} seconds, avg {} iters/sec".format(wend-wstart, 20/(wend-wstart)))
    
  if world_rank==0: 
    logging.info("Starting Training Loop...")

  t1 = time.time()
  for epoch in range(startEpoch, startEpoch+params.num_epochs):
    start = time.time()
    tr_loss = []
    tr_time = 0.
    dat_time = 0.
    log_time = 0.

    model.train()
    for i, data in enumerate(train_data_loader, 0):
      iters += 1
      dat_start = time.time()
      inp, tar = map(lambda x: x.to(device), data)
      tr_start = time.time()
      b_size = inp.size(0)
      
      optimizer.zero_grad()
      with autocast(params.enable_amp):
        gen = model(inp)
        loss = UNet.loss_func(gen, tar, params)
        tr_loss.append(loss.item())

      if params.enable_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()
      
      tr_end = time.time()
      tr_time += tr_end - tr_start
      dat_time += tr_start - dat_start

    end = time.time()
    if world_rank==0:
      logging.info('Time taken for epoch {} is {} sec, avg {} iters/sec'.format(epoch + 1, end-start, params.Nsamples/(end-start)))
      logging.info('  Step breakdown:')
      logging.info('  Data to GPU: %.2f ms, U-Net fwd/back/optim: %.2f ms'%(1e3*dat_time/params.Nsamples, 1e3*tr_time/params.Nsamples))
      logging.info('  Avg train loss=%f'%np.mean(tr_loss))
      args.tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)

    val_start = time.time()
    val_loss = []
    model.eval()
    if not args.no_val and world_rank==0:
      for i, data in enumerate(val_data_loader, 0):
        with autocast(params.enable_amp):
          with torch.no_grad():
            inp, tar = map(lambda x: x.to(device), data)
            gen = model(inp)
            loss = UNet.loss_func(gen, tar, params)
            val_loss.append(loss.item())
      val_end = time.time()
      logging.info('  Avg val loss=%f'%np.mean(val_loss))
      logging.info('  Total validation time: {} sec'.format(val_end - val_start)) 
      args.tboard_writer.add_scalar('Loss/valid', np.mean(val_loss), iters)

  t2 = time.time()
  tottime = t2 - t1
  if world_rank==0:
    logging.info('Total time is {} sec, avg {} iters/sec'.format(tottime, params.Nsamples*params.num_epochs/tottime))



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", default=0, type=int)
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
  parser.add_argument("--config", default='base', type=str)
  parser.add_argument("--no_val", action='store_true', help='skip validation steps (for profiling train only)')
  args = parser.parse_args()
  
  run_num = args.run_num

  params = YParams(os.path.abspath(args.yaml_config), args.config)

  params.distributed = False
  if 'WORLD_SIZE' in os.environ:
    params.distributed = int(os.environ['WORLD_SIZE']) > 1

  world_rank = 0
  if params.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.gpu = args.local_rank
    world_rank = torch.distributed.get_rank() 

  torch.backends.cudnn.benchmark = True

  # Set up directory
  baseDir = params.expdir
  expDir = os.path.join(baseDir, args.config+'/'+str(run_num)+'/')
  if  world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()
    args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

  params.experiment_dir = os.path.abspath(expDir)

  train(params, args, world_rank)
  logging.info('DONE ---- rank %d'%world_rank)


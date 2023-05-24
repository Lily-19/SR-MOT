from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter

# 2022.12.25 前一帧只需要模型输出，不需要计算loss
class ModleWithoutLoss(torch.nn.Module):
  def __init__(self, model):
    super(ModleWithoutLoss, self).__init__()
    self.model = model

  def forward(self, batch):
    outputs = self.model(batch['input']) #2023.04.05,这里只需要输出['id'] 
    return outputs # 保持和输入loss计算的output一致的格式 list里面是个dict

# 2022.12.25 当前帧的loss计算需要前一帧的特征和标签
class ModleWithLossPre(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLossPre, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch, pre_batch, pre_output):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch, pre_batch, pre_output)
    return outputs[-1], loss, loss_stats

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    # 2022.12.25前一帧只输出模型结果，不计算loss
    self.model_without_loss = ModleWithoutLoss(model)
    # 当前帧计算loss时需要输入前一帧的特征
    self.model_with_loss_pre = ModleWithLossPre(model, self.loss)
    # self.model_with_loss = ModleWithLoss(model, self.loss)
    self.optimizer.add_param_group({'params': self.loss.parameters()})

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      # self.model_with_loss = DataParallel(
      #   self.model_with_loss, device_ids=gpus,
      #   chunk_sizes=chunk_sizes).to(device)
      self.model_with_loss_pre = DataParallel(
        self.model_with_loss_pre, device_ids=gpus,
        chunk_sizes=chunk_sizes).to(device)
      self.model_without_loss = DataParallel(
        self.model_without_loss, device_ids=gpus,
        chunk_sizes=chunk_sizes).to(device)
    else:
      # self.model_with_loss = self.model_with_loss.to(device)
      self.model_with_loss_pre = self.model_with_loss_pre.to(device)
      self.model_without_loss = self.model_without_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    # 2022.12.25
    model_without_loss = self.model_without_loss
    model_with_loss_pre = self.model_with_loss_pre
    # model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss_pre.train()
      # model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss_pre = self.model_with_loss_pre.module
        # model_with_loss = self.model_with_loss.module
      # model_with_loss.eval()
      model_with_loss_pre.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      # 2022.12.25 batch是一个list，里面含两帧图片（每一帧是一个dict）
      img_pre = batch[0]
      img = batch[1]

      # for k in batch:
      #   if k != 'meta':
      #     batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      for k in img_pre:
        if k != 'meta':
          img_pre[k] = img_pre[k].to(device=opt.device, non_blocking=True)
      for k in img:
        if k != 'meta':
          img[k] = img[k].to(device=opt.device, non_blocking=True)

      pre_output = model_without_loss(img_pre)
      output, loss, loss_stats = model_with_loss_pre(img, img_pre, pre_output)


      # output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        # avg_loss_stats[l].update(
        #   loss_stats[l].mean().item(), batch['input'].size(0))
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), img['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.test:
        # self.save_result(output, batch, results)
        self.save_result(output, img, results)
      del output, loss, loss_stats, img
      # del output, loss, loss_stats, batch
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results

  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer
import math

class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        # self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        # self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        # 2023.2.7 符合push—pull-loss量级的新权重
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-0.5 * torch.ones(1))

    # 2022.12.25
    def forward(self, outputs, batch, pre_batch, pre_output):
        # def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        frame_loss = 0 # 2022.12.2，用于约束同一帧中不同身份特征向量距离的loss
        
        for s in range(opt.num_stacks):
            output = outputs[s]
            output_pre = pre_output[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:

                    id_loss += self.IDLoss(id_output, id_target)

            # 2023.02.07 对比学习损失 infoNCE
            if opt.id_weight == 0:
                old_id_feature = output['id']
                bs = old_id_feature.size(0)  # batch_size
                id_w = old_id_feature.size(2)
                id_h = old_id_feature.size(3)
                id_loss = torch.tensor(0, dtype=hm_loss.dtype, device=hm_loss.device)
                old_id_feature_pre = output_pre['id']
                ind_x_pre = pre_batch['ind'] // id_h  # 行号
                ind_y_pre = pre_batch['ind'] % id_h  # 列号
                ind_x = batch['ind'] // id_h  # 行号
                ind_y = batch['ind'] % id_h  # 列号
                for b in range(bs):  # 逐个样本考虑
                    # 根据行号和列号选取对应位置处的id_feature
                    cur_feature = old_id_feature[b, :, ind_x[b], ind_y[b]] # 128*500
                    pre_feature = old_id_feature_pre[b, :, ind_x_pre[b], ind_y_pre[b]]
                    # 置换维度
                    cur_feature = cur_feature.permute(1, 0) # 500*128 看作是bs*128
                    pre_feature = pre_feature.permute(1, 0)
                    # 根据有效标志位筛选特征
                    tag = batch['reg_mask'][b] * pre_batch['reg_mask'][b]  # 有效位置标记
                    tag = tag.bool()
                    cur_feature = cur_feature[tag]
                    pre_feature = pre_feature[tag]
                    obj_num = cur_feature.shape[0]
                    if obj_num == 0:
                        continue
                    # 做归一化
                    z_i = F.normalize(cur_feature, dim=1)
                    z_j = F.normalize(pre_feature, dim=1)

                    representations = torch.cat([z_i, z_j], dim=0) # (2*obj_num, 128)
                    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2) #(2*obj_num, 2*obj_num)

                    sim_ij = torch.diag(similarity_matrix, obj_num) # obj_num
                    sim_ji = torch.diag(similarity_matrix, -obj_num)
                    positives = torch.cat([sim_ij, sim_ji], dim=0) # 2*obj_num，相同目标不同数据增强后的相似度

                    nominator = torch.exp(positives / 0.05) # 2*obj_num
                    negatives_mask = (~torch.eye(obj_num*2, obj_num*2, dtype=bool, device=hm_loss.device)).float() # 用于屏蔽自己与自己的相似度（全是1，无意义）
                    denominator = negatives_mask * torch.exp(similarity_matrix / 0.05) # 2*obj_num

                    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=-1)) # 2*obj_num
                    frame_loss += torch.sum(loss_partial) / (2*obj_num)

                id_loss = torch.tensor(0, dtype=hm_loss.dtype, device=hm_loss.device)
            

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.id_weight > 0:
            if opt.multi_loss == 'uncertainty':
                loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
                loss *= 0.5
            else:
                loss = det_loss + 0.1 * id_loss
        else:
            loss = det_loss
            
        # 2022.12.19 调整权重
        if frame_loss == 0:
            frame_loss = torch.tensor(0, dtype=hm_loss.dtype, device=hm_loss.device)
        loss = opt.det_weight * loss + opt.reid_weight * frame_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss,
                      'frame_loss': frame_loss}
        return loss, loss_stats

def xy_dist(x,y):
    return math.sqrt(sum())

class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        # loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'frame_loss'] # 2022.12.23

        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import os,sys
import torch, pdb
import numpy as np
from config import config
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from utils.box_ops import box_iou

class RelationNet(nn.Module):

    def __init__(self, d_model, activation = 'relu', iou_thr = 0.4, neighbors = 10):
        
        super().__init__()

        assert neighbors > 0
        self.d_model = d_model
        self.iou_thr = iou_thr
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

        self.linear3 = nn.Linear(2 * d_model + 64, d_model)
        self.linear4 = nn.Linear(d_model, d_model)

        self.linear5 = nn.Linear(d_model, d_model)

        self.activation = _get_activation_fn(activation)
        self.top_k = neighbors

    def _get_center(self, boxes):

        assert boxes.shape[-1] >= 4
        ctr = 0.5 * (boxes[..., 2:4] + boxes[..., :2])
        return ctr
    
    def _get_hw(self, boxes):

        assert boxes.shape[-1] >= 4
        hw = boxes[..., 2:4] - boxes[..., :2]
        return hw

    @torch.no_grad()
    def sin_cos_encoder_v2(self, boxes, mask, indices):
        
        # 这一部分的主要工作是对detected boxes之间的关系进行sin/cos编码.
        eps = 1e-7
        nums = boxes.shape[1]
        neighbors = torch.gather(boxes.unsqueeze(1).repeat_interleave(nums, 1), 2, indices[..., :4])
        
        cur_ctrs = self._get_center(boxes)
        neighbor_ctrs = self._get_center(neighbors)

        cur_hws = self._get_hw(boxes)
        neighbor_hws = self._get_hw(neighbors)

        delta_ctrs = neighbor_ctrs - cur_ctrs.unsqueeze(2)
        log_ctrs = torch.log(torch.clamp(torch.abs(delta_ctrs), eps))

        delta_hws = neighbor_hws - cur_hws.unsqueeze(2)
        log_hws = torch.log(torch.clamp(torch.abs(delta_hws), eps))

        position_mat = torch.cat([log_ctrs, log_hws], dim = -1)
        masks = (mask * (1 - mask).transpose(1, 2)).unsqueeze(-1)
        cur_mask = torch.gather(masks, 2, indices[..., :1])
        waves = self._extract_position_embedding(position_mat)
        wave_features = waves * cur_mask
        
        return wave_features

    @torch.no_grad()
    def sin_cos_encoder(self, boxes, mask, indices):
        
        # 这一部分的主要工作是对detected boxes之间的关系进行sin/cos编码.
        eps = 1e-7
        cur_boxes, neighbors = boxes, boxes
        
        cur_ctrs = self._get_center(cur_boxes)
        neighbor_ctrs = self._get_center(neighbors)

        cur_hws = self._get_hw(cur_boxes)
        neighbor_hws = self._get_hw(neighbors)

        delta_ctrs = neighbor_ctrs.unsqueeze(2) - cur_ctrs.unsqueeze(1)
        log_ctrs = torch.log(torch.clamp(torch.abs(delta_ctrs), eps))

        delta_hws = neighbor_hws.unsqueeze(2) - cur_hws.unsqueeze(1)
        log_hws = torch.log(torch.clamp(torch.abs(delta_hws), eps))

        position_mat = torch.cat([log_ctrs, log_hws], dim = -1).transpose(1, 2)
        mask = (mask * (1 - mask).transpose(1, 2)).unsqueeze(-1)
        waves = self._extract_position_embedding(position_mat) * mask
        
        wave_features = torch.gather(waves, 2, indices.repeat_interleave(2, -1))
        
        return wave_features
    
    def _extract_position_embedding(self, position_mat, num_pos_feats=64,temperature=1000,):

        num_pos_feats = 128
        temperature = 10000
        scale = 2 * np.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device = position_mat.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_mat = scale * position_mat.unsqueeze(-1) / dim_t.reshape(1, 1, 1, 1, -1)
        pos = torch.stack((pos_mat[:, :, :, 0::2].sin(), pos_mat[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return pos

    def forward(self, target, container):

        assert isinstance(container, dict)

        pred_boxes = container['bboxes'].detach()
        mask = container['gather_mask']
        bs, num_boxes = pred_boxes.shape[:2]

        tgt = target['tgt'].reshape(bs, num_boxes, self.d_model)
        ious = target['ious']

        bs, num_queries = pred_boxes.shape[:2]
        c = tgt.shape[-1]
        
        # use masking to mask the overlap
        neg_mask = (1 - mask)
        overlaps = ious * mask.transpose(1, 2) * neg_mask

        indices = torch.argsort(-overlaps)[..., :self.top_k].unsqueeze(-1).repeat_interleave(c, dim = -1)
        
        nmask = mask.transpose(1, 2).repeat_interleave(num_queries, dim = 1)
        nmk = torch.gather(nmask, 2, indices[..., 0])
        ious = torch.gather(overlaps, 2 ,indices[..., 0])
        
        mk = nmk * (ious >= self.iou_thr)
        overs = (ious * mk).unsqueeze(-1).repeat_interleave(64, dim = -1)
        wave_features = self.sin_cos_encoder_v2(pred_boxes, neg_mask, indices) * mk.unsqueeze(-1)
        features = torch.cat([overs, wave_features], dim = -1)
   
        cur = self.linear2(self.activation(self.linear1(tgt)))
        features = self.linear4(self.activation(self.linear3(features)))
        
        # mask the features.
        cur_tgt = cur * neg_mask + (features * mk.unsqueeze(-1)).max(dim = 2)[0]
        
        # update feature of target
        cur_tgt = self.activation(self.linear5(cur_tgt)) * neg_mask

        return cur_tgt

def _get_activation_fn(activation):

    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_relation_net(d_model=256, activation='relu', iou_thr=0.4, neighbors = 10):

    return RelationNet(d_model, activation, iou_thr, neighbors)

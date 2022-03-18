# !/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Sparse-RCNN(github: https://github.com/PeizeSun/SparseR-CNN) created by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy, math
from typing import Optional, List
import torch, pdb
from torch import nn, Tensor
import torch.nn.functional as F
from config import config
from .relation_net import build_relation_net
from .dynamic_conv import build_dynamic_conv
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from utils.box_ops import box_iou

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class IterRCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model
        self.confidence_score = cfg.MODEL.SparseRCNN.CONFIDENCE_THR
        self.watershed = cfg.MODEL.SparseRCNN.WATERSHED
        self.low_confidence_score = cfg.MODEL.SparseRCNN.LOW_CONFIDENCE_THR
        self.iou_thr = cfg.MODEL.SparseRCNN.RELATION_IOU_THR

        # dynamic.
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.relation_net = build_relation_net(d_model, activation, self.iou_thr, neighbors=10)

        self.inst_interact = build_dynamic_conv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def _predict(self, fc_feature):
        
        cls_feature = reg_feature = fc_feature

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)

        return class_logits, bboxes_deltas

    def ffn_forward(self, tgt):

        # obj_feature.
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
    
    @staticmethod
    def with_pos_embed(tensor, pos = None):
        return tensor if pos is None else tensor + pos
    
    def compute_overlaps(self, bboxes):
        
        N, nr_boxes = bboxes.shape[:2]
        a = b = bboxes.reshape(-1, bboxes.shape[-1])
        overlaps, _ = box_iou(a, b)
        overlaps = overlaps.reshape(N, nr_boxes, N, nr_boxes)
        overlaps = torch.stack([overlaps[i, :, i] for i in range(N)], dim = 0)
        return overlaps

    def forward(self, features, container):
        
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        bboxes, pro_features = container['bboxes'], container['pro_features']
        pooler, level = container['pooler'], container['lid']
        cur_all_mask, query = container['next_mask'], container['auxi_query']

        assert level >= self.watershed
        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = [Boxes(bboxes[i]) for i in range(N)]
        
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        
        overlaps = self.compute_overlaps(bboxes)

        target = {'tgt': pro_features, 'ious': overlaps}
        
        # construct the relationship.
        pro_features = self.relation_net(target, container) 
        
        # modified to local self-attention.
        query = query.reshape(N, nr_boxes, self.d_model)
        q = k = self.with_pos_embed(pro_features, query).transpose(0, 1)
        v = pro_features.transpose(0, 1)
        
        device = overlaps.device
        o_mask = (overlaps < self.iou_thr).float()
        e_mask = torch.eye(nr_boxes).to(device).unsqueeze(0)
        mask = cur_all_mask * cur_all_mask.transpose(1, 2)
        attn_mask = (o_mask * mask + e_mask * (1 - cur_all_mask)) > 0
        attn_mask = attn_mask.repeat_interleave(self.nhead, 0)

        tgt2 = self.self_attn(q, k, value=v, attn_mask = attn_mask)[0].transpose(0, 1)
        pro_features = pro_features + self.dropout1(tgt2)
        pro_features = self.norm1(pro_features)

        pro_features = (pro_features * cur_all_mask).reshape(1, N*nr_boxes, self.d_model)
        
        # inst_interact.
        pro_features2 = self.inst_interact(pro_features, roi_features.detach())
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features = self.ffn_forward(obj_features) 

        fc_feature = obj_features.reshape(N * nr_boxes, -1) * cur_all_mask.reshape(N * nr_boxes, -1)
        
        class_logits, _ = self._predict(fc_feature)

        class_logits = class_logits.view(N, nr_boxes, -1)
        cls_scores = class_logits.sigmoid().max(dim=-1, keepdim=True)[0]

        tmp_container = container.copy()
        tmp = {'class_logits': class_logits, 'mask': cur_all_mask, 'pro_features': obj_features}
        tmp_container.update(tmp)
        
        return tmp_container

    def apply_deltas(self, deltas, boxes):
        
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

def build_iter_rcnn_head(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation):

    return IterRCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)

def _get_activation_fn(activation):
    
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

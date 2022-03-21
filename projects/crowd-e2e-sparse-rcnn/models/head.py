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
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .rcnn_head import build_rcnn_head
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .shared_head import build_iter_rcnn_head

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.iter_nums = cfg.MODEL.SparseRCNN.ITER_NUM
        self.watershed = cfg.MODEL.SparseRCNN.WATERSHED
        
        rcnn_head = build_rcnn_head(cfg, self.d_model, num_classes, dim_feedforward, nhead, dropout, activation)        
        self.head_series = _get_clones(rcnn_head, self.num_heads)

        iter_head = build_iter_rcnn_head(cfg, self.d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.iter_series = _get_clones(iter_head, self.iter_nums)

        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        
        # Init parameters.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features, auxi_proposal_features):

        inter_class_logits, inter_pred_bboxes = [], []
        inter_containers = []
        bs = len(features[0])
        bboxes = init_bboxes
        
        queries = (init_features,)  + auxi_proposal_features.split(self.d_model, -1)
        queries = [q.unsqueeze(0).repeat(1, bs, 1) for q in queries]
        proposal_features = queries[0].clone()
        
        container = {'bboxes': bboxes, 'pro_features':proposal_features, 'pooler': self.box_pooler}
        names = ['pro_features', 'pooler', 'cur_all_mask', 'lid', 'next_mask']
        for i in range(self.watershed):

            container.update({'lid': i})
            rcnn_head = self.head_series[i]
            tmp_container = rcnn_head(features, container)
            if self.return_intermediate:
                cls_logits, pred_boxes = tmp_container['class_logits'], tmp_container['bboxes']
                inter_class_logits.append(cls_logits)
                inter_pred_bboxes.append(pred_boxes)
                t_container = {k:v for k,v in tmp_container.items() if k not in names}
                inter_containers.append(t_container)

            bboxes = tmp_container['bboxes']
            tmp_container['bboxes'] = bboxes.detach() if i < self.watershed - 1 else bboxes
            container = tmp_container
        
        tgt = container['pro_features'].detach()
        features = [f.detach() for f in features]
        for i in range(self.iter_nums):
            
            query = queries[1].clone()
            lid = self.num_heads + i + 1
            container.update({'lid': lid, 'auxi_query': query, 'pro_features': tgt})

            iter_head = self.iter_series[i]
            tmp_container = iter_head(features, container)

            if self.return_intermediate:
                cls_logits, pred_boxes = tmp_container['class_logits'], tmp_container['bboxes']
                inter_class_logits.append(cls_logits)
                inter_pred_bboxes.append(pred_boxes)
                t_container = {k:v for k,v in tmp_container.items() if k not in names}
                inter_containers.append(t_container)

            tmp_container['bboxes'] = tmp_container['bboxes'].detach()
            container = tmp_container
        
        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), inter_containers

        return class_logits[None], pred_bboxes[None], inter_containers


def _get_clones(module, N):
    
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

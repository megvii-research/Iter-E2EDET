# !/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Sparse-RCNN(github: https://github.com/PeizeSun/SparseR-CNN) created by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
SparseRCNN model and criterion classes.
"""
import torch, pdb
import numpy as np
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
from config import config
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_ioa, box_iou
from utils.infrastructure import mask_minimumWeightMatching
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):

    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.ioa_thr = cfg.MODEL.SparseRCNN.IGNORE_THR
        self.watershed = cfg.MODEL.SparseRCNN.WATERSHED
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.SparseRCNN.ALPHA
            self.focal_loss_gamma = cfg.MODEL.SparseRCNN.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def _get_matcher(self, ):

        matcher_map = {True: self.regular_forward,
                       False: self.step_forward}
        return matcher_map

    @torch.no_grad()
    def forward(self, outputs, targets, level=1):
        
        return self._get_matcher()[level < self.watershed](outputs, targets)

        
    @torch.no_grad()
    def regular_forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])
        ign_bbox = torch.cat([v['ignore_xyxy'] for v in targets])
        ign_sizes = [len(v['ignore_xyxy']) for v in targets]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
        
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        ioas = box_ioa(out_bbox, ign_bbox).reshape(bs, num_queries, -1)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        costs = [c[i] for i, c in enumerate(C.split(sizes, -1))]
        ign_ioas = [ioa[i] for i, ioa in enumerate(ioas.split(ign_sizes, -1))]
        shapes = [ign.shape for ign in ign_ioas]

        out_bbox = out_bbox.reshape(bs, num_queries, out_bbox.shape[1])
        res = [self.atomic_matcher(c, ign_ioas[i]) for i, c in enumerate(costs)]
        index = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j, k in res]
        mask = torch.stack([k for _, _, k in res], dim = 0).to(C.device)        

        return index, mask

    @torch.no_grad()
    def atomic_matcher(self, cost, ign_ioa):

        device = cost.device
        num_queries = cost.shape[0]
        rows, cols = np.array([]), np.array([])
        mask = torch.zeros([num_queries, 1]).to(device)

        if cost.shape[0] < 1:
            return rows, cols, mask
        
        rows, cols = linear_sum_assignment(cost.cpu())
        
        mask = torch.ones(num_queries).to(device)
        if ign_ioa.shape[1]:
            ign_mask = (ign_ioa.max(1)[0] > self.ioa_thr).float()
            mask = 1 - ign_mask.scatter_(0, torch.as_tensor(rows).to(device).long(), 0)
      
        return rows, cols, mask.unsqueeze(-1)

    @torch.no_grad()
    def step_forward(self, outputs, targets):
        
        assert 'prev_indices' in outputs
        bs, num_queries = outputs["pred_logits"].shape[:2]
        cur_mask = outputs['container']['mask']
        gather_mask = outputs['container']['gather_mask']
        prev_indices = outputs['prev_indices']
        
        
        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])
        ign_bbox = torch.cat([v['ignore_xyxy'] for v in targets])
        ign_sizes = [len(v['ignore_xyxy']) for v in targets]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
        ign_ioas = box_ioa(out_bbox, ign_bbox).reshape(bs, num_queries, -1)
        overlaps, _ = box_iou(out_bbox, tgt_bbox)
        overlaps = overlaps.reshape(bs, num_queries, -1)
        
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        spatial_mask = self.spatial_prior(out_bbox, tgt_bbox)
        spatial_mask = spatial_mask.reshape(bs, num_queries, -1)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]

        spatial_mask = [s[i] for i,s in enumerate(spatial_mask.split(sizes, -1))]
        costs = [c[i] for i,c in enumerate(C.split(sizes, -1))]
        overlaps = [o[i] for i, o in enumerate(overlaps.split(sizes, -1))]
        ign_ioas = [ign_ioa[i] for i, ign_ioa in enumerate(ign_ioas.split(ign_sizes, -1))]

        out_bbox = out_bbox.reshape(bs, num_queries, out_bbox.shape[1])
        res = [self.match_independently(c, cur_mask[i], spatial_mask[i], gather_mask[i],overlaps[i], ign_ioas[i]) for i, c in enumerate(costs)]
        index =  [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j, k in res]
        mask = torch.stack([k for _,_,k in res], dim = 0)

        return index, mask

    @torch.no_grad()
    def match_independently(self, cost, cur_mask, spatial_mask, gather_mask, iou, ign_ioa):
        
        device = cost.device
        num_queries, num = cost.shape
        tmp = torch.tensor([]).long().to(device)
        
        mask = torch.zeros([num_queries, 1]).to(device)
        valid_mask = gather_mask * spatial_mask * (iou > 0).float()
        _, cols = mask_minimumWeightMatching(cost.cpu(), valid_mask.cpu())
        if cols.size == cost.shape[1]:
            return tmp, tmp, mask
        
        auxi_mask = cur_mask.repeat_interleave(num, 1)
        cols = torch.Tensor(cols).to(device).long()
        auxi_mask[..., cols] = 0
        
        mask = auxi_mask * spatial_mask * (iou > 0).float()
        rows, cols = mask_minimumWeightMatching(cost.cpu(), mask.cpu())
        rows, cols = torch.Tensor(rows).long(), torch.Tensor(cols).long()

        mask = torch.ones(num_queries,).to(device)
        if ign_ioa.shape[1]:
            ign_mask = (ign_ioa.max(1)[0] >= self.ioa_thr).float()
            mask = 1 - ign_mask.scatter_(0, torch.as_tensor(rows).to(device).long(), 0)
        
        mask = cur_mask * mask.unsqueeze(-1)
        rows, cols = rows.to(device), cols.to(device)
        return rows, cols, mask
    
    @torch.no_grad()
    def spatial_prior(self, dtboxes, gtboxes):

        area = self._compute_center_range(gtboxes)
        x1, y1, x2, y2 = torch.split(area, 1, -1)
        
        flag = (x1 <= x2).float() * (y1 <= y2).float()
        assert flag.sum() == flag.shape[0]
        
        x1y1, x2y2 = dtboxes[..., :2], dtboxes[...,2:4]
        ctrs = 0.5 * (x1y1 + x2y2)
        x, y = torch.split(ctrs, 1, -1)

        a = (x >= x1.transpose(0, 1))
        b = (y >= y1.transpose(0, 1))
        c = (x <= x2.transpose(0, 1))
        d = (y <= y2.transpose(0, 1))
        mask = (a & b & c & d).float()

        return mask

    def _compute_center_range(self, boxes, scale = 1):
        
        assert boxes.shape[-1] > 3
        x1y1, x2y2 = boxes[..., :2], boxes[..., 2:4]
        ctrs = 0.5 * (x1y1 + x2y2)
        wh = (x2y2 - x1y1) * scale
        a, b = ctrs - 0.5 * wh, ctrs + 0.5 * wh
        return torch.cat([a, b], dim = -1)

def build_matcher(cfg, cost_class, cost_bbox, cost_giou, use_focal):

    return HungarianMatcher(cfg, cost_class, cost_bbox, cost_giou, use_focal)

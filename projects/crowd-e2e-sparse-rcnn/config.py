# !/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Sparse-RCNN(github: https://github.com/PeizeSun/SparseR-CNN) created by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os, sys
import pdb, getpass
import os.path as osp
from detectron2.config import CfgNode as CN

def add_sparsercnn_config(cfg):
    """
    Add config for SparseRCNN.
    """
    cfg.MODEL.SparseRCNN = CN()
    cfg.MODEL.SparseRCNN.NUM_CLASSES = 80
    cfg.MODEL.SparseRCNN.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.SparseRCNN.NHEADS = 8
    cfg.MODEL.SparseRCNN.DROPOUT = 0.0
    cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.SparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.SparseRCNN.NUM_CLS = 1
    cfg.MODEL.SparseRCNN.NUM_REG = 3
    cfg.MODEL.SparseRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.SparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.SparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.SparseRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SparseRCNN.IGNORE_THR = 0.7
    
    # ITER
    cfg.MODEL.SparseRCNN.WATERSHED = 5
    cfg.MODEL.SparseRCNN.RELATION_IOU_THR = 0.4
    # cfg.MODEL.SparseRCNN.IOA_THR= 0.7
    cfg.MODEL.SparseRCNN.CONFIDENCE_THR = 0.7
    cfg.MODEL.SparseRCNN.ITER_NUM = 1
    cfg.MODEL.SparseRCNN.LOW_CONFIDENCE_THR= 0.05

    # Focal Loss.
    cfg.MODEL.SparseRCNN.USE_FOCAL = True
    cfg.MODEL.SparseRCNN.ALPHA = 0.25
    cfg.MODEL.SparseRCNN.GAMMA = 2.0
    cfg.MODEL.SparseRCNN.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../..'
add_path(osp.join(root_dir, 'utils'))

class Config:

    user = getpass.getuser()
    this_model_dir = osp.split(os.path.realpath(__file__))[0]

    output_dir = osp.join(this_model_dir, 'output', '50e.6h.500pro.ignore')

    model_dir = output_dir
    eval_dir = osp.join(output_dir, 'inference')

    imgDir = '/home/zhenganlin/june/CrowdHuman/images'
    json_dir = '/home/zhenganlin/june/CrowdHuman/annotation_sparse-rcnn'
    train_json = osp.join(json_dir, 'train.json')
    eval_json = osp.join(json_dir, 'val.json')
    
    dirpath = '/home/zhenganlin/june/CrowdHuman'
    train_file = osp.join(dirpath,'crowd_human_train15000_final_unsure_fixempty_fixvis_vboxmerge.odgt')
    anno_file = osp.join(dirpath, 'crowd_human_test4370_final_unsure_fixempty_fixvis_vboxmerge.odgt')

config = Config()
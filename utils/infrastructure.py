#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import os, sys
from box import *
import pickle
from matching import maxWeightMatching
import pdb
def compute_Jaccard(dtboxes, gtboxes, bm_thr):

    assert dtboxes.shape[-1] > 3 and gtboxes.shape[-1] > 3
    if dtboxes.shape[0] < 1 or gtboxes.shape[0] < 1:
        return list()
    N, K = dtboxes.shape[0], gtboxes.shape[0]
    ious = compute_iou_matrix(dtboxes, gtboxes)
    rows, cols = np.where(ious > bm_thr)
    bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    mates = maxWeightMatching(bipartites)
    if len(mates) < 1:
        return list()
    rows = np.where(np.array(mates) > -1)[0]
    indices = np.where(rows < N + 1)[0]
    rows = rows[indices]
    cols = np.array([mates[i] for i in rows])
    matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    return matches

def compute_JC(detection:np.ndarray, gt:np.ndarray, iou_thresh:np.ndarray):

    return compute_Jaccard(detection, gt, iou_thresh)

def compute_ioa_matrix(dboxes: np.ndarray, gboxes: np.ndarray):

    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    eps = 1e-6
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = np.maximum(dtboxes[:,:,2] - dtboxes[:,:,0], 0) * np.maximum(dtboxes[:,:,3] - dtboxes[:,:,1], 0)   
    ioas = inter / (dtarea + eps)
    return ioas

def compute_iou_matrix(dboxes:np.ndarray, gboxes:np.ndarray):
    
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
    gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1])
    ious = inter / (dtarea + gtarea - inter)
    return ious

def compute_maximal_iou(proposals:np.ndarray,gt:np.ndarray):
    
    ious = compute_iou_matrix(proposals, gt)
    return np.max(ious, axis = 1)

def load(fpath):
    
    assert os.path.exists(fpath)
    with open(fpath,'rb') as fid:
        data = pickle.load(fid)
    return data

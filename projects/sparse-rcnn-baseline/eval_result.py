# !/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Sparse-RCNN(github: https://github.com/PeizeSun/SparseR-CNN) created by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from config import config
from common import *
def computeJaccard(fpath, save_path ='results.md'):

    assert os.path.exists(fpath)
    records = load_func(fpath)

    GT = load_func(config.anno_file)
    fid = open(save_path, 'a')
    for i in range(1, 10):
        score_thr = 1e-1 * i
        results = common_process(worker, records, 20, GT, score_thr, 0.5)
        line = strline(results)
        line = 'score_thr:{:.3f}, '.format(score_thr) + line
        print(line)
        fid.write(line + '\n')
        fid.flush()
    fid.close()

def computeIoUs(fpath, score_thr = 0.1, save_file='record.txt'):
    
    print('Processing {}'.format(osp.basename(fpath)))
    name = os.path.basename(fpath)

    records = load_func(fpath)
    save_results(records, fpath)
    mAP, mMR = compute_mAP(fpath)

    fid = open(save_file, 'a')
    fid.write('{}\ndtboxes:\n'.format(name))
    print('{}\ndtboxes:\n'.format(name))
    line = 'mAP:{:.4f}, mMR:{:.4f}, '.format(mAP, mMR)
    print(line)
    fid.write(line + '\n')
    fid.close()
    computeJaccard(fpath, save_path=save_file)
    print('evaluation is done.')

def eval_all():
    
    score_thr = 0.05
    save_file = 'record.txt'
    for epoch in range(75, 100):
        fpath = osp.join(config.eval_dir, 'epoch-{}.human'.format(epoch))
        if not os.path.exists(fpath):
            continue
        computeIoUs(fpath, score_thr, save_file)

def build_image_info(images):

    res = {}
    _ = [res.update({rb['id']: i}) for i, rb in enumerate(images)]
    return res

def coco_to_human(json_results, images):

    indices = build_image_info(images)
    new_json_results = dict()
    score_thr = 0.05

    for res in json_results:
        if res['score'] < score_thr:
            continue
        index = indices[res['image_id']]
        filename = images[index]['file_name'].split('.')[0]
        if res['image_id'] in new_json_results:
            new_json_results[res['image_id']]['dtboxes'].append({'box':res['bbox'], 'score':res['score']})
        else:
            new_json_results[res['image_id']] = dict()
            new_json_results[res['image_id']]['ID'] = res['image_id']
            new_json_results[res['image_id']]['filename'] = filename
            new_json_results[res['image_id']]['dtboxes'] = list()
            new_json_results[res['image_id']]['dtboxes'].append({'box':res['bbox'], 'score':res['score']})
    
    values = []
    for k in new_json_results.keys():

        value = new_json_results[k]
        filename, dtboxes = value['filename'], value['dtboxes']
        values.append({'ID': filename, 'dtboxes': dtboxes})
    return values

def eval_result():

    fpath = 'output/inference/coco_instances_results.json'
    records = load_func(fpath)[0]

    record = load_func(config.eval_json)[0]
    images, annotations = record['images'], record['annotations']
    results =  coco_to_human(records, images)
    fpath = 'mountain.human'
    save_results(results, fpath)

    print('Saving is Done...')
    computeIoUs(fpath, 0.05)

if __name__ == '__main__':
    
    eval_result()

import pickle as pkl
from common import *
from copy import deepcopy
import torch
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

def filter_boxes(result_queue, records, score_thr, nms_thr):

    assert np.logical_and(score_thr >= 0., nms_thr > 0.)
    for i, record in enumerate(records):

        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] > score_thr
        if flag.sum() < 1:
            result_queue.put_nowait(None)
            continue

        cls_dets = dtboxes[flag]
        keep = nms(np.float32(cls_dets), nms_thr)
        res = record
        res['dtboxes'] = boxes_dump(cls_dets[keep])
        result_queue.put_nowait(res)

def compute_iou_worker(result_queue, records, score_thr):

    for record in records:
        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        result = record
        result['dtboxes'] = list(filter(lambda rb: rb['score'] >= score_thr, record['dtboxes']))
        result_queue.put_nowait(result)

def computeIoUs(fpath, score_thr = 0.1, save_file='record.txt'):
    
    print('Processing {}'.format(osp.basename(fpath)))
    name = os.path.basename(fpath)

    records = load_func(fpath)
    results = common_process(compute_iou_worker, records, 16, score_thr)
    
    fpath = 'msra.human'
    save_results(results, fpath)
    mAP, mMR = compute_mAP(fpath)

    fid = open(save_file, 'a')
    fid.write('{}\ndtboxes:\n'.format(name))
    print('{}\ndtboxes:\n'.format(name))
    line = 'mAP:{:.4f}, mMR:{:.4f}, '.format(mAP, mMR)
    print(line)
    fid.write(line + '\n')
    fid.close()
    computeJaccard(fpath, save_path=save_file)
    # os.remove(fpath)

def test_unit():

    fpath = osp.join(config.eval_dir, 'epoch-30.human')
    records = load_func(fpath)
    save_path = 'nms.md'
    
    score_thr = 0.1
    for i in range(1, 10):
        nms_thr = 0.1 * i
        results = common_process(filter_boxes, records, 16, score_thr, nms_thr)
        fpath = 'mountain.human'
        save_results(results, fpath)
        mAP, mMR = compute_mAP(fpath)
        line = 'score_thr:{:.1f}, mAP:{:.4f}, mMR:{:.4f}'.format(score_thr, mAP, mMR)
        print(line)
        fid = open(save_path, 'a')
        fid.write(line + '\n')
        fid.close()
        computeJaccard(fpath, save_path)

def eval_all():
    
    score_thr = 0.05
    save_file = 'record.txt'
    for epoch in range(75, 100):
        fpath = osp.join(config.eval_dir, 'epoch-{}.human'.format(epoch))
        if not os.path.exists(fpath):
            continue
        computeIoUs(fpath, score_thr, save_file)

def capture_target_wkr(result_queue, records, tgts, GT, score_thr):

    assert isinstance(tgts, dict)
    for record in records:

        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] >= score_thr
        
        result = record
        if flag.sum() < 1:
            result_queue.put_nowait(None)
            continue

        gtboxes, ignores = recover_gtboxes(GT[tgts[record['ID']]])
        gtboxes = gtboxes[~ignores]
        matches = compute_JC(dtboxes[flag], gtboxes, 0.5)

        cols = np.array([i for _, i in matches])
        
        rest = dtboxes[~flag]
        indices = np.array(list(set(np.arange(gtboxes.shape[0]))  - set(cols)))
        res, auxi = None, []

        keep = dtboxes[flag]
        if indices.size:

            matches = compute_JC(rest, gtboxes[indices], 0.5)
            
            if len(matches):
                rs = np.array([i for i, _ in matches])
                keep = np.vstack([dtboxes[flag], rest[rs]])
                auxi = boxes_dump(rest[rs])
                

        result['dtboxes'] = boxes_dump(keep)
        result['auxi'] =  auxi
        result['main'] = boxes_dump(dtboxes[flag])
        result_queue.put_nowait(result)

def build_image_info(images):

    res = {}
    _ = [res.update({rb['ID']: i}) for i, rb in enumerate(images)]
    return res

def coco_to_human(json_results, images):

    indices = build_image_info(images)
    new_json_results = dict()
    score_thr = 0.0

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

def paint_images():

    fpath = 'mountain.human'
    if not osp.exists(fpath):
        print('please confirm {} exists.'.format(fpath))
    
    records = load_func(fpath)
    np.random.shuffle(records)

    visDir = 'vis_images'
    ensure_dir(visDir)

    for i, record in enumerate(records):

        filename = record['ID']
        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] >= 0.3
        if flag.sum() < 1:
            continue
        
        boxes = dtboxes[flag]
        
        fpath = osp.join(config.imgDir,filename + '.png')
        image = cv2.imread(fpath)

        draw_xt(boxes, image, Color.Blue, 2)
        fpath = osp.join(visDir, filename + '.png')
        cv2.imwrite(fpath, image)

        if i > 50:
            break

def compute_iou_worker_v2(result_queue, records, nms_thr, score_thr):

    assert nms_thr > 0 and score_thr >= 0.
    for record in records:
        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        result = record
        result['dtboxes'] = list(filter(lambda rb: rb['score'] >= score_thr, record['dtboxes']))
        dtboxes = recover_dtboxes(result)
        keep = nms(np.float(dtboxes), nms_thr)
        result['dtboxes'] = boxes_dump(dtboxes[keep])
        result_queue.put_nowait(result)

def NMS_post_process():

    fpath = 'mountain.human'

    print('Processing {}'.format(osp.basename(fpath)))
    name = os.path.basename(fpath)

    records = load_func(fpath)
    results = common_process(compute_iou_worker_v2, records, 16, 0.4, score_thr)
    pdb.set_trace()
    
    fpath = 'nms.human'
    save_results(results, fpath)
    mAP, mMR = compute_mAP(fpath)

    fid = open(save_file, 'a')
    fid.write('{}\ndtboxes:\n'.format(name))
    print('{}\ndtboxes:\n'.format(name))
    line = 'mAP:{:.4f}, mMR:{:.4f}, '.format(mAP, mMR)
    print(line)
    fid.write(line + '\n')
    fid.close()
    computeJaccard(fpath, save_path=save_file)

def valid_mean_score(result_queue, records, GT, iou_thr = 0.5):
    
    for i, record in enumerate(records):
        index = annotations[record['ID']]
        gtboxes, ignores = recover_gtboxes(GT[index])
        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] >= 0.05
        cls_boxes = dtboxes[flag]
        matches = compute_JC(cls_boxes, gtboxes[ignores < 1], 0.5)
        if len(matches) < 1:
            result_queue.put_nowait(None)
            continue

        rows = np.array([k for (k, _) in matches])
        cols = np.array([k for (_, k) in matches])
        if rows.size < 1:
            result_queue.put_nowait(None)
            continue
        else:
            valid_boxes = cls_boxes[rows]
            result_queue.put_nowait(valid_boxes)

def nms_worker(result_queue, records, nms_thr, score_thr):

    for record in records:
        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4]>= score_thr
        dtboxes = np.float32(dtboxes[flag])
        # keep = soft_nms(dtboxes, sigma=0.5, Nt = nms_thr)
        keep = nms(dtboxes, nms_thr)
        dtboxes = dtboxes[keep]
        result = deepcopy(record)
        result['dtboxes'] = boxes_dump(dtboxes)
        result_queue.put_nowait(result)

def nms_results():

    fpath = 'detr_1000.human'
    records =load_func(fpath)
    for i in range(1):
        nms_thr = 0.5 + 0.1 * i
        results = common_process(nms_worker, records, 20, nms_thr, 0.05)
        save_path = 'mid.human'
        save_results(results, save_path)
        line = 'nms_thr:{:.1f}'.format(nms_thr)
        print(line)
        with open('record.txt', 'a') as fid:
            fid.write(line + '\n')
        computeIoUs(save_path, 0.05)
        os.remove(save_path)

if __name__ == '__main__':
    
    nms_results()
    # 0.28
    # fpath = 'results/iter.sparse-rcnn.human'
    # records = load_func(fpath)
    # GT = load_func(config.anno_file)
    # annotations = build_image_info(GT)
    # results = common_process(valid_mean_score, records, 16, GT, 0.5)
    # dtboxes = np.vstack(results)
    # pdb.set_trace()
    # mean_score = dtboxes[:, 4].mean()
    # print('average score is {:.2f}'.format(mean_score))

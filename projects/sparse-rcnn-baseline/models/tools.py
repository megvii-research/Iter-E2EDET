from common import *
from config import config
import pdb
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

def paint_images():

    fpath = 'mountain.human'
    if not osp.exists(fpath):
        vis_results()

    records = load_func(fpath)
    visDir = 'vis_images'
    ensure_dir(visDir)

    for i, record in enumerate(records):

        filename = record['ID']
        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] >= 0.05
        if flag.sum() < 1:
            continue
        
        boxes = dtboxes[flag]
        
        fpath = osp.join(config.imgDir,filename + '.png')
        image = cv2.imread(fpath)

        draw_xt(boxes, image, Color.Yello, 2)
        fpath = osp.join(visDir, filename + '.png')
        cv2.imwrite(fpath, image)

        if i > 10:
            break
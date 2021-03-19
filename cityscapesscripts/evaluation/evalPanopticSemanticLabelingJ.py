#!/usr/bin/python
#

########################################################################################################
# i.21.3.19.19:58) 동명의 파일을 내가 좀 변수명들 바꿔주면서 읽어보려고 만든 복사본 파일임.
########################################################################################################`

# The evaluation script for panoptic segmentation (https://arxiv.org/abs/1801.00868).
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
# Test set evaluation assumes prediction use 'id' and not 'trainId'
# for categories, i.e. 'person' id is 24.
#
# The script expects both ground truth and predictions to use COCO panoptic
# segmentation format (http://cocodataset.org/#format-data and
# http://cocodataset.org/#format-results respectively). The format has 'image_id' field to
# match prediction and annotation. For cityscapes we assume that the 'image_id' has form
# <city>_123456_123456 and corresponds to the prefix of cityscapes image files.
#
# Note, that panoptic segmentaion in COCO format is not included in the basic dataset distribution.
# To obtain ground truth in this format, please run script 'preparation/createPanopticImgs.py'
# from this repo. The script is quite slow and it may take up to 5 minutes to convert val set.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import sys
import argparse
import functools
import traceback
import json
import time
import multiprocessing
import numpy as np
from collections import defaultdict

# Image processing
from PIL import Image

# Cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.helpers.labels import labels as csLabels


OFFSET = 256 * 256 * 256
VOID = 0


# The decorator is used to prints an error trhown inside process
def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, catId2cat, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for catId, cat in catId2cat.items():
            if isthing is not None:
                cat_isthing = cat['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[catId].iou
            tp = self.pq_per_cat[catId].tp
            fp = self.pq_per_cat[catId].fp
            fn = self.pq_per_cat[catId].fn
            if tp + fp + fn == 0:
                per_class_results[catId] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[catId] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, catId2cat):
    pq_stat = PQStat()

    idx = 0
    # i.21.3.19.22:28) 각 이미지에 대해서.
    for gt_ann, pred_ann in annotation_set:
        if idx % 30 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_id2segInfo = {segment_info['id']: segment_info for segment_info in gt_ann['segments_info']}
        pred_id2segInfo = {segment_info['id']: segment_info for segment_info in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(segment_info['id'] for segment_info in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_id2segInfo:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_id2segInfo[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_id2segInfo[label]['category_id'] not in catId2cat:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_id2segInfo[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_id2segInfo:
                continue
            if pred_label not in pred_id2segInfo:
                continue
            if gt_id2segInfo[gt_label]['iscrowd'] == 1: # i. gt가 iscrowd=1인건 왜 안쳐줌? 사람들이 모여잇는곳(iscrowd=1)을 사람이라고 프레딕션했으면 점수 더줘야하는거 아님?/21.3.19.23:04.
                continue
            if gt_id2segInfo[gt_label]['category_id'] != pred_id2segInfo[pred_label]['category_id']:
                continue

            # i.21.3.19.23:10) 여기서 union 넓이 구할때 gt_pred_map.get((VOID, pred_label), 0) 를 빼주는부분이,
            #  gt VOID 영역을 모델이 foreground 카테고리로 프레딕션했더라도(애시당초 프레딕션 선택지에 VOID 는 없게 해줬으니 어차피 VOID로 프레딕션 할수는 없음) 
            #  그부분은 union 증가되지 않게 해주는것임!! 
            #  이밸류에이션 방식이 이러니, 점수 조금이라도 더 높게 받으려면 프레딕션 선택지에서 VOID 는 빼주는게 더 유리하겠지!!
            #  내가 내 치과파노플젝 할때 데이터셋 레지스터하는 부분에서 Det2 에 나와있는 기존 방식대로 따라해줫다가('unlabeled' 즉 VOID 는 프레딕션할 카테고리에서 제외) 
            #  백그라운드가 죄다 foreground 카테고리들로 프레딕션됐길래(특히 대부분이 sinus 로 프레딕션됏엇지)
            #  내가 뭘 잘못한건가 햇엇는데, 그게아니고 기존 방식이 그런거지. 이밸류에이션에서 불리하게할필요 없으니 VOID 는 프레딕션할 카테고리에서 제외한거지.
            union = pred_id2segInfo[pred_label]['area'] + gt_id2segInfo[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_id2segInfo[gt_label]['category_id']].tp += 1
                pq_stat[gt_id2segInfo[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives 
        gt_crowd_catId2id = {}
        for gt_id, gt_segInfo in gt_id2segInfo.items():
            if gt_id in gt_matched:
                continue
            # crowd segments are ignored
            if gt_segInfo['iscrowd'] == 1:
                # i. TODO Q: 어차피 iscrowd=1 이면 카테고리id 랑 id랑 똑같지 않나..?? 
                #  혹시, 이 코드가 COCO panoptic api 를 기반으로한거니까, COCO 데이터셋에서는 다를수도 있어서 이렇게 해줬던거려나? /21.3.20.0:09.
                gt_crowd_catId2id[gt_segInfo['category_id']] = gt_id   
                continue
            pq_stat[gt_segInfo['category_id']].fn += 1

        # count false positives
        for pred_id, pred_segInfo in pred_id2segInfo.items():
            if pred_id in pred_matched:
                continue
            # intersection of the segment with VOID                 # i. 여기서도 마찬가지로 VOID, CROWD 부분을 프레딕션한것은 무효처리함(점수 안깎임).
            intersection = gt_pred_map.get((VOID, pred_id), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_segInfo['category_id'] in gt_crowd_catId2id:
                intersection += gt_pred_map.get((gt_crowd_catId2id[pred_segInfo['category_id']], pred_id), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_segInfo['area'] > 0.5:
                continue
            pq_stat[pred_segInfo['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, catId2cat):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, catId2cat))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    workers.close()
    return pq_stat


def average_pq(pq_stat, catId2cat):
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(catId2cat, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    return results


def print_results(results, categories):
    metrics = ["All", "Things", "Stuff"]
    print("{:14s}| {:>5s}  {:>5s}  {:>5s}".format("Category", "PQ", "SQ", "RQ"))
    labels = sorted(results['per_class'].keys())
    for label in labels:
        print("{:14s}| {:5.1f}  {:5.1f}  {:5.1f}".format(
            categories[label]['name'],
            100 * results['per_class'][label]['pq'],
            100 * results['per_class'][label]['sq'],
            100 * results['per_class'][label]['rq']
        ))
    print("-" * 41)
    print("{:14s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))

    for name in metrics:
        print("{:14s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n']
        ))


def evaluatePanoptic(gt_json_file, gt_folder, pred_json_file, pred_folder, resultsFile):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)
    catId2cat = {cat['id']: cat for cat in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        printError("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        printError("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    imgId2predAnn = {pred_ann['image_id']: pred_ann for pred_ann in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in imgId2predAnn:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, imgId2predAnn[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, catId2cat)

    results = average_pq(pq_stat, catId2cat)
    with open(resultsFile, 'w') as f:
        print("Saving computed results in {}".format(resultsFile))
        json.dump(results, f, sort_keys=True, indent=4)
    print_results(results, catId2cat)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


# The main method
def main():
    cityscapesPath = os.environ.get(
        'CITYSCAPES_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    )
    gtJsonFile = os.path.join(cityscapesPath, "gtFine", "cityscapes_panoptic_val.json")

    predictionPath = os.environ.get(
        'CITYSCAPES_RESULTS',
        os.path.join(cityscapesPath, "results")
    )
    predictionJsonFile = os.path.join(predictionPath, "cityscapes_panoptic_val.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file",
                        dest="gtJsonFile",
                        help= '''path to json file that contains ground truth in COCO panoptic format.
                            By default it is $CITYSCAPES_DATASET/gtFine/cityscapes_panoptic_val.json.
                        ''',
                        default=gtJsonFile,
                        type=str)
    parser.add_argument("--gt-folder",
                        dest="gtFolder",
                        help= '''path to folder that contains ground truth *.png files. If the
                            argument is not provided this script will look for the *.png files in
                            'name' if --gt-json-file set to 'name.json'.
                        ''',
                        default=None,
                        type=str)
    parser.add_argument("--prediction-json-file",
                        dest="predictionJsonFile",
                        help='''path to json file that contains prediction in COCO panoptic format.
                            By default is either $CITYSCAPES_RESULTS/cityscapes_panoptic_val.json
                            or $CITYSCAPES_DATASET/results/cityscapes_panoptic_val.json if
                            $CITYSCAPES_RESULTS is not set.
                        ''',
                        default=predictionJsonFile,
                        type=str)
    parser.add_argument("--prediction-folder",
                        dest="predictionFolder",
                        help='''path to folder that contains prediction *.png files. If the
                            argument is not provided this script will look for the *.png files in
                            'name' if --prediction-json-file set to 'name.json'.
                        ''',
                        default=None,
                        type=str)
    resultFile = "resultPanopticSemanticLabeling.json"
    parser.add_argument("--results_file",
                        dest="resultsFile",
                        help="File to store computed panoptic quality. Default: {}".format(resultFile),
                        default=resultFile,
                        type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.gtJsonFile):
        printError("Could not find a ground truth json file in {}. Please run the script with '--help'".format(args.gtJsonFile))
    if args.gtFolder is None:
        args.gtFolder = os.path.splitext(args.gtJsonFile)[0]

    if not os.path.isfile(args.predictionJsonFile):
        printError("Could not find a prediction json file in {}. Please run the script with '--help'".format(args.predictionJsonFile))
    if args.predictionFolder is None:
        args.predictionFolder = os.path.splitext(args.predictionJsonFile)[0]

    evaluatePanoptic(args.gtJsonFile, args.gtFolder, args.predictionJsonFile, args.predictionFolder, args.resultsFile)

    return

# call the main method
if __name__ == "__main__":
    main()

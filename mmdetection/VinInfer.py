"""
Dataset: VinBigData Chest X-ray Abnormalities Detection
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
1) 150,000 X-ray images with disease labels and bounding box
2）Label:['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No Finding']
"""
# -*- coding: utf-8 -*-
'''
@data: 2021/03/01
@author: Jason.Fang
'''
import os
import sys
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import mmcv

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
CLASS_NAMES_Vin = ['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']

def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            j[0], j[1], int(j[2][0]), int(j[2][1]), int(j[2][2]), int(j[2][3])))

    return " ".join(pred_strings)

def TestInfer(score_thr=0.8):
    vin_test_file = '/data/pycode/CXRAD/dataset/VinCXR_test.txt'
    vin_test_image = '/data/fjsdata/Vin-CXR/test_jpg/'
    vin_test_data = '/data/comcode/mmdetection/vincxr/test/'
    
    # Specify the path to model config and checkpoint file
    config_file = 'vincxr/code/maskrcnn.py'
    checkpoint_file = 'vincxr/workdir/latest.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:6')

    # test images and show the results
    images = pd.read_csv(vin_test_file, sep=',', header=None).values
    sub_res = []
    for image in images:
        img = vin_test_image + image[0]+'.jpeg'
        result = inference_detector(model, img)
        #extract result
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        #prediction
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        sub_tmp = {'image_id': image[0], 'PredictionString': '14 1.0 0 0 1 1'}
        if len(scores)>0:
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if len(scores)>0:
                sub_tmp['PredictionString'] = format_prediction_string(labels, bboxes, scores)
                #sub = {'image_id': image[0],'PredictionString': format_prediction_string(labels, bboxes, scores)}
        sub_res.append(sub_tmp)
        sys.stdout.write('\r process: = {}'.format(len(sub_res)))
        sys.stdout.flush()
    #Save submission file
    test_df = pd.DataFrame(sub_res, columns=['image_id', 'PredictionString'])
    print("\r set shape: {}".format(test_df.shape)) 
    print("\r set Columns: {}".format(test_df.columns))
    test_df.to_csv(vin_test_data+'submission.csv', index=False)

def compute_IoUs(xywh1, xywh2):
    x1, y1, w1, h1 = xywh1
    w1 = w1-x1
    h1 = h1-y1
    x2, y2, w2, h2 = xywh2
    w2 = w2-x2
    h2 = h2-y2

    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    intersection = dx * dy if (dx >=0 and dy >= 0) else 0.
    
    union = w1 * h1 + w2 * h2 - intersection
    IoUs = intersection / union
    
    return IoUs

def ValInfer(score_thr=0.5, show_thr=0.80):
    vin_val_file = '/data/pycode/CXRAD/dataset/VinCXR_val.txt'
    vin_val_image = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    vin_val_data = '/data/comcode/mmdetection/vincxr/val/'
    
    # Specify the path to model config and checkpoint file
    config_file = 'vincxr/code/maskrcnn.py'
    checkpoint_file = 'vincxr/workdir/latest.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:6')

    # test images and show the results
    images = pd.read_csv(vin_val_file, sep=',', header=None).values
    IoUs = []
    for image in images:
        img = vin_val_image + image[0]+'.jpeg'
        gtlbl = image[1]
        gtbox = [float(eval(i)) for i in image[3].split(' ')]
        #extract result
        result = inference_detector(model, img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        #prediction
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        IoU = 0.0
        if len(scores)>0:
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if gtlbl in labels: #hit ratio
                inds =  labels == gtlbl
                bboxes = bboxes[inds, :]
                for box in bboxes:
                    IoU_tmp = compute_IoUs(gtbox, box[:-1])
                    if IoU_tmp > IoU: 
                        IoU = IoU_tmp
                        if IoU_tmp > show_thr: #show
                            fig, ax = plt.subplots()# Create figure and axes
                            ax.imshow(Image.open(img))
                            rect = patches.Rectangle((gtbox[0], gtbox[1]), gtbox[2]-gtbox[0], gtbox[3]-gtbox[1], linewidth=2, edgecolor='b', facecolor='none')
                            ax.add_patch(rect)# add groundtruth
                            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect) #add predicted boundingbox
                            ax.text(gtbox[0], gtbox[1], CLASS_NAMES_Vin[gtlbl])
                            ax.axis('off')
                            fig.savefig(vin_val_data + image[0]+'.jpeg')
        IoUs.append(IoU)
        sys.stdout.write('\r testing process: = {}'.format(len(IoUs)))
        sys.stdout.flush()
    #evaluation
    print('The average IoU is {:.4f}'.format(np.array(IoUs).mean()))
    print('The Accuracy is {:.4f}'.format(Acc/len(images)))

def main():
    #ValInfer()
    TestInfer()

if __name__ == "__main__":
    main()
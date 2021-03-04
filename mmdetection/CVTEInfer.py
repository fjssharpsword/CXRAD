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
from mmdet.apis import init_detector, inference_detector
import mmcv

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

def main(score_thr=0.3):
    cvte_csv_file = '/data/pycode/CXRAD/dataset/cvte_test.txt'
    cvte_image_dir = '/data/fjsdata/CVTEDR/images/'
    cvte_box_dir = '/data/comcode/mmdetection/vincxr/cvte/'
    
    # Specify the path to model config and checkpoint file
    config_file = 'vincxr/code/maskrcnn.py'
    checkpoint_file = 'vincxr/workdir/latest.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:7')

    # test images and show the results
    images = pd.read_csv(cvte_csv_file, sep=',').values
    gt = []
    pred = []
    prob = []
    name = []
    for image in images:
        img = cvte_image_dir+image[0]
        name.append(img)
        gt.append(image[1])

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
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        #probability
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        #inds = scores > score_thr
        #bboxes = bboxes[inds, :]
        #labels = labels[inds]
        #if segms is not None:
        #    segms = segms[inds, ...]
        if len(scores)>0:
            ind = np.argmax(scores)
            prob.append(bboxes[ind,:][-1])
            if bboxes[ind,:][-1]>score_thr:
                pred.append(1)
                #save the visualization results to image files
                #model.show_result(img, result, score_thr=0.3, out_file=cvte_box_dir + image[0] )
            else: 
                pred.append(0)
        else:
            prob.append(0.0)
            pred.append(0)
        
        sys.stdout.write('\r testing process: = {}'.format(len(name)+1))
        sys.stdout.flush()

    #evaluation
    gt_np = np.array(gt)
    pred_np = np.array(pred)
    prob_np = np.array(prob)
    assert gt_np.shape == pred_np.shape
    #AUROCS
    AUROCs = roc_auc_score(gt_np, prob_np)
    print('AUROC = {:.4f}'.format(AUROCs))
    #F1 = 2 * (precision * recall) / (precision + recall)
    f1score = f1_score(gt_np, pred_np, average='micro')
    print('\r F1 Score = {:.4f}'.format(f1score))
    #sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
    sen = tp /(tp+fn)
    spe = tn /(tn+fp)
    print('\rSensitivity = {:.4f} and specificity = {:.4f}'.format(sen, spe)) 

    #result = pd.concat([pd.DataFrame(np.array(name)),pd.DataFrame(gt_np), pd.DataFrame(pred_np)], axis=1)
    #result.to_csv('/data/comcode/mmdetection/vincxr/workdir/disan.csv', index=False, header=False, sep=',')

if __name__ == "__main__":
    main()
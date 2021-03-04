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

def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            j[0], j[1], j[2][0], j[2][1], j[2][2], j[2][3]))

    return " ".join(pred_strings)

def TestInfer(score_thr=0.5):
    vin_test_file = '/data/pycode/CXRAD/dataset/VinCXR_test.txt'
    vin_test_image = '/data/fjsdata/Vin-CXR/test_jpg/'
    vin_test_data = '/data/comcode/mmdetection/vincxr/test/'
    
    # Specify the path to model config and checkpoint file
    config_file = 'vincxr/code/maskrcnn.py'
    checkpoint_file = 'vincxr/workdir/latest.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:6')

    # test images and show the results
    images = pd.read_csv(vin_test_file, sep=',').values
    results = []
    for image in images:
        result = {
                'image_id': image[0],
                'PredictionString': '14 1.0 0 0 1 1'
            }
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
        if len(scores)>0:
            inds = scores > score_thr
            if len(inds)>0:
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                result = {
                            'image_id': image[0],
                            'PredictionString': format_prediction_string(labels, bboxes, scores)
                        }
        results.append(result)
        sys.stdout.write('\r testing process: = {}'.format(len(results)+1))
        sys.stdout.flush()
    #Save submission file
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv(vin_test_data+'submission.csv', index=False)

def compute_IoUs(xywh1, xywh2):
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2

    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    intersection = dx * dy if (dx >=0 and dy >= 0) else 0.
    
    union = w1 * h1 + w2 * h2 - intersection
    IoUs = intersection / union
    
    return IoUs

def get_box(points):
    min_x = points[0]
    min_y = points[1]
    max_x = points[2]
    max_y = points[3]

    return [min_x, min_y, max_x - min_x, max_y - min_y]

def visHeatmap(self, batch_idx, class_name, image, cam_img, pdbox, gtbox, iou):
        #raw image 
        image = (image + 1).squeeze().permute(1, 2, 0) #[-1,1]->[1, 2]
        image = (image - image.min()) / (image.max() - image.min()) #[1, 2]->[0,1]
        image = np.uint8(255 * image) #[0,1] ->[0,255]
        
        #feature map
        heat_map = cv2.applyColorMap(np.uint8(cam_img * 255.0), cv2.COLORMAP_JET) #L to RGB
        heat_map = Image.fromarray(heat_map)#.convert('RGB')#PIL.Image
        mask_img = Image.new('RGBA', heat_map.size, color=0) #transparency
        #paste heatmap
        x1, y1, w, h = np.array(pdbox).astype(int)
        x2, y2, w, h = np.array(gtbox).astype(int)
        cropped_roi = heat_map.crop((x1,y1,x1+w,y1+h))
        mask_img.paste(cropped_roi, (x1,y1,x1+w,y1+h))
        cropped_roi = heat_map.crop((x2,y2,x2+w,y2+h))
        mask_img.paste(cropped_roi, (x2,y2,x2+w,y2+h))
        #plot
        output_img = cv2.addWeighted(image, 0.7, np.asarray(mask_img.convert('RGB')), 0.3, 0)
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(output_img)
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='b', facecolor='none')# Create a Rectangle patch
        ax.add_patch(rect)# Add the patch to the Axes
        rect = patches.Rectangle((x2, y2), w, h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        ax.add_patch(rect)# Add the patch to the Axes
        ax.axis('off')
        fig.savefig(config['img_path']+ str(batch_idx+1) +'_'+class_name+'_'+ str(iou)[0:6] +'.png')
        """
        image = Image.fromarray(image).convert('RGB')#PIL.Image
        x2, y2, w, h = np.array(gtbox).astype(int)
        cropped_roi = image.crop((x2,y2,x2+w,y2+h))
        width, height = image.size
        cropped_roi = cropped_roi.resize((width, height),Image.ANTIALIAS)
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(cropped_roi)
        ax.axis('off')
        fig.savefig(config['img_path']+str(batch_idx+1)+'_'+class_name+'.png')
        """

def ValInfer(show_thr=0.8):
    vin_val_file = '/data/pycode/CXRAD/dataset/VinCXR_val.txt'
    vin_val_image = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    vin_val_data = '/data/comcode/mmdetection/vincxr/val/'
    
    # Specify the path to model config and checkpoint file
    config_file = 'vincxr/code/maskrcnn.py'
    checkpoint_file = 'vincxr/workdir/latest.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:6')

    # test images and show the results
    images = pd.read_csv(vin_val_file, sep=',').values
    IoUs = []
    Acc = 0
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
        if gtlbl in labels: #hit ratio
            Acc = Acc + 1 
            inds =   labels == gtlbl
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            for box in bboxes:
                IoU_tmp = compute_IoUs(gtbox, box[:-1])
                if IoU_tmp > IoU: 
                    IoU = IoU_tmp
                    if IoU_tmp > show_thr: #show
                        pass
        IoUs.append(IoU)
        sys.stdout.write('\r testing process: = {}'.format(len(results)+1))
        sys.stdout.flush()
        #evaluation
        print('The average IoU is {:.4f}'.format(np.array(IoUs).mean()))
        print('The Accuracy is {:.4f}'.format(Acc/len(images)))

def main():
    #ValInfer()
    TestInfer()

if __name__ == "__main__":
    main()
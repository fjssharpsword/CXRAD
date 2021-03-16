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
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
import sys
from sklearn.model_selection import train_test_split

#class name, 'No finding'/background:0
classname_to_id = {'Aortic enlargement':1, 'Atelectasis':2, 'Calcification':3, 'Cardiomegaly':4,
    		   	   'Consolidation':5, 'ILD':6, 'Infiltration':7, 'Lung Opacity':8, 'Nodule/Mass':9,
               	   'Other lesion':10, 'Pleural effusion':11, 'Pleural thickening':12, 'Pneumothorax':13, 'Pulmonary fibrosis':14}

#https://github.com/Klawens/dataset_prepare/blob/main/csv2coco.py
class Csv2CoCo:
    
    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                annotation = self._annotation(shape, key)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'Jason.Fang created'
        instance['license'] = ['J.F']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        img = cv2.imread(self.image_dir + path + '.jpeg')
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = path
        image['file_name'] = path + '.jpeg'
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, path):
        label = shape[0]
        points = shape[3:]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = path
        annotation['category_id'] = int(classname_to_id[str(label)])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a

def main():
    vin_csv_file = '/data/fjsdata/Vin-CXR/train.csv'
    vin_image_dir = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    vin_coco_path = '/data/comcode/mmdetection/vincxr/data/'

    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(vin_csv_file, sep=',')
    annotations.fillna(0, inplace = True)
    annotations.loc[annotations["class_id"] == 14, ['x_max', 'y_max']] = 1.0
    annotations["class_id"] = annotations["class_id"] + 1
    annotations.loc[annotations["class_id"] == 15, ["class_id"]] = 0
    annotations = annotations[annotations.class_name!='No finding'].reset_index(drop=True)
    annotations = annotations.values #dataframe -> numpy
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1] 
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
        else:
            total_csv_annotations[key] = value
        sys.stdout.write('\r key {} completed'.format(key))
        sys.stdout.flush()  
    # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())
   
    # 把训练集转化为COCO的json格式
    l2c_train = Csv2CoCo(image_dir=vin_image_dir, total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(total_keys)
    l2c_train.save_coco_json(train_instance, vin_coco_path+'vin_coco_ann.json')

def check():
    json_file = '/data/comcode/mmdetection/vincxr/data/vin_coco_ann.json'
    annos = json.loads(open(json_file).read())
    print(annos.keys())   # 键
    print(annos["info"])   # 键值
    print(annos["license"])
    print(annos["categories"])
    print(annos["images"][0]) 
    print(annos["annotations"][0])
    
if __name__ == "__main__":
    main()
    #check()
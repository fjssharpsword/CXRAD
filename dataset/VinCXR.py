import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
import re
import sys
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
import PIL.ImageOps 
from sklearn.utils import shuffle
import shutil
#define by myself
sys.path.append("..") 
from CXRAD.config import *
#from config import *
"""
Dataset: VinBigData Chest X-ray Abnormalities Detection
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
1) 150,000 X-ray images with disease labels and bounding box
2）Label:['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No Finding']
"""
#generate dataset 
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f:
                    items = line.strip().split(',') 
                    image_name= items[0]#.split('/')[1]
                    label = list(items[1].replace(' ', ''))[1:16]
                    label = [int(eval(i)) for i in label]
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

        """
        #statistics of dataset
        labels_np = np.array(labels)
        multi_dis_num = 0
        for i in range(len(CLASS_NAMES_Vin)):
            num = len(np.where(labels_np[:,i]==1)[0])
            multi_dis_num = multi_dis_num + num
            print('Number of {} is {}'.format(CLASS_NAMES[i], num))
        print('Number of Multi Finding is {}'.format(multi_dis_num))

        norm_num = (np.sum(labels_np, axis=1)==0).sum()
        dis_num = (np.sum(labels_np, axis=1)!=0).sum()
        assert norm_num + dis_num==len(labels)
        print('Number of No Finding is {}'.format(norm_num))
        print('Number of Finding is {}'.format(dis_num))
        print('Total number is {}'.format(len(labels)))
        """

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

def get_test_dataloader_NIH(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_NIH,
                                    path_to_dataset_file=[PATH_TO_TEST_BENCHMARK_FILE_NIH], transform=transform_seq_test)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def get_train_dataloader_NIH(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_NIH,
                                          path_to_dataset_file=[PATH_TO_TRAIN_VAL_BENCHMARK_FILE_NIH], transform=transform_seq_train)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

"""
#for cross-validation
def get_train_val_dataloader_NIH(batch_size, shuffle, num_workers, split_ratio=0.1):
    dataset_train_full = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                         path_to_dataset_file=[PATH_TO_TRAIN_VAL_BENCHMARK_FILE], transform=transform_seq_train)

    val_size = int(split_ratio * len(dataset_train_full))
    train_size = len(dataset_train_full) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train_full, [train_size, val_size])

    data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader_train, data_loader_val
"""

#generate box dataset
class BBoxGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        boxes = []
        boxdata = pd.read_csv(path_to_dataset_file, sep=',')
        boxdata = boxdata[['Image Index','Finding Label','Bbox [x', 'y', 'w', 'h]']]
        for _, row in boxdata.iterrows():
            image_name = os.path.join(path_to_img_dir, row['Image Index'])
            image_names.append(image_name)
            label = np.zeros(len(CLASS_NAMES_NIH))
            label[CLASS_NAMES_NIH.index(row['Finding Label'])] = 1
            labels.append(label)
            boxes.append(np.array([row['Bbox [x'], row['y'], row['w'], row['h]']]))#xywh

        self.image_names = image_names
        self.labels = labels
        self.boxes = boxes
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        width, height = image.size 
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        #get gounded-truth boxes
        x_scale = config['TRAN_SIZE']/width
        y_scale = config['TRAN_SIZE']/height
        crop_del = (config['TRAN_SIZE']-config['TRAN_CROP'])/2
        box = self.boxes[index]
        x, y, w, h = int(box[0])*x_scale-crop_del, int(box[1])*y_scale-crop_del, int(box[2])*x_scale, int(box[3])*y_scale
        gtbox = np.array([x,y,w,h])

        return image, torch.FloatTensor(label), gtbox

    def __len__(self):
        return len(self.image_names)

def get_bbox_dataloader_NIH(batch_size, shuffle, num_workers):
    dataset_bbox = BBoxGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_NIH, 
                                 path_to_dataset_file=PATH_TO_BOX_FILE_NIH, transform=transform_seq_test)
    data_loader_bbox = DataLoader(dataset=dataset_bbox, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_bbox

def DicomProcess():
    PATH_TO_IMAGES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train/'
    PATH_TO_IMAGES_DIR_Vin_Train_JPG = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    PATH_TO_FILES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train.csv'
    PATH_TO_IMAGES_DIR_Vin_Test = '/data/fjsdata/Vin-CXR/test/'
    PATH_TO_IMAGES_DIR_Vin_Test_JPG = '/data/fjsdata/Vin-CXR/test_jpg/'

    try:
        train_imageIDs = []
        for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_Vin_Train):
            for file in files:
                #slice = pydicom.read_file(os.path.join(root, file))
                sitk_image = sitk.ReadImage(os.path.join(root, file))
                img = sitk.GetArrayFromImage(sitk_image)
                img = np.squeeze(img, axis=0)
                img = (img-np.min(img))/(np.max(img)-np.min(img)) * 255
                img = Image.fromarray(img.astype('uint8')).convert('RGB')#numpy to PIL
                imgID = os.path.splitext(file)[0]
                img.save(os.path.join(PATH_TO_IMAGES_DIR_Vin_Train_JPG, imgID+'.jpeg'),"JPEG", quality=95, optimize=True, progressive=True)
                train_imageIDs.append(imgID)
                sys.stdout.write('\r Image ID {} completed'.format(imgID))
                sys.stdout.flush()     
    except Exception as e:
        print("Unable to read file. %s" % e)
        continue

    try:
        test_imageIDs = []
        for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_Vin_Test):
            for file in files:
                #slice = pydicom.read_file(os.path.join(root, file))
                sitk_image = sitk.ReadImage(os.path.join(root, file))
                img = sitk.GetArrayFromImage(sitk_image)
                img = np.squeeze(img, axis=0)
                img = (img-np.min(img))/(np.max(img)-np.min(img)) * 255
                img = Image.fromarray(img.astype('uint8')).convert('RGB')#numpy to PIL
                imgID = os.path.splitext(file)[0]
                img.save(os.path.join(PATH_TO_IMAGES_DIR_Vin_Test_JPG, imgID+'.jpeg'),"JPEG", quality=95, optimize=True, progressive=True)
                test_imageIDs.append(imgID)
                sys.stdout.write('\r Image ID {} completed'.format(imgID))
                sys.stdout.flush()     
    except Exception as e:
        print("Unable to read file. %s" % e)
        continue



if __name__ == "__main__":

    DicomProcess()
    #preprocess()

    #for debug   
    #data_loader = get_test_dataloader(batch_size=10, shuffle=False, num_workers=0)
    #data_loader = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    #data_loader = get_bbox_dataloader(batch_size=10, shuffle=False, num_workers=0)
    #for batch_idx, (image, label, gtbox) in enumerate(data_loader):
    #   print(label.shape)
    #    print(image.shape)
    #    print(gtbox.shape)
    #    break
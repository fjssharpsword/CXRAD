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
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
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
#generate 
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None, is_test=False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names, gtlabels, gtboxes = [], [], []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
              for line in f: 
                    items = line.strip().split(',') 
                    image_name = os.path.join(path_to_img_dir, items[0]+'.jpeg')
                    image_names.append(image_name)
                    if is_test==False:
                        label = [float(eval(i)) for i in items[2].split(' ')]
                        gtlabels.append(label)#class_label
                        box = [float(eval(i)) for i in items[3].split(' ')]
                        gtboxes.append(box)#gtbox

        self.image_names = image_names
        self.gtlabels = gtlabels
        self.gtboxes = gtboxes
        self.transform = transform
        self.is_test = is_test

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
        if self.transform is not None:
            image = self.transform(image)
        if self.is_test==False:
            gtlabel = np.array(self.gtlabels[index])
            gtbox = np.array(self.gtboxes[index])
            if np.argwhere(gtlabel==1.0)<14:
                x_scale = config['TRAN_SIZE']/width
                y_scale = config['TRAN_SIZE']/height
                crop_del = (config['TRAN_SIZE']-config['TRAN_CROP'])/2
                x_min, y_min = int(gtbox[0])*x_scale-crop_del, int(gtbox[1])*y_scale-crop_del
                x_max, y_max = int(gtbox[2])*x_scale-crop_del, int(gtbox[3])*y_scale-crop_del
                gtbox = np.array([x_min, y_min, x_max, y_max])

            return image, torch.FloatTensor(gtlabel), gtbox
        else: return image

    def __len__(self):
        return len(self.image_names)

def get_test_dataloader_VIN(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_Vin_Test,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE_VIN], transform=transform_seq_test, is_test=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def get_train_dataloader_VIN(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_Vin_Train,
                                          path_to_dataset_file=[PATH_TO_TRAIN_FILE_VIN], transform=transform_seq_train, is_test=False)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_val_dataloader_VIN(batch_size, shuffle, num_workers):
    dataset_val = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_Vin_Train,
                                          path_to_dataset_file=[PATH_TO_VAL_FILE_VIN], transform=transform_seq_test, is_test=False)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_val

def SplitSet():
    PATH_TO_IMAGES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train/'
    PATH_TO_FILES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train.csv'
    PATH_TO_IMAGES_DIR_Vin_Test = '/data/fjsdata/Vin-CXR/test/'

    train_imageIDs = []
    for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_Vin_Train):
        for file in files:
            imgID = os.path.splitext(file)[0]
            train_imageIDs.append(imgID)
    valIDs = random.sample(train_imageIDs, int(0.1*len(train_imageIDs)))
    trainIDs = list(set(train_imageIDs).difference(set(valIDs)))

    datas = pd.read_csv(PATH_TO_FILES_DIR_Vin_Train, sep=',')
    print("\r CXR shape: {}".format(datas.shape)) 
    print("\r CXR Columns: {}".format(datas.columns))
    #datas = datas[['image_id','class_id','x_min','y_min','x_max','y_max']]
    datas['x_min'] = datas['x_min'].fillna(0)
    datas['y_min'] = datas['y_min'].fillna(0)
    datas['x_max'] = datas['x_max'].fillna(1)
    datas['y_max'] = datas['y_max'].fillna(1)
    trainset, valset = [], []
    for index, row in datas.iterrows():
        label = np.zeros(len(CLASS_NAMES_Vin))
        label[int(row['class_id'])] = 1
        label = ' '.join(str(e) for e in label.tolist())
        gtbox = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
        gtbox = ' '.join(str(e) for e in gtbox)
        if row['image_id'] in trainIDs:
            trainset.append([row['image_id'], row['class_id'], label, gtbox])
        elif row['image_id'] in valIDs:
            valset.append([row['image_id'], row['class_id'], label, gtbox])
        else:
            print('\r Image_ID {} is not exist'.format(row['image_id']))
        sys.stdout.write('\r index {} completed'.format(index+1))
        sys.stdout.flush()
    trainset =  pd.DataFrame(trainset, columns=['image_id', 'class_id', 'class_label','gtbox'])
    print("\r trainset shape: {}".format(trainset.shape)) 
    print("\r trainset Columns: {}".format(trainset.columns))
    print("\r Num of disease: {}".format(trainset['class_id'].value_counts()) )
    trainset.to_csv('/data/pycode/CXRAD/dataset/VinCXR_train.txt', index=False, header=False, sep=',')
    valset =  pd.DataFrame(valset, columns=['image_id', 'class_id', 'class_label','gtbox'])
    print("\r valset shape: {}".format(valset.shape)) 
    print("\r valset Columns: {}".format(valset.columns))
    print("\r Num of disease: {}".format(valset['class_id'].value_counts()) )
    valset.to_csv('/data/pycode/CXRAD/dataset/VinCXR_val.txt', index=False, header=False, sep=',')

    test_imageIDs = []
    for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_Vin_Test):
        for file in files:
            imgID = os.path.splitext(file)[0]
            test_imageIDs.append(imgID)
    test_set =  pd.DataFrame(test_imageIDs, columns=['image_id'])
    print("\r test_set shape: {}".format(test_set.shape)) 
    test_set.to_csv('/data/pycode/CXRAD/dataset/VinCXR_test.txt', index=False, header=False, sep=',')

def DicomProcess():
    PATH_TO_IMAGES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train/'
    PATH_TO_IMAGES_DIR_Vin_Train_JPG = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    PATH_TO_FILES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train.csv'
    PATH_TO_IMAGES_DIR_Vin_Test = '/data/fjsdata/Vin-CXR/test/'
    PATH_TO_IMAGES_DIR_Vin_Test_JPG = '/data/fjsdata/Vin-CXR/test_jpg/'

    #handle train_val set
    try:
        train_imageIDs = []
        for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_Vin_Train):
            for file in files:
                slice = pydicom.read_file(os.path.join(root, file))
                sitk_image = sitk.ReadImage(os.path.join(root, file))
                img = sitk.GetArrayFromImage(sitk_image)
                img = np.squeeze(img, axis=0)
                img = (img-np.min(img))/(np.max(img)-np.min(img)) * 255
                img = Image.fromarray(img.astype('uint8')).convert('RGB')#numpy to PIL
                imgID = os.path.splitext(file)[0]
                #ss.PhotometricInterpretation: 'MONOCHROME1'=flip and 'MONOCHROME2'=normal
                if 'MONOCHROME1' in slice.PhotometricInterpretation:
                    img = PIL.ImageOps.invert(img) #flip the white and black, RGB
                img.save(os.path.join(PATH_TO_IMAGES_DIR_Vin_Train_JPG, imgID+'.jpeg'),"JPEG", quality=95, optimize=True, progressive=True)
                train_imageIDs.append(imgID)
                sys.stdout.write('\r Image ID {} completed'.format(imgID))
                sys.stdout.flush()     
    except Exception as e:
        print("Unable to read file. %s" % e)
    """
    train_val_set =  pd.DataFrame(train_val_set, columns=['image_id', 'class_id', 'class_label','gtbox'])
    print("\r CXR shape: {}".format(train_val_set.shape)) 
    print("\r CXR Columns: {}".format(train_val_set.columns))
    print("\r Num of disease: {}".format(train_val_set['class_id'].value_counts()) )
    images = train_val_set[['image_id']]
    labels = train_val_set[['class_id', 'class_label','gtbox']]
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.10, random_state=11)
    print("\r trainset shape: {}".format(y_train.shape)) 
    print("\r trainset distribution: {}".format(y_train['class_id'].value_counts()))
    print("\r valset shape: {}".format(y_val.shape)) 
    print("\r valset distribution: {}".format(y_val['class_id'].value_counts()))
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/pycode/CXRAD/dataset/VinCXR_train.txt', index=False, header=False, sep=',')
    valset = pd.concat([X_val, y_val], axis=1).to_csv('/data/pycode/CXRAD/dataset/VinCXR_val.txt', index=False, header=False, sep=',')
    """  
    #handle test set
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
    """
    test_set =  pd.DataFrame(test_imageIDs, columns=['image_id'])
    print("\r test_set shape: {}".format(test_set.shape)) 
    test_set.to_csv('/data/pycode/CXRAD/dataset/VinCXR_test.txt', index=False, header=False, sep=',')
    """
    
if __name__ == "__main__":

    #DicomProcess()
    #SplitSet()

    #for debug   
    
    data_loader = get_train_dataloader_VIN(batch_size=10, shuffle=True, num_workers=0)
    #data_loader = get_val_dataloader_VIN(batch_size=10, shuffle=False, num_workers=0)
    for batch_idx, (image, label, gtbox) in enumerate(data_loader):
        print(label.shape)
        print(image.shape)
        print(gtbox.shape)
        break

    #data_loader = get_test_dataloader_VIN(batch_size=10, shuffle=False, num_workers=0)
    #for batch_idx, (image) in enumerate(data_loader):
    #    print(image.shape)
    #    break
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
from PIL import Image
import PIL.ImageOps 
from sklearn.utils import shuffle
import shutil
from matplotlib import pyplot as plt
import cv2
#define by myself
sys.path.append("..") 
from CXRAD.config import *
#from config import *
"""
Dataset: CVTE ChestXRay
"""

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
                    image_name = os.path.join(path_to_img_dir, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[1])) #eval for 
                        if label ==0:  labels.append([0]) #negative
                        else: labels.append([1]) #label == 1: #positive  
                        image_names.append(image_name) 

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        try:
            image_name = self.image_names[index]
            """
            if self.labels[index][0] == 0:
                image = cv2.imread(image_name)
                chans = cv2.split(image)
                colors = ("b","g","r")
                plt.figure()
                plt.title("Flattened Color Histogram of CVTE-CXR")
                plt.xlabel("Bins")
                plt.ylabel("# of Pixels")

                for (chan,color) in zip(chans,colors):
                    hist = cv2.calcHist([chan],[0],None,[256],[0,256]) #histogram
                    plt.plot(hist,color = color)
                    plt.xlim([0,256])
                plt.savefig(config['img_path']+"his_cvte.png")
            """
            image = Image.open(image_name).convert('RGB')
            #image.save('/data/pycode/ChestXRay/Imgs/test.jpeg',"JPEG", quality=95, optimize=True, progressive=True)
            label = self.labels[index]
            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image, torch.FloatTensor(label), image_name.split('/')[-1]

    def __len__(self):
        return len(self.image_names)


def get_test_dataloader_CVTE(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_CVTE,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE_CVTE], transform=transform_seq_test)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def splitCVTEDR(dataset_path, pos_dataset_path): 
    #read all samples and drop positive sample(remain negative sample)
    datas = pd.read_csv(dataset_path, sep=',')
    neg_datas = datas.drop(datas[datas['label']==3.0].index)
    neg_datas = neg_datas.drop(neg_datas[neg_datas['label']==1.0].index)
    neg_images = neg_datas['name'].tolist()

    #read positive samples and validation
    pos_datas = pd.read_csv(pos_dataset_path, sep=',',encoding='gbk')
    print("\r CXR Columns: {}".format(pos_datas.columns))
    #pos_datas = pos_datas[pos_datas['12-肺膜增厚']==1]  #select specific disease
    print("\r shape: {}".format(pos_datas.shape)) 
    pos_images = pos_datas['图片路径'].tolist()
    pos_images = [x.split('\\')[-1].split('_')[0]+'.jpeg' for x in pos_images]

    #assert
    assert len(set(neg_images) & set(pos_images)) == 0
    
    #split trainset and testset
    neg_test = []
    for x in neg_images:
        if x[2:4]=='20':
            neg_test.append(x)
    
    #merge positive and negative
    pos_datas_test = pd.DataFrame(pos_images, columns=['name'])
    pos_datas_test['label'] = 1
    neg_data_test = pd.DataFrame(neg_test, columns=['name'])
    neg_data_test['label'] = 0
    testset = pd.concat([pos_datas_test, neg_data_test], axis=0)
    testset = shuffle(testset)
    print("\r testset shape: {}".format(testset.shape)) 
    print("\r testset distribution: {}".format(testset['label'].value_counts()))
    #save 
    testset.to_csv('/data/pycode/CXRAD/dataset/cvte_test.txt', index=False, header=False, sep=',')
 
"""
def copyimage(dataset_path):
    with open(dataset_path, "r") as f:
        for line in f: 
            items = line.strip().split(',') 
            image_name = os.path.join(PATH_TO_IMAGES_DIR, items[0])
            if os.path.isfile(image_name) == True:
                label = int(eval(items[1])) #eval for
                shutil.copyfile(image_name, '/data/fjsdata/CVTEDR/test_images/'+str(label)+ '_' + items[0])
"""

if __name__ == "__main__":

    #generate test lists
    #splitCVTEDR('/data/fjsdata/CVTEDR/CXR20201210.csv', '/data/fjsdata/CVTEDR/CVTE-DR-Pos-939.csv')
    
    #for debug   
    data_loader_test = get_test_dataloader_CVTE(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, label, name) in enumerate(data_loader_test):
        print(batch_idx)
        print(image.shape)
        print(label.shape)
        break

    """
    pos_datas = pd.read_csv('/data/fjsdata/CVTEDR/CVTE-DR-Pos-939.csv', sep=',',encoding='gbk')
    pos_datas['name'] = pos_datas['图片路径'].apply(lambda x: x.split('\\')[-1].split('_')[0]+'.jpeg')
    pos_datas = pos_datas.drop(['图片路径'], axis=1)
    print("\r CXR Columns: {}".format(pos_datas.columns))
    labels = pd.read_csv('/data/pycode/CXRAD/log/disan.csv', sep=',')
    print(labels.head())
    #result = pd.concat([pos_datas, labels], axis=1, join='inner')
    result = pd.merge(labels, pos_datas, how='left', on='name')
    result.to_csv(config['log_path']+'disan_join.csv', index=False, header=True, sep=',')
    """
       


 
    
   
    
    
        
    
    
    
    
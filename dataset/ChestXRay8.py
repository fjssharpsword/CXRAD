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
import sys
#define by myself
#sys.path.append("..") 
#from CXRAD.config import *
from config import *

"""
Dataset: Chest X-Ray8
https://www.kaggle.com/nih-chest-xrays/data
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
1) 112,120 X-ray images with disease labels from 30,805 unique patients
2）Label:['Atelectasis', 'Cardiomegaly', 'Effusion','Infiltration', 'Mass', 'Nodule', 'Pneumonia', \
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
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
                    items = line.split()
                    image_name= items[0].split('/')[1]
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    if np.sum(label)==0: 
                        labels.append([0]) #normal
                    else:
                        labels.append([1]) #disease

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
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.LongTensor(label)

    def __len__(self):
        return len(self.image_names)
   
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                     path_to_dataset_file=[PATH_TO_TRAIN_COMMON_FILE], transform=transform_seq_train)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=[PATH_TO_VAL_COMMON_FILE], transform=transform_seq_test)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation

def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_COMMON_FILE], transform=transform_seq_test)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

#for cross-validation
def get_train_dataloader_full(batch_size, shuffle, num_workers, split_ratio=0.1):
    dataset_train_full = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                         path_to_dataset_file=[PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE], transform=transform_seq_train)

    val_size = int(split_ratio * len(dataset_train_full))
    train_size = len(dataset_train_full) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train_full, [train_size, val_size])

    data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train, data_loader_val

if __name__ == "__main__":
    #for debug   
    data_loader = get_train_dataloader(batch_size=10, shuffle=False, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader):
        print(image.shape)
        print(label.shape)
        break
# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 11/11/2020
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
#define by myself
from config import *
from net.CXRNet import CXRNet
from net.UNet import UNet
from util.logger import get_logger
from dataset.ChestXRay8 import get_train_dataloader, get_validation_dataloader, get_test_dataloader

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='CXRNet', help='CXRNet')
args = parser.parse_args()
#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'CXRNet':
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
        #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)

        model_unet = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet.eval()
    else: 
        print('No required model')
        return #over

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    ce_criterion = nn.CrossEntropyLoss()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    f1_best = 0.50 
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                optimizer.zero_grad()

                var_mask = model_unet(var_image)
                var_output = model(var_image, var_mask)#forward
                loss_tensor = ce_criterion(var_output, var_label.squeeze())#backward
                loss_tensor.backward()
                optimizer.step()##update parameters
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model.eval()#turn to test mode
        val_loss = []
        gt = torch.LongTensor().cuda()
        pred = torch.LongTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label.cuda()), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_mask = model_unet(var_image)
                var_output = model(var_image, var_mask)#forward
                loss_tensor = ce_criterion(var_output, var_label.squeeze())
                var_output = F.log_softmax(var_output,dim=1) 
                var_output = var_output.max(1,keepdim=True)[1]
                pred = torch.cat((pred, var_output.data), 0)
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                val_loss.append(loss_tensor.item())

        #evaluation  
        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        f1score = f1_score(gt_np, pred_np, average='micro')
        logger.info("\r Eopch: %5d validation loss = %.6f, Validataion F1 Score = %.4f" % (epoch + 1, np.mean(val_loss), f1score)) 

        if f1_best < f1score:
            f1_best = f1score
            CKPT_PATH = config['CKPT_PATH'] +'/best_model.pkl'
            torch.save(model.state_dict(), CKPT_PATH)
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
        CKPT_PATH = config['CKPT_PATH'] +  'best_model.pkl'
        if os.path.isfile(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+CKPT_PATH)
        model.eval()

        model_unet = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet.eval() 
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    # initialize the ground truth and output tensor
    gt = torch.LongTensor().cuda()
    pred = torch.LongTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (_, image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            var_mask = model_unet(var_image)
            var_output = model(var_image, var_mask)#forward
            var_output = F.log_softmax(var_output,dim=1) 
            var_output = var_output.max(1,keepdim=True)[1]
            pred = torch.cat((pred, var_output.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #evaluation
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    #F1 = 2 * (precision * recall) / (precision + recall)
    f1score = f1_score(gt_np, pred_np, average='micro')
    print('\r F1 Score = {:.4f}'.format(f1score))
    #sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
    sen = tp /(tp+fn)
    spe = tn /(tn+fp)
    print('\rSensitivity = {:.4f} and specificity = {:.4f}'.format(sen, spe))

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()
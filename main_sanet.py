# encoding: utf-8
"""
Training implementation for Mirror Attention
Author: Jason.Fang
Update time: 11/03/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#define by myself
from config import *
from util.logger import get_logger
from util.evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from dataset.CVTECXR import get_test_dataloader_CVTE
from dataset.VinCXR import get_train_dataloader_VIN, get_val_dataloader_VIN, get_test_dataloader_VIN
from net.SANet import SANet

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='SANet', help='SANet')
parser.add_argument('--dataset', type=str, default='VinCXR', help='VinCXR')
parser.add_argument('--testset', type=str, default='CVTECXR', help='CVTECXR')
args = parser.parse_args()
#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    if args.dataset == 'VinCXR':
        dataloader_train = get_train_dataloader_VIN(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
        dataloader_val = get_val_dataloader_VIN(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    else:
        print('No required dataset')
        return
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'SANet' and args.dataset == 'VinCXR':
        N_CLASSES = len(CLASS_NAMES_Vin)
        model = SANet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained SANet model checkpoint of NIH-CXR dataset: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label, box) in enumerate(dataloader_train):
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()

                optimizer_model.zero_grad()
                _, var_output = model(var_image)
                loss_tensor = bce_criterion(var_output, var_label)#backward
                loss_tensor.backward()
                optimizer_model.step()##update parameters
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model.eval()#turn to test mode
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label, box) in enumerate(dataloader_val):
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                _, var_output = model(var_image)#forward
                loss_tensor = bce_criterion(var_output, var_label)#backward
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                val_loss.append(loss_tensor.item())
                gt = torch.cat((gt, label.cuda()), 0)
                pred = torch.cat((pred, var_output.data), 0)
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        logger.info("\r Eopch: %5d validation loss = %.6f, Validataion AUROC = %.4f" % (epoch + 1, np.mean(val_loss), AUROC_avg)) 

        if AUROC_best < AUROC_avg:
            AUROC_best = AUROC_avg
            CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    if args.testset == 'CVTECXR':
        dataloader_test = get_test_dataloader_CVTE(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    elif args.testset == 'VinCXR':
        dataloader_test = get_val_dataloader_VIN(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    else:
        print('No required dataset')
        return
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'SANet' and args.dataset == 'VinCXR':
        CLASS_NAMES = CLASS_NAMES_Vin
        N_CLASSES = len(CLASS_NAMES_Vin)
        model = SANet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained SANet model checkpoint of NIH-CXR dataset: "+CKPT_PATH) 
    else: 
        print('No required model')
        return #over
    model.eval()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    name_list = []
    with torch.autograd.no_grad():
        for batch_idx, (image, label, name) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            _, var_output = model(var_image)#forward
            gt = torch.cat((gt, label.cuda()), 0)
            pred = torch.cat((pred, var_output.data), 0)
            name_list.extend(name)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #evaluation
    if args.testset == 'VinCXR':
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        for i in range(N_CLASSES):
            print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROCs[i]))
        print('The average AUROC is {:.4f}'.format(AUROC_avg))
        compute_ROCCurve(gt, pred, N_CLASSES, CLASS_NAMES, args.dataset) #plot ROC Curve
    elif args.testset == 'CVTECXR':
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        AUROCs = roc_auc_score(1-gt_np, pred_np[:,-1])
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[-1], AUROCs))

        pred_np_ad = np.where(pred_np[:,-1]>config['PROB'], 0, 1) #normal=0, abnormal=1
        pred_np = np.where(pred_np[:, :-1]>1-config['PROB'], 1, 0).sum(axis=1)
        pred_np_ad = np.logical_or(pred_np_ad, pred_np)
        #F1 = 2 * (precision * recall) / (precision + recall)
        f1score = f1_score(gt_np, pred_np_ad, average='micro')
        print('\r F1 Score = {:.4f}'.format(f1score))
        #sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(gt_np, pred_np_ad).ravel()
        sen = tp /(tp+fn)
        spe = tn /(tn+fp)
        print('\rSensitivity = {:.4f} and specificity = {:.4f}'.format(sen, spe)) 

        #result = pd.concat([pd.DataFrame(np.array(name_list)),pd.DataFrame(gt_np), pd.DataFrame(pred_np_ad)], axis=1)
        #result.to_csv(config['log_path']+'disan.csv', index=False, header=False, sep=',')
    else:
        print('No dataset need to evaluate')

def main():
    #Train()
    Test()

if __name__ == '__main__':
    main()
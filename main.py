# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
E-mail: fangjiansheng@cvte.com
Update time: 02/02/2021
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
#define by myself
from config import *
from net.CXRNet import CXRNet
from util.logger import get_logger
from util.evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from util.CAM import CAM
from dataset.NIHCXR import get_train_dataloader_NIH, get_test_dataloader_NIH, get_bbox_dataloader_NIH
from dataset.CVTECXR import get_test_dataloader_CVTE
from dataset.VinCXR import get_train_dataloader_VIN, get_val_dataloader_VIN, get_test_dataloader_VIN

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='CXRNet', help='CXRNet')
parser.add_argument('--dataset', type=str, default='VinCXR', help='VinCXR')
parser.add_argument('--testset', type=str, default='CVTECXR', help='CVTECXR')
args = parser.parse_args()
#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    if args.dataset == 'NIHCXR':
        dataloader_train = get_train_dataloader_NIH(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
        dataloader_val = get_test_dataloader_NIH(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    elif args.dataset == 'VinCXR':
        dataloader_train = get_train_dataloader_VIN(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
        dataloader_val = get_val_dataloader_VIN(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    else:
        print('No required dataset')
        return
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'CXRNet' and args.dataset == 'NIHCXR':
        N_CLASSES = len(CLASS_NAMES_NIH)
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True)#initialize model
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained CXRNet model checkpoint of NIH-CXR dataset: "+CKPT_PATH)
    elif args.model == 'CXRNet' and args.dataset == 'VinCXR':
        N_CLASSES = len(CLASS_NAMES_Vin)
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True)#initialize model
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained CXRNet model checkpoint of NIH-CXR dataset: "+CKPT_PATH)
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
            for batch_idx, (image, label, _) in enumerate(dataloader_train):
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
            for batch_idx, (image, label, _) in enumerate(dataloader_val):
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
    if args.testset == 'NIHCXR':
        dataloader_test = get_test_dataloader_NIH(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    elif args.testset == 'CVTECXR':
        dataloader_test = get_test_dataloader_CVTE(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    elif args.testset == 'VinCXR':
        dataloader_test = get_test_dataloader_VIN(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    else:
        print('No required dataset')
        return
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet' and args.dataset == 'NIHCXR':
        CLASS_NAMES = CLASS_NAMES_NIH
        N_CLASSES = len(CLASS_NAMES_NIH)
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained CXRNet model checkpoint of NIH-CXR dataset: "+CKPT_PATH) 
    elif args.model == 'CXRNet' and args.dataset == 'VinCXR':
        CLASS_NAMES = CLASS_NAMES_Vin
        N_CLASSES = len(CLASS_NAMES_Vin)
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained CXRNet model checkpoint of NIH-CXR dataset: "+CKPT_PATH) 

    else: 
        print('No required model')
        return #over
    model.eval()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            _, var_output = model(var_image)#forward
            gt = torch.cat((gt, label.cuda()), 0)
            pred = torch.cat((pred, var_output.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #evaluation
    if args.testset == 'NIHCXR':
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        for i in range(N_CLASSES):
            print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROCs[i]))
        print('The average AUROC is {:.4f}'.format(AUROC_avg))
        compute_ROCCurve(gt, pred, N_CLASSES, CLASS_NAMES, args.dataset) #plot ROC Curve
    elif args.testset == 'CVTECXR':
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()[:,-1]#No Finding
        AUROCs = roc_auc_score(gt_np, pred_np)
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[-1], AUROCs))
    else:
        print('No dataset need to evaluate')

def BoxTest():
    print('********************load data********************')
    if args.dataset == 'NIHCXR':
        dataloader_bbox = get_bbox_dataloader_NIH(batch_size=1, shuffle=False, num_workers=0)
    else:
        print('No required dataset')
        return
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet' and args.dataset == 'NIHCXR':
        CLASS_NAMES = CLASS_NAMES_NIH
        N_CLASSES = len(CLASS_NAMES_NIH)
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_' + args.dataset + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained CXRNet model checkpoint of NIH-CXR dataset: "+CKPT_PATH) 
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin bounding box testing!*********')
    #np.set_printoptions(suppress=True) #to float
    #for name, layer in model.named_modules():
    #    print(name, layer)
    cls_weights = list(model.parameters())
    weight_softmax = np.squeeze(cls_weights[-5].data.cpu().numpy())
    cam = CAM()
    IoUs = []
    IoU_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    with torch.autograd.no_grad():
        for batch_idx, (image, label, gtbox) in enumerate(dataloader_bbox):
            var_image = torch.autograd.Variable(image).cuda()
            var_feaConv, _ = model(var_image)#forward
            idx = torch.where(label[0]==1)[0] #true label
            cam_img = cam.returnCAM(var_feaConv.cpu().data.numpy(), weight_softmax, idx)
            pdbox = cam.returnBox(cam_img, gtbox[0].numpy())
            iou = compute_IoUs(pdbox, gtbox[0].numpy())
            IoU_dict[idx.item()].append(iou)
            IoUs.append(iou) #compute IoU
            if iou>0.99: 
                cam.visHeatmap(batch_idx, CLASS_NAMES[idx], image, cam_img, pdbox, gtbox[0].numpy(), iou) #visulization
            sys.stdout.write('\r box process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    print('The average IoU is {:.4f}'.format(np.array(IoUs).mean()))
    for i in range(len(IoU_dict)):
        print('The average IoU of {} is {:.4f}'.format(CLASS_NAMES[i], np.array(IoU_dict[i]).mean())) 

def main():
    Train()
    Test()
    #BoxTest()

if __name__ == '__main__':
    main()
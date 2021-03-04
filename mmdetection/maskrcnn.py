# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
E-mail: fangjiansheng@cvte.com
Update time: 01/03/2021
"""
# The new config inherits a base config to highlight the necessary modification
_base_ = '/data/comcode/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=14),
        mask_head=dict(num_classes=14)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    		'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
            'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax','Pulmonary fibrosis')
data = dict(
    train=dict(
        img_prefix='/data/fjsdata/Vin-CXR/train_val_jpg/',
        classes=classes,
        ann_file='/data/comcode/mmdetection/vincxr/data/vin_coco_ann.json'),
    val=dict(
        img_prefix='/data/fjsdata/Vin-CXR/train_val_jpg/',
        classes=classes,
        ann_file='/data/comcode/mmdetection/vincxr/data/vin_coco_ann.json'),
    test=dict(
        img_prefix='/data/fjsdata/Vin-CXR/train_val_jpg/',
        classes=classes,
        ann_file='/data/comcode/mmdetection/vincxr/data/vin_coco_ann.json'))

work_dir = '/data/comcode/mmdetection/vincxr/workdir/'
#load_from = 'vincxr/workdir/latest.pth'
evaluation = dict(metric=['bbox', 'segm'], interval=20)
checkpoint_config = dict(interval=20)
gpu_ids = range(0,5) #gpus = 6
runner = dict(type='EpochBasedRunner', max_epochs=100)

# SingleGPU for training: python tools/train.py vincxr/code/maskrcnn.py
# MultiGPU for training: ./tools/dist_train.sh vincxr/code/maskrcnn.py 6
# Test: python tools/test.py vincxr/code/maskrcnn.py vincxr/workdir/latest.pth --eval bbox segm
# Evaluation: https://cocodataset.org/#detection-eval
# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
E-mail: fangjiansheng@cvte.com
Update time: 08/03/2021
"""
# The new config inherits a base config to highlight the necessary modification
_base_ = '/data/comcode/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        num_classes=14
        )
    )

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
load_from = '/data/comcode/mmdetection/vincxr/workdir/latest.pth'
evaluation = dict(interval=20, metric=['bbox'])
checkpoint_config = dict(interval=20)
gpu_ids = range(0,7) #gpus = 6
runner = dict(type='EpochBasedRunner', max_epochs=200)

# SingleGPU for training: python tools/train.py vincxr/code/yolo.py
# MultiGPU for training: ./tools/dist_train.sh vincxr/code/yolo.py 6
# Test: python tools/test.py vincxr/code/yolo.py vincxr/workdir/latest.pth --eval bbox
# Evaluation: https://cocodataset.org/#detection-eval
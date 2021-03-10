# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
E-mail: fangjiansheng@cvte.com
Update time: 10/03/2021
"""
# The new config inherits a base config to highlight the necessary modification
_base_ = '/data/comcode/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=14,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
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
#load_from = '/data/comcode/mmdetection/vincxr/workdir/latest.pth'
evaluation = dict(interval=20, metric=['bbox'])
checkpoint_config = dict(interval=20)
gpu_ids = range(0,7) #gpus = 6
runner = dict(type='EpochBasedRunner', max_epochs=200)

# SingleGPU for training: python tools/train.py vincxr/code/CascadeRCNN.py
# MultiGPU for training: ./tools/dist_train.sh vincxr/code/CascadeRCNN.py 6
# Test: python tools/test.py vincxr/code/CascadeRCNN.py vincxr/workdir/latest.pth --eval bbox
# Evaluation: https://cocodataset.org/#detection-eval
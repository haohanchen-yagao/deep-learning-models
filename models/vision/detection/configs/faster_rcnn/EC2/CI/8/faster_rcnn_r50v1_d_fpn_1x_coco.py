base_files = ['../../../../common/datasets/coco.py',
              '../../../../common/lr_policy.py',
              '../../../../common/runtime.py',]

# model settings
model = dict(
    type='FasterRCNN',
    norm_type='BN',
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1_d',
        weights_path='/shared/awsdet/data/weights/resnet50v1_d', # SavedModel format
        weight_decay=1e-4
    ),
    neck=dict(
        type='FPN',
        in_channels=[('C2', 256), ('C3', 512), ('C4', 1024), ('C5', 2048)],
        out_channels=256,
        num_outs=5,
        interpolation_method='bilinear',
        weight_decay=1e-4,
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_scales=[8.],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds= [1.0, 1.0, 1.0, 1.0],
        feat_channels=512,
        num_samples=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=6000,
        num_post_nms_train=2000,
        num_pre_nms_test=2000,
        num_post_nms_test=1000,
        weight_decay=1e-4,
        use_smooth_l1=False,
    ),
    bbox_roi_extractor=dict(
        type='PyramidROIAlign',
        pool_shape=[7, 7],
        pool_type='avg',
    ),
    bbox_head=dict(
        type='BBoxHead',
        num_classes=81,
        pool_size=[7, 7],
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        min_confidence=0.005,
        nms_threshold=0.75,
        max_instances=100,
        weight_decay=1e-4,
        use_conv=False,
        use_bn=False,
        use_smooth_l1=False,
        soft_nms_sigma=0.5
    ),
)


# optimizer
optimizer = dict(
    _overwrite_=True,
    type='MomentumOptimizer',
    learning_rate=5e-3,
    momentum=0.9,
    nesterov=False,
)

# overwrite lr policy
# learning policy
lr_config = dict(
    _overwrite_=True,
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[8, 11]
)

# extra options related to optimizers
optimizer_config = dict(
    _overwrite_=True,
    amp_enabled=True,
    gradient_clip=5.0,
)


work_dir = './work_dirs/faster_rcnn_r50v1_d_fpn_1x_coco'
total_epochs = 12
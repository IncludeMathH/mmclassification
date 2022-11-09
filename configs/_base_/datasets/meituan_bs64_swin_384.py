# dataset settings
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=384, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='ClassBalancedDataset',        # 因为训练集是类别不平衡的。  验证集和测试集类别平衡。
        oversample_thr=1e-3,                # 对小于千分之一的类别进行重采样
        dataset=dict(
            type=dataset_type,
            data_prefix='/ssd/data/meituan/Train_qtc',
            ann_file='/ssd/data/meituan/meta/train_qtcom.txt',
            pipeline=train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/ssd/data/meituan/val',
        ann_file='/ssd/data/meituan/meta/val_qtcom.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/ssd/data/meituan/val',
        ann_file='/ssd/data/meituan/meta/val_qtcom.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=10, metric='accuracy')

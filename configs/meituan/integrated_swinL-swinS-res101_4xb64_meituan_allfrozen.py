_base_ = [
    '../_base_/datasets/meituan_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model1 = '/ssd/mmclassification/work_dir/swin-large_4xb32_meituan_100e/latest.pth'
model2 = '//ssd/mmclassification/work_dir/swin-small_4xb64_meituan_lr2-5e-4_100e/latest.pth'
model3 = '/ssd/mmclassification/work_dir/resnet101_4xb64_meituan/latest.pth'
model = dict(
    type='IntegratedClassifier',
    num_models=3,
    backbone=[
        dict(
            frozen_stages=3,
            type='SwinTransformer',
            arch='large',
            img_size=224,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=model1,
                prefix='backbone')),
        dict(
            frozen_stages=3,
            type='SwinTransformer',
            arch='small',
            img_size=224,
            drop_path_rate=0.3,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=model2,
                prefix='backbone')),
        dict(
            frozen_stages=3,
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(3,),
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                checkpoint=model3,
                prefix='backbone')),
    ],
    neck=[
        dict(type='GlobalAveragePooling'),
        dict(type='GlobalAveragePooling'),
        dict(type='GlobalAveragePooling'),
    ],
    head=[
        dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1536,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=model1,
                prefix='head')),
        dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=768,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=model2,
                prefix='head'),
            loss=dict(
                type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
            cal_acc=False),
        dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=model3,
                prefix='head')),
        dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=3000,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5)),
    ],
    )

# datasets config
data = dict(samples_per_gpu=64)         # batch_size
evaluation = dict(interval=1, metric='accuracy')

# schedules config
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    lr=5e-5 * 64*4 / 512)
runner = dict(type='EpochBasedRunner', max_epochs=100)

# run_time config
# checkpoint saving
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

find_unused_parameters=True  # 如果冻结权重，则需要加上这一句
_base_ = [
    '../_base_/models/resnet152.py', '../_base_/datasets/meituan_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]


# 1. model config file
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth',
            prefix='backbone',
        ))
)

# 2. datasets config file
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4)

# 3. shedules config file
optimizer = dict(type='SGD', lr=0.01*(64*4)/(8*32), momentum=0.9, weight_decay=0.0001)  # original lr=0.1
runner = dict(type='EpochBasedRunner', max_epochs=100)

# 4. runtime config file
# checkpoint saving
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
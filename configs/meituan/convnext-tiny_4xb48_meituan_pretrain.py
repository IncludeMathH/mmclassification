_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/meituan_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128-noema_in1k_20220222-2908964a.pth',
            prefix='backbone',
        )),

data = dict(samples_per_gpu=128)

optimizer = dict(lr=4e-3)

runner = dict(type='EpochBasedRunner', max_epochs=100)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
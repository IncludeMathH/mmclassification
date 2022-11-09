# Only for evaluation
_base_ = [
    '../_base_/models/swin_transformer/large_384.py',
    '../_base_/datasets/meituan_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model config
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth',
            prefix='backbone',
        ))
)

# datasets config
data = dict(samples_per_gpu=8)         # batch_size
evaluation = dict(interval=1, metric='accuracy')

# schedules config
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    lr=5e-5 * 8*4 / 512)
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
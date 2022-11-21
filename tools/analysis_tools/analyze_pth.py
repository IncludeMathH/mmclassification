import torch  # 命令行是逐行立即执行的

pth_dir = '/home/dn/.cache/torch/hub/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
# pth_dir = '/ssd/mmclassification/work_dir/convnext-tiny_4xb128_lr4e-4_100e_pretrain/epoch_100.pth'
content = torch.load(pth_dir)
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
for item in content:
    print(f'item={item}\n')

weights = content['state_dict']
print(f'value={weights}')

metas = content['meta']
print(f'metas={metas}')
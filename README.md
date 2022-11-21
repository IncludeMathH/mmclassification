# 目前的实验结果
<<<<<<< HEAD

|  Backbone  | Params(M) | Epoch | Top1-acc  | Top5-acc  |
|:----------:|:---------:|:-----:|:---------:|:---------:|
| ConvNext-T |   28.59   |  100  |   73.04   |   93.33   |
|   Swin-T   |   28.29   |  100  |   80.16   |   96.53   |
|   Swin-S   |   49.61   |  100  |   80.13   |   96.47   |
|   Swin-B   |   87.77   |  100  |   79.98   |   96.54   |
|   Swin-L   |  196.53   |  100  | **85.12** | **97.98** |
| ResNet-50  |   25.56   |  100  |   78.89   |   95.51   |
| ResNet-101 |   44.55   |  100  |   79.48   |   95.53   |
| ResNet-152 |   60.19   |  100  |   79.69   |   95.74   |
|   ViT-B    |   86.86   |  100  |   79.73   |   95.91   | 

# 数据集结构
需要将测试集图片放到一个名字为'0'的文件夹中，然后把’0‘文件夹放到’test‘文件夹中。

例如：
- data
  - meituan
    - meta
    - test
      - 0
        - xxx.jpg
    - train
    - val

可以参见CustomDataset的代码说明。



## 在使用代码的同时一起开发：
先fork到自己的github中，clone之后进行Pull request即可。

<<<<<<< HEAD
:point_right: **MMClassification 1.0 branch is in trial, welcome every to [try it](https://github.com/open-mmlab/mmclassification/tree/1.x) and [discuss with us](https://github.com/open-mmlab/mmclassification/discussions)!** :point_left:
=======

|  Backbone  | Params(M) | Epoch | Top1-acc  | Top5-acc  |
|:----------:|:---------:|:-----:|:---------:|:---------:|
| ConvNext-T |   28.59   |  100  |   73.04   |   93.33   |
|   Swin-T   |   28.29   |  100  |   80.16   |   96.53   |
|   Swin-S   |   49.61   |  100  |   80.13   |   96.47   |
|   Swin-B   |   87.77   |  100  |   79.98   |   96.54   |
|   Swin-L   |  196.53   |  100  | **85.12** | **97.98** |
| ResNet-50  |   25.56   |  100  |   78.89   |   95.51   |
| ResNet-101 |   44.55   |  100  |   79.48   |   95.53   |
| ResNet-152 |   60.19   |  100  |   79.69   |   95.74   |
|   ViT-B    |   86.86   |  100  |   79.73   |   95.91   | 

# 数据集结构
需要将测试集图片放到一个名字为'0'的文件夹中，然后把’0‘文件夹放到’test‘文件夹中。

例如：
- data
  - meituan
    - meta
    - test
      - 0
        - xxx.jpg
    - train
    - val
>>>>>>> 2e8a49b0011f42b0efd253fc073f8eeb2408adc4

可以参见CustomDataset的代码说明。



## 在使用代码的同时一起开发：
先fork到自己的github中，clone之后进行Pull request即可。

## 只使用代码而不一起开发：

<<<<<<< HEAD
<div align="center">
  <img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width="70%"/>
</div>

### Major features

- Various backbones and pretrained models
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility
- Powerful toolkits

## What's new

The MMClassification 1.0 has released! It's still unstable and in release candidate. If you want to try it, go
to [the 1.x branch](https://github.com/open-mmlab/mmclassification/tree/1.x) and discuss it with us in
[the discussion](https://github.com/open-mmlab/mmclassification/discussions).

v0.24.1 was released in 31/10/2022.
Highlights of the new version:

- Support HUAWEI Ascend device.

v0.24.0 was released in 30/9/2022.
Highlights of the new version:

- Support **HorNet**, **EfficientFormerm**, **SwinTransformer V2** and **MViT** backbones.
- Support Standford Cars dataset.

v0.23.0 was released in 1/5/2022.
Highlights of the new version:

- Support **DenseNet**, **VAN** and **PoolFormer**, and provide pre-trained models.
- Support training on IPU.
- New style API docs, welcome [view it](https://mmclassification.readthedocs.io/en/master/api/models.html).

Please refer to [changelog.md](docs/en/changelog.md) for more details and other release history.

## Installation

Below are quick steps for installation:
=======
## 只使用代码而不一起开发：

以下是安装的简要步骤：
>>>>>>> 更新了ReadMe文件。在Swin-L模型上微调
=======
以下是安装的简要步骤：
>>>>>>> 2e8a49b0011f42b0efd253fc073f8eeb2408adc4

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone git@github.com:IncludeMathH/mmclassification.git
cd mmclassification
pip3 install -e .
```

更详细的步骤请参考 [安装指南](https://mmclassification.readthedocs.io/zh_CN/latest/install.html) 进行安装。

## 基础教程

请参考 [基础教程](https://mmclassification.readthedocs.io/zh_CN/latest/getting_started.html) 来了解 MMClassification 的基本使用。MMClassification 也提供了其他更详细的教程：

- [如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)
- [如何微调模型](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/finetune.html)
- [如何增加新数据集](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_dataset.html)
- [如何设计数据处理流程](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html)
- [如何增加新模块](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_modules.html)
- [如何自定义优化策略](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/schedule.html)
- [如何自定义运行参数](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/runtime.html)

我们也提供了相应的中文 Colab 教程：

- 了解 MMClassification **Python API**：[预览 Notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_python_cn.ipynb) 或者直接[在 Colab 上运行](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_python_cn.ipynb)。
- 了解 MMClassification **命令行工具**：[预览 Notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_tools_cn.ipynb) 或者直接[在 Colab 上运行](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_tools_cn.ipynb)。

## 模型库

相关结果和模型可在 [model zoo](https://mmclassification.readthedocs.io/en/latest/model_zoo.html) 中获得

<details open>
<summary>支持的主干网络</summary>

- [x] [VGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/vgg)
- [x] [ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)
- [x] [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)
- [x] [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [SE-ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [RegNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/regnet)
- [x] [ShuffleNetV1](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)
- [x] [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)
- [x] [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)
- [x] [MobileNetV3](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v3)
- [x] [Swin-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/swin_transformer)
- [x] [RepVGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/repvgg)
- [x] [Vision-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/vision_transformer)
- [x] [Transformer-in-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/tnt)
- [x] [Res2Net](https://github.com/open-mmlab/mmclassification/tree/master/configs/res2net)
- [x] [MLP-Mixer](https://github.com/open-mmlab/mmclassification/tree/master/configs/mlp_mixer)
- [x] [DeiT](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit)
- [x] [Conformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/conformer)
- [x] [T2T-ViT](https://github.com/open-mmlab/mmclassification/tree/master/configs/t2t_vit)
- [x] [Twins](https://github.com/open-mmlab/mmclassification/tree/master/configs/twins)
- [x] [EfficientNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/efficientnet)
- [x] [ConvNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/convnext)
- [x] [HRNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/hrnet)
- [x] [VAN](https://github.com/open-mmlab/mmclassification/tree/master/configs/van)
- [x] [ConvMixer](https://github.com/open-mmlab/mmclassification/tree/master/configs/convmixer)
- [x] [CSPNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/cspnet)
- [x] [PoolFormer](https://github.com/open-mmlab/mmclassification/tree/master/configs/poolformer)
- [x] [MViT](https://github.com/open-mmlab/mmclassification/tree/master/configs/mvit)
- [x] [EfficientFormer](https://github.com/open-mmlab/mmclassification/tree/master/configs/efficientformer)
- [x] [HorNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/hornet)

</details>

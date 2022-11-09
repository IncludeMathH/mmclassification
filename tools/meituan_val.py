# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import (init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcls.apis import multi_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)

# for single_gpu_test
from mmcv.image import tensor2imgs
import os.path as osp
# for writing csv file
import csv

def write_csv(s, filename):
    '''
    :param s: (numpy array) Contents to be writen in csv file.
    :param filename: (str) The name of csv file.
    :return:
    '''
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerows(s)  # writerows方法是一下子写入多行内容


def single_gpu_test(model,
                    data_loader,
                    out_dir=None):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.

    Returns:
        None
    """
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    assert out_dir, \
        'Please specify the dir to save csv file.'
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)  # result: 一个batch出来的结果。 32*1000

        scores = np.vstack(result)          # 由元组转换为numpy数组

        top5_label = np.argsort(-scores, axis=1)[:, :5]   # 获取前k个索引
        # print(len(top5_label), len(top5_label[0]))
        # print(top5_label[0])
        file_name = []
        # print(data['img_metas'].data[0])
        # print('==================')
        for data_info in data['img_metas'].data[0]:
            # file_name.append(data_info['ori_filename'].split('/')[-1])           # 测试集
            file_name.append(data_info['ori_filename'])                          # 验证集

        # test_result = np.insert(top5_label, 0, values=np.array(file_name), axis=1)
        test_result = np.c_[np.array(file_name), top5_label]
        write_csv(test_result, out_dir)

        batch_size = data['img'].size(0)
        prog_bar.update(batch_size)
    return


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
             f'result file. You can choose some of ({", ".join(out_options)}), '
             'or use "all" to include all above, or use "none" to disable all of '
             'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be parsed as a dict metric_options for dataset.evaluate()'
             ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
             'Check available options in `model.show_result`.')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    if not distributed:
        model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids)
        if cfg.device == 'ipu':
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper
            opts = cfg2options(cfg.runner.get('options_cfg', {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
            data_loader.init(opts['inference'])
        model.CLASSES = CLASSES
        show_kwargs = args.show_options or {}
        outputs = single_gpu_test(model, data_loader, args.out)
    else:
        model = wrap_distributed_model(
            model,
            device=cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)


if __name__ == '__main__':
    main()

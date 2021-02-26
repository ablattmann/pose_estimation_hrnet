# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
import math
from tqdm import tqdm

import _init_paths
from config import cfg
from config import update_config
from core.inference import get_final_preds,get_max_preds
from utils.transforms import flip_back
from core.evaluate import accuracy



from utils.utils import create_logger
from dataset.infer_datasets import InferenceDataset

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            #joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                # if joint_vis[0]:

                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    ndarr = cv2.cvtColor(ndarr,cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)


def save_debug_images(config, input,# meta,
                      joints_pred,
                      prefix):
    if not config.DEBUG.DEBUG:
        return


    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, [],
            #meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )

def infer(config, val_loader, val_dataset, model, output_dir,device,num_keypoints=16):

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    data_dict = val_dataset.data.copy()
    data_dict.update({'keypoints_abs': np.zeros((len(val_dataset),num_keypoints,2)),
                      'keypoints_rel': np.zeros((len(val_dataset),num_keypoints,2))})
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader,desc=f'Estimating keypoints')):
            # compute output
            input = batch['img'].to(device)
            ids = batch['id'].numpy()
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            # if config.TEST.FLIP_TEST:
            #     input_flipped = input.flip(3)
            #     outputs_flipped = model(input_flipped)
            #
            #     if isinstance(outputs_flipped, list):
            #         output_flipped = outputs_flipped[-1]
            #     else:
            #         output_flipped = outputs_flipped
            #
            #     output_flipped = flip_back(output_flipped.cpu().numpy(),
            #                                val_dataset.flip_pairs)
            #     output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
            #
            #
            #     # feature is not aligned, shift flipped heatmap for higher accuracy
            #     if config.TEST.SHIFT_HEATMAP:
            #         output_flipped[:, :, :, 1:] = \
            #             output_flipped.clone()[:, :, :, 0:-1]
            #
            #     output = (output + output_flipped) * 0.5



            num_images = input.size(0)
            # # measure accuracy and record loss
            # losses.update(loss.item(), num_images)
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())

            # multiply by four,to obtain resolution of 256x256
            pred, _ = get_max_preds(output.cpu().numpy()) * 4

            # acc.update(avg_acc, cnt)
            #
            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            #
            # c = meta['center'].numpy()
            # s = meta['scale'].numpy()
            # score = meta['score'].numpy()

            # preds, maxvals = get_final_preds(
            #     config, output.clone().cpu().numpy(), c, s)
            #
            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            # all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            # all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            # all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            # all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            # all_boxes[idx:idx + num_images, 5] = score
            # image_path.extend(meta['image'])

            idx += num_images
            data_dict['keypoints_abs'][ids] = pred
            data_dict['keypoints_rel'][ids] = pred / 256.

            if i % config.PRINT_FREQ == 0:
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input,
                                  #meta,
                                  pred,
                                  prefix)



def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    device = torch.device(f'cuda:{cfg.GPUS}' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()
    valid_dataset = InferenceDataset(dataset=cfg.DATASET.DATASET)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    infer(cfg, valid_loader, valid_dataset, model,final_output_dir,device=device)


if __name__ == '__main__':
    main()

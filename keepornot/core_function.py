from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np

from utils.vis import save_debug_images_multi
from utils.vis import save_debug_3d_images
from utils.vis import save_debug_3d_cubes

logger = logging.getLogger(__name__)


def train_2d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_2d = AverageMeter()

    model.train()

    end = time.time()
    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)

        if 'panoptic' in config.DATASET.test_dataset:
            pred, loss_2d = model.training_step(batch)

        loss_2d = loss_2d.mean()

        losses_2d.update(loss_2d.item())
        loss = loss_2d

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed= None, # len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss=loss, loss_2d=losses_2d, loss_3d=None,
                    loss_cord=None, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1



def validate_3d(config, model, loader, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
            data_time.update(time.time() - end)
            if 'panoptic' in config.DATASET.test_dataset:
                pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                              weights_2d=weights_2d, targets_3d=targets_3d[0])
            elif 'campus' in config.DATASET.test_dataset or 'shelf' in config.DATASET.test_dataset:
                pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                              input_heatmaps=input_heatmap)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

                for k in range(len(inputs)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{:08}_{}'.format(
                        os.path.join(output_dir, 'validation'), i, view_name)
                    save_debug_images_multi(config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix)
                prefix2 = '{}_{:08}'.format(
                    os.path.join(output_dir, 'validation'), i)

                save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                save_debug_3d_images(config, meta[0], pred, prefix2)

    metric = None
    if 'panoptic' in config.DATASET.test_dataset:
        aps, _, mpjpe, recall = loader.dataset.evaluate(preds)
        msg = 'ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        logger.info(msg)
        metric = np.mean(aps)

    return metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

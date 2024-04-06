import argparse
import cv2
from configs.config import config
import torch
from configs.config import update_config
import dataset
import numpy as np
from utils import view_skeleton_batch
from utils.axu import (
    heatmap_visualization,
    create_distinct_color_palette,
    draw_pose_skeleton,
)
from utils.heatmap_related import GeneratePoseTarget
from pyskl.datasets.builder import build_from_cfg, build_dataset
from pyskl.models.builder import build_model
from mmcv import Config
from operator import itemgetter
import configs.nturgbd.ntu120_limb_xsub
import random
from utils.get_dVAE import get_dVAE


import torch
from vector_quantize_pytorch import VectorQuantize


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def load_ntu_pkl():
    args = parse_args()
    device = torch.device(config.device)

    cfg_file = "./configs/nturgbd/ntu120_limb_xsub.py"
    cfg = Config.fromfile(cfg_file)

    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model)
    # from dataset import ntu_mmcv

    # ds = ntu_mmcv(config, False)
    size = len(ds)

    idx = random.randint(0, size)
    data = ds.__getitem__(idx)
    imgs = data["imgs"]
    imgs = imgs.squeeze(0)
    imgs = imgs.permute((1, 0, 2, 3))
    J = imgs.shape[1]
    color_palette = create_distinct_color_palette(J)
    draw_pose_skeleton(imgs[0].numpy(), color_palette)
    imgs, _ = imgs.max(axis=1)
    heatmap_visualization(imgs.numpy())

    dVAE = get_dVAE(config)
    dVAE = dVAE.to(device)

    data = np.expand_dims(imgs, 0)
    data = torch.from_numpy(data).to(device)

    temp = 0.8146

    _, recons = dVAE(data, return_loss=True, return_recons=True, temp=temp)

    recons = np.array(recons[0].cpu().detach().numpy())
    heatmap_visualization(recons, "ntu_recon_mmcv.mp4")

    print()


def main():
    args = parse_args()

    train_dataset = eval("dataset." + config.DATASET.train_dataset)(
        config, is_training=True
    )
    train_dataset = eval("dataset." + config.DATASET.test_dataset)(
        config, is_training=False
    )

    # vf_idx = 95000
    vf_idx = 5724

    # data, gt, _, mask, meta, padding = train_dataset.__getitem__(vf_idx)
    data = train_dataset.__getitem__(vf_idx)
    combined_heatmaps = data[0]

    video_name = "heatmap_video.mp4"
    frame_height, frame_width = combined_heatmaps.shape[1], combined_heatmaps.shape[2]
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height)
    )

    for i in range(combined_heatmaps.shape[0]):
        # Normalize the heatmap for display
        normalized_heatmap = cv2.normalize(
            combined_heatmaps[i], None, 0, 255, cv2.NORM_MINMAX
        )
        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        # Write to video
        out.write(colored_heatmap)

    out.release()
    print()


if __name__ == "__main__":
    load_ntu_pkl()

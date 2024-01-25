import argparse
import cv2
from configs.config import config
import torch
from configs.config import update_config
from bert_pytorch import BERT, PoseBERT
from bert_pytorch import BERTTrainer
import dataset
import numpy as np
from utils import view_skeleton_batch
from train_2d import load_latest_model, return_dataloaders
from utils.heatmap_related import GeneratePoseTarget


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()

    # train_dataset = eval('dataset.' + config.DATASET.train_dataset)(
    #    config, config.DATASET.train_subset, is_train=True)
    HeatPose = GeneratePoseTarget(
        use_gaussian_score=False, with_limb=True, with_kp=False, heatmap_size=256
    )
    train_dataset = eval("dataset." + config.DATASET.train_dataset)(
        config, config.DATASET.train_subset, is_train=True, heatmap_generator=HeatPose
    )
    train_dataset = eval("dataset." + config.DATASET.test_dataset)(
        config, config.DATASET.test_subset, is_train=False, heatmap_generator=HeatPose
    )

    vf_idx = 2967  # Strange head motion?

    # data, gt, _, mask, meta, padding = train_dataset.__getitem__(vf_idx)
    data = train_dataset.__getitem__(vf_idx)
    combined_heatmaps = data

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
    main()

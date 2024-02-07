import argparse
import cv2
from configs.config import config
import torch
from configs.config import update_config
import dataset
import numpy as np
from utils import view_skeleton_batch
from utils.heatmap_related import GeneratePoseTarget

from torch.utils.data import DataLoader

import time


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

    train_dataset = eval("dataset." + config.DATASET.train_dataset)(
        config, is_train=True
    )

    dl = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    for _ in range(len(dl)):
        start = time.time()
        x = next(iter(dl))
        end = time.time()

        print("Elapsed time {}".format(end - start))


if __name__ == "__main__":
    main()

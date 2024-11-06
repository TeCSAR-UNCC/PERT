import math
from math import sqrt
import argparse
from pathlib import Path

# torch

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


# For DS
import dataset

# heatmap to color
from utils.axu import convert_to_rgb_3d, AverageMeter

# argument parsing

parser = argparse.ArgumentParser()

# Dataset Config

from configs.config import config
from configs.config import update_config


from utils.get_dVAE import get_dVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Overall Training for dVAE")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


args = parse_args()


ds = eval("dataset." + config.DATASET.train_dataset)(config, is_training=False)

device = torch.device(config.device)



dVAE = get_dVAE(config)
dVAE = dVAE.to(device)

idx = 0
heatmaps, labels = ds[idx]
heatmaps = heatmaps.to(device)
input_ids = dVAE.get_codebook_indices(heatmaps).flatten(1)

collapsed_heatmap = heatmaps.numpy().max(axis=1).astype("float32")
heatmaps_rgb = convert_to_rgb_3d(collapsed_heatmap)

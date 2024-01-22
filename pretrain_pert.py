import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

import math
from math import sqrt
import argparse
from pathlib import Path

# torch

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR

# dalle classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch import DiscreteVAE


# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


# For DS
import dataset

# Dataset Config
from configs.config import config
from configs.config import update_config

# Heatmap
from utils.heatmap_related import GeneratePoseTarget
from utils import init_distributed_mode, get_rank
from utils.args_handler import get_args

from timm.models import create_model


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    return model


def main(args):
    init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (
        args.input_size // patch_size[0],
        args.input_size // patch_size[1],
    )
    args.patch_size = patch_size

    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()

    using_deepspeed = distributed_utils.using_backend(
        distributed_utils.DeepSpeedBackend
    )

    # data
    HeatPose = GeneratePoseTarget(
        use_gaussian_score=False,
        with_limb=True,
        with_kp=False,
        heat_map_size=args.heatmap_size,
    )

    ds = eval("dataset." + config.DATASET.test_dataset)(
        config, config.DATASET.test_subset, is_train=False, heatmap_generator=HeatPose
    )

    if distributed_utils.using_backend(distributed_utils.HorovodBackend):
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=distr_backend.get_world_size(),
            rank=distr_backend.get_rank(),
        )
    else:
        data_sampler = None

    dl = DataLoader(ds, args.batch_size, shuffle=not data_sampler, sampler=data_sampler)

    vae_params = dict(
        image_size=args.heatmap_size,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        channels=config.DATASET.window_size,
        codebook_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_resnet_blocks=args.num_resnet_blocks,
        normalization=None,
    )

    vae = DiscreteVAE(**vae_params)

    if not using_deepspeed:
        vae = vae.cuda()


if __name__ == "__main__":
    args = get_args()
    main(args)

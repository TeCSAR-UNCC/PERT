import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import torch.nn as nn

from pathlib import Path

import math
from math import sqrt
import argparse
from pathlib import Path

# torch

import torch
from deepspeed.ops.lamb import FusedLamb
from torch.optim.lr_scheduler import OneCycleLR

# dalle classes and utils

from dalle_pytorch import distributed_utils


# vision imports
from torch.utils.data import DataLoader


from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm


# For DS
import dataset

# Dataset Config
from configs.config import config
from configs.config import update_config

# Heatmap
from utils.heatmap_related import GeneratePoseTarget
from utils import init_distributed_mode, get_rank
from utils.args_handler import get_args_finetune

from timm.models import create_model
from timm.utils import accuracy

# Note: Just to enforce to update timm.models dictionary
import models

from utils.axu import fill_the_model, AverageMeter, reduce_mean


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=config.DATASET.Heatmap_Generator.heatmap_size,
        in_chans=config.DATASET.window_size,
        num_classes=config.DATASET.num_classes,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    return model


def validate(model_engin, dl_validation, distr_backend, device):
    model_engin.eval()  # Set the model to evaluation mode
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for i, (data, target) in enumerate(dl_validation):
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                outputs = model_engin(data)
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            torch.distributed.barrier()

            top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True)
            top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True)
            total += target.size(0)

            if distr_backend.is_root_worker():
                print(
                    "{:3.2f} of validation is passed. Top-1 partial result is: {:3.1f}".format(
                        (i * 100) / len(dl_validation),
                        top1_correct.item() / total * 100,
                    ),
                    end="\r",
                )

    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100

    if distr_backend.is_root_worker():
        print(
            "\nFinal results:\n\tTop-1:{:3.2f},\n\tTop-5:{:3.2f}.".format(
                top1_accuracy.item(),
                top5_accuracy.item(),
            )
        )

    return top1_accuracy.item(), top5_accuracy.item()


def main(args):
    init_distributed_mode(args)

    print(args)

    args.config = config

    device = torch.device(config.device)

    cudnn.benchmark = True

    model = get_model(args)
    state_dict = torch.load(config.PeIT.finetune_saved_file, map_location="cpu")[
        "weights"
    ]
    model.load_state_dict(state_dict)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    config.PeIT.window_size = (
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[0],
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[1],
    )
    args.patch_size = patch_size

    model.to(device)

    distr_backend = distributed_utils.set_backend_from_args(config)
    distr_backend.initialize()

    using_deepspeed = distributed_utils.using_backend(
        distributed_utils.DeepSpeedBackend
    )

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)

    dl_val = DataLoader(
        ds_eval,
        config.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
        drop_last=False,
    )

    distr_backend.check_batch_size(config.batch_size)

    deepspeed_config = {
        "fp16": {"enabled": True},
        "train_batch_size": config.batch_size,
        "train_micro_batch_size_per_gpu": (
            config.batch_size // distr_backend.get_world_size()
        ),
        "wall_clock_breakdown": False,
    }

    (dist_model, _, dl_dist, _) = distr_backend.distribute(
        args=config,
        model=model,
        optimizer=None,
        model_parameters=model.parameters(),
        training_data=ds_eval if using_deepspeed else dl_val,
        lr_scheduler=None,
        config_params=deepspeed_config,
    )

    validate(dist_model, dl_dist, distr_backend, device)


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)

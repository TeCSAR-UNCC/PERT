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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR

# dalle classes and utils

from dalle_pytorch import distributed_utils


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

from utils.get_dVAE import get_dVAE

# Note: Just to enforce to update timm.models dictionary
import models


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

    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    config.PERT.window_size = (
        config.Heatmap_Generator.heatmap_size // patch_size[0],
        config.Heatmap_Generator.heatmap_size // patch_size[1],
    )
    args.patch_size = patch_size
    model.to(device)

    distr_backend = distributed_utils.set_backend_from_args(config)
    distr_backend.initialize()

    using_deepspeed = distributed_utils.using_backend(
        distributed_utils.DeepSpeedBackend
    )

    dVAE = get_dVAE(config, using_deepspeed=using_deepspeed)

    if not using_deepspeed:
        vae = vae.to(device)

    # data
    HeatPose = GeneratePoseTarget(**config.Heatmap_Generator)
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

    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    sched = CyclicLR(
        optimizer=opt,
        base_lr=args.base_learning_rate,
        max_lr=args.max_learning_rate,
        mode="triangular2",
        step_size_up=args.coeff_step_size_up * len(ds),
        step_size_down=args.coeff_step_size_down * len(ds),
    )

    if distr_backend.is_root_worker():
        # weights & biases experiment tracking

        import wandb

        run = wandb.init(
            project="PERT_window_{}_layer_{}".format(
                config.DATASET.window_size, config.VAE_Params.num_layers
            ),
            job_type="training",
            config=args.model,
        )

    # distribute

    distr_backend.check_batch_size(config.batch_size)
    deepspeed_config = {"train_batch_size": config.batch_size}

    (distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=args,
        model=vae,
        optimizer=opt,
        model_parameters=model.parameters(),
        training_data=ds if using_deepspeed else dl,
        lr_scheduler=sched if not using_deepspeed else None,
        config_params=deepspeed_config,
    )

    # starting temperature

    global_step = 0
    temp = config.starting_temp

    for epoch in range(config.epochs):
        for i, batch in enumerate(distr_dl):
            model.train()

            heatmaps = heatmaps.to(device)

            heatmaps, bool_masked_pos = batch
            heatmaps = heatmaps.to(device, non_blocking=True)
            # samples = samples.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

            with torch.no_grad():
                input_ids = vae.get_codebook_indices(heatmaps).flatten(1)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = input_ids[bool_masked_pos]

            with torch.cuda.amp.autocast():
                outputs = model(
                    heatmaps, bool_masked_pos=bool_masked_pos, return_all_tokens=False
                )
                loss = nn.CrossEntropyLoss()(input=outputs, target=labels)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                wandb.finish()

            if using_deepspeed:
                # Gradients are automatically zeroed after the step
                distr_vae.backward(loss)
                distr_vae.step()
            else:
                distr_opt.zero_grad()
                loss.backward()
                distr_opt.step()

            logs = {}

            if i % 100 == 0:
                if distr_backend.is_root_worker():
                    k = 4

                    with torch.no_grad():
                        codes = vae.get_codebook_indices(heatmaps[:k])
                        hard_recons = vae.decode(codes)

                    heatmaps, recons = map(lambda t: t[:k], (heatmaps, recons))
                    heatmaps, recons, hard_recons, codes = map(
                        lambda t: t.detach().cpu(),
                        (heatmaps, recons, hard_recons, codes),
                    )
                    """
                        heatmaps, recons, hard_recons = map(
                            lambda t: make_grid(
                                t.float(),
                                nrow=int(sqrt(k)),
                                normalize=True,
                                value_range=(-1, 1),
                            ),
                            (heatmaps, recons, hard_recons),
                        )
                    """

                    heatmaps_rgb = convert_to_rgb_3d(heatmaps.numpy())
                    recons_rgb = convert_to_rgb_3d(recons.numpy())
                    hard_recons_rgb = convert_to_rgb_3d(hard_recons.numpy())
                    logs = {
                        **logs,
                        "sample heatmap": wandb.Video(
                            heatmaps_rgb, fps=30, caption="original heatmap"
                        ),
                        "reconstructions": wandb.Video(
                            recons_rgb, fps=30, caption="reconstructions"
                        ),
                        "hard reconstructions": wandb.Video(
                            hard_recons_rgb, fps=30, caption="hard reconstructions"
                        ),
                        "codebook_indices": wandb.Histogram(codes),
                        "temperature": temp,
                    }

                    wandb.save("./vae.pt")

                save_model(f"./vae.pt")

                # temperature anneal

                temp = max(
                    temp * math.exp(-config.anneal_rate * global_step), config.temp_min
                )

            # lr decay
            # Do not advance schedulers from `deepspeed_config`.
            if not using_deepspeed_sched:
                distr_sched.step()

            # Collective loss, averaged
            avg_loss = distr_backend.average_all(loss)

            if distr_backend.is_root_worker():
                if i % 10 == 0:
                    lr = distr_sched.get_last_lr()[0]
                    print(epoch, i, f"lr - {lr:6f} loss - {avg_loss.item()}")

                    logs = {
                        **logs,
                        "epoch": epoch,
                        "iter": i,
                        "loss": avg_loss.item(),
                        "lr": lr,
                    }

                wandb.log(logs)
            global_step += 1

        if distr_backend.is_root_worker():
            # save trained model to wandb as an artifact every epoch's end

            model_artifact = wandb.Artifact(
                "trained-vae", type="model", metadata=dict(model_config)
            )
            model_artifact.add_file("vae.pt")
            run.log_artifact(model_artifact)

    if distr_backend.is_root_worker():
        # save final vae and cleanup

        save_model("./vae-final.pt")
        wandb.save("./vae-final.pt")

        model_artifact = wandb.Artifact(
            "trained-vae", type="model", metadata=dict(model_config)
        )
        model_artifact.add_file("vae-final.pt")
        run.log_artifact(model_artifact)

        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)

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
from torch.profiler import profile, record_function, ProfilerActivity

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
from timm.utils import accuracy

from utils.get_dVAE import get_dVAE

# Note: Just to enforce to update timm.models dictionary
import models


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=config.Heatmap_Generator.heatmap_size,
        in_chans=config.DATASET.window_size,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    return model


"""
def save_model(path, dist_model, using_deepspeed):
    save_obj = {
        "hparams": config.VAE_Params,
    }
    if using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + "-ds-cp"

        dist_model.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.
"""


def trace_handler(p):
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./Profiling/trace_" + str(p.step_num) + ".json")


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
    dVAE = dVAE.to(device)

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

    dl = DataLoader(
        ds,
        config.batch_size,
        shuffle=not data_sampler,
        sampler=data_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
    )

    opt = AdamW(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
    )

    step_size_up = int(config.coeff_step_size_up * len(dl))
    step_size_down = int(config.coeff_step_size_down * len(dl))

    sched = CyclicLR(
        optimizer=opt,
        base_lr=config.base_learning_rate,
        max_lr=config.max_learning_rate,
        mode="triangular2",
        step_size_up=step_size_up,
        step_size_down=step_size_down,
        cycle_momentum=False,
    )

    # distribute

    distr_backend.check_batch_size(config.batch_size)
    deepspeed_config = {"train_batch_size": config.batch_size}

    (dist_model, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=config,
        model=model,
        optimizer=opt,
        model_parameters=model.parameters(),
        training_data=ds if using_deepspeed else dl,
        lr_scheduler=sched if not using_deepspeed else None,
        config_params=deepspeed_config,
    )

    using_deepspeed_sched = False
    # Prefer scheduler in `deepspeed_config`.
    if distr_sched is None:
        distr_sched = sched
    elif using_deepspeed:
        # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
        # handle its scheduling.
        using_deepspeed_sched = True

    if distr_backend.is_root_worker():
        # weights & biases experiment tracking

        import wandb

        model_config = dict(
            name=args.model,
            img_size=config.Heatmap_Generator.heatmap_size,
            in_chans=config.DATASET.window_size,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
        )

        run = wandb.init(
            project="BEiT_window_{}_model_{}".format(
                config.DATASET.window_size, args.model
            ),
            job_type="training",
            config=model_config,
        )

    global_step = 0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=trace_handler,
    ) as prof:
        for epoch in range(config.epochs):
            for i in range(len(distr_dl)):
                with record_function("Data_loader"):
                    batch = next(iter(distr_dl))
                model.train()

                heatmaps, bool_masked_pos = batch
                heatmaps = heatmaps.to(device, non_blocking=True)
                # samples = samples.to(device, non_blocking=True)
                bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

                with torch.no_grad():
                    with record_function("Encoder"):
                        input_ids = dVAE.get_codebook_indices(heatmaps).flatten(1)
                        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                        labels = input_ids[bool_masked_pos]

                with torch.cuda.amp.autocast():
                    with record_function("BeIT"):
                        outputs = model(
                            heatmaps,
                            bool_masked_pos=bool_masked_pos,
                            return_all_tokens=False,
                        )
                    loss = nn.CrossEntropyLoss()(input=outputs, target=labels)

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    wandb.finish()

                if using_deepspeed:
                    # Gradients are automatically zeroed after the step
                    dist_model.backward(loss)
                    dist_model.step()
                else:
                    distr_opt.zero_grad()
                    loss.backward()
                    distr_opt.step()

                logs = {}

                # lr decay
                # Do not advance schedulers from `deepspeed_config`.
                if not using_deepspeed_sched:
                    distr_sched.step()

                # Collective loss, averaged
                avg_loss = distr_backend.average_all(loss)
                acc = accuracy(outputs, labels, topk=(1, 5))

                if distr_backend.is_root_worker():
                    if i % 10 == 0:
                        lr = distr_sched.get_last_lr()[0]
                        print(epoch, i, f"lr - {lr:6f} loss - {avg_loss.item()}")

                        logs = {
                            **logs,
                            "Top-1": acc[0].item(),
                            "Top-5": acc[1].item(),
                            "epoch": epoch,
                            "iter": i,
                            "loss": avg_loss.item(),
                            "lr": lr,
                        }

                    wandb.log(logs)
                global_step += 1

                prof.step()

        if distr_backend.is_root_worker():
            # save trained model to wandb as an artifact every epoch's end

            model_artifact = wandb.Artifact(
                "trained-model", type="model", metadata=dict(args.model)
            )
            model_artifact.add_file("vae.pt")
            run.log_artifact(model_artifact)

    if distr_backend.is_root_worker():
        # save final vae and cleanup

        # save_model("./vae-final.pt")
        wandb.save("./model-final.pt")

        model_artifact = wandb.Artifact(
            "trained-vae", type="model", metadata=dict(args.model)
        )
        model_artifact.add_file("vae-final.pt")
        run.log_artifact(model_artifact)

        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    update_config(args.cfg)
    main(args)

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
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

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

from utils.axu import AverageMeter, reduce_mean

# Note: Just to enforce to update timm.models dictionary
import models


def get_model(args):
    print(f"Creating model: {args.model}")
    in_chan = (
        config.DATASET.window_size
        if config.PeIT.embed_2dpatch
        else config.DATASET.joint_number
    )
    model = create_model(
        args.model,
        img_size=config.DATASET.Heatmap_Generator.heatmap_size,
        in_chans=in_chan,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=config.VAE_Params.num_tokens,
        single_cnn=not (args.disable_single_cnn),
        embed_2dpatch=config.PeIT.embed_2dpatch,
        patch_size=config.PeIT.patch_size,
    )

    # Do we have a checkpoint to start from?
    if config.Pretrained_Models.prefix_saved_file != "":
        print(
            "--> We are starting from here: {}".format(
                config.Pretrained_Models.prefix_saved_file
            )
        )
        state_dict = torch.load(
            config.Pretrained_Models.prefix_saved_file, map_location="cpu"
        )["weights"]
        model.load_state_dict(state_dict)

    return model


def save_model(
    full_path,
    using_deepspeed,
    distr_model,
    distr_backend,
    model,
    epoch,
    acc_top1=0,
    run_name="",
):
    save_obj = {"epoch": epoch, "top1_val": acc_top1}

    if using_deepspeed:
        chk_path = str(full_path)
        distr_model.save_checkpoint(chk_path, client_state=save_obj)

    if not distr_backend.is_root_worker():
        return

    # Fixme: Add optimizer and scheduler states here.
    save_obj = {**save_obj, "weights": model.state_dict()}

    file_name = full_path / "beit_best_{}_pretraining.pt".format(run_name)

    torch.save(save_obj, str(file_name))


def main(args):

    debug = False

    if debug:
        print("X---> Debug mode is activated <---X")

    init_distributed_mode(args)

    print(args)

    args.config = config
    args.model = config.PeIT.model

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

    dVAE = get_dVAE(config)
    dVAE = dVAE.to(device)

    # data
    ds = eval("dataset." + config.DATASET.train_dataset)(config, is_training=True)

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
        1 if debug else config.batch_size,
        shuffle=not data_sampler,
        sampler=data_sampler,
        num_workers=0 if debug else config.num_workers,
        pin_memory=True,
        persistent_workers=False if debug else (config.num_workers > 0),
    )

    iteration_size = len(dl)

    opt = Adam(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
    )

    step_size_up = int(config.coeff_step_size_up * iteration_size)
    step_size_down = int(config.coeff_step_size_down * iteration_size)

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

    deepspeed_config = {
        "fp16": {"enabled": config.fp16_training},
        "bf16": {"enabled": False},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.max_learning_rate,
                "weight_decay": config.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": config.epochs * iteration_size,
                "warmup_min_ratio": 0,
                "warmup_num_steps": step_size_up,
                "cos_min_ratio": config.base_learning_rate,
            },
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": config.gradient_clipping,
        "steps_per_print": 2000,
        "train_batch_size": config.batch_size,
        "train_micro_batch_size_per_gpu": (
            config.batch_size // distr_backend.get_world_size()
        ),
        "wall_clock_breakdown": False,
    }

    parameters = [p for p in model.parameters() if p.requires_grad]

    (dist_model, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=config,
        model=model,
        optimizer=opt if not using_deepspeed else None,
        model_parameters=parameters,
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

    print(
        " We are using DeepSpeed Schedular {} and it is {}".format(
            using_deepspeed_sched, distr_sched
        )
    )

    chk_top1_val = 0
    chk_epoch = 0

    if config.PeIT.finetune_checkpoint != "":
        _, states = dist_model.load_checkpoint(
            config.PeIT.finetune_checkpoint,
        )
        chk_epoch, chk_top1_val = states["epoch"], states["top1_val"]
        print(
            "---> Model loaded from: {}.\n\tLast epoch: {}.\n\tTop1 Validation: {:3.2f}".format(
                config.PeIT.finetune_checkpoint, chk_epoch, chk_top1_val
            )
        )
    # Let's create the sub folder for saving the checkpoints
    base_directory = Path(config.PeIT.checkpoint_root_folder)
    full_path = base_directory / config.PeIT.custom_run_name
    if not full_path.exists():
        # Let's error if the folder exist!
        full_path.mkdir(parents=True, exist_ok=True)
    elif any(full_path.iterdir()):
        raise FileExistsError(
            "XXX> The folder has already been created and it's not empty."
        )

    run = None
    if distr_backend.is_root_worker():
        # weights & biases experiment tracking

        import wandb

        model_config = dict(
            name=args.model,
            img_size=config.DATASET.Heatmap_Generator.heatmap_size,
            in_chans=config.DATASET.window_size,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            saved_dir=full_path,
            iteration_size=iteration_size,
            wandb_dir=config.wandb_log_dir,
            embed_2dpatch=config.PeIT.embed_2dpatch,
        )

        run = wandb.init(
            project="BEiT_window_{}_model_{}".format(
                config.DATASET.window_size, args.model
            ),
            job_type="pre-training",
            config=model_config,
            dir=config.wandb_log_dir,
        )

    start_epoch = max(0, chk_epoch)
    acc_top = max(0.0, chk_top1_val)
    global_step = 0

    nprocs = distr_backend.get_world_size()
    print("===> The word size is {}".format(nprocs))

    for epoch in range(start_epoch, config.epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, batch in enumerate(distr_dl):
            dist_model.train()

            if len(batch) == 2:
                heatmaps, bool_masked_pos = batch
                dvae_heatmap = heatmaps
            else:
                heatmaps, dvae_heatmap, bool_masked_pos = batch

            heatmaps = heatmaps.to(device, non_blocking=True)
            dvae_heatmap = dvae_heatmap.to(device, non_blocking=True)
            # samples = samples.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

            with torch.no_grad():
                input_ids = dVAE.get_codebook_indices(dvae_heatmap).flatten(1)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = input_ids[bool_masked_pos]

            with torch.cuda.amp.autocast():
                outputs = dist_model(
                    heatmaps,
                    bool_masked_pos=bool_masked_pos,
                    return_all_tokens=False,
                )
                loss = nn.CrossEntropyLoss()(input=outputs, target=labels)

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

            if using_deepspeed:
                torch.distributed.barrier()

            acc = accuracy(outputs, labels, topk=(1, 5))

            reduced_acc1 = reduce_mean(acc[0], nprocs)
            reduced_acc5 = reduce_mean(acc[1], nprocs)

            top1.update(reduced_acc1, heatmaps.size(0))
            top5.update(reduced_acc5, heatmaps.size(0))

            if using_deepspeed:
                total_norm = dist_model.get_global_grad_norm()
            else:
                parameters = [
                    p
                    for p in dist_model.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                if len(parameters) == 0:
                    total_norm = 0.0
                else:
                    device = parameters[0].grad.device
                    total_norm = torch.norm(
                        torch.stack(
                            [torch.norm(p.grad.detach()).to(device) for p in parameters]
                        ),
                        2.0,
                    ).item()

            if distr_backend.is_root_worker():
                if i % 10 == 0:
                    # Will be engin.step() will be ignored if the fp16 has overflow
                    try:
                        lr = distr_sched.get_last_lr()[0]
                    except:
                        lr = float("nan")
                    print(epoch, i, f"lr - {lr:6f} loss - {avg_loss.item()}")

                    logs = {
                        **logs,
                        "Top-1 (Training)": top1.avg,
                        "Top-5 (Training)": top5.avg,
                        "epoch": epoch + 1,
                        "loss": avg_loss.item(),
                        "Grad. Norm.": total_norm,
                        "label hist.": wandb.Histogram(labels.detach().cpu().numpy()),
                        "lr": lr,
                    }

                wandb.log(logs)
            global_step += 1

        if acc_top < top1.avg:

            print("-> Saving model for acc: {:.2f} ".format(top1.avg))
            save_model(
                full_path,
                using_deepspeed,
                dist_model,
                distr_backend,
                model,
                epoch=epoch + 1,
                acc_top1=top1.avg,
                run_name=run.name if run is not None else "",
            )
            acc_top = top1.avg

    if distr_backend.is_root_worker():
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    update_config(args.cfg)
    main(args)

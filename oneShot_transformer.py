import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from pathlib import Path

import argparse
from pathlib import Path

# torch

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler

# dalle classes and utils

from dalle_pytorch import distributed_utils


# vision imports
from torch.utils.data import DataLoader

from datetime import datetime


# For DS
import dataset

# Dataset Config
from configs.config import config
from configs.config import update_config

# Heatmap
from utils import init_distributed_mode, get_rank, get_model_size
from utils.args_handler import get_args_finetune

from timm.models import create_model

# Note: For updating timm.models dictionary
import models

from utils.axu import fill_the_model, AverageMeter

from pytorch_metric_learning.samplers import MPerClassSampler
from catalyst.data import DistributedSamplerWrapper
from utils.loss_functions import SupConLoss
import torch.nn.functional as F
from utils.lr_scheduler import get_cosine_schedule_with_warmup

from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from deepspeed.accelerator import get_accelerator
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=config.DATASET.Heatmap_Generator.heatmap_size,
        in_chans=(
            config.DATASET.window_size
            if config.DATASET.Heatmap_Generator.joint_reduction
            else config.DATASET.joint_number
        ),
        num_classes=config.DATASET.num_classes,
        pretrained=False,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        embed_2dpatch=config.PeIT.embed_2dpatch,
        patch_size=config.PeIT.patch_size,
        drop_rate=config.PeIT.drop_rate,
        attn_drop_rate=config.PeIT.attn_drop_rate,
        drop_path_rate=config.PeIT.drop_path_rate,
    )

    return model


def save_model(
    path,
    using_deepspeed,
    distr_model,
    distr_backend,
    model,
    epoch,
    top1_val=0,
):
    save_obj = {"epoch": epoch, "top1_val": top1_val}

    if using_deepspeed:

        distr_model.save_checkpoint(path, client_state=save_obj)
        return
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    # Fixme: Add optimizer and scheduler states here.
    save_obj = {**save_obj, "weights": model.state_dict()}

    torch.save(save_obj, path)


def extract_feats(dataloader_x, model, distr_backend):
    all_feats = []
    all_gts = []

    model.eval()

    device = torch.device(get_accelerator().device_name(args.local_rank))

    with torch.no_grad():
        for _, (batch_input, batch_gt) in enumerate(dataloader_x):
            batch_input = batch_input.to(device, non_blocking=True)
            batch_gt = batch_gt.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                feat = model(batch_input)
            all_feats.append(feat)
            all_gts.append(batch_gt)

    # Concatenate local results
    all_feats = torch.cat(all_feats)
    all_gts = torch.cat(all_gts)

    # Gather results from all processes
    world_size = distr_backend.get_world_size()
    all_feats_list = [torch.zeros_like(all_feats) for _ in range(world_size)]
    all_gts_list = [torch.zeros_like(all_gts) for _ in range(world_size)]

    dist.all_gather(all_feats_list, all_feats)
    dist.all_gather(all_gts_list, all_gts)

    # Concatenate results from all processes
    all_feats = torch.cat(all_feats_list)
    all_gts = torch.cat(all_gts_list)

    return all_feats, all_gts


def validate(anchor_loader, test_loader, model, distr_backend):
    train_feats, train_labels = extract_feats(anchor_loader, model, distr_backend)
    test_feats, test_labels = extract_feats(test_loader, model, distr_backend)
    M = len(train_feats)
    N = len(test_feats)
    train_feats = train_feats.unsqueeze(1)
    test_feats = test_feats.unsqueeze(0)
    dis = F.cosine_similarity(train_feats, test_feats, dim=-1)
    pred = train_labels[torch.argmax(dis, dim=0)]
    assert len(pred) == len(test_labels)
    acc = sum(pred == test_labels) / len(pred)
    return acc


def main(args):
    debug = False

    init_distributed_mode(args)

    print(args)

    args.config = config
    args.model = config.PeIT.model

    # fix the seed for reproducibility
    if False:
        seed = args.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    _ = get_model_size(model)
    filled = fill_the_model(model, args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    config.PeIT.window_size = (
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[0],
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[1],
    )
    args.patch_size = patch_size

    if config.PeIT.finetune_checkpoint != "":
        if ".pt" in config.PeIT.finetune_checkpoint:
            print("==> Loading the model from a pretrained PT file.")
            model_sd = torch.load(config.PeIT.finetune_checkpoint)
            model.load_state_dict(model_sd)

    distr_backend = distributed_utils.set_backend_from_args(config)
    distr_backend.initialize()

    using_deepspeed = distributed_utils.using_backend(
        distributed_utils.DeepSpeedBackend
    )

    # data training
    ds_train = eval("dataset." + config.DATASET.train_dataset)(
        config, loader_type="train"
    )

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, loader_type="val")

    # data validation
    ds_anchor = eval("dataset." + config.DATASET.test_dataset)(
        config, loader_type="anchors"
    )

    sampler = MPerClassSampler(
        ds_train.labels,
        m=config.DATASET.n_views,
        batch_size=config.batch_size,
        length_before_new_iter=len(ds_train),
    )

    nprocs = distr_backend.get_world_size()
    print("===> The word size is {}".format(nprocs))

    ds_sampler = DistributedSampler(
        sampler=sampler, num_replicas=nprocs, rank=args.local_rank, shuffle=False
    )

    dl_train = DataLoader(
        ds_train,
        config.batch_size,
        shuffle=sampler is None,
        sampler=ds_sampler,
        prefetch_factor=4,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=False if debug else (config.num_workers > 0),
    )

    val_sampler = None
    anchor_sampler = None
    if using_deepspeed:
        val_sampler = DistributedSampler(
            ds_eval,
            num_replicas=nprocs,
            rank=args.local_rank,
            shuffle=False,
            drop_last=False,
        )

        anchor_sampler = DistributedSampler(
            ds_anchor,
            num_replicas=nprocs,
            rank=args.local_rank,
            shuffle=False,
            drop_last=False,
        )

    dl_val = DataLoader(
        ds_eval,
        config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        prefetch_factor=4,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    dl_anchor = DataLoader(
        ds_anchor,
        config.batch_size,
        shuffle=False,
        sampler=anchor_sampler,
        prefetch_factor=4,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    opt = Adam(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
    )

    step_size_up = int(config.coeff_step_size_up * len(dl_train))

    sched = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=step_size_up,
        num_training_steps=config.epochs * len(dl_train),
        num_cycles=0.5,
        last_epoch=-1,
    )

    # distribute

    distr_backend.check_batch_size(config.batch_size)

    deepspeed_config = {
        "fp16": {"enabled": True},
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
                "total_num_steps": config.epochs * len(dl_train),
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
        "gradient_clipping": 0,
        "steps_per_print": 2000,
        "train_batch_size": config.batch_size,
        "train_micro_batch_size_per_gpu": (
            config.batch_size // distr_backend.get_world_size()
        ),
        "wall_clock_breakdown": False,
    }

    parameters = [p for p in model.parameters() if p.requires_grad]

    dist_model = None
    if using_deepspeed:
        (dist_model, opt, _, sched) = distr_backend.distribute(
            args=config,
            model=model,
            model_parameters=parameters,
            config_params=deepspeed_config,
        )

    chk_top1_val = 0
    chk_epoch = 0

    if config.PeIT.finetune_checkpoint != "":
        if not ".pt" in config.PeIT.finetune_checkpoint:
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
            "XXX> The folder `{}` has already been created and it's not empty.".format(
                str(full_path)
            )
        )

    criterion = SupConLoss(temperature=config.PeIT.contrastive_temp)

    if distr_backend.is_root_worker():
        model_config = dict(
            name=args.model,
            img_size=config.DATASET.Heatmap_Generator.heatmap_size,
            in_chans=config.DATASET.window_size,
            pretrained=filled,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            saved_dir=full_path,
        )

        run = wandb.init(
            project="BEiT_{}_{}_Fine_Tunning_{}_OneShot".format(
                config.DATASET.train_dataset, config.extra_project_name, args.model
            ),
            job_type="training",
            config=model_config,
        )

    start_epoch = max(0, chk_epoch)
    val_top1 = max(0.0, chk_top1_val)

    device = torch.device(get_accelerator().device_name(args.local_rank))

    print("Staring training...")
    model_candidate = dist_model if using_deepspeed else model
    for epoch in range(start_epoch, config.epochs):
        losses_train = AverageMeter()
        for i, (heatmaps, labels) in enumerate(dl_train):
            model_candidate.train()

            heatmaps = heatmaps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output_features = model_candidate(heatmaps)
                # Num_classes here acts the hidden dimension size.
                output_features = output_features.reshape(
                    config.batch_size, -1, config.DATASET.num_classes
                )
                loss = criterion(output_features, labels)

            if using_deepspeed:
                # Gradients are automatically zeroed after the step
                dist_model.backward(loss)
                dist_model.step()
                avg_loss = distr_backend.average_all(loss)
            else:
                opt.zero_grad()
                loss.backward()
                losses_train.update(loss.item(), config.batch_size)
                avg_loss = losses_train.avg
                opt.step()
                sched.step()

            logs = {}

            if (i % 100 == 0) and distr_backend.is_root_worker():
                # Will be engin.step() will be ignored if the fp16 has overflow
                try:
                    lr = sched.get_last_lr()[0]
                except:
                    lr = float("nan")
                print(epoch + 1, i, f"lr - {lr:6f} loss - {avg_loss}")

                logs = {
                    **logs,
                    "Max Top-1 (Validation)": val_top1,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "lr": lr,
                }

                wandb.log(logs)

            if (i % (len(dl_train) // config.repeated_size_of_dataset) == 0) and i > 0:
                if using_deepspeed:
                    dist.barrier()

                print("-> Starting validation...")
                start = time.time()
                acc = validate(dl_anchor, dl_val, dist_model, distr_backend)
                end = time.time()
                print("Validation took: {}".format(end - start))

                if val_top1 < acc:
                    # save trained model to wandb as an artifact every epoch's end

                    print("-> Saving model for acc: {:.2f} ".format(acc))
                    chk_path = str(
                        full_path / "beit_best_{}_finetuning.pt".format(run.name)
                    )
                    save_model(
                        chk_path,
                        using_deepspeed,
                        dist_model,
                        distr_backend,
                        model,
                        epoch,
                        acc,
                    )
                    val_top1 = acc

    if distr_backend.is_root_worker():
        wandb.finish()


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)

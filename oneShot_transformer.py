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

import wandb


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


def save_env(
    path,
    model,
    epoch,
    optimizer,
    lr_sched,
    top1_val=0.0,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched,
        "top1_val": top1_val,
    }

    torch.save(checkpoint, path)


def load_env(
    path,
    model,
    optimizer,
    lr_sched,
):

    chk_pt = torch.load(path)
    epoch, model_state, optimizer_state, lr_sched_state, top1_val = (
        chk_pt["epoch"],
        chk_pt["model"],
        chk_pt["optimizer"],
        chk_pt["lr_sched"],
        chk_pt["top1_val"],
    )

    optimizer.load_state_dict(optimizer_state)
    model.load_state_dict(model_state)
    lr_sched.load_state_dict(lr_sched_state)

    return epoch, top1_val


def extract_feats(dataloader_x, model):
    all_feats = []
    all_gts = []
    with torch.no_grad():
        for _, (batch_input, batch_gt) in enumerate(dataloader_x):
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            feat = model(batch_input)
            all_feats.append(feat)
            all_gts.append(batch_gt)
    all_feats = torch.cat(all_feats)
    all_gts = torch.cat(all_gts)
    return all_feats, all_gts


def validate(anchor_loader, test_loader, model):
    train_feats, train_labels = extract_feats(anchor_loader, model)
    test_feats, test_labels = extract_feats(test_loader, model)
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

    device = torch.device(config.device)

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

    dl_train = DataLoader(
        ds_train,
        config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        prefetch_factor=4,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False if debug else (config.num_workers > 0),
    )

    dl_val = DataLoader(
        ds_eval,
        config.batch_size,
        shuffle=False,
        prefetch_factor=4,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False if debug else (config.num_workers > 0),
        drop_last=False,
    )

    dl_anchor = DataLoader(
        ds_anchor,
        config.batch_size,
        shuffle=False,
        prefetch_factor=4,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False if debug else (config.num_workers > 0),
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

    chk_top1_val = 0
    chk_epoch = 0
    if config.PeIT.finetune_checkpoint != "":
        if ".pt" in config.PeIT.finetune_checkpoint:
            print("==> Loading the model from a pretrained PT file.")
            chk_epoch, chk_top1_val = load_env(
                config.PeIT.finetune_checkpoint,
                model=model,
                optimizer=opt,
                lr_sched=sched,
            )

    model.to(device)

    criterion = SupConLoss(temperature=config.PeIT.contrastive_temp)

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
    global_step = 0

    print("Staring training...")
    for epoch in range(start_epoch, config.epochs):
        losses_train = AverageMeter()
        # for i, (heatmaps, labels) in enumerate(dl_train):
        for i in range(4):
            (heatmaps, labels) = next(iter(dl_train))
            model.train()

            heatmaps = heatmaps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output_features = model(heatmaps)
            # Num_classes here acts the hidden dimension size.
            output_features = output_features.reshape(
                config.batch_size, -1, config.DATASET.num_classes
            )
            loss = criterion(output_features, labels)

            opt.zero_grad()
            losses_train.update(loss.item(), config.batch_size)
            loss.backward()
            opt.step()
            sched.step()

            logs = {}

            if i % 100 == 0:
                # Will be engin.step() will be ignored if the fp16 has overflow
                try:
                    lr = sched.get_last_lr()[0]
                except:
                    lr = float("nan")
                print(epoch + 1, i, f"lr - {lr:6f} loss - {losses_train.avg}")

                logs = {
                    **logs,
                    "Max Top-1 (Validation)": val_top1,
                    "epoch": epoch + 1,
                    "loss": losses_train.avg,
                    "lr": lr,
                }

            wandb.log(logs)

            global_step += 1

            if (
                True
            ):  # (i % (len(dl_train) // config.repeated_size_of_dataset) == 0) and i > 0:
                print("-> Starting validation...")
                start = time.time()
                acc = validate(dl_anchor, dl_val, model)
                end = time.time()
                print("Validation took: {}".format(end - start))

                if val_top1 < acc:
                    # save trained model to wandb as an artifact every epoch's end

                    print("-> Saving model for acc: {:.2f} ".format(acc))
                    chk_path = str(
                        full_path / "beit_best_{}_finetuning.pt".format(run.name)
                    )
                    save_env(chk_path, model, epoch, opt, sched, acc)
                    val_top1 = acc

    wandb.finish()


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)

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
from torch.utils.data import DataLoader

# Dataset Config
from configs.config import config
from configs.config import update_config

# Heatmap
from utils.args_handler import get_args_finetune

from timm.models import create_model
from timm.utils import accuracy

import dataset

# Note: For updating timm.models dictionary
import models

from utils.axu import fill_the_model, AverageMeter


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


def save_model(path, model, epoch, top1_val=0):
    save_obj = {"epoch": epoch, "top1_val": top1_val, "weights": model.state_dict()}
    torch.save(save_obj, path)


def validate(model, dl_validation, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        top1 = AverageMeter()
        top5 = AverageMeter()

        for _, batch in enumerate(dl_validation):
            if len(batch) > 2:
                (data, _, target) = batch
            else:
                data, target = batch
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(data)

            acc = accuracy(outputs, target, topk=(1, 5))

            top1.update(acc[0], data.size(0))
            top5.update(acc[1], data.size(0))

    return top1.avg, top5.avg


def main(args):
    print(args)

    args.config = config
    args.model = config.PeIT.model

    device = torch.device(config.device)

    cudnn.benchmark = True

    model = get_model(args)
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

    model.to(device)

    # data training
    ds_train = eval("dataset." + config.DATASET.train_dataset)(config, is_training=True)

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)

    dl_train = DataLoader(
        ds_train,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    dl_val = DataLoader(
        ds_eval,
        config.batch_size,
        shuffle=False,
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

    sched = OneCycleLR(
        optimizer=opt,
        max_lr=config.max_learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(dl_train),
    )

    base_directory = Path(config.PeIT.checkpoint_root_folder)
    full_path = base_directory / config.PeIT.custom_run_name
    if not full_path.exists():
        full_path.mkdir(parents=True, exist_ok=True)
    elif any(full_path.iterdir()):
        raise FileExistsError(
            "XXX> The folder `{}` has already been created and it's not empty.".format(
                str(full_path)
            )
        )

    import wandb

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
        project="BEiT_{}_{}_Fine_Tunning_{}".format(
            config.DATASET.train_dataset, config.extra_project_name, args.model
        ),
        job_type="training",
        config=model_config,
    )

    start_epoch = 0
    val_top1 = 0.0
    val_top5 = 0.0

    for epoch in range(start_epoch, config.epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, batch in enumerate(dl_train):
            model.train()

            if len(batch) == 2:
                heatmaps, labels = batch
            else:
                heatmaps, _, labels = batch

            heatmaps = heatmaps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(heatmaps)
                loss = nn.CrossEntropyLoss(label_smoothing=0.1)(
                    input=outputs, target=labels
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            logs = {}

            sched.step()

            avg_loss = loss.item()

            acc = accuracy(outputs, labels, topk=(1, 5))

            top1.update(acc[0], heatmaps.size(0))
            top5.update(acc[1], heatmaps.size(0))

            if i % 100 == 0:
                lr = sched.get_last_lr()[0]
                print(epoch + 1, i, f"lr - {lr:6f} loss - {avg_loss}")

                logs = {
                    **logs,
                    "Top-1 (Training)": top1.avg,
                    "Top-5 (Training)": top5.avg,
                    "Max Top-1 (Validation)": val_top1,
                    "Max Top-5 (Validation)": val_top5,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "lr": lr,
                }

                wandb.log(logs)

            if (i % (len(dl_train) // config.repeated_size_of_dataset) == 0) and i > 0:
                print("-> Starting validation...")
                start = time.time()
                acc_val = validate(model, dl_val, device)
                end = time.time()
                print("Validation took: {}".format(end - start))

                if val_top1 < acc_val[0]:
                    print("-> Saving model for acc: {:.2f} ".format(top1.avg))
                    chk_path = str(full_path / "beit_best_{}_finetuning.pt".format(run.name))
                    save_model(chk_path, model, epoch=epoch, top1_val=acc_val[0])
                    val_top1 = acc_val[0]
                    val_top5 = acc_val[1]

    wandb.finish()


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)

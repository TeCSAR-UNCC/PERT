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


def save_model(
    path,
    hparam,
    using_deepspeed,
    distr_model,
    distr_backend,
    model,
):
    save_obj = {
        "hparams": hparam,
    }
    print("Saved at {}".format(path))

    if using_deepspeed and False:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + "-ds-cp"

        distr_model.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    save_obj = {**save_obj, "weights": model.state_dict()}

    torch.save(save_obj, path)


def validate(model_engin, dl_validation, device):
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

            top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True)
            top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True)
            total += target.size(0)
            print(
                "{:3.2f} of validation is passed.".format(
                    (i * 100) / len(dl_validation)
                ),
                end="\r",
            )

    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100

    print()

    return top1_accuracy.item(), top5_accuracy.item()


def main(args):
    init_distributed_mode(args)

    print(args)

    args.config = config

    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    fill_the_model(model, args)
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

    # data training
    ds_train = eval("dataset." + config.DATASET.train_dataset)(config, is_train=True)

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_train=False)

    if distributed_utils.using_backend(distributed_utils.HorovodBackend):
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            ds_train,
            num_replicas=distr_backend.get_world_size(),
            rank=distr_backend.get_rank(),
        )
    else:
        data_sampler = None

    dl_train = DataLoader(
        ds_train,
        config.batch_size,
        shuffle=data_sampler is None,
        sampler=data_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
    )

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

    opt = FusedLamb(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
    )

    step_size_up = int(config.coeff_step_size_up * len(dl_train))
    step_size_down = int(config.coeff_step_size_down * len(dl_train))

    sched = OneCycleLR(
        optimizer=opt,
        max_lr=config.max_learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(dl_train),
    )

    # distribute

    distr_backend.check_batch_size(config.batch_size)

    deepspeed_config = {
        "fp16": {"enabled": True},
        "bf16": {"enabled": False},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.base_learning_rate,
                "weight_decay": config.weight_decay,
            },
        },
        "scheduler": {
            "type": "OneCycle",
            "params": {
                "cycle_first_step_size": step_size_up,
                "cycle_second_step_size": step_size_down,
                "cycle_min_lr": config.base_learning_rate,
                "cycle_max_lr": config.max_learning_rate,
                "decay_lr_rate": config.lr_decay,
            },
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 3.0,
        "steps_per_print": 2000,
        "train_batch_size": config.batch_size,
        "train_micro_batch_size_per_gpu": (
            config.batch_size // distr_backend.get_world_size()
        ),
        "wall_clock_breakdown": False,
    }

    nprocs = distr_backend.get_world_size()
    print("===> The word size is {}".format(nprocs))

    (dist_model, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=config,
        model=model,
        optimizer=opt if not using_deepspeed else None,
        model_parameters=model.parameters(),
        training_data=ds_train if using_deepspeed else dl_train,
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
        )

        run = wandb.init(
            project="BEiT_Fine_Tunning_{}".format(args.model),
            job_type="training",
            config=model_config,
        )

    global_step = 0
    val_top1 = 0.0

    for epoch in range(config.epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, batch in enumerate(distr_dl):
            model.train()

            heatmaps, labels = batch
            heatmaps = heatmaps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(
                    heatmaps,
                )
                loss = nn.CrossEntropyLoss()(input=outputs, target=labels)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                if distr_backend.is_root_worker():
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

            torch.distributed.barrier()

            reduced_acc1 = reduce_mean(acc[0], nprocs)
            reduced_acc5 = reduce_mean(acc[1], nprocs)

            top1.update(reduced_acc1, heatmaps.size(0))
            top5.update(reduced_acc5, heatmaps.size(0))

            if distr_backend.is_root_worker():
                if i % 100 == 0:

                    acc_val = [0, 0]  # validate(model, dl_val, device)

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
                        # "Top-1 (Validation)": acc_val[0],
                        # "Top-5 (Validation)": acc_val[1],
                        "epoch": epoch,
                        "iter": i,
                        "loss": avg_loss.item(),
                        "lr": lr,
                    }

                wandb.log(logs)
            global_step += 1

            if distr_backend.is_root_worker() and val_top1 < top1.avg:
                # save trained model to wandb as an artifact every epoch's end

                print("-> Saving model for acc: {:.2f} ".format(top1.avg))
                save_model(
                    "./saved_models/beit_best_{}_finetuning.pt".format(run.name),
                    model_config,
                    using_deepspeed,
                    dist_model,
                    distr_backend,
                    model,
                )
                # model_artifact = wandb.Artifact(
                #    "trained-model", type="model", metadata=model_config
                # )
                # model_artifact.add_file(
                #    "./saved_models/beit_best_{}_finetuning.pt".format(run.name)
                # ),
                # run.log_artifact(model_artifact)

                val_top1 = top1.avg

    if distr_backend.is_root_worker():
        # save final vae and cleanup

        save_model(
            "./saved_models/beit_final_{}_finetuning.pt".format(run.name),
            model_config,
            using_deepspeed,
            dist_model,
            distr_backend,
            model,
        )
        wandb.save(
            "./saved_models/beit_final_{}_finetuning.pt".format(run.name),
        )

        if epoch % 5 == 0:
            model_artifact = wandb.Artifact(
                "trained-finetuining", type="model", metadata=dict(args.model)
            )
            model_artifact.add_file(
                "./saved_models/beit_final_{}_finetuning.pt".format(run.name),
            )
            run.log_artifact(model_artifact)

        wandb.finish()


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)

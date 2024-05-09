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
from utils import init_distributed_mode, get_rank
from utils.args_handler import get_args_finetune

from timm.models import create_model
from timm.utils import accuracy

# Note: For updating timm.models dictionary
import models

from utils.axu import fill_the_model, AverageMeter, reduce_mean


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


def validate(model_engin, dl_validation, device, using_deepspeed, nprocs):
    model_engin.eval()  # Set the model to evaluation mode
    # print("---> Hello from validation.")
    with torch.no_grad():  # Disable gradient calculation
        top1 = AverageMeter()
        top5 = AverageMeter()

        for _, batch in enumerate(dl_validation):
            # for _ in range(1):
            #    batch = next(iter(dl_validation))
            if len(batch) > 2:
                (data, _, target) = batch
            else:
                data, target = batch
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                outputs = model_engin(data)

            acc = accuracy(outputs, target, topk=(1, 5))

            torch.distributed.barrier()

            if using_deepspeed:
                reduced_acc1 = reduce_mean(acc[0], nprocs)
                reduced_acc5 = reduce_mean(acc[1], nprocs)
            else:
                reduced_acc1 = acc[0]
                reduced_acc5 = acc[1]

            top1.update(reduced_acc1, data.size(0))
            top5.update(reduced_acc5, data.size(0))

    return top1.avg, top5.avg


def main(args):
    debug = False

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
    filled = fill_the_model(model, args)
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
    ds_train = eval("dataset." + config.DATASET.train_dataset)(config, is_training=True)

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)

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
        1 if debug else config.batch_size,
        shuffle=data_sampler is None,
        sampler=data_sampler,
        num_workers=0 if debug else config.num_workers,
        pin_memory=True,
        persistent_workers=False if debug else (config.num_workers > 0),
    )

    nprocs = distr_backend.get_world_size()
    print("===> The word size is {}".format(nprocs))

    val_sampler = None
    if using_deepspeed:
        val_sampler = DistributedSampler(
            ds_eval,
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

    (dist_model, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=config,
        model=model,
        optimizer=opt if not using_deepspeed else None,
        model_parameters=parameters,
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
            "XXX> The folder `{}` has already been created and it's not empty.".format(
                str(full_path)
            )
        )

    if distr_backend.is_root_worker():
        # weights & biases experiment tracking

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

    start_epoch = max(0, chk_epoch)
    val_top1 = max(0.0, chk_top1_val)
    val_top5 = 0.0
    global_step = 0

    for epoch in range(start_epoch, config.epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, batch in enumerate(distr_dl):
            # for i in range(4):
            #    batch = next(iter(distr_dl))
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
            if using_deepspeed:
                reduced_acc1 = reduce_mean(acc[0], nprocs)
                reduced_acc5 = reduce_mean(acc[1], nprocs)
            else:
                reduced_acc1 = acc[0]
                reduced_acc5 = acc[1]

            top1.update(reduced_acc1, heatmaps.size(0))
            top5.update(reduced_acc5, heatmaps.size(0))

            torch.distributed.barrier()

            if distr_backend.is_root_worker():
                if i % 100 == 0:
                    # Will be engin.step() will be ignored if the fp16 has overflow
                    try:
                        lr = distr_sched.get_last_lr()[0]
                    except:
                        lr = float("nan")
                    print(epoch + 1, i, f"lr - {lr:6f} loss - {avg_loss.item()}")

                    logs = {
                        **logs,
                        "Top-1 (Training)": top1.avg,
                        "Top-5 (Training)": top5.avg,
                        "Max Top-1 (Validation)": val_top1,
                        "Max Top-5 (Validation)": val_top5,
                        "epoch": epoch + 1,
                        "loss": avg_loss.item(),
                        "lr": lr,
                    }

                wandb.log(logs)

            global_step += 1

            if (i % (len(dl_train) // config.repeated_size_of_dataset) == 0) and i > 0:
                print("-> Starting validation...")
                start = time.time()
                acc_val = validate(
                    model,
                    dl_val,
                    device,
                    using_deepspeed=using_deepspeed,
                    nprocs=nprocs,
                )
                end = time.time()
                print("Validation took: {}".format(end - start))

                if val_top1 < acc_val[0]:
                    # save trained model to wandb as an artifact every epoch's end

                    print("-> Saving model for acc: {:.2f} ".format(top1.avg))
                    if not using_deepspeed:
                        chk_path = str(
                            full_path / "beit_best_{}_finetuning.pt".format(run.name)
                        )
                    else:
                        chk_path = str(full_path)
                    save_model(
                        chk_path,
                        using_deepspeed,
                        dist_model,
                        distr_backend,
                        model,
                        epoch=epoch,
                        top1_val=acc_val[0],
                    )
                    val_top1 = acc_val[0]
                    val_top5 = acc_val[1]

    if distr_backend.is_root_worker():
        wandb.finish()


if __name__ == "__main__":
    args = get_args_finetune()
    update_config(args.cfg)
    main(args)

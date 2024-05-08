import math
from math import sqrt
import argparse
from pathlib import Path

# torch

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch import Discrete3DVAE

# For DS
import dataset

# heatmap to color
from utils.axu import convert_to_rgb_3d, AverageMeter

# argument parsing

parser = argparse.ArgumentParser()

# Dataset Config

from configs.config import config
from configs.config import update_config

# Heatmap
from utils.heatmap_related import GeneratePoseTarget


def parse_args():
    parser = argparse.ArgumentParser(description="Overall Training for dVAE")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


args = parse_args()

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(config)
distr_backend.initialize()

using_deepspeed = distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

ds = eval("dataset." + config.DATASET.train_dataset)(config, is_training=True)

device = torch.device(config.device)

debug = False

if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=distr_backend.get_world_size(), rank=distr_backend.get_rank()
    )
else:
    data_sampler = None

dl_train = DataLoader(
    ds,
    1 if debug else config.batch_size,
    shuffle=data_sampler is None,
    sampler=data_sampler,
    num_workers=0 if debug else config.num_workers,
    pin_memory=True,
    persistent_workers=False if debug else (config.num_workers > 0),
)

vae = Discrete3DVAE(**config.VAE_Params)
d3d_encoder = config.VAE_Params.d3d_decoder

resume = False
if config.Resume.chk_ptn != "":
    print("--> Loading data from: {}".format(config.Resume.chk_ptn))
    state_dict = torch.load(config.Resume.chk_ptn, map_location="cpu")["weights"]
    vae.load_state_dict(state_dict)
    resume = True

if not using_deepspeed:
    vae = vae.cuda()


assert len(ds) > 0, "folder does not contain any images"
if distr_backend.is_root_worker():
    print(f"{len(ds)} images found for training")

# optimizer

opt = AdamW(
    vae.parameters(), lr=config.base_learning_rate, weight_decay=config.weight_decay
)
step_size_up = int(config.coeff_step_size_up * len(dl_train))
step_size_down = int(config.coeff_step_size_down * len(dl_train))
# New schedular.
sched = CyclicLR(
    optimizer=opt,
    base_lr=config.base_learning_rate,
    max_lr=config.max_learning_rate,
    mode="triangular2",
    step_size_up=step_size_up,
    step_size_down=step_size_down,
    cycle_momentum=False,
)

if distr_backend.is_root_worker():
    # weights & biases experiment tracking

    import wandb

    model_config = dict(
        num_tokens=config.VAE_Params.num_tokens,
        smooth_l1_loss=config.VAE_Params.smooth_l1_loss,
        num_resnet_blocks=config.VAE_Params.num_resnet_blocks,
        kl_loss_weight="Dynamic",
        wandb_dir=config.wandb_log_dir,
    )

    run = wandb.init(
        project="3DdVAE_window_{}_layer_{}_{}".format(
            config.DATASET.window_size,
            config.VAE_Params.num_layers,
            config.DATASET.train_dataset,
        ),
        job_type="dVAE_model",
        config=model_config,
        dir=config.wandb_log_dir,
    )

    if resume:
        print("--> Starting from: {}".format(config.Resume.chk_ptn))

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
            "total_num_steps": step_size_down,
            "warmup_min_ratio": 0,
            "warmup_num_steps": step_size_up,
            "cos_min_ratio": config.base_learning_rate,
        },
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

(distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=config,
    model=vae,
    optimizer=opt if not using_deepspeed else None,
    model_parameters=vae.parameters(),
    training_data=ds if using_deepspeed else dl_train,
    lr_scheduler=sched if not using_deepspeed else None,
    config_params=deepspeed_config,
)

print("The schedular that is going to be used: {}".format(distr_sched))

using_deepspeed_sched = False
# Prefer scheduler in `deepspeed_config`.
if distr_sched is None:
    distr_sched = sched
elif using_deepspeed:
    # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
    # handle its scheduling.
    using_deepspeed_sched = True


def save_model(path):
    save_obj = {
        "hparams": config.VAE_Params,
    }
    if False:  # using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + "-dvae-ds-cp"

        distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    save_obj = {**save_obj, "weights": vae.state_dict()}

    torch.save(save_obj, path)


# starting temperature

global_step = 0
temp = config.starting_temp
kl_weight_loss = config.starting_kl_weight
global_loss = float("inf")

for epoch in range(config.epochs):
    loss_per_epoch = AverageMeter()
    for i, heatmaps in enumerate(distr_dl):

        heatmaps = heatmaps.to(device)
        with torch.cuda.amp.autocast():
            loss, recons, recon_loss = distr_vae(
                heatmaps,
                return_loss=True,
                return_recons=True,
                temp=temp,
                kl_div_loss_weight=kl_weight_loss,
            )

        if using_deepspeed:
            # Gradients are automatically zeroed after the step
            distr_vae.backward(loss)
            distr_vae.step()
        else:
            distr_opt.zero_grad()
            loss.backward()
            distr_opt.step()

        logs = {}

        # lr decay
        # Do not advance schedulers from `deepspeed_config`.
        if not using_deepspeed_sched:
            distr_sched.step()

        if using_deepspeed:
            torch.distributed.barrier()

        if using_deepspeed:
            total_norm = distr_vae.get_global_grad_norm()
        else:
            parameters = [
                p
                for p in distr_vae.parameters()
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

        if i % 100 == 0:
            if distr_backend.is_root_worker():
                k = 4

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        codes = vae.get_codebook_indices(heatmaps[:k])
                        hard_recons = vae.decode(codes)

                heatmaps, recons = map(lambda t: t[:k], (heatmaps, recons))
                heatmaps, recons, hard_recons, codes = map(
                    lambda t: t.detach().cpu(), (heatmaps, recons, hard_recons, codes)
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
                collapsed_heatmap = heatmaps.numpy().max(axis=1).astype("float32")
                recons_heatmap = recons.numpy().astype("float32")
                if d3d_encoder:
                    recons_heatmap = recons_heatmap.max(axis=2)
                heatmaps_rgb = convert_to_rgb_3d(collapsed_heatmap)
                recons_rgb = convert_to_rgb_3d(recons_heatmap)
                # hard_recons_rgb = convert_to_rgb_3d(hard_recons.numpy())
                logs = {
                    **logs,
                    "sample heatmap": wandb.Video(
                        heatmaps_rgb, fps=30, caption="original heatmap"
                    ),
                    "reconstructions": wandb.Video(
                        recons_rgb, fps=30, caption="reconstructions"
                    ),
                    # "hard reconstructions": wandb.Video(
                    #    hard_recons_rgb, fps=30, caption="hard reconstructions"
                    # ),
                    "codebook_indices": wandb.Histogram(codes),
                    "temperature": temp,
                }
            # temperature anneal

            temp = max(
                temp * math.exp(-config.anneal_rate * global_step), config.temp_min
            )

            kl_weight_loss = kl_weight_loss * (
                math.exp(-config.kl_div_loss_weight_anneal_rate * global_step)
            )
            kl_weight_loss = (
                kl_weight_loss
                if kl_weight_loss > config.zero_kl_div_loss_weight_at
                else 0
            )

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)
        avg_recons_loss = distr_backend.average_all(recon_loss)
        loss_per_epoch.update(avg_recons_loss.item())

        if distr_backend.is_root_worker():
            if i % 10 == 0:
                try:
                    lr = distr_sched.get_last_lr()[0]
                except:
                    lr = float("nan")

                print(epoch + 1, i, f"lr - {lr:6f} loss - {avg_loss.item()}")

                logs = {
                    **logs,
                    "epoch": epoch + 1,
                    "iter": i,
                    "loss": avg_loss.item(),
                    "Epcoh loss": loss_per_epoch.avg,
                    "KL Div. Weight": kl_weight_loss,
                    "Grad. Norm": total_norm,
                    "lr": lr,
                }

            wandb.log(logs)
        global_step += 1

    if distr_backend.is_root_worker():
        # save trained model to wandb as an artifact every epoch's end

        if global_loss > loss_per_epoch.avg:
            file_name = "./saved_models/vae_{}_{}_{}.pt".format(
                config.DATASET.window_size, config.DATASET.train_dataset, run.name
            )
            print("-->Saving file: {}".format(file_name))
            save_model(file_name)
            global_loss = loss_per_epoch.avg

            model_artifact = wandb.Artifact(
                "trained-vae", type="model", metadata=dict(model_config)
            )

            model_artifact.add_file(file_name)
            run.log_artifact(model_artifact)

if distr_backend.is_root_worker():
    # save final vae and cleanup

    save_model("./vae-final_{}.pt".format(run.name))
    wandb.save("./vae-final_{}.pt".format(run.name))

    model_artifact = wandb.Artifact(
        "trained-vae", type="model", metadata=dict(model_config)
    )
    model_artifact.add_file("vae-final.pt")
    run.log_artifact(model_artifact)

    wandb.finish()

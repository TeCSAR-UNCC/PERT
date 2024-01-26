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
from dalle_pytorch import DiscreteVAE

# For DS
import dataset

# heatmap to color
from utils.axu import convert_to_rgb_3d

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

ds = eval("dataset." + config.DATASET.test_dataset)(
    config, config.DATASET.test_subset, is_train=False
)

if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=distr_backend.get_world_size(), rank=distr_backend.get_rank()
    )
else:
    data_sampler = None

dl = DataLoader(ds, config.batch_size, shuffle=not data_sampler, sampler=data_sampler)

vae = DiscreteVAE(**config.VAE_Params)

if not using_deepspeed:
    vae = vae.cuda()


assert len(ds) > 0, "folder does not contain any images"
if distr_backend.is_root_worker():
    print(f"{len(ds)} images found for training")

# optimizer

opt = AdamW(vae.parameters(), lr=config.lr, weight_decay=config.weight_decay)
sched = CyclicLR(
    optimizer=opt,
    base_lr=config.base_learning_rate,
    max_lr=config.max_learning_rate,
    mode="triangular2",
    step_size_up=config.coeff_step_size_up * len(ds),
    step_size_down=config.coeff_step_size_down * len(ds),
    cycle_momentum=False,
)

if distr_backend.is_root_worker():
    # weights & biases experiment tracking

    import wandb

    model_config = dict(
        num_tokens=config.VAE_Params.num_tokens,
        smooth_l1_loss=config.VAE_Params.smooth_l1_loss,
        num_resnet_blocks=config.VAE_Params.num_resnet_blocks,
        kl_loss_weight=config.VAE_Params.kl_div_loss_weight,
    )

    run = wandb.init(
        project="heatmap_train_vae_window_{}_layer_{}".format(
            config.DATASET.window_size, config.VAE_Params.num_layers
        ),
        job_type="dVAE_model",
        config=model_config,
    )

# distribute

distr_backend.check_batch_size(config.batch_size)
deepspeed_config = {"train_batch_size": config.batch_size}

(distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=args,
    model=vae,
    optimizer=opt,
    model_parameters=vae.parameters(),
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


def save_model(path):
    save_obj = {
        "hparams": config.VAE_Params,
    }
    if using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + "-ds-cp"

        distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    save_obj = {**save_obj, "weights": vae.state_dict()}

    torch.save(save_obj, path)


# starting temperature

global_step = 0
temp = config.starting_temp

for epoch in range(config.epochs):
    for i, heatmaps in enumerate(distr_dl):
        heatmaps = heatmaps.cuda()

        loss, recons = distr_vae(
            heatmaps, return_loss=True, return_recons=True, temp=temp
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

        if i % 100 == 0:
            if distr_backend.is_root_worker():
                k = 4

                with torch.no_grad():
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

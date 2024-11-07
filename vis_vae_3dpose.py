import argparse
from math import sqrt

import torch


# For DS
import dataset

# heatmap to color
from utils.axu import convert_to_rgb_3d, save_array_as_video, plot_heatmap

# argument parsing

parser = argparse.ArgumentParser()

# Dataset Config

from configs.config import config
from configs.config import update_config


from utils.get_dVAE import get_dVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Overall Training for dVAE")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


args = parse_args()


ds = eval("dataset." + config.DATASET.train_dataset)(config, is_training=False)

device = torch.device(config.device)



dVAE = get_dVAE(config)
if torch.cuda.is_available():
    print("Using cuda...")
    dVAE = dVAE.cuda()

#idx = 1245
import random
idx = random.randint(0, len(ds)-1)
heatmaps, labels = ds[idx]
heatmaps = torch.unsqueeze(heatmaps, 0)
if torch.cuda.is_available():
    heatmaps = heatmaps.cuda()
codes = dVAE.get_codebook_indices(heatmaps)
hard_recons = dVAE.decode(codes)

heatmaps, hard_recons = map(
    lambda t: t.detach().cpu(), (heatmaps,  hard_recons)
)

collapsed_heatmap = heatmaps.numpy().max(axis=1).astype("float32")
heatmaps_rgb = convert_to_rgb_3d(collapsed_heatmap)
recons_heatmap = hard_recons.numpy().astype("float32")
recons_heatmap_rgb = convert_to_rgb_3d(recons_heatmap)

save_array_as_video(heatmaps_rgb, "ntu_gt_{}.mp4".format(idx))
save_array_as_video(recons_heatmap_rgb, "ntu_recons_{}.mp4".format(idx))

b, n = codes.shape
h = w = int(sqrt(n))

from einops import rearrange

codes_sqr = rearrange(codes, "b (h w) -> b h w", h=h, w=w)

codes_sqr = torch.squeeze(codes_sqr, 0).cpu().numpy()
plot_heatmap(codes_sqr, save_path='figs/code_{}.pdf'.format(idx))

print("Finished")

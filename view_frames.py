import argparse
from configs.config import config
import cv2
import torch
import dataset
from configs.config import update_config
import numpy as np
from utils.get_dVAE import get_dVAE
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    device = torch.device(config.device)

    train_dataset = eval("dataset." + config.DATASET.test_dataset)(
        config, is_training=True
    )

    vf_idx = 3150

    ds_data = train_dataset.__getitem__(vf_idx)
    if len(ds_data) > 2:
        data_np, _, _ = ds_data
    else:
        data_np, _ = ds_data

    dVAE = get_dVAE(config)
    dVAE = dVAE.to(device)

    vf_idx = 3150  # random.randrange(0, len(test_dataset))

    data = np.expand_dims(data_np, 0)
    data = torch.from_numpy(data).to(device)

    temp = 0.5

    loss, recons = dVAE(data, return_loss=True, return_recons=True, temp=temp)

    recons = np.array(recons[0].cpu().detach().numpy())
    video_name = "ntu_recons_{}.mp4".format(vf_idx)
    frame_height, frame_width = recons.shape[1], recons.shape[2]
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height)
    )

    for i in range(recons.shape[0]):
        # Normalize the heatmap for display
        normalized_heatmap = cv2.normalize(recons[i], None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        # Write to video
        out.write(colored_heatmap)

    out.release()

    video_name = "ntu_sample_scnd.mp4"
    frame_height, frame_width = data_np.shape[1], data_np.shape[2]
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height)
    )

    for i in range(data_np.shape[0]):
        # Normalize the heatmap for display
        normalized_heatmap = cv2.normalize(data_np[i], None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        # Write to video
        out.write(colored_heatmap)

    out.release()


if __name__ == "__main__":
    main()

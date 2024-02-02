import argparse
from configs.config import config
import cv2
import torch
import dataset
from configs.config import update_config
import numpy as np
from utils.get_dVAE import get_dVAE
from utils.args_handler import get_args
from timm.models import create_model
import models


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=config.DATASET.Heatmap_Generator.heatmap_size,
        in_chans=config.DATASET.window_size,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    if config.Pretrained_Models.PERT.prefix_saved_file != "":
        state_dict = torch.load(
            config.Pretrained_Models.PERT.prefix_saved_file, map_location="cpu"
        )["weights"]
        model.load_state_dict(state_dict)

    return model


def main():
    args = get_args()
    update_config(args.cfg)

    device = torch.device(config.device)

    beit = get_model(args)
    beit = beit.to(device)

    patch_size = beit.patch_embed.patch_size

    config.PERT.window_size = (
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[0],
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[1],
    )

    test_dataset = eval("dataset." + config.DATASET.test_dataset)(
        config, config.DATASET.test_subset, is_training=False
    )

    vf_idx = 3150

    data, bool_masked_pos = test_dataset.__getitem__(vf_idx)
    data = np.expand_dims(data, 0)
    bool_masked_pos = np.expand_dims(bool_masked_pos, 0)
    data = torch.from_numpy(data).to(device, non_blocking=True)
    bool_masked_pos = torch.from_numpy(bool_masked_pos).to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

    dVAE = get_dVAE(config)
    dVAE = dVAE.to(device)

    temp = 0.5

    recons = dVAE(data, return_loss=False, return_recons=True, temp=temp)

    outputs = beit(
        data,
        bool_masked_pos=bool_masked_pos,
        return_all_tokens=True,
    )
    recon_beit = dVAE.decode(outputs.argmax(dim=2).flatten(1))

    recon_beit = np.array(recon_beit[0].cpu().detach().numpy())
    video_name = "beit_sample.mp4"
    frame_height, frame_width = recon_beit.shape[1], recon_beit.shape[2]
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height)
    )

    for i in range(recon_beit.shape[0]):
        # Normalize the heatmap for display
        normalized_heatmap = cv2.normalize(recon_beit[i], None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        # Write to video
        out.write(colored_heatmap)

    out.release()

    recons = np.array(recons[0].cpu().detach().numpy())
    video_name = "ntu_recons.mp4"
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

    data = np.array(data[0].cpu().detach().numpy())
    video_name = "ntu_sample.mp4"
    frame_height, frame_width = data.shape[1], data.shape[2]
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height)
    )

    for i in range(data.shape[0]):
        # Normalize the heatmap for display
        normalized_heatmap = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        # Write to video
        out.write(colored_heatmap)

    out.release()

    print()


if __name__ == "__main__":
    main()

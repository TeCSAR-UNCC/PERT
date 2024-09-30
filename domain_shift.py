import argparse
from configs.config import config
import torch

from torch.utils.data import DataLoader, SequentialSampler
from configs.config import update_config
import numpy as np
from pyskl.datasets.builder import build_dataset, build_dataloader
from pyskl.models import build_model
from mmcv import Config
from utils.multi_mapping import mapping_ucla2ntu, calculate_accuracies
import models
import dataset
from timm.models import create_model
from utils.args_handler import get_parsser_fintuned
from tqdm import tqdm
from pyskl.models.cnns import ResNet3dSlowOnly, ResNet


def parse_args():
    parser = get_parsser_fintuned()
    parser.add_argument(
        "--py-cfg", help="PYSKL configure file name", required=True, type=str
    )

    parser.add_argument(
        "--pysk-model-weight",
        help="Saved model for PYSL-based model configuration",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--peit-model-weight",
        help="Saved model for peit configure",
        required=True,
        type=str,
    )

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args


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


def main(args):
    '''
    m_3d = False
    if m_3d:
        test = ResNet3dSlowOnly(
            in_channels=17,
            base_channels=32,
            num_stages=3,
            out_indices=(2,),
            stage_blocks=(4, 6, 3),
            conv1_stride=(1, 1),
            pool1_stride=(1, 1),
            inflate=(0, 1, 1),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 2),
        )
        data = torch.rand((1, 17, 48, 64, 64))
    else:
        test = ResNet(
            in_channels=48,
            num_stages=3,
            out_indices=(2,),
            strides=(1, 2, 2),
        )
        data = torch.rand((1, 48, 64, 64))
    r = test(data)
    '''
    
    cfg = Config.fromfile(args.py_cfg)
    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model)
    dl = build_dataloader(
        ds,
        videos_per_gpu=4,
        workers_per_gpu=4,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )
    state = torch.load(args.pysk_model_weight)
    model.load_state_dict(state)
    model.cuda()
    model.eval()

    predicted = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dl):
            test_val, label = batch["imgs"], batch["label"]
            test_val = test_val.cuda()
            output = model(test_val, return_loss=False)
            top_5_indices = np.argsort(-output, axis=1)[:, :5]
            predicted.append(top_5_indices)
            targets.append(label.cpu().numpy())

    top1_model, top5_model = calculate_accuracies(predicted, targets, mapping_ucla2ntu)
    print("Top@1={}, top@5={}".format(top1_model, top5_model))
    peit = get_model(args)
    state_peit = torch.load(args.peit_model_weight)
    peit.load_state_dict(state_peit)
    peit.cuda()

    patch_size = peit.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    config.PeIT.window_size = (
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[0],
        config.DATASET.Heatmap_Generator.heatmap_size // patch_size[1],
    )
    args.patch_size = patch_size

    # data validation
    ds_eval = eval("dataset." + config.DATASET.test_dataset)(config, is_training=False)
    dl_val = DataLoader(
        ds_eval,
        config.batch_size,
        shuffle=False,
        sampler=SequentialSampler(ds_eval),
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    peit_predicted = []
    peit_targets = []
    with torch.no_grad():
        for batch in tqdm(dl_val):
            test_val, label = batch
            test_val = test_val.cuda()
            output = peit(test_val).cpu().numpy()
            top_5_indices = np.argsort(-output, axis=1)[:, :5]
            peit_predicted.append(top_5_indices)
            peit_targets.append(label.cpu().numpy())

    top1_peit, top5_peit = calculate_accuracies(
        peit_predicted, peit_targets, mapping_ucla2ntu
    )
    print("PeIT: Top@1={}, top@5={}".format(top1_peit, top5_peit))
    pass


if __name__ == "__main__":

    args = parse_args()
    args.config = config
    args.model = config.PeIT.model
    main(args)

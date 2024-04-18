import argparse
from configs.config import config
import torch
from configs.config import update_config
import numpy as np
from pyskl.datasets.builder import build_dataset, build_dataloader
from pyskl.models import build_model
from operator import itemgetter
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    parser.add_argument(
        "--py-cfg", help="PYSKL configure file name", required=True, type=str
    )

    parser.add_argument(
        "--pysk-model-weight",
        help="Saved model for PYSKL configure file name",
        required=True,
        type=str,
    )
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main(args):
    cfg = Config.fromfile(args.py_cfg)
    ds = build_dataset(cfg.data.val)
    model = build_model(cfg.model)
    dl = build_dataloader(
        ds,
        videos_per_gpu=1,
        workers_per_gpu=12,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )
    state = torch.load(args.pysk_model_weight)
    model.load_state_dict(state)
    batch = next(iter(dl))
    test_val, label = batch["imgs"], batch["label"]
    test_val = test_val.cuda()
    label = label.cuda()
    model.cuda()
    model.eval()
    with torch.no_grad():
        output = model(test_val, return_loss=False)
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)

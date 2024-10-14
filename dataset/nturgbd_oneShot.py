import os.path as osp
import numpy as np
import json_tricks as json
import logging
import pandas as pd
import mmcv

from torch.utils.data import Dataset
from mmcv import Config
from easydict import EasyDict as edict
from pyskl.datasets import build_dataset
from operator import itemgetter
from torch import Tensor


def get_class_label(filename):
    # Remove the file extension
    filename = filename.split(".")[0]

    # Extract the last 3 digits which represent the action class
    class_label = int(filename[-3:]) - 1

    return class_label


class NTU_oneShot(Dataset):
    def __init__(self, cfg, loader_type="train", **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        mmcv_cfg = Config.fromfile(cfg.DATASET.mmcv_config)

        self.loader_type = loader_type
        self.is_training = self.loader_type == "train"

        skl_filename = None
        skl_meta = None
        if loader_type == "train":
            self.ds = build_dataset(mmcv_cfg.data.train)
            skl_meta = mmcv.load(mmcv_cfg.data.train.dataset.ann_file)["split"]
            skl_filenames = skl_meta["oneShot_train"]
            self.action_set = skl_meta["action_set"]

        elif loader_type == "val":
            self.ds = build_dataset(mmcv_cfg.data.val)
        else:
            self.ds = build_dataset(mmcv_cfg.data.exemplar)

        self.labels = []
        if skl_meta is not None:
            for skl_filename in skl_filenames:
                self.labels.append(get_class_label(skl_filename))
        self.labels = self.labels * mmcv_cfg.data.train.times

        print("Finished creating one shot dataset...")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        imgs, label = itemgetter("imgs", "label")(self.ds.__getitem__(index))
        imgs = imgs.squeeze(0)
        if self.cfg.DATASET.Heatmap_Generator.joint_reduction:
            imgs = imgs.permute((1, 0, 2, 3))
            imgs, _ = imgs.max(axis=1)
        if self.is_training:
            return [imgs, label[0]]
        else:
            return [imgs, label]


# DATASET = edict({
#   'train_dataset': "ntu_mmcv",
#   'test_dataset': "ntu_mmcv",
#   'window_size': 48, # Dataset Window Size
#   'second_heatmap': None,
#   'num_classes': 120,
#   'joint_number': 17,
#   'mmcv_config': "configs/nturgbd/ntu120_limb_oneShot.py",

#   'Heatmap_Generator':{
#     # You need to also update `mmcv_config` if you change this value to something else.
#     'heatmap_size': 72,
#     'joint_reduction': False}
# })
# config = edict({'DATASET':DATASET})

# ds_train = NTU_oneShot(config, loader_type='train')
# ds_val = NTU_oneShot(config, loader_type='val')
# ds_exemplar = NTU_oneShot(config, loader_type='exemplar')

# x = ds_train.__getitem__(0)
# x = ds_val.__getitem__(0)
# print(x)

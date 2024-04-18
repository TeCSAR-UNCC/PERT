import os.path as osp
import numpy as np
import json_tricks as json
import logging
import pandas as pd

from torch.utils.data import Dataset
from mmcv import Config
from pyskl.datasets import build_dataset
from operator import itemgetter
from torch import Tensor


class UCLA_MMCV(Dataset):
    def __init__(self, cfg, is_training, **kwargs):
        super().__init__(**kwargs)
        cfg = Config.fromfile(cfg.DATASET.mmcv_config)

        self.is_training = is_training
        if is_training:
            self.ds = build_dataset(cfg.data.train)
        else:
            self.ds = build_dataset(cfg.data.val)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        imgs, label = itemgetter("imgs", "label")(self.ds.__getitem__(index))
        imgs = imgs.squeeze(0)
        imgs = imgs.permute((1, 0, 2, 3))
        imgs, _ = imgs.max(axis=1)
        if self.is_training:
            return [imgs, label[0]]
        else:
            return [imgs, label]

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



class NTU_oneShot(Dataset):
    def __init__(self, cfg, loader_type='train', joint_reduction=True, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        mmcv_cfg = Config.fromfile(cfg.DATASET.mmcv_config)

        self.loader_type = loader_type
        self.is_training = self.loader_type == 'train'
        if loader_type == 'train':
            self.ds = build_dataset(mmcv_cfg.data.train)
            self.action_set = mmcv.load(mmcv_cfg.data.train.dataset.ann_file)['split']['action_set']
        elif loader_type == 'val':
            self.ds = build_dataset(mmcv_cfg.data.val)
        else:
            self.ds = build_dataset(mmcv_cfg.data.exemplar)
        

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        imgs, label = itemgetter("imgs", "label")(self.ds.__getitem__(index))
        imgs = imgs.squeeze(0)
        if self.cfg.DATASET.Heatmap_Generator.joint_reduction:
            imgs = imgs.permute((1, 0, 2, 3))
            imgs, _ = imgs.max(axis=1)
        if self.is_training:
            pos_act = int(label) + 1
            pos_idx = np.random.choice(self.action_set[pos_act])

            keys = np.array(list(self.action_set.keys()))
            pos_mask = keys != pos_act
            neg_act = np.random.choice(keys[pos_mask])
            neg_idx = np.random.choice(self.action_set[neg_act])

            pos = itemgetter("imgs")(self.ds.__getitem__(pos_idx)).squeeze(0)
            neg = itemgetter("imgs")(self.ds.__getitem__(neg_idx)).squeeze(0)
            
            if self.cfg.DATASET.Heatmap_Generator.joint_reduction:
                pos = pos.permute((1, 0, 2, 3))
                pos, _ = pos.max(axis=1)
                neg = neg.permute((1, 0, 2, 3))
                neg, _ = neg.max(axis=1)

            return [imgs, pos, neg, label[0]]
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

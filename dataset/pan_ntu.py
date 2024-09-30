from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import torch
import os.path as osp
import numpy as np

np.set_printoptions(suppress=True, precision=10)
import json_tricks as json
import pickle
import logging
import os
import cv2
import copy
from tqdm import tqdm
import pandas as pd

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints
from dataset.panoptic import (
    TRAIN_LIST as pan_train,
    VALIDATION_LIST as pan_val,
    CAMERA_LIST as pan_cam,
    JOINTS_DEF as pan_joints,
    SKELETON as pan_skel,
    LEFT_LIMB as pan_llimb,
    RIGHT_LIMB as pan_rlimb,
)

from dataset.nturgbd import (
    X_SUB_TRAIN_LIST as ntu_train,
    X_SUB_VAL_LIST as ntu_val,
    JOINTS_DEF as ntu_joints,
    SKELETON as ntu_skel,
    LEFT_LIMB as ntu_llimb,
    RIGHT_LIMB as ntu_rlimb,
)
from dataset.kalman_filter import KeypointsKalmanFilter
from utils.heatmap_related import GeneratePoseTarget


class Pan_Ntu(JointsDataset):
    def __init__(self, cfg, is_training, **kwargs):
        super().__init__(cfg, **cfg.DATASET, is_training=is_training, **kwargs)
        self.joints_def = {"panoptic": pan_joints, "nturgbd": ntu_joints}
        self.joint_indices = {
            "panoptic": list(pan_joints.values()),
            "nturgbd": list(ntu_joints.values()),
        }

        self.heatmap_generator = {}
        self.heatmap_generator["panoptic"] = GeneratePoseTarget(
            **cfg.DATASET.Heatmap_Generator,
            skeletons=pan_skel,
            left_kp=pan_llimb,
            left_limb=pan_llimb,
            right_kp=pan_rlimb,
            right_limb=pan_rlimb
        )
        self.heatmap_generator["nturgbd"] = GeneratePoseTarget(
            **cfg.DATASET.Heatmap_Generator,
            skeletons=ntu_skel,
            left_kp=ntu_llimb,
            left_limb=ntu_llimb,
            right_kp=ntu_rlimb,
            right_limb=ntu_rlimb
        )

        self.cam_list = [(0, i) for i in pan_cam]
        self.num_views = len(self.cam_list)
        if self.image_set == "train":
            self.sequence_list = {"panoptic": pan_train, "nturgbd": ntu_train}

        elif self.image_set == "validation":
            self.sequence_list = {"panoptic": pan_val, "nturgbd": ntu_val}

        self.db_file = {
            "panoptic": os.path.join(
                self.dataset_root["panoptic"],
                "ts_group_{}_cam{}.pkl".format(self.image_set, self.num_views),
            ),
            "nturgbd": os.path.join(
                self.dataset_root["nturgbd"], "all_group_{}.pkl".format(self.image_set)
            ),
        }

        self.vf = {}
        self.db = {}
        self.meta = {}
        self.lengths = {}

        if osp.exists(self.db_file["panoptic"]) and osp.exists(self.db_file["nturgbd"]):
            # Panoptic db Loading
            info = pickle.load(open(self.db_file["panoptic"], "rb"))
            assert info["sequence_list"] == self.sequence_list["panoptic"]
            assert info["cam_list"] == self.cam_list
            self.vf["panoptic"] = info["valid_frames"]
            self.db["panoptic"] = info["data"]
            self.meta["panoptic"] = info["meta"]
            self.lengths["panoptic"] = len(self.vf["panoptic"])

            # NTU db Loading
            info = pickle.load(open(self.db_file["nturgbd"], "rb"))
            assert info["sequence_list"] == self.sequence_list["nturgbd"]
            self.vf["nturgbd"] = info["valid_frames"]
            self.db["nturgbd"] = info["data"]
            self.meta["nturgbd"] = info["meta"]
            self.lengths["nturgbd"] = len(self.vf["nturgbd"])

        else:
            raise Exception("Database has not been created properly, Missing files")

    def __len__(self):
        return (self.lengths["panoptic"] // self.stride) + self.lengths["nturgbd"]

    def __getitem__(self, index):

        if index < self.lengths["panoptic"] // self.stride:
            # center_idx = pan_joints["mid-hip"]
            dataset, stride = ("panoptic", self.stride)
        else:
            # center_idx = ntu_joints["spine-base"]
            dataset, stride = ("nturgbd", 1)
            index -= self.lengths["panoptic"] // self.stride

        idx, num_frames = self.vf[dataset][::stride][index]
        data = self.db[dataset][idx : idx + num_frames][:: self.frame_interval]
        data = np.nan_to_num(data, nan=1.0)

        # data = self._filter_data(data)
        # Center Skeletons
        # data = data - np.median(data[:, center_idx, :], axis=0)
        # data = data + (np.array(self.resolution) / 2)
        # top = self.resolution[1] * 0.9
        # data = data * (top / data[:,:,1].max())

        # Select random sequence of frames
        start_idx = 0
        if num_frames > self.window_size:
            start_idx = np.random.randint(
                0, high=num_frames - self.window_size, size=1
            )[0]
            data = data[start_idx : start_idx + self.window_size]

            if self.heatmap_generator[dataset] is not None:
                data = self.heatmap_generator[dataset](np.expand_dims(data, axis=0))

        elif num_frames < self.window_size:

            if self.heatmap_generator[dataset] is not None:
                data = self.heatmap_generator[dataset](np.expand_dims(data, axis=0))

            pad_size = ((0, self.window_size - num_frames), (0, 0), (0, 0))
            data = np.pad(data, pad_size, "constant")
        else:
            if self.heatmap_generator[dataset] is not None:
                data = self.heatmap_generator[dataset](np.expand_dims(data, axis=0))

        if self.masked_position_generator is not None:
            data = [data, self.masked_position_generator()]

        assert data.shape[1] == 256 and data.shape[2] == 256

        return data

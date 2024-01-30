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
    VAL_LIST as pan_val,
    CAMERA_LIST as pan_cam,
    JOINTS_DEF as pan_joints,
    SKELETON as pan_skel,
    LEFT_LIMB as pan_llimb,
    RIGHT_LIMB as pan_rlimb,
)

from dataset.nturgbd import (
    TRAIN_LIST as ntu_train,
    VAL_LIST as ntu_val,
    JOINTS_DEF as ntu_joints,
    SKELETON as ntu_skel,
    LEFT_LIMB as ntu_llimb,
    RIGHT_LIMB as ntu_rlimb,
)
from dataset.kalman_filter import KeypointsKalmanFilter
from utils.heatmap_related import GeneratePoseTarget


class Pan_Ntu(JointsDataset):
    def __init__(self, cfg, image_set, **kwargs):
        super().__init__(cfg, **cfg.DATASET, image_set=image_set, **kwargs)
        self.joints_def = {"panoptic": pan_joints, "nturgbd": ntu_joints}
        self.joint_indices = {
            "panoptic": list(pan_joints.values()),
            "nturgbd": list(ntu_joints.values()),
        }

        # self.num_joints = len(JOINTS_DEF)
        # self.kf_filter = KeypointsKalmanFilter(n_keypoints=len(self.joint_indices)-1)
        self.panoptic_heatmap = GeneratePoseTarget(
            **cfg.DATASET.Heatmap_Generator,
            skeletons=pan_skel,
            left_kp=pan_llimb,
            left_limb=pan_llimb,
            right_kp=pan_rlimb,
            right_limb=pan_rlimb
        )
        self.nturgbd_heatmap = GeneratePoseTarget(
            **cfg.Heatmap_Generator,
            skeletons=ntu_skel,
            left_kp=ntu_llimb,
            left_limb=ntu_llimb,
            right_kp=ntu_rlimb,
            right_limb=ntu_rlimb
        )
        self.heatmap_generator = True

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
                self.dataset_root["nturgbd"], "ts_group_{}.pkl".format(self.image_set)
            ),
        }

        if osp.exists(self.db_file["panoptic"]) and osp.exists(self.db_file["nturgbd"]):
            # Panoptic db Loading
            info = pickle.load(open(self.db_file["panoptic"], "rb"))
            assert info["sequence_list"] == self.sequence_list["panoptic"]
            assert info["cam_list"] == self.cam_list
            self.vf = info["valid_frames"]
            self.db_pan = info["data"]
            self.meta = info["meta"]
            self.panoptic_len = len(self.vf) - 1

            # NTU db Loading
            info = pickle.load(open(self.db_file["nturgbd"], "rb"))
            assert info["sequence_list"] == self.sequence_list["nturgbd"]
            self.vf = np.concatenate((self.vf, info["valid_frames"]), axis=0)
            self.db_ntu = info["data"]
            self.meta = pd.concat((self.meta, info["meta"]))

        else:
            raise Exception("Database has not been created properly, Missing files")

        self.vf_size = len(self.vf)

    def __len__(self):
        return self.vf_size // self.stride

    def __getitem__(self, index):
        idx, num_frames = self.vf[:: self.stride][index]

        if index < self.panoptic_len:
            db = self.db_pan
            heatmap_generator = self.panoptic_heatmap
        else:
            heatmap_generator = self.nturgbd_heatmap
            db = self.db_ntu

        data = db[idx : idx + num_frames][:: self.frame_interval]

        data = np.nan_to_num(data, nan=1.0)

        # data = self._filter_data(data)

        # Select random sequence of frames
        start_idx = 0
        if num_frames > self.window_size:
            start_idx = np.random.randint(
                0, high=num_frames - self.window_size, size=1
            )[0]
        elif num_frames < self.window_size:
            pad_size = ((0, self.window_size - num_frames), (0, 0), (0, 0))
            data = np.pad(data, pad_size, "constant")

        data = data[start_idx : start_idx + self.window_size]

        if heatmap_generator is not None:
            data = heatmap_generator(np.expand_dims(data, axis=0))

        if self.masked_position_generator is not None:
            data = [data, self.masked_position_generator()]

        return data

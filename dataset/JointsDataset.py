# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import copy
import logging
from filterpy.kalman import KalmanFilter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):

    def __init__(self, cfg, image_set, is_train):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.norm = [1920, 1080]

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.root)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.rootIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.test_dataset
        self.window_size = cfg.DATASET.window_size
        self.stride = cfg.DATASET.stride

        self.mask_chance = cfg.DATASET.mask_chance
        self.mix_chance = cfg.DATASET.mix_chance
        self.token_window_size = cfg.DATASET.token_window_size
        self.add_cls = cfg.DATASET.add_cls

        self.num_views = cfg.DATASET.camera_num
        self.joints_weight = 1
        self.db = []
        self.cls_token = False

    def normalize(self, data):
        pose_center = data[:, 2].mean(0, keepdims=True)
        data = data - pose_center
        data = data / self.norm.to(data.device)
        data = data + torch.Tensor([0.5, 0.5], device=data.device)
        return data

    def unnormalize(self, data):
        data = data * self.norm.to(data.device)
        return data

    def normalize_pose(self, pose_data):
        """
        Normalize keypoint values to the range of [-1, 1]
        :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
        :param vid_res:
        :param symm_range:
        :return:
        """
        vid_res = np.array(self.norm)
        symm_range = True
        pose_data = np.nan_to_num(pose_data)
        pose_data_normalized = pose_data / vid_res
        pose_data_centered = pose_data_normalized
        if symm_range:  # Means shift data to [-1, 1] range
            pose_data_centered = 2 * pose_data_centered - 1

        pose_data_zero_mean = pose_data_centered
        mean = pose_data_centered.mean(axis=(1,), keepdims=True)
        std = pose_data_centered.std(axis=(1,), keepdims=True)
        pose_data_zero_mean = (pose_data_centered - mean) / std

        return pose_data_zero_mean, (mean, std)
    
    def unnormalize_pose(self, pose_data_zero_mean, features):
        """
        unNormalize keypoint values from the range of [-1, 1]
        :param vid_res:
        :param symm_range:
        :return:
        Add for incase of zero padding
        """
        if pose_data_zero_mean.shape[1] > self.window_size:
            pose_data_zero_mean = pose_data_zero_mean[:, 1:]

        if isinstance(features, dict):
            mean, std = features['mean'], features['std']
        else:
            mean, std = features
        device = pose_data_zero_mean.device

        mean, std = mean.to(device), std.to(device)
        vid_res = torch.Tensor(self.norm).to(device)
        symm_range = True

        pose_data_centered = (pose_data_zero_mean * std) + mean

        if symm_range:  # Means shift data to [-1, 1] range
            pose_data_normalized = (pose_data_centered + 1) / 2

        pose_data = pose_data_normalized * vid_res

        return pose_data

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        pass

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import copy
import logging
import math
from filterpy.kalman import KalmanFilter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, image_set, is_train, heatmap_generator=None):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.norm = [1920, 1080]
        self.heatmap_size = (256, 256)
        self.heatmap_generator = heatmap_generator

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, "../..", cfg.DATASET.root)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.rootIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.test_dataset
        self.frame_interval = cfg.DATASET.frame_interval
        self.window_size = cfg.DATASET.window_size
        self.total_window = self.window_size * self.frame_interval
        self.stride = cfg.DATASET.stride
        self.token_window_size = cfg.DATASET.token_window_size
        self.num_views = cfg.DATASET.camera_num
        self.db = []

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

        pose_data_normalized = pose_data / vid_res
        pose_data_centered = pose_data_normalized
        if symm_range:  # Means shift data to [-1, 1] range
            pose_data_centered = 2 * pose_data_centered - 1

        pose_data_zero_mean = pose_data_centered
        mean = pose_data_centered.mean(axis=0, keepdims=True)
        std = pose_data_centered.std(axis=0, keepdims=True)
        std[std == 0] = 1

        pose_data_zero_mean = (pose_data_centered - mean) / std

        pose_data_zero_mean = torch.from_numpy(pose_data_zero_mean).float()
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()

        return pose_data_zero_mean, (mean, std)

    def unnormalize_pose(self, pose_data_zero_mean, features):
        """
        unNormalize keypoint values from the range of [-1, 1]
        :param vid_res:
        :param symm_range:
        :return:
        Add for incase of zero padding
        """
        if pose_data_zero_mean.shape[1] > (self.window_size // self.token_window_size):
            pose_data_zero_mean = pose_data_zero_mean[:, 1:]

        if isinstance(features, dict):
            mean, std = features["mean"], features["std"]
        else:
            mean, std = features

        if len(pose_data_zero_mean.shape) == len(mean.shape) + 1:
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)

        device = pose_data_zero_mean.device

        mean, std = mean.to(device), std.to(device)
        vid_res = torch.Tensor(self.norm).to(device)
        symm_range = True

        pose_data_centered = (pose_data_zero_mean * std) + mean

        if symm_range:  # Means shift data to [-1, 1] range
            pose_data_centered = (pose_data_centered + 1) / 2

        pose_data = pose_data_centered * vid_res

        return pose_data

    def _compute_velocity(self, keypoints, delta_t=1):
        # keypoints is an array of shape (window, num_keypoints, 2)
        velocities = np.zeros_like(keypoints)

        velocities[1:] = (keypoints[1:] - keypoints[:-1]) / delta_t

        return velocities

    def _filter_data(self, data):
        filtered_data = np.zeros_like(data)
        vel = self._compute_velocity(data)
        data_vel = np.concatenate((data, vel), axis=2)

        for i in range(data_vel.shape[0]):
            # Apply the Kalman Filter on the concatenated data and velocity
            filtered_data[i] = self.kf_filter.apply(data_vel[i])[:, :2]

        # First 10 unfiltered because idk whats wrong
        return np.concatenate((data[:10], filtered_data[10:]), axis=0)

    def _tokenize(self, data, mean=None, std=None):
        pass

    def __getitem__(self, index):
        idx, num_frames = self.vf[:: self.stride][index]
        data = self.db[idx : idx + num_frames][:: self.frame_interval]
        
        data = np.nan_to_num(data, nan=1.0)

        data = self._filter_data(data)

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

        if self.heatmap_generator is not None:
            data = self.heatmap_generator(np.expand_dims(data, axis=0))

        return data

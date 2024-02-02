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
from .masking_generator import MaskingGenerator
from typing import List

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(
        self,
        cfg,
        root="",
        stride=15,
        joint_req=0.9,
        camera_num=10,
        window_size=300,
        frame_interval=1,
        training_mode="",
        image_set="validation",
        resolution=[1920, 1080],
        is_training=True,
        **kwargs,
    ):
        this_dir = os.path.dirname(__file__)
        self.dataset_root = root
        if isinstance(root, str):
            dataset_root = os.path.join(this_dir, "../..", root)
            self.dataset_root = os.path.abspath(dataset_root)

        self.cfg = cfg
        self.stride = stride
        self.joint_req = joint_req
        self.image_set = image_set
        self.num_views = camera_num
        self.resolution = resolution
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.num_views = camera_num
        self.resolution = resolution
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.total_window = self.window_size * self.frame_interval
        self.db = []

        self.training_mode = None
        if training_mode.lower() in ["d_vae", "pert"]:
            self.training_mode = training_mode.lower()

        # Heatmap generator defined in child dataset class
        self.heatmap_generator = None

        self.masked_position_generator = None
        if self.training_mode == "pert":
            self.masked_position_generator = MaskingGenerator(
                cfg.PERT.window_size,
                num_masking_patches=cfg.PERT.num_mask_patches if is_training else 0,
                max_num_patches=cfg.PERT.max_mask_patches_per_block,
                min_num_patches=cfg.PERT.min_mask_patches_per_block,
            )

    def __getitem__(self, index):
        idx, num_frames = self.vf[:: self.stride][index]
        data = self.db[idx : idx + num_frames][:: self.frame_interval]

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

        if self.heatmap_generator is not None:
            data = self.heatmap_generator(np.expand_dims(data, axis=0))

        if self.masked_position_generator is not None:
            data = [data, self.masked_position_generator()]

        return data

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
import random

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
        resolution=[1920, 1080],
        drop_frames_rate = 0.3,
        max_num_frame_rate = 0.9,
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


        if is_training:
            self.image_set = cfg.DATASET.train_subset
        else:
            self.image_set = cfg.DATASET.test_subset

        self.drop_frames_rate = drop_frames_rate
        self.max_num_frame_rate = max_num_frame_rate

        self.num_views = camera_num
        self.resolution = resolution
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.num_views = camera_num
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.total_window = self.window_size * self.frame_interval
        self.db = []

        self.training_mode = None
        if training_mode.lower() in ["d_vae", "pert", "fine-tuning"]:
            self.training_mode = training_mode.lower()

        # Heatmap generator defined in child dataset class
        self.heatmap_generator = None

        self.masked_position_generator = None
        if self.training_mode == "pert":
            self.masked_position_generator = MaskingGenerator(
                cfg.PeIT.window_size,
                num_masking_patches=cfg.PeIT.num_mask_patches if is_training else 0,
                max_num_patches=cfg.PeIT.max_mask_patches_per_block,
                min_num_patches=cfg.PeIT.min_mask_patches_per_block,
            )

    def __getitem__(self, index):
        idx, num_frames = self.vf[:: self.stride][index]
        data = self.db[idx : idx + num_frames][:: self.frame_interval]

        data = np.nan_to_num(data, nan=1.0)

        end_idx = num_frames
        if random.random() <= self.drop_frames_rate and self.training_mode:
            # Let's drop some frame
            cur_rate = random.uniform(0, self.max_num_frame_rate)

            end_idx = round((1 - cur_rate) * self.window_size)

        start_idx = 0
        if num_frames > self.window_size:
            start_idx = np.random.randint(
                0, high=num_frames - self.window_size, size=1
            )[0]
        # elif num_frames < self.window_size:
        #    pad_size = ((0, self.window_size - num_frames), (0, 0), (0, 0))
        #    data = np.pad(data, pad_size, "constant")

        data = data[start_idx : start_idx + end_idx]

        if self.heatmap_generator is not None:
            data = self.heatmap_generator(np.expand_dims(data, axis=0))

        diff = self.window_size - len(data)

        if diff > 0:
            pad_size = ((0, diff), (0, 0), (0, 0))
            data = np.pad(data, pad_size, "constant")

        if self.masked_position_generator is not None:
            data = [data, self.masked_position_generator()]

        return data

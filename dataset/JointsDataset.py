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
        self.frame_interval = cfg.DATASET.frame_interval
        self.window_size = cfg.DATASET.window_size
        self.total_window = self.window_size * self.frame_interval
        self.stride = cfg.DATASET.stride
        self.stage = cfg.train_stage

        self.mask_chance = cfg.DATASET.mask_chance
        self.mix_chance = 0.0 if self.stage == 0 else cfg.DATASET.mix_chance
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
            mean, std = features['mean'], features['std']
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

    def _create_mask(self, total_window):
        # Create the mask
        rd_idx = total_window//(self.stage%2+1)
        masked_amount = int(rd_idx * self.mask_chance)
        mask = torch.cat((
                        torch.ones(masked_amount, dtype=torch.bool), 
                        torch.zeros(rd_idx - masked_amount, dtype=torch.bool)
                        ))
        mask = mask[torch.randperm(rd_idx)]
        padding = torch.zeros(total_window - rd_idx, dtype=torch.bool)
        mask = torch.cat((mask, padding))
        
        if self.mix_chance:
            half = rd_idx
            mask = torch.cat((mask[:half], torch.tensor([0], dtype=torch.bool), mask[half:]))
        
        if self.add_cls:
            mask = torch.cat((torch.tensor([0], dtype=torch.bool), mask))

        return mask
    
    def _tokenize(self, data, mean=None, std=None):
        
        window, keypoints, channels = data.shape

        # Calculate the number of tokens in each window
        tokens = int(window / self.token_window_size)

        # Reshape input to process in tokens
        data = data.view(tokens, self.token_window_size, keypoints, channels)
        data = data.view(tokens, self.token_window_size * keypoints, channels)            

        return data
    
    def _class_token(self, data):
        rd_idx = int(data.shape[0]//2)
        if self.mix_chance:
            # half = int(data.shape[0] // 2)
            sep_token = torch.ones_like(data[0:1]) * 0.5
            data = torch.cat((data[:rd_idx], sep_token, data[rd_idx:]), dim=0).float()

        # Add class token to data
        if self.add_cls:
            cls_token = torch.ones_like(data[0:1]) * -1
            data = torch.cat((cls_token, data), dim=0).float()

        return data
    
    def _mix(self, data, type=0):
        length = data.shape[0]

        # Draw a random number to determine if this example should be mixed
        mixed = torch.rand(1).item() < self.mix_chance
        # mixed = True
        if mixed:
            if type == 0:
                half_window = int(self.window_size // 2)
                mixed_data = data
                mixed_data[half_window:] = self.__getrandom__()[:half_window]

                mixed = torch.eye(2)[1]
                return mixed_data, mixed
            
            if type == 1:
                # Compute random index for the example
                min_index = int(length//self.token_window_size * 0.1)
                max_index = int(length//self.token_window_size * 0.9)
                random_index = torch.randint(min_index, max_index, (1,)).item()

                mixed_data = data[:random_index*self.token_window_size]
                random_data = self.__getrandom__()[random_index*self.token_window_size:]
                mixed_data = np.concatenate((mixed_data, random_data), axis=0)

                mixed = torch.eye(2)[1]
                return mixed_data, mixed, random_index
        
        # If not mixed, just return the original data and zero for the shift index
        mixed = torch.eye(2)[0]
        return data, mixed
    
    def __getrandom__(self):
        index = int(torch.rand(1).item() * self.__len__())
        idx, num_frames = self.vf[::self.stride][index]
        data = self.db[idx:idx + self.total_window][::self.frame_interval]
        data = self._filter_data(data)

        return data
    
    def __getitem__(self, index): # Buggy NTU 3139Tr, 3150 Te
        idx, num_frames = self.vf[::self.stride][index]
        data = self.db[idx:idx + num_frames][::self.frame_interval]
        data = np.nan_to_num(data, nan=1.0)

        data = self._filter_data(data)
        data, mixed = self._mix(data)
        data, (mean, std) = self.normalize_pose(data)

        # Add zero padding
        prev_length = data.shape[0]
        padding = torch.zeros((self.window_size - prev_length, *data.shape[1:]))
        data = torch.cat((data, padding))

        data = self._tokenize(data)

        mask = self._create_mask(data.shape[0])
        data = self._class_token(data)
        
        gt = data[mask].clone()

        gt = gt.view(gt.shape[0], self.token_window_size, -1, gt.shape[-1])
        data[mask] = 1.0

        meta = self.meta.iloc[idx:idx + num_frames][::self.frame_interval]
        unq_videos = meta[meta.columns[meta.columns != 'frame']].drop_duplicates()

        if len(unq_videos) > 1:
            print(meta)
            raise Exception("Multiple videos in one segment")
        
        meta = self.meta.iloc[idx].to_dict()
        meta['mean'] = mean
        meta['std'] = std

        special_tokens = [bool(self.mix_chance), self.add_cls]
        additional_tokens = len(torch.tensor([0, 0])[special_tokens])
        padding = math.ceil(prev_length / self.token_window_size) + additional_tokens

        if self.stage == 2:
            mixed = int(unq_videos.values[0, 0][-3:])
            mixed = torch.eye(120)[mixed-1]
        
        return (data, gt, mixed, mask, meta, padding)

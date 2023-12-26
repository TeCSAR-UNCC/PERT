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
        self.heatmap_size = (100, 100)

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

    def generate_a_heatmap(self, arr, centers, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        EPS = 1e-3
        sigma = 0.6
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_heatmap(self, kps):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: kps * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: 1 * kps * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        num_kp = kps.shape[0]
        kps = np.expand_dims(kps, axis=0)
        arr = np.zeros([num_kp, 1080, 1920], dtype=np.float32)
        max_values = np.ones([1, num_kp])
        for i in range(num_kp):
            self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])
        
        combined_heatmaps = arr.max(axis=0)
        
        # Assuming kps is your keypoints array of shape (15, 2)
        min_x, min_y = np.min(kps[0], axis=0)
        max_x, max_y = np.max(kps[0], axis=0)

        # Add some padding around the keypoints to ensure no information is lost
        padding = 10  # You can adjust the padding as needed
        min_x = max(int(min_x) - padding, 0)
        min_y = max(int(min_y) - padding, 0)
        max_x = min(int(max_x) + padding, combined_heatmaps.shape[1])
        max_y = min(int(max_y) + padding, combined_heatmaps.shape[0])
        cropped_heatmap = combined_heatmaps[min_y:max_y, min_x:max_x]
        
        desired_height, desired_width = max(cropped_heatmap.shape), max(cropped_heatmap.shape)

        # Current dimensions of the cropped_heatmap
        current_height, current_width = cropped_heatmap.shape

        # Calculate padding
        delta_width = max(desired_width - current_width, 0)
        delta_height = max(desired_height - current_height, 0)

        padding_top = delta_height // 2
        padding_bottom = delta_height - padding_top
        padding_left = delta_width // 2
        padding_right = delta_width - padding_left

        # Add padding
        padded_heatmap = np.pad(cropped_heatmap, ((padding_top, padding_bottom), (padding_left, padding_right)), 'constant')

        resized_heatmap = cv2.resize(padded_heatmap, self.heatmap_size)
        
        return resized_heatmap

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
    
    def __getitem__(self, index): 
        idx, num_frames = self.vf[::self.stride][index]
        data = self.db[idx:idx + num_frames][::self.frame_interval]
        data = np.nan_to_num(data, nan=1.0)

        data = self._filter_data(data)

        # data, mixed = self._mix(data)
        # data, (mean, std) = self.normalize_pose(data)

        # Add zero padding
        # prev_length = data.shape[0]
        # padding = torch.zeros((self.window_size - prev_length, *data.shape[1:]))
        # data = torch.cat((data, padding))

        # data = self._tokenize(data)

        # mask = self._create_mask(data.shape[0])
        # data = self._class_token(data)
        
        # gt = data[mask].clone()

        # gt = gt.view(gt.shape[0], self.token_window_size, -1, gt.shape[-1])
        # data[mask] = 1.0

        # meta = self.meta.iloc[idx:idx + num_frames][::self.frame_interval]
        # unq_videos = meta[meta.columns[meta.columns != 'frame']].drop_duplicates()

        # if len(unq_videos) > 1:
        #     print(meta)
        #     raise Exception("Multiple videos in one segment")
        
        # meta = self.meta.iloc[idx].to_dict()
        # meta['mean'] = mean
        # meta['std'] = std

        # special_tokens = [bool(self.mix_chance), self.add_cls]
        # additional_tokens = len(torch.tensor([0, 0])[special_tokens])
        # padding = math.ceil(prev_length / self.token_window_size) + additional_tokens

        # if self.stage == 2:
        #     mixed = int(unq_videos.values[0, 0][-3:])
        #     mixed = torch.eye(120)[mixed-1]
        
        return data
        return (data, gt, mixed, mask, meta, padding)

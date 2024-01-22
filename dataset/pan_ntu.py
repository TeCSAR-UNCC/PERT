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
from dataset.panoptic import TRAIN_LIST as pan_train, VAL_LIST as pan_val, \
    CAMERA_LIST as pan_cam, JOINTS_DEF as pan_joints, LIMBS as pan_limbs

from dataset.nturgbd import TRAIN_LIST as ntu_train, VAL_LIST as ntu_val,\
    JOINTS_DEF as ntu_joints, KeypointsKalmanFilter

class Pan_Ntu(JointsDataset):
    def __init__(self, cfg, image_set, is_train, heatmap_generator=None, **kwargs):
        super().__init__(cfg, image_set, is_train, heatmap_generator=heatmap_generator)
        self.joints_def = {"panoptic": pan_joints, "nturgbd": ntu_joints}
        self.joint_indices = {"panoptic": list(pan_joints.values()), "nturgbd": list(ntu_joints.values())}

        self.joint_req = 0.5
        # self.num_joints = len(JOINTS_DEF)
        self.kf_filter = KeypointsKalmanFilter(n_keypoints=len(self.joint_indices)-1)
        
        self.cam_list = [(0, i) for i in pan_cam]
        self.num_views = len(self.cam_list)
        if self.image_set == 'train':
            self.sequence_list = {"panoptic": pan_train, "nturgbd": ntu_train}

        elif self.image_set == 'validation':
            self.sequence_list = {"panoptic": pan_val, "nturgbd": ntu_val}

        self.db_file = {"panoptic": os.path.join(self.dataset_root["panoptic"], 
                                                 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)),

                        "nturgbd": os.path.join(self.dataset_root["nturgbd"], 
                                                'group_{}.pkl'.format(self.image_set))}

        if osp.exists(self.db_file["panoptic"]) and osp.exists(self.db_file["nturgbd"]):
            # Panoptic db Loading
            info = pickle.load(open(self.db_file["panoptic"], 'rb'))
            assert info['sequence_list'] == self.sequence_list["panoptic"]
            assert info['cam_list'] == self.cam_list
            self.vf = info['valid_frames']
            self.db_pan = info['data']
            self.meta = info['meta']
            self.panoptic_len = len(self.vf)-1

            # NTU db Loading
            info = pickle.load(open(self.db_file["nturgbd"], 'rb'))
            assert info['sequence_list'] == self.sequence_list["nturgbd"]
            self.vf = np.concatenate((self.vf, info['valid_frames']), axis=0)
            self.db_ntu = info['data']
            self.meta = pd.concat((self.meta, info['meta']))

        else:
            raise Exception("Database has not been created properly, Missing files")
        
        self.vf_size = len(self.vf)

    def __len__(self):
        return self.vf_size // self.stride

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
        if index > self.panoptic_len:
            print("NTU FRAMES")
        idx, num_frames = self.vf[:: self.stride][index]
        db = self.db_pan if index < self.panoptic_len else self.db_ntu
        data = db[idx : idx + num_frames][:: self.frame_interval]
        
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
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

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

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160224_haggling1',
    '160226_haggling1',
    '170221_haggling_b1',
    '170221_haggling_b3',
    '170221_haggling_m1',
    '170221_haggling_m2',
    '170221_haggling_m3',
    '170224_haggling_a2',
    '170224_haggling_a3',
    '170224_haggling_b1',
    '170224_haggling_b2',
    '170224_haggling_b3',
    '170228_haggling_a1',
    '170228_haggling_a2',
    '170228_haggling_b1',
    '170228_haggling_b2',
    '170228_haggling_b3',
    '170404_haggling_a1',
    '170404_haggling_a3',
    '170404_haggling_b1',
    '170404_haggling_b2',
    '170404_haggling_b3',
    '170407_haggling_a2',
    '170407_haggling_a3',
    '170407_haggling_b1',
    '170407_haggling_b2',
    '160422_ultimatum1',
    '160906_band1',
    '160906_band2',
    '160906_ian2',
    '160906_ian3',
    '160906_ian5',
    '160906_pizza1',
    '161029_flute1',
    '161029_piano1',
    '161029_piano2',
    '161029_piano4',
    '170307_dance5',
    '170407_office2',
    '171026_cello3',
    '171026_pose1',
    '171026_pose3',
    '171204_pose1',
    '171204_pose2',
    '171204_pose3',
    '171204_pose4',
    '171204_pose6',    
]

VAL_LIST = [
    '170407_haggling_b3',
    '170407_haggling_a1',
    '170404_haggling_a2',
    '170228_haggling_a3',
    '170224_haggling_a1',
    '170221_haggling_b2',
    '160422_haggling1',
    '160906_band3',
    '160906_ian1',
    '161029_piano3',
    '170915_office1',
    '171026_pose2',
    '171204_pose5',
]

CAMERA_LIST = [1, 2, 4, 6, 7, 10, 13, 17, 19, 28]

JOINTS_DEF = {
    'neck': 0,
    # 'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    'l-ear': 16,
    # 'r-eye': 17,
    'r-ear': 18,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]


class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train):
        super().__init__(cfg, image_set, is_train)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.joint_indices = list(JOINTS_DEF.values())
        self.kf_filter = KeypointsKalmanFilter(n_keypoints=len(self.joint_indices)-1)
        self.joint_req = 0.9
        # self.num_joints = len(JOINTS_DEF)

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = [(0, i) for i in CAMERA_LIST]# range(0, self.num_views)]
            # self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)]
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            self.cam_list = [(0, i) for i in CAMERA_LIST]
            self.num_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.vf = info['valid_frames']
            self.db = info['data']
            self.meta = info['meta']
        else:
            self.vf, self.db, self.meta, _ = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'valid_frames': self.vf,
                'data': self.db,
                'meta': self.meta,
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.vf_size = len(self.vf)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in tqdm(self.sequence_list, desc="Sequences", position=0):

            cameras = self._get_cam(seq)

            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            prev_pose2d = {}
            for i, file in tqdm(enumerate(anno_files), desc=f"Files in {seq}", total=len(anno_files), position=1, leave=False):
                try:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                except:
                    print(file)
                if len(bodies) == 0:
                    continue
                
                for k, v in cameras.items():
                    postfix = osp.basename(file).replace('body3DScene', '')
                    prefix = '{:02d}_{:02d}'.format(k[0], k[1])

                    all_poses_3d = []
                    for body in bodies:
                        full_id = f"{prefix}{body['id']}"
                        
                        pose3d = np.array(body['joints19']).reshape((-1, 4))
                        pose3d = pose3d[self.joint_indices]

                        joints_vis = pose3d[:, -1] > 0.1

                        if not joints_vis[self.root_id]:
                            continue

                        # Coordinate transformation
                        M = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0],
                                    [0.0, 1.0, 0.0]])
                        pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                        pose2d = np.zeros((pose3d.shape[0], 2))
                        pose2d[:, :2] = projectPoints(
                            pose3d[:, 0:3].transpose(), v['K'], v['R'],
                            v['t'], v['distCoef']).transpose()[:, :2]

                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                    pose2d[:, 1] <= height - 1)
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0
                        vis_perc = np.sum(joints_vis)/len(joints_vis)

                        if vis_perc <= self.joint_req:
                            continue

                        head_kp = (pose2d[14] + pose2d[15])/2
                        pose2d = np.delete(pose2d, [14, 15], axis=0)
                        pose2d = np.insert(pose2d, 1, head_kp, axis=0)

                        if full_id not in prev_pose2d.keys():
                            prev_pose2d[full_id] = copy.deepcopy(pose2d)

                        glitch_check = np.linalg.norm(pose2d - prev_pose2d[full_id], axis=1) > 100
                        if glitch_check.any():
                            pose2d[glitch_check] = copy.deepcopy(prev_pose2d[full_id][glitch_check])
                        
                        prev_pose2d[full_id] = copy.deepcopy(pose2d)

                        if len(pose2d) > 0:

                            db.append({
                                'frame': postfix[1:-5],
                                'video': seq,
                                'joints_2d': pose2d,
                                'camera': prefix,
                                'id': body['id']
                            })
        db = sorted(db, key=lambda x: (x['video'], x['id'], int(x['camera'])))

        frames_from_end = -1

        # Display the unique combinations
        num_sep_videos = 1
        valid_frames = []

        video = {'camera':db[-1]['camera'],
                 'video': db[-1]['video'], 
                 'id':db[-1]['id'],
                }
        
        for i, datapoint in tqdm(enumerate(db[::-1]), desc="Formating", total=len(db)):

            new_frame = {'camera':datapoint['camera'],
                         'video': datapoint['video'], 
                         'id':datapoint['id'],
                        }
            
            frames_from_end += 1
            # Check if we are at a new video
            if video != new_frame: 
                video = new_frame
                num_sep_videos += 1
                frames_from_end = 0
                continue

            if frames_from_end < self.total_window-1:
                continue
                
            index = len(db)-i-1
            valid_frames.append(index)
            
        valid_frames = np.array(valid_frames[::-1])
        skel_array = np.array([i['joints_2d'] for i in db])
        meta_db = pd.DataFrame([{k: v for k, v in d.items() if k != 'joints_2d'} for d in db])
        unique_combinations = meta_db[['video', 'camera', 'id']].drop_duplicates()

        return valid_frames, skel_array, meta_db, unique_combinations

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def _compute_velocity(self, keypoints, delta_t=1):
    # keypoints is an array of shape (window, num_keypoints, 2)
        velocities = np.zeros_like(keypoints)
        
        velocities[1:] = (keypoints[1:] - keypoints[:-1]) / delta_t

        return velocities

    def _filter_data(self, data):
        filtered_data = np.zeros_like(data)
        vel = self._compute_velocity(data)
        data_vel = np.concatenate((data, vel), axis=2)

        for i in range(self.window_size):
            # Apply the Kalman Filter on the concatenated data and velocity
            filtered_data[i] = self.kf_filter.apply(data_vel[i])[:, :2]

        # First 10 unfiltered because idk whats wrong
        return np.concatenate((data[:10], filtered_data[10:]), axis=0)

    def _create_mask(self, window):
        # Create the mask
        masked_amount = int(window * self.mask_chance)
        mask = torch.cat((torch.ones(masked_amount, dtype=torch.bool), 
                        torch.zeros(window - masked_amount, dtype=torch.bool)))
        mask = mask[torch.randperm(window)]
        
        if self.add_cls:
            mask = torch.cat((torch.tensor([0], dtype=torch.bool), mask))

        return mask
    
    def _tokenize(self, data):
        
        window, keypoints, channels = data.shape

        # Calculate the number of tokens in each window
        tokens = int(window / self.token_window_size)

        # Reshape input to process in tokens
        data = data.view(tokens, self.token_window_size, keypoints, channels)
        data = data.view(tokens, self.token_window_size * keypoints, channels)

        return data
    
    def _class_token(self, data):
        # Add class token to data
        if self.add_cls:
            cls_token = torch.ones_like(data[0]).unsqueeze(0) * -1
            data = torch.cat((cls_token, data), dim=0).float()

        return data
    
    def _mix(self, data, type=1):
        length = data.shape[0]

        # Draw a random number to determine if this example should be mixed
        mixed = torch.rand(1).item() > self.mix_chance

        if mixed:
            if type == 0:
                # Compute random index for the example
                min_index = int(length * 0.1)
                max_index = int(length * 0.9)
                random_index = torch.randint(min_index, max_index, (1,)).item()

                # Apply rotation
                mixed_data = torch.cat((data[random_index:], data[:random_index]), dim=0)
                mixed = torch.eye(2)[1]

                return mixed_data, mixed
            elif type == 1:

                # Reshape to make pairs of elements adjacent: (15, 2, 15, 2)
                data = data.view(15, 2, 225, 2)
                data[:, 0, :, :], data[:, 1, :, :] = data[:, 1, :, :].clone(), data[:, 0, :, :].clone()

                # Reshape back to the original shape
                mixed_data = data.view(30, 225, 2)
                mixed = torch.eye(2)[1]

                return mixed_data, mixed
        
        # If not mixed, just return the original data and zero for the shift index
        mixed = torch.eye(2)[0]
        return data, mixed

    def __getitem__(self, idx):
        idx = self.vf[::self.stride][idx]
        data = self.db[idx:idx + self.total_window][::self.frame_interval]
        data = self._filter_data(data)
        data, (mean, std) = self.normalize_pose(data)
        data = torch.from_numpy(data).float()

        data = self._tokenize(data)
        mixed = torch.eye(2)[1]
        if self.mix_chance > 0.0:
            data, mixed = self._mix(data)

        mask = self._create_mask(data.shape[0])
        data = self._class_token(data)
        
        gt = data[mask].clone()
        gt = gt.view(-1, int(gt.shape[1] // self.token_window_size), gt.shape[2])
        data[mask] = 1.0

        meta = self.meta.iloc[idx:idx + self.total_window][::self.frame_interval]
        unq_videos = meta[['video', 'camera', 'id']].drop_duplicates()

        if len(unq_videos) > 1:
            print(meta)
            raise Exception("Multiple videos in one segment")
        
        meta = self.meta.iloc[idx].to_dict()
        meta['mean'] = mean
        meta['std'] = std

        padding = int(self.window_size // self.token_window_size) + 1
        
        return (data, gt, mixed, mask, meta, padding)

    def __len__(self):
        return self.vf_size // self.stride

class KeypointsKalmanFilter:
    def __init__(self, n_keypoints, dt=1):
        self.n_keypoints = n_keypoints
        self.filters = [self._create_kalman_filter(dt) for _ in range(n_keypoints)]

    @staticmethod
    def _create_kalman_filter(dt):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-2
        kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-1
        return kf

    def apply(self, keypoints):
        # Initialize filtered keypoints and velocities with zeros
        filtered_data = np.zeros_like(keypoints)
        
        for i, kf in enumerate(self.filters):
            prediction = kf.predict()
            measurement = np.array(keypoints[i, :2], dtype=np.float32).reshape(2, 1)
            corrected = kf.correct(measurement)
            # Fill in both the position and velocity parts of the output
            filtered_data[i, :2] = corrected[:2].ravel()
            filtered_data[i, 2:] = corrected[2:].ravel()
        
        return filtered_data
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
from dataset.kalman_filter import KeypointsKalmanFilter
from utils.heatmap_related import GeneratePoseTarget

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    "160224_haggling1",
    "160226_haggling1",
    "170221_haggling_b1",
    "170221_haggling_b3",
    "170221_haggling_m1",
    "170221_haggling_m2",
    "170221_haggling_m3",
    "170224_haggling_a2",
    "170224_haggling_a3",
    "170224_haggling_b1",
    "170224_haggling_b2",
    "170224_haggling_b3",
    "170228_haggling_a1",
    "170228_haggling_a2",
    "170228_haggling_b1",
    "170228_haggling_b2",
    "170228_haggling_b3",
    "170404_haggling_a1",
    "170404_haggling_a3",
    "170404_haggling_b1",
    "170404_haggling_b2",
    "170404_haggling_b3",
    "170407_haggling_a2",
    "170407_haggling_a3",
    "170407_haggling_b1",
    "170407_haggling_b2",
    "160422_ultimatum1",
    "160906_band1",
    "160906_band2",
    "160906_ian2",
    "160906_ian3",
    "160906_ian5",
    "160906_pizza1",
    "161029_flute1",
    "161029_piano1",
    "161029_piano2",
    "161029_piano4",
    "170307_dance5",
    "170407_office2",
    "171026_cello3",
    "171026_pose1",
    "171026_pose3",
    "171204_pose1",
    "171204_pose2",
    "171204_pose3",
    "171204_pose4",
    "171204_pose6",
]

VALIDATION_LIST = [
    "170407_haggling_b3",
    "170407_haggling_a1",
    "170404_haggling_a2",
    "170228_haggling_a3",
    "170224_haggling_a1",
    "170221_haggling_b2",
    "160422_haggling1",
    "160906_band3",
    "160906_ian1",
    "161029_piano3",
    "170915_office1",
    "171026_pose2",
    "171204_pose5",
]

CAMERA_LIST = [1, 2, 4, 6, 7, 10, 13, 17, 19, 28]

JOINTS_DEF = {
    "neck": 0,
    "nose": 1,
    "mid-hip": 2,
    "l-shoulder": 3,
    "l-elbow": 4,
    "l-wrist": 5,
    "l-hip": 6,
    "l-knee": 7,
    "l-ankle": 8,
    "r-shoulder": 9,
    "r-elbow": 10,
    "r-wrist": 11,
    "r-hip": 12,
    "r-knee": 13,
    "r-ankle": 14,
    "l-eye": 15,
    "l-ear": 16,
    "r-eye": 17,
    "r-ear": 18,
}

JOINTS_PAIRS = [
    ("l-eye", "l-ear"),
    ("r-eye", "r-ear"),
    ("neck", "nose"),
    ("neck", "l-shoulder"),
    ("l-shoulder", "l-elbow"),
    ("l-elbow", "l-wrist"),
    ("neck", "r-shoulder"),
    ("r-shoulder", "r-elbow"),
    ("r-elbow", "r-wrist"),
    ("neck", "mid-hip"),
    ("mid-hip", "l-hip"),
    ("l-hip", "l-knee"),
    ("l-knee", "l-ankle"),
    ("mid-hip", "r-hip"),
    ("r-hip", "r-knee"),
    ("r-knee", "r-ankle"),
]


SKELETON = [(JOINTS_DEF[joint1], JOINTS_DEF[joint2]) for joint1, joint2 in JOINTS_PAIRS]

LEFT_LIMB = (3, 4, 5, 6, 7, 8)
RIGHT_LIMB = (9, 10, 11, 12, 13, 14)


class Panoptic(JointsDataset):
    def __init__(self, cfg, is_train, **kwargs):
        super().__init__(cfg, **cfg.DATASET, is_train=is_train,  **kwargs)
        self.joints_def = JOINTS_DEF
        self.joint_indices = list(JOINTS_DEF.values())
        self.heatmap_generator = GeneratePoseTarget(
            **cfg.DATASET.Heatmap_Generator,
            skeletons=SKELETON,
            left_kp=LEFT_LIMB,
            left_limb=LEFT_LIMB,
            right_kp=RIGHT_LIMB,
            right_limb=RIGHT_LIMB,
        )
        # self.kf_filter = KeypointsKalmanFilter(n_keypoints=len(self.joint_indices) - 1)

        self.sequence_list = eval(self.image_set.upper()+"_LIST")
        self._interval = 3
        self.cam_list = [(0, i) for i in CAMERA_LIST]
        self.num_views = len(self.cam_list)

        self.db_file = "v3_group_{}_cam{}.pkl".format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, "rb"))
            assert info["sequence_list"] == self.sequence_list
            assert info["interval"] == self._interval
            assert info["cam_list"] == self.cam_list
            self.vf = info["valid_frames"]
            self.db = info["data"]
            # self.hms = info['heatmap']
            self.meta = info["meta"]
        else:
            self.vf, self.db, self.meta, _ = self._get_db()
            info = {
                "sequence_list": self.sequence_list,
                "interval": self._interval,
                "cam_list": self.cam_list,
                "valid_frames": self.vf,
                "data": self.db,
                #'heatmap': self.hms,
                "meta": self.meta,
            }
            pickle.dump(info, open(self.db_file, "wb"))
        self.vf_size = len(self.vf)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in tqdm(self.sequence_list, desc="Sequences", position=0):
            cameras = self._get_cam(seq)

            curr_body_anno = osp.join(self.dataset_root, seq, "hdPose3d_stage1_coco19")
            anno_body_files = sorted(glob.iglob("{:s}/*.json".format(curr_body_anno)))

            curr_hand_anno = osp.join(self.dataset_root, seq, "hdHand3d")
            anno_hand_files = sorted(glob.iglob("{:s}/*.json".format(curr_hand_anno)))

            prev_pose2d = {}
            for i, (b_file, h_file) in tqdm(
                enumerate(zip(anno_body_files, anno_hand_files)),
                desc=f"Files in {seq}",
                total=len(anno_body_files),
                position=1,
                leave=False,
            ):
                try:
                    with open(b_file) as dfile:
                        bodies = json.load(dfile)["bodies"]
                    with open(h_file) as dfile:
                        hands = json.load(dfile)["people"]

                        
                except:
                    print(b_file, h_file)
                    continue

                if len(bodies) == 0:
                    continue

                for k, v in cameras.items():
                    postfix = osp.basename(b_file).replace("body3DScene", "")
                    prefix = "{:02d}_{:02d}".format(k[0], k[1])

                    all_poses_3d = []
                    for body, hand in zip(bodies, hands):
                        full_id = f"{prefix}{body['id']}"

                        pose3d = np.array(body["joints19"]).reshape((-1, 4))
                        pose3d = pose3d[self.joint_indices]

                        joints_vis = pose3d[:, -1] > 0.1

                        # Coordinate transformation
                        M = np.array(
                            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
                        )
                        pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                        pose2d = np.zeros((pose3d.shape[0], 2))
                        pose2d[:, :2] = projectPoints(
                            pose3d[:, 0:3].transpose(),
                            v["K"],
                            v["R"],
                            v["t"],
                            v["distCoef"],
                        ).transpose()[:, :2]

                        x_check = np.bitwise_and(
                            pose2d[:, 0] >= 0, pose2d[:, 0] <= width - 1
                        )
                        y_check = np.bitwise_and(
                            pose2d[:, 1] >= 0, pose2d[:, 1] <= height - 1
                        )
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0
                        vis_perc = np.sum(joints_vis) / len(joints_vis)

                        if vis_perc <= self.joint_req:
                            continue

                        # head_kp = (pose2d[14] + pose2d[15]) / 2
                        # pose2d = np.delete(pose2d, [14, 15], axis=0)
                        # pose2d = np.insert(pose2d, 1, head_kp, axis=0)

                        # if full_id not in prev_pose2d.keys():
                        #     prev_pose2d[full_id] = copy.deepcopy(pose2d)

                        # glitch_check = np.linalg.norm(pose2d - prev_pose2d[full_id], axis=1) > 100
                        # if glitch_check.any():
                        #     pose2d[glitch_check] = copy.deepcopy(prev_pose2d[full_id][glitch_check])

                        # prev_pose2d[full_id] = copy.deepcopy(pose2d)

                        if len(pose2d) > 0:
                            db.append(
                                {
                                    "frame": postfix[1:-5],
                                    "video": seq,
                                    "joints_2d": pose2d,
                                    "camera": prefix,
                                    "id": body["id"],
                                }
                            )
        db = sorted(db, key=lambda x: (x["video"], x["id"], int(x["camera"])))

        frames_from_end = -1

        # Display the unique combinations
        num_sep_videos = 1
        valid_frames = []

        video = {
            "camera": db[-1]["camera"],
            "video": db[-1]["video"],
            "id": db[-1]["id"],
        }

        for i, datapoint in tqdm(enumerate(db[::-1]), desc="Formating", total=len(db)):
            new_frame = {
                "camera": datapoint["camera"],
                "video": datapoint["video"],
                "id": datapoint["id"],
            }

            frames_from_end += 1
            # Check if we are at a new video
            if video != new_frame:
                video = new_frame
                num_sep_videos += 1
                frames_from_end = 0

            if frames_from_end < self.total_window - 1:
                continue

            index = len(db) - i - 1
            valid_frames.append([index, self.total_window])

        valid_frames = np.array(valid_frames[::-1])
        skel_array = np.array([i["joints_2d"] for i in db])
        meta_db = pd.DataFrame(
            [{k: v for k, v in d.items() if k != "joints_2d"} for d in db]
        )
        unique_combinations = meta_db[["video", "camera", "id"]].drop_duplicates()

        return valid_frames, skel_array, meta_db, unique_combinations

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, "calibration_{:s}.json".format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib["cameras"]:
            if (cam["panel"], cam["node"]) in self.cam_list:
                sel_cam = {}
                sel_cam["K"] = np.array(cam["K"])
                sel_cam["distCoef"] = np.array(cam["distCoef"])
                sel_cam["R"] = np.array(cam["R"]).dot(M)
                sel_cam["t"] = np.array(cam["t"]).reshape((3, 1))
                cameras[(cam["panel"], cam["node"])] = sel_cam
        return cameras

    def __len__(self):
        return self.vf_size // self.stride

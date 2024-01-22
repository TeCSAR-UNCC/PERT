import glob
import torch
import cv2
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy
from tqdm import tqdm
import pandas as pd

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    1,
    2,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    25,
    27,
    28,
    31,
    34,
    35,
    38,
    45,
    46,
    47,
    49,
    50,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    70,
    74,
    78,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    89,
    91,
    92,
    93,
    94,
    95,
    97,
    98,
    100,
    103,
]

VAL_LIST = [i for i in range(1, 121) if i not in TRAIN_LIST]


JOINTS_DEF = {
    "neck": 2,
    "head": 3,
    "mid-hip": 0,
    "l-shoulder": 4,
    "l-elbow": 5,
    "l-wrist": 6,
    "l-hip": 12,
    "l-knee": 13,
    "l-ankle": 14,
    "r-shoulder": 8,
    "r-elbow": 9,
    "r-wrist": 10,
    "r-hip": 16,
    "r-knee": 17,
    "r-ankle": 18,
}


class Nturgbd(JointsDataset):
    def __init__(self, cfg, image_set, is_train, heatmap_generator):
        super().__init__(cfg, image_set, is_train, heatmap_generator=heatmap_generator)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.joint_indices = list(JOINTS_DEF.values())
        self.joint_req = 0.5
        # self.num_joints = len(JOINTS_DEF)
        self.kf_filter = KeypointsKalmanFilter(n_keypoints=len(self.joint_indices) - 1)

        if self.image_set == "train":
            self.sequence_list = TRAIN_LIST

        elif self.image_set == "validation":
            self.sequence_list = VAL_LIST

        self.db_file = "group_{}.pkl".format(self.image_set)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, "rb"))
            assert info["sequence_list"] == self.sequence_list
            self.vf = info["valid_frames"]
            self.db = info["data"]
            self.meta = info["meta"]
        else:
            self.vf, self.db, self.meta, _ = self._get_db()
            info = {
                "sequence_list": self.sequence_list,
                "valid_frames": self.vf,
                "data": self.db,
                "meta": self.meta,
            }
            pickle.dump(info, open(self.db_file, "wb"))
        self.vf_size = len(self.vf)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []

        txt_files = os.listdir(os.path.join(self.dataset_root, "raw_txt"))

        for tfile in tqdm(txt_files, desc="NTU Text Files", position=0):
            subject = int(tfile[1:4])

            if subject not in self.sequence_list:
                continue

            tfile_path = os.path.join(self.dataset_root, "raw_txt", tfile)
            mat = self.read_skeleton(tfile_path, save_depthxy=False, save_skelxyz=False)

            if len(mat["nbodys"]) == 0:
                continue

            all_poses_3d = []
            frame = -1
            for pose2d in mat["rgb_body0"]:
                pose2d = pose2d[self.joint_indices]

                x_check = np.bitwise_and(pose2d[:, 0] >= 0, pose2d[:, 0] <= width - 1)
                y_check = np.bitwise_and(pose2d[:, 1] >= 0, pose2d[:, 1] <= height - 1)

                joints_vis = np.bitwise_and(x_check, y_check)

                vis_perc = np.sum(joints_vis) / len(joints_vis)

                if vis_perc <= self.joint_req:
                    continue

                if np.sum(pose2d) == 0.0:
                    continue

                frame += 1
                if len(pose2d) > 0:
                    video = tfile.split(".")[0]
                    db.append(
                        {"frame": frame, "video": video, "joints_2d": pose2d, "id": 0}
                    )

        frames_from_end = -1

        # Display the unique combinations
        num_sep_videos = 1
        valid_frames = []

        video = {
            "video": db[-1]["video"],
            "id": db[-1]["id"],
        }

        for i, datapoint in tqdm(enumerate(db[::-1]), desc="Formating", total=len(db)):
            new_frame = {
                "video": datapoint["video"],
                "id": datapoint["id"],
            }

            frames_from_end += 1
            # Check if we are at a new video
            if video != new_frame:
                video = new_frame
                num_sep_videos += 1

                if frames_from_end < self.total_window - 1:
                    index = len(db) - i
                    valid_frames.append([int(index), int(frames_from_end)])

                frames_from_end = 0

            if frames_from_end < self.total_window - 1:
                continue

            index = len(db) - i - 1
            valid_frames.append([int(index), int(self.total_window)])

        valid_frames = np.array(valid_frames[::-1])
        skel_array = np.array([i["joints_2d"] for i in db])
        meta_db = pd.DataFrame(
            [{k: v for k, v in d.items() if k != "joints_2d"} for d in db]
        )
        unique_combinations = meta_db[["video", "id"]].drop_duplicates()

        return valid_frames, skel_array, meta_db, unique_combinations

    def read_skeleton(
        self, file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True
    ):
        with open(file_path, "r") as f:
            datas = f.readlines()
        max_body = 4
        njoints = 25

        # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
        # abundant bodys.
        # read all lines into the pool to speed up, less io operation.
        nframe = int(datas[0][:-1])
        bodymat = {"file_name": file_path[-29:-9], "nbodys": [], "njoints": njoints}

        for body in range(max_body):
            if save_skelxyz:
                bodymat[f"skel_body{body}"] = np.zeros(shape=(nframe, njoints, 3))
            if save_rgbxy:
                bodymat[f"rgb_body{body}"] = np.zeros(shape=(nframe, njoints, 2))
            if save_depthxy:
                bodymat[f"depth_body{body}"] = np.zeros(shape=(nframe, njoints, 2))

        # above prepare the data holder
        cursor = 0
        for frame in range(nframe):
            cursor += 1
            bodycount = int(datas[cursor][:-1])
            if bodycount == 0:
                continue
            # skip the empty frame
            bodymat["nbodys"].append(bodycount)
            for body in range(bodycount):
                cursor += 1
                skel_body = f"skel_body{body}"
                rgb_body = f"rgb_body{body}"
                depth_body = f"depth_body{body}"

                bodyinfo = datas[cursor][:-1].split(" ")
                cursor += 1

                njoints = int(datas[cursor][:-1])
                for joint in range(njoints):
                    cursor += 1
                    jointinfo = datas[cursor][:-1].split(" ")
                    jointinfo = np.array(list(map(float, jointinfo)))
                    if save_skelxyz:
                        bodymat[skel_body][frame, joint] = jointinfo[:3]
                    if save_depthxy:
                        bodymat[depth_body][frame, joint] = jointinfo[3:5]
                    if save_rgbxy:
                        bodymat[rgb_body][frame, joint] = jointinfo[5:7]
        # prune the abundant bodys
        for each in range(max_body):
            if len(bodymat["nbodys"]):
                if each >= max(bodymat["nbodys"]):
                    if save_skelxyz:
                        del bodymat[f"skel_body{each}"]
                    if save_rgbxy:
                        del bodymat[f"rgb_body{each}"]
                    if save_depthxy:
                        del bodymat[f"depth_body{each}"]

        return bodymat

    def __len__(self):
        return self.vf_size // self.stride

    # def __getitem__(self, idx):
    #     idx, num_frames = self.vf[idx]

    #     data = torch.from_numpy(self.db[idx:idx + num_frames])

    #     data = torch.nan_to_num(data, nan=0.0)
    #     data = self.normalize(data)

    #     # Add zero padding
    #     data = torch.cat((data, torch.zeros((self.window_size - num_frames, *data.shape[1:]))))

    #     # Create the mask
    #     masked_amount = int(data.shape[0] * self.mask_chance)
    #     mask = torch.cat((torch.ones(masked_amount, dtype=torch.bool),
    #                     torch.zeros(data.shape[0] - masked_amount, dtype=torch.bool)))
    #     mask = mask[torch.randperm(data.shape[0])]
    #     mask = torch.cat((torch.tensor([0], dtype=torch.bool), mask))

    #     # Get indices of True values directly using PyTorch
    #     indices = torch.nonzero(mask).squeeze().tolist()

    #     # Add class token to data
    #     cls_token = torch.ones_like(data[0]).unsqueeze(0) * -1
    #     data = torch.cat((cls_token, data), dim=0)
    #     data = data.view(data.shape[0], -1).float()

    #     # Create a copy of data at specified indices for gt
    #     gt = data[indices].clone()

    #     # Mask the data tensor
    #     data[mask] = 1.0

    #     meta = self.meta.iloc[idx:idx + num_frames]
    #     unq_videos = meta[['video', 'id']].drop_duplicates()

    #     if len(unq_videos) > 1:
    #         print(meta)
    #         raise Exception("Multiple videos in one segment")

    #     cls = int(unq_videos.values[0, 0][-3:])

    #     return (data, gt, mask, cls, num_frames+1)


class KeypointsKalmanFilter:
    def __init__(self, n_keypoints, dt=1):
        self.n_keypoints = n_keypoints
        self.filters = [self._create_kalman_filter(dt) for _ in range(n_keypoints)]

    @staticmethod
    def _create_kalman_filter(dt):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov = (
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            * 1e-2
        )
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

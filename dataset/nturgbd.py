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
from dataset.kalman_filter import KeypointsKalmanFilter
from utils.heatmap_related import GeneratePoseTarget

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
    "spine-base": 0,
    "spine-mid": 1,
    "neck": 2,
    "head": 3,
    "l-shoulder": 4,
    "l-elbow": 5,
    "l-wrist": 6,
    "l-hand": 7,
    "r-shoulder": 8,
    "r-elbow": 9,
    "r-wrist": 10,
    "r-hand": 11,
    "l-hip": 12,
    "l-knee": 13,
    "l-ankle": 14,
    "l-foot": 15,
    "r-hip": 16,
    "r-knee": 17,
    "r-ankle": 18,
    "r-foot": 19,
    "spine-shoulder": 20,
    "l-handtip": 21,
    "l-thumb": 22,
    "r-handtip": 23,
    "r-thumb": 24,
}

JOINTS_PAIRS = [
    ("head", "neck"),
    ("neck", "spine-shoulder"),
    ("spine-shoulder", "spine-mid"),
    ("spine-mid", "spine-base"),
    ("spine-shoulder", "l-shoulder"),
    ("l-shoulder", "l-elbow"),
    ("l-elbow", "l-wrist"),
    ("l-wrist", "l-hand"),
    ("l-hand", "l-handtip"),
    ("l-wrist", "l-thumb"),
    ("spine-shoulder", "r-shoulder"),
    ("r-shoulder", "r-elbow"),
    ("r-elbow", "r-wrist"),
    ("r-wrist", "r-hand"),
    ("r-hand", "r-handtip"),
    ("r-wrist", "r-thumb"),
    ("spine-base", "l-hip"),
    ("l-hip", "l-knee"),
    ("l-knee", "l-ankle"),
    ("l-ankle", "l-foot"),
    ("spine-base", "r-hip"),
    ("r-hip", "r-knee"),
    ("r-knee", "r-ankle"),
    ("r-ankle", "r-foot"),
]

SKELETON = [(JOINTS_DEF[joint1], JOINTS_DEF[joint2]) for joint1, joint2 in JOINTS_PAIRS]

LEFT_LIMB = (20, 4, 5, 6, 7, 0, 12, 13, 14, 15)
RIGHT_LIMB = (20, 8, 9, 10, 11, 0, 16, 17, 18, 19)


class Nturgbd(JointsDataset):
    def __init__(self, cfg, image_set, **kwargs):
        super().__init__(cfg, **cfg.DATASET, image_set=image_set, **kwargs)
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

        if self.image_set == "train":
            self.sequence_list = TRAIN_LIST

        elif self.image_set == "validation":
            self.sequence_list = VAL_LIST

        self.db_file = "ts_group_{}.pkl".format(self.image_set)
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


class Action_Nturgbd(Nturgbd):
    def __init__(self, cfg, image_set, **kwargs):
        super().__init__(cfg, **cfg.DATASET, image_set=image_set, **kwargs)

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

        meta = self.meta.iloc[idx : idx + num_frames]
        unq_videos = meta[["video", "id"]].drop_duplicates()
        cls = int(unq_videos.values[0, 0][-3:])

        if self.heatmap_generator is not None:
            data = self.heatmap_generator(np.expand_dims(data, axis=0))

        if self.masked_position_generator is not None:
            data = [data, self.masked_position_generator(), cls]

        return data

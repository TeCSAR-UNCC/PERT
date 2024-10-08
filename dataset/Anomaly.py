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
from PIL import Image

import json
import math
import os
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import normalize_pose
from utils.pose_utils import gen_clip_seg_data_np, get_ab_labels
from torch.utils.data import DataLoader
import cv2
from tsmoothie.smoother import *
from utils.heatmap_related import CV2BasedLimbGenerated, GeneratePoseTarget
from pyskl.datasets.pipelines.augmentations import PoseCompact, Resize
from pyskl.datasets.pipelines.formatting import FormatShape

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]


logger = logging.getLogger(__name__)


# COCO17_JOINTS_DEF = {
#     "nose": 0,
#     "l-eye": 1,
#     "r-eye": 2,
#     "l-ear": 3,
#     "r-ear": 4,
#     "l-shoulder": 5,
#     "r-shoulder": 6,
#     "l-elbow": 7,
#     "r-elbow": 8,
#     "l-wrist": 9,
#     "r-wrist": 10,
#     "l-hip": 11,
#     "r-hip": 12,
#     "l-knee": 13,
#     "r-knee": 14,
#     "l-ankle": 15,
#     "r-ankle": 16,
# }

# COCO17_JOINTS_PAIRS = [
#     ("nose", "l-eye"),
#     ("nose", "r-eye"),
#     ("l-eye", "l-ear"),
#     ("r-eye", "r-ear"),
#     ("l-shoulder", "r-shoulder"),
#     ("l-shoulder", "l-elbow"),
#     ("l-elbow", "l-wrist"),
#     ("r-shoulder", "r-elbow"),
#     ("r-elbow", "r-wrist"),
#     ("l-hip", "r-hip"),
#     ("l-shoulder", "l-hip"),
#     ("r-shoulder", "r-hip"),
#     ("l-hip", "l-knee"),
#     ("l-knee", "l-ankle"),
#     ("r-hip", "r-knee"),
#     ("r-knee", "r-ankle"),
# ]

SKELETON = [
    [0, 5],
    [0, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [11, 13],
    [13, 15],
    [6, 12],
    [12, 14],
    [14, 16],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [11, 12],
]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
# Create the skeleton by mapping joint names to their indices
# SKELETON = [(COCO17_JOINTS_DEF[joint1], COCO17_JOINTS_DEF[joint2]) for joint1, joint2 in COCO17_JOINTS_PAIRS]


class AnomalyDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    def __init__(self, 
                 path_to_json_dir, 
                 cfg, 
                 is_training=True,
                 path_to_vid_dir=None, 
                 normalize_pose_segs=True, 
                 return_indices=False,
                 return_metadata=False, 
                 debug=False, 
                 return_global=True, 
                 evaluate=False, 
                 abnormal_train_path=None,
                 **dataset_args):
        super().__init__()
        self.cfg = cfg
        # self.heatmap_generator = CV2BasedLimbGenerated(
        #     eval(cfg.DATASET.resolution),
        #     (
        #         cfg.DATASET.Heatmap_Generator.heatmap_size,
        #         cfg.DATASET.Heatmap_Generator.heatmap_size,
        #     ),
        #     limb_pairs=SKELETON,
        #     min_down_scaling=0.65,
        #     max_up_scaling=0.8,
        #     is_training=False,
        # )
        self.compact = PoseCompact ( hw_ratio=1.0, allow_imgpad=True)
        self.resize = Resize (scale=(72, 72), keep_ratio=False)
        self.heatmap_generator = GeneratePoseTarget(
            with_kp=False,
            with_limb=True,
            skeletons=SKELETON,
            # double=True,
            # left_kp=left_kp,
            # right_kp=right_kp,
            # left_limb=left_limb,
            # right_limb=right_limb,
            heatmap_size=72
        )
       

    
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.patches_db = None
        self.use_patches = False
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = dataset_args.get('headless', False)
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.debug = debug
        num_clips = None
        # num_clips = 1
        # num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.return_global = return_global
        self.transform_list = dataset_args.get('trans_list', None)
        self.is_training = is_training
    
        
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        self.segs_data_np, self.segs_meta, self.person_keys, self.global_data_np, \
        self.global_data, self.segs_score_np = \
            gen_dataset(path_to_json_dir, num_clips=num_clips, ret_keys=True,
                        ret_global_data=return_global, **dataset_args)
        self.segs_meta = np.array(self.segs_meta)
        if abnormal_train_path is not None:
            self.segs_data_np_ab, self.segs_meta_ab, self.person_keys_ab, self.global_data_np_ab, \
            self.global_data_ab, self.segs_score_np_ab = \
                gen_dataset(abnormal_train_path, num_clips=num_clips, ret_keys=True,
                            ret_global_data=return_global, **dataset_args)
            self.segs_meta_ab = np.array(self.segs_meta_ab)
            ab_labels = get_ab_labels(self.segs_data_np_ab, self.segs_meta_ab, path_to_vid_dir, abnormal_train_path)
            num_normal_samp = self.segs_data_np.shape[0]
            num_abnormal_samp = (ab_labels == -1).sum()
            total_num_normal_samp = num_normal_samp + (ab_labels == 1).sum()
            print("Num of abnormal sapmles: {}  | Num of normal samples: {}  |  Precent: {}".format(
                num_abnormal_samp, total_num_normal_samp, num_abnormal_samp / total_num_normal_samp))
            self.labels = np.concatenate((np.ones(num_normal_samp), ab_labels),
                                         axis=0).astype(int)
            self.segs_data_np = np.concatenate((self.segs_data_np, self.segs_data_np_ab), axis=0)
            self.segs_meta = np.concatenate((self.segs_meta, self.segs_meta_ab), axis=0)
            self.global_data_np = np.concatenate((self.global_data_np, self.global_data_np_ab), axis=0)
            self.segs_score_np = np.concatenate(
                (self.segs_score_np, self.segs_score_np_ab), axis=0)
            self.global_data += self.global_data_ab
            self.person_keys.update(self.person_keys_ab)
        else:
            self.labels = np.ones(self.segs_data_np.shape[0])
        # Convert person keys to ints
        self.person_keys = {k: [int(float(i)) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

   
    
    def visualize_segment(self, tensor, output_video_path):
        tensor = tensor[:2, :, :]
        num_frames = tensor.shape[1]
        frame_size = (856, 480)  # Adjust the frame size as needed
        fps = 24.0  # Frames per second
        limbs = [(1, 18), (18, 6), (18, 7), (6, 8),
         (7, 9), (8, 10), (9, 11), (10, 12), (18, 3),
         (3, 4), (4, 5), (3, 13), (13, 14), (14, 15), (6, 12), (7, 13)]


        # Define the codec for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed

        # Create a VideoWriter object to write the video
        out = cv2.VideoWriter(output_video_path+".mp4", fourcc, fps, frame_size)
        tensor = np.transpose(tensor, (1, 2, 0))
        for frame in range(num_frames):
            # keypoints = tensor[:, frame, :]
            
            # keypoints = keypoints.reshape(15,2)
            keypoints = tensor[frame]
            # Create an image for visualization
            image = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # White background

            # Plot keypoints
            for x, y in keypoints:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circles for keypoints

            # Plot skeletal limbs
            for limb in limbs:
                start_idx, end_idx = limb
                start_point = keypoints[start_idx - 1]
                end_point = keypoints[end_idx - 1]
                cv2.line(image, (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)  # Green lines for limbs

            # Write the frame to the video
            out.write(image)

        # Release the VideoWriter
        out.release()
    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations
    

           
        
        # if self.normalize_pose_segs:    
        #     data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
        #                                       **self.args).squeeze(axis=0).transpose(2, 0, 1)
            
            
            
        ret_arr = [data_transformed, trans_index]

        ret_arr += [self.segs_score_np[sample_index]]
        ret_arr += [self.labels[sample_index]]
   
        
        
        # Generate heatmaps
        if self.heatmap_generator is not None:
            results = {}
            results["img_shape"] = eval(self.cfg.DATASET.resolution)
            kp = ret_arr[0]
            kp = kp[0:2, :, :]
            kp = kp.transpose(1, 2, 0)
            results["keypoint"] = np.expand_dims(kp, axis=0)
            results = self.compact(results)
            results["modality"] = "pose"
            results = self.resize(results)
            results["imgs"], results["keypoint"] = self.heatmap_generator(results["keypoint"])
            
            
            # kp = ret_arr[0]
            # kp = kp[0:2, :, :]
            # kp = kp.transpose(1, 2, 0)
            # heatmap, kpts = self.heatmap_generator(kp, eval(self.cfg.DATASET.resolution))
            
        
        # if self.is_training:
        #     shuffled, order = self.permute_spatial_temporal_cube(results["imgs"], self.cfg.DATASET.puzzle_patch_size)
        #     ret_arr += [shuffled]
        #     ret_arr += [order]
        # else:
        ret_arr += [results["imgs"]]
        return ret_arr

    def get_all_data(self, normalize_pose_segs=False):
        # if normalize_pose_segs:
        #     segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
        #         (0, 3, 1, 2))
        segs_data_np = self.segs_data_np
        if self.num_transform == 1 or self.eval:
            return list(segs_data_np)
        return segs_data_np
    

    def permute_spatial_temporal_cube(self, cube, patch_size):
        """
        Divides a (48, 72, 72) spatial-temporal cube into 9 pieces of shape (48, 24, 24),
        applies a random permutation, and returns a new cube with the same shape (48, 72, 72)
        but with permuted patches.

        Args:
        - cube: numpy array of shape (48, 72, 72) representing the spatial-temporal data

        Returns:
        - permuted_cube: numpy array of shape (48, 72, 72) with permuted patches
        - permutation: the order of the applied permutation (list of integers)
        """
        # Check that input dimensions are as expected
        if cube.shape != (48, 72, 72):
            raise ValueError("Input cube must have dimensions (48, 72, 72)")

        # Step 1: Divide the cube into 9 (48, 24, 24) pieces
        pieces = []
        for i in range(0, 72, patch_size):
            for j in range(0, 72, patch_size):
                patch = cube[:, i:i+patch_size, j:j+patch_size]
                pieces.append(patch)

        # Step 2: Apply a random permutation
        num_pieces = len(pieces)  # Should be 9 for a 3x3 grid of patches
        permutation = np.random.permutation(num_pieces)  # Randomly permute the indices

        # Step 3: Create a new cube with the permuted patches
        permuted_cube = np.zeros_like(cube)  # Create an empty tensor with the same shape as the input
        n_patches = cube.shape[1]//patch_size
        for idx, perm_idx in enumerate(permutation):
            # Find the location where this patch should go in the final cube
            y = idx // n_patches
            x = idx % n_patches
            y_offset = y * patch_size
            x_offset = x * patch_size
            # Place the permuted patch in the correct location in the permuted_cube
            permuted_cube[:, y_offset:y_offset + patch_size, x_offset:x_offset + patch_size] = pieces[perm_idx]

        return permuted_cube, permutation




    def __len__(self):
        return self.num_transform * self.num_samples


def get_dataset_and_loader(args, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'headless': False,'seg_len': args.DATASET.window_size, 'return_indices': True, 'return_metadata': True,
                    'train_seg_conf_th': 0.0, 'specific_clip': args.DATASET.specific_clip, 'vid_res': args.DATASET.resolution}
    dataset, loader = dict(), dict()
    splits = ['train', 'test'] if not only_test else ['test']
    for split in splits:
        evaluate = split == 'test'
        normalize_pose_segs = False
        # dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None
        dataset_args['trans_list'] = None
        dataset_args['seg_stride'] = args.DATASET.stride if split == 'train' else 1  # No strides for test set
        dataset_args['dataset'] = args.DATASET.train_dataset if split == 'train' else args.DATASET.test_dataset 
        dataset_args['puzzle'] = True if split == 'train' else False  # No puzzle needed for inference
        dataset_args['vid_path'] = args.vid_path[split]
        dataset[split] = AnomalyDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                        is_training=dataset_args['puzzle'],
                                        normalize_pose_segs=normalize_pose_segs,
                                        cfg=args,
                                        evaluate=evaluate,
                                        **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader


def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False


def gen_dataset(person_json_root, num_clips=None, kp18_format=True, ret_keys=False, ret_global_data=True,
                **dataset_args):
    segs_data_np = []
    segs_score_np = []
    segs_meta = []
    global_data = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'ShanghaiTech')

    dir_list = os.listdir(person_json_root)
    if dataset=='c1' or dataset=='c2' or dataset=='c3' or dataset=='c4' or dataset=='combined' or dataset=='corridor' or dataset=='Avenue' or dataset=='CPCC' or dataset=='CPCC0' or dataset=='CPCC1' or dataset=='CPCC2' or dataset=='CPCC3' or dataset=='CPCC4' or dataset=='CPCC5' or dataset=='CPCC6':
        json_list = sorted([fn for fn in dir_list])
    else:
        json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
    if num_clips is not None:
        json_list = [json_list[num_clips]]  # For debugging purposes
    for person_dict_fn in tqdm(json_list):
        if dataset == "UBnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        else:
            if dataset == 'corridor' or dataset == 'Avenue' or dataset=='CPCC' or dataset=='CPCC0' or dataset=='CPCC1' or dataset=='CPCC2' or dataset=='CPCC3' or dataset=='CPCC4' or dataset=='CPCC5' or dataset=='CPCC6':
                scene_id, clip_id = (person_dict_fn.split('.')[0]).split('_')[:2]
            else:
                scene_id, clip_id = person_dict_fn.split('_')[:2]
            if shanghaitech_hr_skip(dataset=="ShanghaiTech-HR", scene_id, clip_id):
                continue
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = gen_clip_seg_data_np(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset=dataset)

        _, _, _, global_data_np, global_data, _ = gen_clip_seg_data_np(clip_dict, start_ofst, 1, 1, scene_id=scene_id,
                                                                       clip_id=clip_id,
                                                                       ret_keys=ret_keys,
                                                                       global_pose_data=global_data,
                                                                       dataset=dataset)
        segs_data_np.append(clip_segs_data_np)
        segs_score_np.append(score_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

    # Global data
    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)


    # if kp18_format and segs_data_np.shape[-2] == 17:
    #     segs_data_np = keypoints17_to_coco18(segs_data_np)
    #     global_data_np = keypoints17_to_coco18(global_data_np)
    #     global_data = [keypoints17_to_coco18(data) for data in global_data]
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
        global_data_np = global_data_np[:, :, 5:]
        global_data = [data[:, 5:, :] for data in global_data]

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = \
            seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)
    if ret_global_data:
        if ret_keys:
            return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np
        else:
            return segs_data_np, segs_meta, global_data_np, global_data, segs_score_np
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_score_np
    else:
        return segs_data_np, segs_meta, segs_score_np


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int_)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np
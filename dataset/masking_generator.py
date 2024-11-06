"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""

# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np
from numpy.typing import NDArray
from typing import Optional


def generate_mask_from_points(points, H, W, S, min_masked_number, max_masked_number):
    """
    Generate a mask based on randomly selected points, where each point's location
    dictates the center of an SxS square to be masked.

    Parameters:
    - points: np.array of shape (F, N, 2) containing [x, y] points for F frames.
    - H: Height of the original heatmap.
    - W: Width of the original heatmap.
    - S: Size of each square side.
    - min_masked_number: Minimum number of points to be randomly selected for masking.
    - max_masked_number: Maximum number of points to be randomly selected for masking.

    Returns:
    - final_mask: A single mask of shape (total_rows_squares, total_cols_squares) with 1s where squares are masked.
    """
    # Calculate total rows and columns based on S
    total_rows_squares = (H // S) + (1 if H % S else 0)
    total_cols_squares = (W // S) + (1 if W % S else 0)

    # Initialize the final mask
    final_mask = np.zeros((total_rows_squares, total_cols_squares), dtype=np.int32)

    # Flatten all points from all frames
    all_points = points.reshape(
        -1, 2
    )  # Flatten to Nx2, where N is total number of points across frames

    # Adjust min and max masked numbers based on the total number of available points
    total_points = all_points.shape[0]
    # Adjust min and max masked numbers based on the total number of squares within the bbox
    if max_masked_number < 1:
        max_masked_number = round(max_masked_number * total_points)
    if min_masked_number < 1:
        min_masked_number = round(min_masked_number * total_points)

    adjusted_min_masked_number = max(1, min(min_masked_number, total_points))
    adjusted_max_masked_number = min(max_masked_number, total_points)

    # Determine the actual number of points to mask
    num_to_mask = np.random.randint(
        adjusted_min_masked_number, adjusted_max_masked_number + 1
    )

    # Randomly select points
    selected_indices = np.random.choice(
        range(total_points), size=num_to_mask, replace=False
    )
    selected_points = all_points[selected_indices]

    # Mask the squares based on the selected points
    for x, y in selected_points:
        col = int(x // S)
        row = int(y // S)
        if 0 <= row < total_rows_squares and 0 <= col < total_cols_squares:
            final_mask[row, col] = 1

    return final_mask


def find_bounding_box_3d(heatmap):
    frame_min, row_min, col_min = np.inf, np.inf, np.inf
    frame_max, row_max, col_max = -np.inf, -np.inf, -np.inf

    for f in range(heatmap.shape[0]):
        frame = heatmap[f]
        rows = np.any(frame > 0, axis=1)
        cols = np.any(frame > 0, axis=0)

        if np.any(rows) and np.any(cols):
            frame_min = min(frame_min, f)
            frame_max = max(frame_max, f)
            row_min_temp, row_max_temp = np.where(rows)[0][[0, -1]]
            col_min_temp, col_max_temp = np.where(cols)[0][[0, -1]]
            row_min = min(row_min, row_min_temp)
            row_max = max(row_max, row_max_temp)
            col_min = min(col_min, col_min_temp)
            col_max = max(col_max, col_max_temp)

    if frame_min == np.inf:
        return None

    return frame_min, row_min, col_min, frame_max, row_max, col_max


def select_squares_normally(num_rows_squares, num_cols_squares, num_to_mask, sigma=8):
    # Select an initial square randomly
    initial_square_idx = np.random.randint(0, num_rows_squares * num_cols_squares)

    # Calculate its row and column index
    initial_row, initial_col = divmod(initial_square_idx, num_cols_squares)

    # Initialize a list to store selected indices
    selected_indices = set([initial_square_idx])

    # Normal distribution parameters
    std_dev = (
        max(num_rows_squares, num_cols_squares) / sigma
    )  # Adjust the standard deviation as needed

    num_try = 0
    while len(selected_indices) < num_to_mask and num_try < 100:
        # Generate a row and column offset based on a normal distribution
        row_offset = int(np.random.normal(0, std_dev))
        col_offset = int(np.random.normal(0, std_dev))

        # Calculate new row and column indices
        new_row = np.clip(initial_row + row_offset, 0, num_rows_squares - 1)
        new_col = np.clip(initial_col + col_offset, 0, num_cols_squares - 1)

        # Convert back to a single index and add to the selection
        new_idx = new_row * num_cols_squares + new_col
        selected_indices.add(new_idx)
        num_try += 1

    return list(selected_indices)


def mask_random_squares(heatmap, S, min_masked_number, max_masked_number):
    bbox = find_bounding_box_3d(heatmap)
    if bbox is None:
        return None  # No bounding box found

    _, row_min, col_min, _, row_max, col_max = bbox

    # Dimensions of the entire heatmap in terms of SxS squares
    total_rows_squares = (heatmap.shape[1] // S) + (1 if heatmap.shape[1] % S else 0)
    total_cols_squares = (heatmap.shape[2] // S) + (1 if heatmap.shape[2] % S else 0)

    # Compute the number of SxS squares within the bounding box
    num_rows_squares = ((row_max - row_min) // S) + 1
    num_cols_squares = ((col_max - col_min) // S) + 1

    # Adjust min and max masked numbers based on the total number of squares within the bbox
    total_squares_bbox = num_rows_squares * num_cols_squares
    if max_masked_number < 1:
        max_masked_number = round(max_masked_number * total_cols_squares)
    if min_masked_number < 1:
        min_masked_number = round(min_masked_number * total_cols_squares)

    max_masked_number = min(max_masked_number, total_squares_bbox)
    min_masked_number = min(max(1, min_masked_number), max_masked_number)

    num_to_mask = np.random.randint(min_masked_number, max_masked_number + 1)

    # Create a mask for the bounding box area
    mask_bbox = np.zeros((num_rows_squares, num_cols_squares), dtype=np.int32)

    # Select squares to mask using a normal distribution around a randomly chosen square
    indices_to_mask = select_squares_normally(
        num_rows_squares, num_cols_squares, num_to_mask, sigma=S
    )

    for index in indices_to_mask:
        row = index // num_cols_squares
        col = index % num_cols_squares
        mask_bbox[row, col] = 1

    # Create the final mask with the shape of the entire heatmap's SxS grid
    final_mask = np.zeros((total_rows_squares, total_cols_squares), dtype=np.int32)

    # Calculate the starting indices for the bbox in the final mask
    start_row = row_min // S
    start_col = col_min // S

    # Place the mask_bbox within the final mask at the correct position
    final_mask[
        start_row : start_row + num_rows_squares,
        start_col : start_col + num_cols_squares,
    ] = mask_bbox

    return final_mask


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches,
        min_masked_number=4,
        max_masked_number=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        # if not isinstance(input_size, tuple):
        #    input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        if num_masking_patches < 1:
            num_masking_patches = round(num_masking_patches * self.num_patches)
        if max_masked_number is not None and max_masked_number < 1:
            max_masked_number = round(max_masked_number * self.num_patches)
        if min_masked_number < 1:
            min_masked_number = round(min_masked_number * self.num_patches)

        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_masked_number
        self.max_num_patches = (
            num_masking_patches if max_masked_number is None else max_masked_number
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(
        self,
        heatmap=None,
        keypoints: Optional[NDArray] = None,
        normal_blocking_chance=0.7,
        is_training=True,
    ):
        if not is_training:
            mask = np.zeros(self.get_shape(), dtype=np.int32)
            return mask
        chance = random.random()
        if chance <= normal_blocking_chance or heatmap is None:
            mask = np.zeros(self.get_shape(), dtype=np.int32)
            mask_count = 0
            while mask_count < self.num_masking_patches:
                max_mask_patches = self.num_masking_patches - mask_count
                max_mask_patches = min(max_mask_patches, self.max_num_patches)

                delta = self._mask(mask, max_mask_patches)
                if delta == 0:
                    break
                else:
                    mask_count += delta
            return mask

        S = heatmap.shape[-1] // self.height
        # mask = mask_random_squares(
        #    heatmap, S, self.num_masking_patches, self.max_num_patches
        # )
        mask = generate_mask_from_points(
            keypoints,
            heatmap.shape[-2],
            heatmap.shape[-1],
            S,
            self.num_masking_patches,
            self.max_num_patches,
        )

        return mask

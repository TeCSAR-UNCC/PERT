# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import imgaug as ia
from multiprocessing.pool import Pool
from functools import partial
from utils.axu import scale_and_center, scale, scale_center_per_frame, resize_keypoints
from imgaug.augmentables import Keypoint
import random
import cv2

EPS = 1e-3


class CV2BasedLimbGenerated:
    def __init__(
        self,
        original_shape,
        heatmap_shape,
        limb_pairs,
        min_down_scaling,
        max_up_scaling,
        is_training=False,
    ) -> None:
        self.original_shape = original_shape
        self.heatmap_shape = heatmap_shape
        self.limb_pairs = limb_pairs
        self.min_down_scaling = min_down_scaling
        self.max_up_scaling = max_up_scaling
        self.is_training = is_training

    def __call__(self, keypoints, resolution=None):
        """
        Generate limb heatmaps from keypoints, scaling keypoints from their original resolution to the output heatmap resolution.

        :param keypoints: Array of shape (F, K, 2) containing keypoint coordinates for each frame.
        :param original_shape: Tuple (H_prime, W_prime) defining the size of the original coordinate space.
        :param output_shape: Tuple (H, W) defining the size of the output heatmap.
        :param limb_pairs: List of tuples defining pairs of keypoints that form limbs.
        :return: Array of shape (F, H, W) containing limb heatmaps for each frame.
        """

        kps = [
            [Keypoint(x=x, y=y) for (x, y) in kps_per_image]
            for kps_per_image in keypoints
        ]
        if self.is_training:
            scaling = random.uniform(self.min_down_scaling, self.max_up_scaling)
        else:
            scaling = 0.65
        aug_kpt = resize_keypoints(keypoints, resolution)
        aug_kpt = scale_and_center(kps, scaling * self.heatmap_shape[0])

        val_x = self.heatmap_shape[0] // 4
        val_y = self.heatmap_shape[0] // 4
        x_offset = random.randrange(-val_x, val_x)
        y_offset = random.randrange(-val_y, val_y)

        keypoints_scales = np.zeros_like(keypoints)
        for i in range(0, len(aug_kpt)):
            kp_on_image = aug_kpt[i]
            for j in range(0, len(kp_on_image)):
                kps = kp_on_image[j]
                if self.is_training:
                    keypoints_scales[i, j, :] = [kps.x + x_offset, kps.y + y_offset]
                else:
                    keypoints_scales[i, j, :] = [kps.x, kps.y]

        F, K, _ = keypoints_scales.shape
        H, W = self.heatmap_shape
        heatmaps = np.zeros((F, H, W), dtype=np.float32)

        for f in range(F):
            frame_heatmap = np.zeros((H, W), dtype=np.float32)
            for start_idx, end_idx in self.limb_pairs:
                # Scale keypoints from original resolution to heatmap resolution
                start_point = tuple(
                    np.round(keypoints_scales[f, start_idx]).astype(np.int32)
                )
                end_point = tuple(
                    np.round(keypoints_scales[f, end_idx]).astype(np.int32)
                )

                # Draw line for the limb
                cv2.line(frame_heatmap, start_point, end_point, color=1, thickness=2)

            # Optionally apply Gaussian blur to smooth the heatmap
            frame_heatmap = cv2.GaussianBlur(
                frame_heatmap, (0, 0), sigmaX=1.25, sigmaY=1.25
            )

            # Normalize the heatmap to [0, 1]
            frame_heatmap = cv2.normalize(
                frame_heatmap,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

            heatmaps[f] = frame_heatmap

        return heatmaps, keypoints_scales


def flatten_generate_a_heatmap(sigma, arr, centers, max_values):
    """Generate pseudo heatmap for one keypoint in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
        max_values (np.ndarray): The max values of each keypoint. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    img_h, img_w = arr.shape

    for center, max_value in zip(centers, max_values):
        if max_value < EPS:
            continue

        mu_x, mu_y = round(center[0]), round(center[1])
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

        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma**2)
        patch = patch * max_value
        arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)


def fallten_generate_a_limb_heatmap(sigma, arr, starts, ends, start_values, end_values):
    """Generate pseudo heatmap for one limb in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
        ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
        start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
        end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    img_h, img_w = arr.shape
    for start, end, start_value, end_value in zip(
        starts, ends, start_values, end_values
    ):
        value_coeff = min(start_value, end_value)
        if value_coeff < EPS:
            continue

        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)

        x = np.arange(min_x, max_x, 1, np.float32)
        y = np.arange(min_y, max_y, 1, np.float32)

        if not (len(x) and len(y)):
            continue

        y = y[:, None]
        x_0 = np.zeros_like(x)
        y_0 = np.zeros_like(y)

        # distance to start keypoints
        d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2

        # distance to end keypoints
        d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2

        # the distance between start and end keypoints.
        d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2

        if d2_ab < 1:
            flatten_generate_a_heatmap(sigma, arr, start[None], start_value[None])
            continue

        coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab

        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate

        position = np.stack([x + y_0, y + x_0], axis=-1)
        projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

        patch = np.exp(-d2_seg / 2.0 / sigma**2)
        patch = patch * value_coeff

        arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)


def flatten_generate_heatmap(data, sigma, num_c, img_h, img_w):
    kps, max_values = data

    arr = np.zeros([num_c, img_h, img_w], dtype=np.float32)

    skeletons = (
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 4),
        (4, 5),
        (0, 9),
        (9, 10),
        (10, 11),
        (2, 6),
        (2, 12),
        (6, 7),
        (7, 8),
        (12, 13),
        (13, 14),
    )

    for i, limb in enumerate(skeletons):
        start_idx, end_idx = limb
        starts = kps[:, start_idx]
        ends = kps[:, end_idx]

        start_values = max_values[:, start_idx]
        end_values = max_values[:, end_idx]
        fallten_generate_a_limb_heatmap(
            sigma, arr[i], starts, ends, start_values, end_values
        )

    return arr


# --------------------------------------------------------------------------------------
class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right limbs of skeletons we defined for COCO-17p.
    """

    def __init__(
        self,
        sigma=1.0,
        use_score=False,
        use_gaussian_score=False,
        mean_gaussian_score=0.85,
        scale_gaussian_score=0.16,
        with_kp=True,
        with_limb=False,
        skeletons=(
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (0, 9),
            (9, 10),
            (10, 11),
            (2, 6),
            (2, 12),
            (6, 7),
            (7, 8),
            (12, 13),
            (13, 14),
        ),
        double=False,
        left_kp=(3, 4, 5, 6, 7, 8),
        right_kp=(9, 10, 11, 12, 13, 14),
        # Not sure what to do with the limbs
        left_limb=(3, 4, 5, 6, 7, 8),
        right_limb=(9, 10, 11, 12, 13, 14),
        heatmap_size=256,
        img_dims=(1080, 1920),
        scaling=1.0,
        is_training=False,
        collapse_joints=False,
    ):
        self.collapse_joints = collapse_joints
        self.sigma = sigma
        self.use_score = use_score
        self.use_gaussian_score = use_gaussian_score
        self.mean_gaussian_score = mean_gaussian_score
        self.scale_gaussian_score = scale_gaussian_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.heatmap_size = heatmap_size
        self.img_dims = img_dims

        assert (
            self.with_kp + self.with_limb == 1
        ), 'One of "with_limb" and "with_kp" should be set as True.'
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling
        self.is_training = is_training

    def generate_a_heatmap(self, arr, centers, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = round(center[0]), round(center[1])
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

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr, starts, ends, start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape
        for start, end, start_value, end_value in zip(
            starts, ends, start_values, end_values
        ):
            value_coeff = min(start_value, end_value)
            if value_coeff < EPS:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2

            # distance to end keypoints
            d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2

            # the distance between start and end keypoints.
            d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None])
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line
            )

            patch = np.exp(-d2_seg / 2.0 / sigma**2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = np.maximum(
                arr[min_y:max_y, min_x:max_x], patch
            )

    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: kps * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: 1 * kps * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(
                    arr[i], starts, ends, start_values, end_values
                )

    def gen_an_aug(
        self, results, keypoint_score=None, min_down_scaling=0.65, max_up_scaling=1.0
    ):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results
        kp_shape = all_kps.shape

        if self.use_score:
            all_kpscores = keypoint_score
        else:
            if self.use_gaussian_score:
                all_kpscores = np.random.normal(
                    loc=self.mean_gaussian_score,
                    scale=self.scale_gaussian_score,
                    size=kp_shape[:-1],
                )
            else:
                all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        all_kpscores = np.clip(all_kpscores, 0, 1)

        img_h, img_w = self.img_dims

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        kps = [
            [Keypoint(x=x, y=y) for (x, y) in kps_per_image]
            for kps_per_image in all_kps[0]
        ]
        if self.is_training:
            scaling = random.uniform(min_down_scaling, max_up_scaling)
        else:
            scaling = 0.65
        aug_kpt = scale_and_center(kps, scaling * self.heatmap_size)

        val = self.heatmap_size // 8
        x_offset = random.randrange(-val, val)
        y_offset = random.randrange(-val, val)

        for i in range(0, len(aug_kpt)):
            kp_on_image = aug_kpt[i]
            for j in range(0, len(kp_on_image)):
                kps = kp_on_image[j]
                # assert kps.x < self.heatmap_size
                # assert kps.y < self.heatmap_size
                # kps.x = min(max(kps.x, 0), self.heatmap_size - 1)
                # kps.y = min(max(kps.y, 0), self.heatmap_size - 1)
                if self.is_training:
                    all_kps[0, i, j, :] = [kps.x + x_offset, kps.y + y_offset]
                else:
                    all_kps[0, i, j, :] = [kps.x, kps.y]

        num_frame = kp_shape[1]
        new_img_h, new_img_w = self.heatmap_size, self.heatmap_size  # aug_kpt[0].shape
        pad_width = new_img_h == self.heatmap_size
        pad_size = (
            (self.heatmap_size - new_img_w) // 2
            if pad_width
            else (self.heatmap_size - new_img_h) // 2
        )
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, new_img_h, new_img_w], dtype=np.float32)

        par_kps = []
        par_kpscores = []

        # ----------------Pool implementation
        """
        for i in range(num_frame):
            par_kps.append(all_kps[:, i])
            par_kpscores.append(all_kpscores[:, i])

        start = time.time()
        with Pool(60) as pool:
            partial_filled_func = partial(
                flatten_generate_heatmap,
                sigma=self.sigma,
                num_c=num_c,
                img_h=new_img_h,
                img_w=new_img_w,
            )
            data = zip(par_kps, par_kpscores)
            par_ret = pool.map(partial_filled_func, data)
        end = time.time()

        print("Elappsed time: {}ms\n".format((end - start) * 1000))

        for i in range(num_frame):
            ret[i] = par_ret[i]
        """
        # ---------------- Endof Pool implementation

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i]

            self.generate_heatmap(ret[i], kps, kpscores)

        return ret, all_kps[0]

    def __call__(self, results):
        heatmap, kpt = self.gen_an_aug(results)
        key = "heatmap_imgs" if "imgs" in results else "imgs"

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (
                (self.left_kp, self.right_kp)
                if self.with_kp
                else (self.left_limb, self.right_limb)
            )
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])

        if self.collapse_joints:
            return heatmap.max(axis=1), kpt
        else:
            return heatmap, kpt

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma}, "
            f"use_score={self.use_score}, "
            f"with_kp={self.with_kp}, "
            f"with_limb={self.with_limb}, "
            f"skeletons={self.skeletons}, "
            f"double={self.double}, "
            f"left_kp={self.left_kp}, "
            f"right_kp={self.right_kp})"
        )
        return repr_str

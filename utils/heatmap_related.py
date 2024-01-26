# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import torch
from multiprocess import Pool


EPS = 1e-3


def flatten_gen_heat(kps, skeletons):
    for i, limb in enumerate(skeletons):
        start_idx, end_idx = limb
        starts = kps[:, start_idx]
        ends = kps[:, end_idx]

        start_values = max_values[:, start_idx]
        end_values = max_values[:, end_idx]
        self.generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values)


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
        mean_gaussian_score=0.65,
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
    ):
        self.sigma = sigma
        self.use_score = use_score
        self.use_gaussian_score = (use_gaussian_score,)
        self.mean_gaussian_score = (mean_gaussian_score,)
        self.scale_gaussian_score = (scale_gaussian_score,)
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

    def gen_an_aug(self, results, keypoint_score=None):
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
            KeypointsOnImage(
                [Keypoint(x=x, y=y) for (y, x) in kps_per_image], shape=(img_h, img_w)
            )
            for kps_per_image in all_kps[0]
        ]

        scale = (
            {"height": self.heatmap_size, "width": "keep-aspect-ratio"}
            if (img_h > img_w)
            else {"width": self.heatmap_size, "height": "keep-aspect-ratio"}
        )

        seq = iaa.Sequential(
            [
                iaa.Resize(scale),
            ]
        )

        aug_kpt = seq(keypoints=kps)

        for i in range(0, len(aug_kpt)):
            kp_on_image = aug_kpt[i]
            for j in range(0, len(kp_on_image)):
                kps = kp_on_image[j]
                all_kps[0, i, j, :] = [kps.y, kps.x]

        num_frame = kp_shape[1]
        new_img_h, new_img_w = aug_kpt[0].shape
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

        for i in range(num_frame):
            par_kps.append(all_kps[:, i])
            par_kpscores.append(all_kpscores[:, i])

        def generate_heatmap_per_fram(fram_idx):
            local_ret = np.zeros([num_c, new_img_h, new_img_w], dtype=np.float32)
            self.generate_heatmap(local_ret, par_kps[fram_idx], par_kpscores[fram_idx])
            return local_ret

        with Pool(1) as pool:
            all_ret = pool.map(generate_heatmap_per_fram, range(num_frame))

        for i in range(num_frame):
            ret[i] = all_ret[i]

        pad_size_tuple = (
            ((0, 0), (0, 0), (0, 0), (pad_size, pad_size))
            if pad_width
            else ((0, 0), (0, 0), (pad_size, pad_size), (0, 0))
        )

        ret = np.pad(ret, pad_size_tuple, "constant", constant_values=0)
        return ret

    def __call__(self, results):
        heatmap = self.gen_an_aug(results)
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

        combined_heatmaps = heatmap.max(axis=1)

        return combined_heatmaps

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

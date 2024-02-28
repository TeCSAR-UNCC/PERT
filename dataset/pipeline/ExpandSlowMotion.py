from pyskl.datasets import PIPELINES
import numpy as np
import copy


@PIPELINES.register_module()
class ExpandSlowMotion:
    def __init__(self, window_size=300):
        self.window_size = window_size

    def __call__(self, results):
        keypoint = results["keypoint"]
        keypoint_score = results["keypoint_score"]
        N, f, K, C = keypoint.shape
        if f == self.window_size:
            return results
        # Create a new frame axis for B_prime
        B_prime = np.zeros((N, self.window_size, K, C), dtype=keypoint.dtype)
        ks_prime = np.zeros((N, self.window_size, K), dtype=keypoint_score.dtype)

        # Calculate the new frame indices (target)
        new_frame_indices = np.linspace(0, f - 1, self.window_size)

        # Original frame indices (source)
        original_frame_indices = np.arange(f)

        # Interpolate for each batch, keypoint, and x,y coordinate
        for k in range(K):
            ks_prime[0, :, k] = np.interp(
                new_frame_indices, original_frame_indices, keypoint_score[0, :, k]
            )
            for c in range(C):
                B_prime[0, :, k, c] = np.interp(
                    new_frame_indices, original_frame_indices, keypoint[0, :, k, c]
                )

        results["keypoint"] = B_prime
        results["keypoint_score"] = ks_prime
        return results

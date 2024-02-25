import numpy as np


def expand_to_slow_motion_np(B, F):
    f, K, C = B.shape
    # Create a new frame axis for B_prime
    B_prime = np.zeros((F, K, C), dtype=B.dtype)

    # Calculate the new frame indices (target)
    new_frame_indices = np.linspace(0, f - 1, F)

    # Original frame indices (source)
    original_frame_indices = np.arange(f)

    # Interpolate for each batch, keypoint, and x,y coordinate
    for k in range(K):
        for c in range(C):
            B_prime[:, k, c] = np.interp(
                new_frame_indices, original_frame_indices, B[:, k, c]
            )

    return B_prime

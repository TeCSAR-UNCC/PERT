import numpy as np
import cv2


def convert_to_rgb_3d(array, offset=10):
    """
    Convert a 3D numpy array (T, H, W) to a 3D RGB representation (T, 3, H, W).

    Parameters:
    array (numpy.ndarray): Input array of shape (N, T, H, W).
    colormap (str): The name of the colormap to use.

    Returns:
    numpy.ndarray: Output array of shape (T, 3, H, N * W).
    """
    N, T, H, W = array.shape
    output = np.zeros((T, 3, H, N * W), dtype=np.uint8)

    # Apply colormap to each time step
    for n in range(N):
        for t in range(T):
            # Normalize the heatmap for display
            normalized_heatmap = cv2.normalize(
                array[n][t], None, 0, 255, cv2.NORM_MINMAX
            )

            colored_heatmap = cv2.applyColorMap(
                normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
            )

            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            colored_heatmap = colored_heatmap.transpose((2, 0, 1))

            output[t, 0:, 0:, n * W : (n + 1) * W] = colored_heatmap

    return output


if __name__ == "__main__":
    # Example usage
    N, T, H, W = 4, 60, 256, 256  # dimensions
    example_array = np.zeros((N, T, H, W))
    for n in range(1, N):
        example_array[n] = np.ones((T, H, W)) * (n / N)  # Gaussian distribution

    rgb_array = convert_to_rgb_3d(example_array).transpose((0, 3, 2, 1))

    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    _, new_W, new_H, _ = rgb_array.shape

    out = cv2.VideoWriter(
        "test_rgb.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 30, (new_W, new_H)
    )

    for i in range(T):
        # Write to video
        out.write(rgb_array[i])

    out.release()

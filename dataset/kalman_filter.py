import cv2
import numpy as np

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
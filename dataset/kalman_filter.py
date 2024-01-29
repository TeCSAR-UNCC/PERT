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

    def _apply(self, keypoints):
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
    
    def _compute_velocity(self, keypoints, delta_t=1):
        # keypoints is an array of shape (window, num_keypoints, 2)
        velocities = np.zeros_like(keypoints)

        velocities[1:] = (keypoints[1:] - keypoints[:-1]) / delta_t

        return velocities

    def filter_data(self, data):
        filtered_data = np.zeros_like(data)
        vel = self._compute_velocity(data)
        data_vel = np.concatenate((data, vel), axis=2)

        for i in range(data_vel.shape[0]):
            # Apply the Kalman Filter on the concatenated data and velocity
            filtered_data[i] = self._apply(data_vel[i])[:, :2]

        # First 10 unfiltered because idk whats wrong
        return np.concatenate((data[:10], filtered_data[10:]), axis=0)
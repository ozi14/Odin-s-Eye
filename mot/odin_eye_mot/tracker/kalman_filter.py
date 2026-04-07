"""
kalman_filter.py — Kalman Filter for ByteTrack Motion Prediction

State vector: [cx, cy, w, h, vcx, vcy, vw, vh]  (8-D)
  cx, cy : bounding box center
  w,  h  : bounding box width and height
  v*     : respective velocities (constant-velocity model)

Measurement: [cx, cy, w, h]  (4-D)

Adapted from the ByteTrack reference implementation with minor
restructuring for clarity.
"""

import numpy as np


class KalmanFilter:
    """
    Kalman filter for axis-aligned bounding box tracking.

    Uses a constant-velocity motion model.  Bounding boxes are
    parameterised as (cx, cy, w, h) — centre coordinates plus dimensions.
    """

    def __init__(self):
        ndim = 4          # measurement dimension
        dt = 1.0          # time step (one frame)

        # Transition matrix F  (8×8)
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = dt          # position += velocity × dt

        # Measurement matrix H  (4×8)
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

        # Process-noise weight (tuned empirically, following ByteTrack)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    # ── initialisation ──────────────────────────────────────────────

    def initiate(self, measurement: np.ndarray):
        """
        Create a new track from a raw measurement (cx, cy, w, h).

        Returns:
            mean  (8,)  initial state estimate
            cov   (8,8) initial covariance
        """
        mean_pos = measurement.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[2],   # cx
            2 * self._std_weight_position * measurement[3],   # cy
            2 * self._std_weight_position * measurement[2],   # w
            2 * self._std_weight_position * measurement[3],   # h
            10 * self._std_weight_velocity * measurement[2],  # vcx
            10 * self._std_weight_velocity * measurement[3],  # vcy
            10 * self._std_weight_velocity * measurement[2],  # vw
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    # ── predict ─────────────────────────────────────────────────────

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        """
        Propagate the state distribution one step forward.

        Returns:
            mean_pred  (8,)
            cov_pred   (8,8)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        Q = np.diag(np.square(std_pos + std_vel))  # process noise

        mean_pred = self._F @ mean
        cov_pred = self._F @ cov @ self._F.T + Q
        return mean_pred, cov_pred

    # ── update ──────────────────────────────────────────────────────

    def update(self, mean: np.ndarray, cov: np.ndarray,
               measurement: np.ndarray):
        """
        Condition the state on a new measurement.

        Returns:
            mean_upd  (8,)
            cov_upd   (8,8)
        """
        projected_mean = self._H @ mean
        projected_cov = self._H @ cov @ self._H.T + self._innovation_cov(mean)

        # Kalman gain
        K = cov @ self._H.T @ np.linalg.inv(projected_cov)

        innovation = measurement - projected_mean
        mean_upd = mean + K @ innovation
        cov_upd = (np.eye(len(mean)) - K @ self._H) @ cov
        return mean_upd, cov_upd

    def _innovation_cov(self, mean: np.ndarray) -> np.ndarray:
        """Measurement noise covariance R."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        return np.diag(np.square(std))

    # ── helpers ─────────────────────────────────────────────────────

    def project(self, mean: np.ndarray, cov: np.ndarray):
        """
        Project state into measurement space (cx, cy, w, h).

        Returns:
            projected_mean (4,)
            projected_cov  (4,4)
        """
        projected_mean = self._H @ mean
        projected_cov = self._H @ cov @ self._H.T + self._innovation_cov(mean)
        return projected_mean, projected_cov

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """(x, y, w, h) top-left → (cx, cy, aspect, h) for legacy compat."""
        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        return np.array([cx, cy, tlwh[2] / tlwh[3], tlwh[3]], dtype=np.float64)

    @staticmethod
    def xywh_to_measurement(xywh: np.ndarray) -> np.ndarray:
        """(cx, cy, w, h) already in measurement space, just cast."""
        return xywh.astype(np.float64)

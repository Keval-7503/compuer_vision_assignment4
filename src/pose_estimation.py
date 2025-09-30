"""
Camera Pose Estimation Module
Implements homography-based and solvePnP methods for estimating camera pose from planar objects.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class PoseEstimator:
    """
    Estimates camera pose (rotation and translation) from a planar object.
    """

    def __init__(self, camera_matrix: np.ndarray):
        """
        Initialize the pose estimator.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = np.zeros(5)  # Assuming no lens distortion

    def estimate_pose_homography(
        self,
        image_points: np.ndarray,
        object_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate camera pose using homography decomposition.

        Args:
            image_points: Nx2 array of 2D image points
            object_points: Nx3 array of 3D object points (planar, z=0)

        Returns:
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector
            rmse: Reprojection error (RMSE)
        """
        # Ensure points are in correct format
        image_points = np.float32(image_points).reshape(-1, 2)
        object_points_2d = np.float32(object_points[:, :2]).reshape(-1, 2)

        # Compute homography
        H, mask = cv2.findHomography(object_points_2d, image_points, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("Failed to compute homography")

        # Normalize homography
        H = H / H[2, 2]

        # Decompose homography to get rotation and translation
        K = self.camera_matrix
        K_inv = np.linalg.inv(K)

        # H = K [r1 r2 t]
        H_norm = K_inv @ H

        # Extract rotation columns
        h1 = H_norm[:, 0]
        h2 = H_norm[:, 1]
        h3 = H_norm[:, 2]

        # Normalize to get proper rotation matrix
        lambda_val = 1.0 / np.linalg.norm(h1)
        r1 = lambda_val * h1
        r2 = lambda_val * h2
        t = lambda_val * h3

        # Compute r3 as cross product to ensure orthogonality
        r3 = np.cross(r1, r2)

        # Build rotation matrix
        R = np.column_stack([r1, r2, r3])

        # Ensure proper rotation matrix (closest orthogonal matrix)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            R = -R
            t = -t

        # Calculate reprojection error
        rmse = self._compute_reprojection_error(
            object_points, image_points, R, t.reshape(3, 1)
        )

        return R, t.reshape(3, 1), rmse

    def estimate_pose_solvepnp(
        self,
        image_points: np.ndarray,
        object_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate camera pose using OpenCV's solvePnP.

        Args:
            image_points: Nx2 array of 2D image points
            object_points: Nx3 array of 3D object points

        Returns:
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector
            rmse: Reprojection error (RMSE)
        """
        # Ensure correct format
        image_points = np.float32(image_points).reshape(-1, 1, 2)
        object_points = np.float32(object_points).reshape(-1, 1, 3)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise ValueError("solvePnP failed to find a solution")

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        # Calculate reprojection error
        rmse = self._compute_reprojection_error(
            object_points.reshape(-1, 3),
            image_points.reshape(-1, 2),
            R,
            tvec
        )

        return R, tvec, rmse

    def _compute_reprojection_error(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> float:
        """
        Compute RMSE of reprojection error.

        Args:
            object_points: Nx3 array of 3D points
            image_points: Nx2 array of 2D points
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            RMSE value
        """
        # Project 3D points to image
        projected_points = self.project_points(object_points, R, t)

        # Compute error
        errors = image_points - projected_points
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)

        return rmse

    def project_points(
        self,
        object_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Project 3D points to image plane.

        Args:
            object_points: Nx3 array of 3D points
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            Nx2 array of projected 2D points
        """
        # Transform points to camera coordinates
        points_3d = object_points.reshape(-1, 3)
        points_cam = (R @ points_3d.T + t).T

        # Project to image plane
        points_2d_hom = (self.camera_matrix @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

        return points_2d

    def get_extrinsic_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Construct 4x4 extrinsic matrix from rotation and translation.

        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            4x4 extrinsic matrix [R|t; 0 0 0 1]
        """
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t.flatten()
        return extrinsic

    def get_projection_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Construct 3x4 projection matrix P = K[R|t].

        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            3x4 projection matrix
        """
        Rt = np.column_stack([R, t])
        P = self.camera_matrix @ Rt
        return P


def create_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Create camera intrinsic matrix.

    Args:
        fx: Focal length in x
        fy: Focal length in y
        cx: Principal point x
        cy: Principal point y

    Returns:
        3x3 camera matrix
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def create_planar_object_points(width: float, height: float, num_points: int = 4) -> np.ndarray:
    """
    Create planar object points (z=0) for a rectangular object.

    Args:
        width: Object width
        height: Object height
        num_points: Number of corner points (4 for rectangle)

    Returns:
        Nx3 array of 3D points
    """
    if num_points == 4:
        # Four corners of a rectangle
        points = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ], dtype=np.float32)
    else:
        raise NotImplementedError("Only 4-point rectangles are currently supported")

    return points

"""
Utility Functions
Helper functions for image processing, coordinate transformations, etc.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import torch


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image array in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: Input image
        target_size: (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image (uint8)

    Returns:
        Normalized image (float32)
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image back to uint8.

    Args:
        image: Normalized image [0, 1]

    Returns:
        Image in uint8 format
    """
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def get_image_corners(
    image_shape: Tuple[int, int],
    pixel_coords: bool = True
) -> np.ndarray:
    """
    Get corner coordinates of an image.

    Args:
        image_shape: (height, width)
        pixel_coords: If True, return pixel coordinates; else normalized [0,1]

    Returns:
        4x2 array of corner coordinates [top-left, top-right, bottom-right, bottom-left]
    """
    h, w = image_shape[:2]

    if pixel_coords:
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
    else:
        corners = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)

    return corners


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (in radians).

    Args:
        roll: Rotation around X-axis
        pitch: Rotation around Y-axis
        yaw: Rotation around Z-axis

    Returns:
        3x3 rotation matrix
    """
    # Rotation around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation around Y-axis
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around Z-axis
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    return R


def euler_from_rotation_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from rotation matrix.

    Args:
        R: 3x3 rotation matrix

    Returns:
        (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


def points_to_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinates.

    Args:
        points: (N, D) array

    Returns:
        (N, D+1) array in homogeneous coordinates
    """
    ones = np.ones((points.shape[0], 1))
    return np.hstack([points, ones])


def homogeneous_to_points(points_h: np.ndarray) -> np.ndarray:
    """
    Convert from homogeneous to Euclidean coordinates.

    Args:
        points_h: (N, D+1) array in homogeneous coordinates

    Returns:
        (N, D) array in Euclidean coordinates
    """
    return points_h[:, :-1] / points_h[:, -1:]


def compute_image_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients.

    Args:
        image: Input image (grayscale or color)

    Returns:
        (grad_x, grad_y) - gradient images
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    return grad_x, grad_y


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    Draw points on image.

    Args:
        image: Input image
        points: Nx2 array of 2D points
        color: Point color (BGR)
        radius: Point radius
        thickness: -1 for filled, positive for outline

    Returns:
        Image with drawn points
    """
    img_copy = image.copy()

    for point in points:
        pt = tuple(point.astype(int))
        cv2.circle(img_copy, pt, radius, color, thickness)

    return img_copy


def draw_correspondences(
    image1: np.ndarray,
    image2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw point correspondences between two images.

    Args:
        image1: First image
        image2: Second image
        points1: Points in first image (Nx2)
        points2: Points in second image (Nx2)
        color: Line color (if None, random colors)

    Returns:
        Combined image with correspondences
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Create side-by-side image
    h = max(h1, h2)
    w = w1 + w2
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[:h1, :w1] = image1
    vis[:h2, w1:w1 + w2] = image2

    # Offset points2 by width of image1
    points2_offset = points2.copy()
    points2_offset[:, 0] += w1

    # Draw correspondences
    for i, (pt1, pt2) in enumerate(zip(points1, points2_offset)):
        if color is None:
            # Random color for each correspondence
            c = tuple(np.random.randint(0, 255, 3).tolist())
        else:
            c = color

        pt1_int = tuple(pt1.astype(int))
        pt2_int = tuple(pt2.astype(int))

        cv2.circle(vis, pt1_int, 5, c, -1)
        cv2.circle(vis, pt2_int, 5, c, -1)
        cv2.line(vis, pt1_int, pt2_int, c, 2)

    return vis


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box of points.

    Args:
        points: Nx2 or Nx3 array of points

    Returns:
        (min_point, max_point)
    """
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)

    return min_point, max_point


def check_torch_device() -> torch.device:
    """
    Check and return available PyTorch device.

    Returns:
        torch.device (cuda if available, else cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_grid_points(
    width: float,
    height: float,
    num_x: int,
    num_y: int,
    z: float = 0.0
) -> np.ndarray:
    """
    Create a grid of 3D points on a plane.

    Args:
        width: Grid width
        height: Grid height
        num_x: Number of points in X direction
        num_y: Number of points in Y direction
        z: Z coordinate (constant)

    Returns:
        (num_x * num_y, 3) array of 3D points
    """
    x = np.linspace(0, width, num_x)
    y = np.linspace(0, height, num_y)

    xv, yv = np.meshgrid(x, y)

    points = np.stack([
        xv.flatten(),
        yv.flatten(),
        np.full(xv.size, z)
    ], axis=1)

    return points.astype(np.float32)

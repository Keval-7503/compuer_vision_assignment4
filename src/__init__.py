"""
Assignment 4: Augmented Reality with PyTorch3D
Main package initialization
"""

from .pose_estimation import PoseEstimator, create_camera_matrix, create_planar_object_points
from .renderer import PyTorch3DRenderer, create_mesh_from_numpy, opencv_to_pytorch3d_projection
from .object_placement import ObjectPlacer, compute_plane_coordinate_system
from .visualization import ImageCompositor, Visualizer, save_image
from .utils import (
    load_image,
    resize_image,
    normalize_image,
    denormalize_image,
    check_torch_device,
    set_random_seed
)

__all__ = [
    'PoseEstimator',
    'create_camera_matrix',
    'create_planar_object_points',
    'PyTorch3DRenderer',
    'create_mesh_from_numpy',
    'opencv_to_pytorch3d_projection',
    'ObjectPlacer',
    'compute_plane_coordinate_system',
    'ImageCompositor',
    'Visualizer',
    'save_image',
    'load_image',
    'resize_image',
    'normalize_image',
    'denormalize_image',
    'check_torch_device',
    'set_random_seed'
]

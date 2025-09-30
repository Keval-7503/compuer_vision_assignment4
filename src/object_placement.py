"""
3D Object Placement and Transformation Module
Handles loading, transforming, and positioning 3D objects in the scene.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.transforms import Transform3d, Rotate, Translate, Scale
from pytorch3d.renderer import TexturesVertex
import os


class ObjectPlacer:
    """
    Handles 3D object transformations and placement in the scene.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the object placer.

        Args:
            device: torch device (cuda/cpu)
        """
        self.device = device if device is not None else torch.device("cpu")

    def load_mesh(self, obj_path: str) -> Meshes:
        """
        Load a mesh from an OBJ file.

        Args:
            obj_path: Path to .obj file

        Returns:
            PyTorch3D Meshes object
        """
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Mesh file not found: {obj_path}")

        # Load mesh
        mesh = load_objs_as_meshes([obj_path], device=self.device)

        return mesh

    def create_primitive_mesh(
        self,
        shape: str = "cube",
        size: float = 1.0,
        color: Optional[np.ndarray] = None
    ) -> Meshes:
        """
        Create a primitive 3D shape (cube, pyramid, etc.).

        Args:
            shape: Type of primitive ("cube", "pyramid", "tetrahedron")
            size: Size of the object
            color: RGB color [0-1] (optional)

        Returns:
            PyTorch3D Meshes object
        """
        if shape == "cube":
            verts, faces = self._create_cube(size)
        elif shape == "pyramid":
            verts, faces = self._create_pyramid(size)
        elif shape == "tetrahedron":
            verts, faces = self._create_tetrahedron(size)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        # Create default color if not provided
        if color is None:
            color = np.array([0.7, 0.3, 0.3])  # Default red

        # Expand color to all vertices
        colors = np.tile(color, (len(verts), 1))

        # Convert to torch
        verts_torch = torch.from_numpy(verts).float().unsqueeze(0).to(self.device)
        faces_torch = torch.from_numpy(faces).long().unsqueeze(0).to(self.device)
        colors_torch = torch.from_numpy(colors).float().unsqueeze(0).to(self.device)

        # Create texture
        textures = TexturesVertex(verts_features=colors_torch)

        # Create mesh
        mesh = Meshes(
            verts=verts_torch,
            faces=faces_torch,
            textures=textures
        )

        return mesh

    def _create_cube(self, size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create cube vertices and faces."""
        s = size / 2.0
        verts = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # back
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]   # front
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # back
            [4, 6, 5], [4, 7, 6],  # front
            [0, 4, 5], [0, 5, 1],  # bottom
            [2, 6, 7], [2, 7, 3],  # top
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ], dtype=np.int64)

        return verts, faces

    def _create_pyramid(self, size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create pyramid vertices and faces."""
        s = size / 2.0
        h = size
        verts = np.array([
            [-s, 0, -s], [s, 0, -s], [s, 0, s], [-s, 0, s],  # base
            [0, h, 0]  # apex
        ], dtype=np.float32)

        faces = np.array([
            [0, 2, 1], [0, 3, 2],  # base
            [0, 1, 4],  # front
            [1, 2, 4],  # right
            [2, 3, 4],  # back
            [3, 0, 4]   # left
        ], dtype=np.int64)

        return verts, faces

    def _create_tetrahedron(self, size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create tetrahedron vertices and faces."""
        s = size / 2.0
        verts = np.array([
            [0, s, 0],
            [-s, -s, s],
            [s, -s, s],
            [0, -s, -s]
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ], dtype=np.int64)

        return verts, faces

    def transform_mesh(
        self,
        mesh: Meshes,
        translation: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        scale: Optional[Union[float, np.ndarray]] = None
    ) -> Meshes:
        """
        Apply transformations to a mesh.

        Args:
            mesh: Input mesh
            translation: 3D translation vector [x, y, z]
            rotation: 3x3 rotation matrix or Euler angles
            scale: Uniform scale or [sx, sy, sz]

        Returns:
            Transformed mesh
        """
        # Get vertices
        verts = mesh.verts_packed()

        # Apply transformations
        if scale is not None:
            if isinstance(scale, (int, float)):
                verts = verts * scale
            else:
                scale_tensor = torch.from_numpy(scale).float().to(self.device)
                verts = verts * scale_tensor

        if rotation is not None:
            if rotation.shape == (3, 3):
                # Rotation matrix
                R = torch.from_numpy(rotation).float().to(self.device)
                verts = torch.matmul(verts, R.T)
            elif rotation.shape == (3,):
                # Euler angles (in radians)
                # Not implemented here - use rotation matrix instead
                raise NotImplementedError("Use rotation matrix instead")

        if translation is not None:
            t = torch.from_numpy(translation).float().to(self.device)
            verts = verts + t

        # Create new mesh with transformed vertices
        new_mesh = mesh.update_padded(verts.unsqueeze(0))

        return new_mesh

    def place_on_plane(
        self,
        mesh: Meshes,
        plane_center: np.ndarray,
        plane_normal: np.ndarray = np.array([0, 0, 1]),
        height_offset: float = 0.0,
        scale: float = 1.0
    ) -> Meshes:
        """
        Place object on a planar surface.

        Args:
            mesh: Input mesh
            plane_center: 3D position of plane center
            plane_normal: Normal vector of the plane (default: +Z)
            height_offset: Height above the plane
            scale: Scale factor

        Returns:
            Transformed mesh positioned on the plane
        """
        # Normalize plane normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Compute rotation to align object with plane
        # Assume object's "up" direction is +Z
        object_up = np.array([0, 0, 1])

        # Rotation to align object_up with plane_normal
        if not np.allclose(object_up, plane_normal):
            # Compute rotation axis (cross product)
            axis = np.cross(object_up, plane_normal)
            axis_norm = np.linalg.norm(axis)

            if axis_norm > 1e-6:
                axis = axis / axis_norm
                # Compute rotation angle
                angle = np.arccos(np.clip(np.dot(object_up, plane_normal), -1.0, 1.0))

                # Rodrigues' rotation formula
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                # Parallel or anti-parallel
                if np.dot(object_up, plane_normal) < 0:
                    # Anti-parallel: rotate 180 degrees
                    R = -np.eye(3)
                    R[2, 2] = 1
                else:
                    R = np.eye(3)
        else:
            R = np.eye(3)

        # Compute final position (center + offset along normal)
        position = plane_center + plane_normal * height_offset

        # Apply transformations
        transformed_mesh = self.transform_mesh(
            mesh,
            scale=scale,
            rotation=R,
            translation=position
        )

        return transformed_mesh

    def set_mesh_color(self, mesh: Meshes, color: np.ndarray) -> Meshes:
        """
        Set uniform color for the entire mesh.

        Args:
            mesh: Input mesh
            color: RGB color [0-1]

        Returns:
            Mesh with updated color
        """
        num_verts = mesh.verts_packed().shape[0]
        colors = np.tile(color, (num_verts, 1))
        colors_torch = torch.from_numpy(colors).float().unsqueeze(0).to(self.device)

        # Create new texture
        textures = TexturesVertex(verts_features=colors_torch)

        # Update mesh
        new_mesh = mesh.clone()
        new_mesh.textures = textures

        return new_mesh


def compute_plane_coordinate_system(
    plane_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a coordinate system for a plane defined by points.

    Args:
        plane_points: Nx3 array of points on the plane

    Returns:
        origin: Origin point (center)
        x_axis: X-axis direction
        y_axis: Y-axis direction
        z_axis: Normal direction (Z-axis)
    """
    # Compute center
    origin = np.mean(plane_points, axis=0)

    # Compute two edge vectors
    if len(plane_points) >= 4:
        v1 = plane_points[1] - plane_points[0]
        v2 = plane_points[3] - plane_points[0]
    else:
        raise ValueError("Need at least 4 points to define plane coordinate system")

    # Normalize vectors
    x_axis = v1 / np.linalg.norm(v1)
    y_axis = v2 / np.linalg.norm(v2)

    # Compute normal (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Re-orthogonalize y_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return origin, x_axis, y_axis, z_axis

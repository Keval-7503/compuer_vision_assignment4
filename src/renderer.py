"""
PyTorch3D Renderer Module
Sets up PyTorch3D renderer with correct camera parameters for AR applications.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesVertex,
    PointLights,
    Materials,
)
from pytorch3d.transforms import Transform3d


class PyTorch3DRenderer:
    """
    Wrapper class for PyTorch3D rendering with camera alignment.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        camera_matrix: np.ndarray,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the PyTorch3D renderer.

        Args:
            image_size: (height, width) of the output image
            camera_matrix: 3x3 camera intrinsic matrix
            device: torch device (cuda/cpu)
        """
        self.device = device if device is not None else torch.device("cpu")
        self.image_size = image_size  # (H, W)
        self.camera_matrix = camera_matrix

        # Extract intrinsic parameters
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

        # Initialize renderer components
        self.rasterizer = None
        self.shader = None
        self.renderer = None
        self.lights = None

    def setup_camera(
        self,
        R: np.ndarray,
        t: np.ndarray,
        znear: float = 0.01,
        zfar: float = 100.0
    ) -> PerspectiveCameras:
        """
        Create PyTorch3D camera from OpenCV camera parameters.

        Args:
            R: 3x3 rotation matrix (camera to world)
            t: 3x1 translation vector
            znear: Near clipping plane
            zfar: Far clipping plane

        Returns:
            PyTorch3D PerspectiveCameras object
        """
        # Convert numpy to torch
        R_torch = torch.from_numpy(R).float().unsqueeze(0).to(self.device)
        t_torch = torch.from_numpy(t.flatten()).float().unsqueeze(0).to(self.device)

        # PyTorch3D uses a different coordinate system than OpenCV
        # OpenCV: +X right, +Y down, +Z forward (into scene)
        # PyTorch3D: +X left, +Y up, +Z into scene
        # We need to convert from OpenCV to PyTorch3D coordinates

        # Coordinate system transformation matrix
        coord_transform = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Transform rotation
        R_pytorch3d = coord_transform @ R_torch[0]
        R_pytorch3d = R_pytorch3d.unsqueeze(0)

        # Transform translation
        t_pytorch3d = (coord_transform @ t_torch[0]).unsqueeze(0)

        # Convert to NDC (Normalized Device Coordinates)
        # PyTorch3D expects principal point at origin
        H, W = self.image_size

        # Focal lengths in NDC
        focal_length = torch.tensor([[self.fx, self.fy]], device=self.device)

        # Principal point in NDC (normalized to [-1, 1])
        principal_point = torch.tensor([[self.cx, self.cy]], device=self.device)

        # Create camera
        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R_pytorch3d,
            T=t_pytorch3d,
            image_size=torch.tensor([[H, W]], device=self.device),
            in_ndc=False,
            device=self.device
        )

        return cameras

    def setup_renderer(
        self,
        cameras: PerspectiveCameras,
        image_size: Optional[Tuple[int, int]] = None,
        blur_radius: float = 0.0,
        faces_per_pixel: int = 1,
        shader_type: str = "soft"
    ):
        """
        Setup the mesh renderer with lighting and shading.

        Args:
            cameras: PyTorch3D cameras object
            image_size: (height, width) for rendering
            blur_radius: Blur radius for soft rasterization
            faces_per_pixel: Number of faces to track per pixel
            shader_type: "soft" or "hard" phong shading
        """
        if image_size is None:
            image_size = self.image_size

        H, W = image_size

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=True
        )

        # Setup lights (positioned above and in front of the scene)
        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, -3.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.6, 0.6, 0.6]],
            specular_color=[[0.3, 0.3, 0.3]]
        )

        # Rasterizer
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        # Shader
        if shader_type == "soft":
            self.shader = SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights
            )
        else:
            self.shader = HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights
            )

        # Create renderer
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def render_mesh(
        self,
        meshes: Meshes,
        cameras: PerspectiveCameras
    ) -> torch.Tensor:
        """
        Render a mesh using the configured renderer.

        Args:
            meshes: PyTorch3D Meshes object
            cameras: PyTorch3D cameras

        Returns:
            Rendered image tensor (H, W, 4) with RGBA channels
        """
        if self.renderer is None:
            self.setup_renderer(cameras)

        # Render
        images = self.renderer(meshes, cameras=cameras, lights=self.lights)

        return images

    def render_to_numpy(
        self,
        meshes: Meshes,
        cameras: PerspectiveCameras
    ) -> np.ndarray:
        """
        Render mesh and return as numpy array.

        Args:
            meshes: PyTorch3D Meshes object
            cameras: PyTorch3D cameras

        Returns:
            Rendered image as numpy array (H, W, 4) with RGBA channels
        """
        images = self.render_mesh(meshes, cameras)

        # Convert to numpy (remove batch dimension)
        image_np = images[0, ..., :4].cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        image_np = (image_np * 255).astype(np.uint8)

        return image_np


def create_mesh_from_numpy(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None
) -> Meshes:
    """
    Create PyTorch3D mesh from numpy arrays.

    Args:
        vertices: (V, 3) array of vertex positions
        faces: (F, 3) array of face indices
        colors: (V, 3) array of RGB colors per vertex (optional)
        device: torch device

    Returns:
        PyTorch3D Meshes object
    """
    if device is None:
        device = torch.device("cpu")

    # Convert to torch tensors
    verts = torch.from_numpy(vertices).float().unsqueeze(0).to(device)
    faces = torch.from_numpy(faces).long().unsqueeze(0).to(device)

    # Create vertex colors
    if colors is None:
        # Default white color
        colors = np.ones_like(vertices)

    colors_torch = torch.from_numpy(colors).float().unsqueeze(0).to(device)
    textures = TexturesVertex(verts_features=colors_torch)

    # Create mesh
    mesh = Meshes(
        verts=verts,
        faces=faces,
        textures=textures
    )

    return mesh


def opencv_to_pytorch3d_projection(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert OpenCV camera parameters to PyTorch3D format.

    Args:
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        image_size: (height, width)

    Returns:
        R_pytorch3d: Rotation matrix in PyTorch3D format
        T_pytorch3d: Translation vector in PyTorch3D format
    """
    # Coordinate system transformation
    coord_transform = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Apply transformation
    R_pytorch3d = coord_transform @ R
    t_pytorch3d = coord_transform @ t.flatten()

    # Convert to torch
    R_torch = torch.from_numpy(R_pytorch3d).float()
    T_torch = torch.from_numpy(t_pytorch3d).float()

    return R_torch, T_torch

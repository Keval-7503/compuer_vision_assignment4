"""
Visualization and Image Compositing Module
Handles overlaying synthetic objects on real images and creating visualizations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import torch
from PIL import Image


class ImageCompositor:
    """
    Handles compositing of synthetic renders with real images.
    """

    def __init__(self):
        pass

    def composite_images(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        blend_mode: str = "alpha"
    ) -> np.ndarray:
        """
        Composite foreground (rendered) image onto background (real) image.

        Args:
            background: Real image (H, W, 3) BGR or RGB
            foreground: Rendered image (H, W, 4) RGBA or (H, W, 3) RGB
            alpha: Custom alpha channel (H, W) or None
            blend_mode: "alpha" or "add"

        Returns:
            Composited image (H, W, 3)
        """
        # Ensure same dimensions
        if background.shape[:2] != foreground.shape[:2]:
            foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

        # Convert to float [0, 1]
        bg = background.astype(np.float32) / 255.0
        fg = foreground.astype(np.float32) / 255.0

        # Extract alpha channel
        if alpha is not None:
            alpha_channel = alpha.astype(np.float32)
            if alpha_channel.max() > 1.0:
                alpha_channel = alpha_channel / 255.0
        elif foreground.shape[2] == 4:
            # Use alpha from RGBA image
            alpha_channel = fg[:, :, 3]
            fg = fg[:, :, :3]
        else:
            # No alpha, use full opacity where foreground is not black
            alpha_channel = np.any(fg > 0.01, axis=2).astype(np.float32)

        # Expand alpha to 3 channels
        alpha_3ch = np.stack([alpha_channel] * 3, axis=2)

        # Composite based on blend mode
        if blend_mode == "alpha":
            # Standard alpha blending: C = α*F + (1-α)*B
            result = alpha_3ch * fg + (1 - alpha_3ch) * bg
        elif blend_mode == "add":
            # Additive blending
            result = np.clip(bg + fg * alpha_3ch, 0, 1)
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

        # Convert back to uint8
        result = (result * 255).astype(np.uint8)

        return result

    def create_mask_from_render(
        self,
        rendered_image: np.ndarray,
        threshold: float = 0.01
    ) -> np.ndarray:
        """
        Create binary mask from rendered image.

        Args:
            rendered_image: Rendered RGBA image
            threshold: Threshold for considering pixel as foreground

        Returns:
            Binary mask (H, W)
        """
        if rendered_image.shape[2] == 4:
            # Use alpha channel
            mask = rendered_image[:, :, 3] > (threshold * 255)
        else:
            # Use brightness
            gray = cv2.cvtColor(rendered_image[:, :, :3], cv2.COLOR_RGB2GRAY)
            mask = gray > (threshold * 255)

        return mask.astype(np.uint8) * 255

    def overlay_with_edges(
        self,
        background: np.ndarray,
        rendered: np.ndarray,
        edge_color: Tuple[int, int, int] = (0, 255, 0),
        edge_thickness: int = 2
    ) -> np.ndarray:
        """
        Composite with edge highlighting for better visualization.

        Args:
            background: Real image
            rendered: Rendered image (RGBA)
            edge_color: Color for edges (RGB)
            edge_thickness: Thickness of edge lines

        Returns:
            Composited image with edges
        """
        # First, do standard compositing
        composited = self.composite_images(background, rendered)

        # Extract mask and find edges
        mask = self.create_mask_from_render(rendered)
        edges = cv2.Canny(mask, 50, 150)

        # Dilate edges for better visibility
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Draw edges on composited image
        composited[edges > 0] = edge_color

        return composited


class Visualizer:
    """
    Creates visualizations for AR results.
    """

    def __init__(self):
        self.compositor = ImageCompositor()

    def visualize_single_result(
        self,
        background: np.ndarray,
        rendered: np.ndarray,
        title: str = "AR Result",
        save_path: Optional[str] = None
    ):
        """
        Visualize a single AR result.

        Args:
            background: Original image
            rendered: Rendered synthetic object
            title: Plot title
            save_path: Path to save figure (optional)
        """
        # Composite images
        result = self.compositor.composite_images(background, rendered)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Rendered object only
        if rendered.shape[2] == 4:
            # Show with transparency
            axes[1].imshow(rendered)
        else:
            axes[1].imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Rendered Object")
        axes[1].axis('off')

        # Composited result
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title(title)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def visualize_multiple_results(
        self,
        results: List[Tuple[np.ndarray, np.ndarray, str]],
        save_path: Optional[str] = None
    ):
        """
        Visualize multiple AR results in a grid.

        Args:
            results: List of (background, rendered, title) tuples
            save_path: Path to save figure (optional)
        """
        n_results = len(results)
        fig, axes = plt.subplots(n_results, 3, figsize=(15, 5 * n_results))

        if n_results == 1:
            axes = axes.reshape(1, -1)

        for i, (background, rendered, title) in enumerate(results):
            # Composite
            composited = self.compositor.composite_images(background, rendered)

            # Original
            axes[i, 0].imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f"Original - {title}")
            axes[i, 0].axis('off')

            # Rendered
            if rendered.shape[2] == 4:
                axes[i, 1].imshow(rendered)
            else:
                axes[i, 1].imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f"Rendered - {title}")
            axes[i, 1].axis('off')

            # Result
            axes[i, 2].imshow(cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title(f"Result - {title}")
            axes[i, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def visualize_with_pose(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        axis_length: float = 1.0,
        save_path: Optional[str] = None
    ):
        """
        Visualize image with camera pose (coordinate axes).

        Args:
            image: Input image
            camera_matrix: 3x3 camera intrinsics
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            axis_length: Length of coordinate axes
            save_path: Path to save figure (optional)
        """
        # Draw coordinate axes on image
        img_with_axes = self.draw_axes(image.copy(), camera_matrix, R, t, axis_length)

        # Display
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_with_axes, cv2.COLOR_BGR2RGB))
        plt.title("Camera Pose Visualization")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def draw_axes(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        length: float = 1.0,
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw 3D coordinate axes on image.

        Args:
            image: Input image
            camera_matrix: Camera intrinsics
            R: Rotation matrix
            t: Translation vector
            length: Axis length
            thickness: Line thickness

        Returns:
            Image with drawn axes
        """
        # Define 3D points for axes
        origin = np.array([0, 0, 0], dtype=np.float32)
        x_axis = np.array([length, 0, 0], dtype=np.float32)
        y_axis = np.array([0, length, 0], dtype=np.float32)
        z_axis = np.array([0, 0, length], dtype=np.float32)

        # Project to 2D
        points_3d = np.array([origin, x_axis, y_axis, z_axis])
        points_cam = (R @ points_3d.T + t).T
        points_2d_hom = (camera_matrix @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        points_2d = points_2d.astype(int)

        # Draw axes
        origin_2d = tuple(points_2d[0])
        x_end = tuple(points_2d[1])
        y_end = tuple(points_2d[2])
        z_end = tuple(points_2d[3])

        # X-axis (red)
        cv2.line(image, origin_2d, x_end, (0, 0, 255), thickness)
        # Y-axis (green)
        cv2.line(image, origin_2d, y_end, (0, 255, 0), thickness)
        # Z-axis (blue)
        cv2.line(image, origin_2d, z_end, (255, 0, 0), thickness)

        return image

    def create_comparison_view(
        self,
        original: np.ndarray,
        result: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Create side-by-side comparison.

        Args:
            original: Original image
            result: AR result
            save_path: Path to save (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Augmented Reality Result")
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


def save_image(image: np.ndarray, path: str):
    """
    Save image to file.

    Args:
        image: Image array (BGR or RGB)
        path: Output path
    """
    # Convert RGB to BGR if needed for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume RGB, convert to BGR for cv2.imwrite
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    cv2.imwrite(path, image_bgr)
    print(f"Image saved to: {path}")

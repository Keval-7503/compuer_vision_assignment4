"""
Main AR Pipeline - Run everything with one function call
"""

def run_ar_pipeline():
    """
    Complete AR pipeline - runs all components automatically.
    """
    print("=" * 60)
    print("Assignment 4: Augmented Reality with PyTorch3D")
    print("=" * 60)

    # Import all required libraries
    print("\n[1/7] Importing libraries...")
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import torch
    import sys
    import os
    from pytorch3d.structures import join_meshes_as_scene

    # Add src to path
    if 'src' not in sys.path:
        sys.path.insert(0, 'src')

    from src.pose_estimation import PoseEstimator, create_camera_matrix, create_planar_object_points
    from src.renderer import PyTorch3DRenderer
    from src.object_placement import ObjectPlacer
    from src.visualization import ImageCompositor, Visualizer
    from src.utils import check_torch_device, set_random_seed

    print("✓ All imports successful")

    # Setup
    print("\n[2/7] Setting up...")
    set_random_seed(42)
    device = check_torch_device()
    print(f"✓ Using device: {device}")

    # Camera parameters
    print("\n[3/7] Configuring camera...")
    image_width = 1280
    image_height = 720
    focal_length = 1000

    K = create_camera_matrix(focal_length, focal_length, image_width/2, image_height/2)

    # Object and image points
    object_width = 0.210  # A4 paper width
    object_height = 0.297  # A4 paper height
    object_points_3d = create_planar_object_points(object_width, object_height)

    image_points_2d = np.array([
        [200, 150], [800, 200], [850, 600], [150, 550]
    ], dtype=np.float32)

    print("✓ Camera configured")

    # Estimate pose
    print("\n[4/7] Estimating camera pose...")
    pose_estimator = PoseEstimator(K)
    R, t, rmse = pose_estimator.estimate_pose_solvepnp(image_points_2d, object_points_3d)
    print(f"✓ Pose estimated (RMSE: {rmse:.4f} pixels)")

    # Setup renderer
    print("\n[5/7] Setting up PyTorch3D renderer...")
    renderer = PyTorch3DRenderer(image_size=(image_height, image_width), camera_matrix=K, device=device)
    cameras = renderer.setup_camera(R=R, t=t)
    renderer.setup_renderer(cameras=cameras, shader_type="soft")
    print("✓ Renderer ready")

    # Create and position objects
    print("\n[6/7] Creating 3D objects...")
    object_placer = ObjectPlacer(device=device)

    # Create objects
    cube_mesh = object_placer.create_primitive_mesh('cube', size=0.05, color=np.array([0.8, 0.2, 0.2]))
    pyramid_mesh = object_placer.create_primitive_mesh('pyramid', size=0.06, color=np.array([0.2, 0.8, 0.2]))
    tetrahedron_mesh = object_placer.create_primitive_mesh('tetrahedron', size=0.04, color=np.array([0.2, 0.2, 0.8]))

    # Position objects
    plane_center = np.array([object_width/2, object_height/2, 0])
    position_1 = np.array([object_width/4, object_height/4, 0])
    position_2 = np.array([3*object_width/4, 3*object_height/4, 0])

    cube_positioned = object_placer.place_on_plane(cube_mesh, plane_center, height_offset=0.025)
    pyramid_positioned = object_placer.place_on_plane(pyramid_mesh, position_1, height_offset=0.0)
    tetrahedron_positioned = object_placer.place_on_plane(tetrahedron_mesh, position_2, height_offset=0.02)

    # Render
    combined_mesh = join_meshes_as_scene([cube_positioned, pyramid_positioned, tetrahedron_positioned])
    rendered_combined = renderer.render_to_numpy(combined_mesh, cameras)
    print("✓ Objects created and rendered")

    # Create background and composite
    print("\n[7/7] Creating AR visualization...")
    background_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 240

    # Draw plane outline
    for i in range(len(image_points_2d)):
        pt1 = tuple(image_points_2d[i].astype(int))
        pt2 = tuple(image_points_2d[(i+1) % len(image_points_2d)].astype(int))
        cv2.line(background_image, pt1, pt2, (100, 100, 100), 3)
        cv2.circle(background_image, pt1, 8, (0, 0, 255), -1)

    # Composite
    compositor = ImageCompositor()
    visualizer = Visualizer()

    ar_result = compositor.composite_images(background_image, rendered_combined, blend_mode="alpha")

    # Display results
    print("✓ AR compositing complete")
    print("\n" + "=" * 60)
    print("Displaying Results...")
    print("=" * 60)

    visualizer.visualize_single_result(
        background=background_image,
        rendered=rendered_combined,
        title="Augmented Reality Result"
    )

    # Save results
    os.makedirs('results', exist_ok=True)
    from src.visualization import save_image
    save_image(ar_result, 'results/ar_result.png')
    save_image(rendered_combined, 'results/rendered_objects.png')

    print("\n" + "=" * 60)
    print("✓ Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults saved to 'results/' folder")
    print(f"Camera Pose RMSE: {rmse:.4f} pixels")
    print(f"Device: {device}")
    print("\n🎉 All tasks completed successfully!")

    return ar_result, rendered_combined, R, t, rmse


def run_ar_pipeline_custom(image_path, image_points_2d, object_width, object_height):
    """
    Run AR pipeline with custom image and points.

    Args:
        image_path: Path to your image file
        image_points_2d: 4x2 array of corner points [top-left, top-right, bottom-right, bottom-left]
        object_width: Width of object in meters
        object_height: Height of object in meters
    """
    print("=" * 60)
    print("Assignment 4: AR Pipeline with Your Custom Image")
    print("=" * 60)

    # Import libraries
    print("\n[1/7] Importing libraries...")
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import torch
    import sys
    import os
    from pytorch3d.structures import join_meshes_as_scene

    if 'src' not in sys.path:
        sys.path.insert(0, 'src')

    from src.pose_estimation import PoseEstimator, create_camera_matrix, create_planar_object_points
    from src.renderer import PyTorch3DRenderer
    from src.object_placement import ObjectPlacer
    from src.visualization import ImageCompositor, Visualizer
    from src.utils import load_image, check_torch_device, set_random_seed

    print("✓ Imports successful")

    # Load image
    print(f"\n[2/7] Loading image: {image_path}")
    background_image = cv2.imread(image_path)
    if background_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_height, image_width = background_image.shape[:2]
    print(f"✓ Image loaded: {image_width}x{image_height}")

    # Setup
    set_random_seed(42)
    device = check_torch_device()

    # Camera parameters (estimate from image size)
    focal_length = max(image_width, image_height)  # Rough estimate
    K = create_camera_matrix(focal_length, focal_length, image_width/2, image_height/2)

    object_points_3d = create_planar_object_points(object_width, object_height)

    print(f"\n[3/7] Object size: {object_width}m x {object_height}m")
    print(f"Corner points: {len(image_points_2d)} points")

    # Estimate pose
    print("\n[4/7] Estimating camera pose...")
    pose_estimator = PoseEstimator(K)
    R, t, rmse = pose_estimator.estimate_pose_solvepnp(image_points_2d, object_points_3d)
    print(f"✓ Pose estimated (RMSE: {rmse:.4f} pixels)")

    # Setup renderer
    print("\n[5/7] Setting up renderer...")
    renderer = PyTorch3DRenderer(image_size=(image_height, image_width), camera_matrix=K, device=device)
    cameras = renderer.setup_camera(R=R, t=t)
    renderer.setup_renderer(cameras=cameras, shader_type="soft")
    print("✓ Renderer ready")

    # Create objects
    print("\n[6/7] Creating 3D objects...")
    object_placer = ObjectPlacer(device=device)

    cube_mesh = object_placer.create_primitive_mesh('cube', size=0.05, color=np.array([0.8, 0.2, 0.2]))
    pyramid_mesh = object_placer.create_primitive_mesh('pyramid', size=0.06, color=np.array([0.2, 0.8, 0.2]))
    tetrahedron_mesh = object_placer.create_primitive_mesh('tetrahedron', size=0.04, color=np.array([0.2, 0.2, 0.8]))

    plane_center = np.array([object_width/2, object_height/2, 0])
    position_1 = np.array([object_width/4, object_height/4, 0])
    position_2 = np.array([3*object_width/4, 3*object_height/4, 0])

    cube_positioned = object_placer.place_on_plane(cube_mesh, plane_center, height_offset=0.025)
    pyramid_positioned = object_placer.place_on_plane(pyramid_mesh, position_1, height_offset=0.0)
    tetrahedron_positioned = object_placer.place_on_plane(tetrahedron_mesh, position_2, height_offset=0.02)

    combined_mesh = join_meshes_as_scene([cube_positioned, pyramid_positioned, tetrahedron_positioned])
    rendered_combined = renderer.render_to_numpy(combined_mesh, cameras)
    print("✓ Objects rendered")

    # Draw corner points on image
    for i, pt in enumerate(image_points_2d):
        pt_int = tuple(pt.astype(int))
        cv2.circle(background_image, pt_int, 8, (0, 0, 255), -1)
        cv2.putText(background_image, str(i+1), (pt_int[0]+10, pt_int[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw plane outline
    for i in range(len(image_points_2d)):
        pt1 = tuple(image_points_2d[i].astype(int))
        pt2 = tuple(image_points_2d[(i+1) % len(image_points_2d)].astype(int))
        cv2.line(background_image, pt1, pt2, (0, 255, 0), 2)

    # Composite
    print("\n[7/7] Creating AR result...")
    compositor = ImageCompositor()
    visualizer = Visualizer()

    ar_result = compositor.composite_images(background_image, rendered_combined, blend_mode="alpha")

    print("✓ AR compositing complete")
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Display
    visualizer.visualize_single_result(
        background=background_image,
        rendered=rendered_combined,
        title="AR Result with Your Image"
    )

    # Save
    os.makedirs('results', exist_ok=True)
    from src.visualization import save_image
    save_image(ar_result, 'results/custom_ar_result.png')
    save_image(background_image, 'results/image_with_corners.png')

    print(f"\n✓ Results saved to 'results/' folder")
    print(f"Camera Pose RMSE: {rmse:.4f} pixels")
    print("🎉 Custom image AR complete!")

    return ar_result, rendered_combined, R, t, rmse


if __name__ == "__main__":
    # Run the default AR pipeline
    result = run_ar_pipeline()
    print("\n✓ Pipeline execution complete!")

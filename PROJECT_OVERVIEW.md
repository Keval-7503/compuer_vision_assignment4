# Project Overview - Assignment 4

## 🎯 What This Project Does

This project creates **Augmented Reality (AR)** by:
1. Taking a real photo with a flat surface (like a book or table)
2. Figuring out where the camera is positioned (pose estimation)
3. Rendering 3D objects using PyTorch3D
4. Placing those 3D objects perfectly onto the real photo

---

## 🔄 Project Workflow

```
┌─────────────────────┐
│   Real Image        │
│   (Your Photo)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Detect Plane        │
│ (4 corner points)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Estimate Camera     │
│ Pose (R, t)         │ ◄─── pose_estimation.py
└──────┬──────────────┘
       │
       ├─────────────┬─────────────┬──────────────┐
       │             │             │              │
       ▼             ▼             ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Cube    │  │ Pyramid  │  │Tetrahedron│  │ Custom   │
│          │  │          │  │          │  │ Object   │
└──────┬───┘  └────┬─────┘  └────┬─────┘  └─────┬────┘
       │           │             │              │
       └───────────┴─────────────┴──────────────┘
                   │
                   ▼
       ┌─────────────────────┐
       │  Position Objects   │
       │  on Plane           │ ◄─── object_placement.py
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │  Render with        │
       │  PyTorch3D          │ ◄─── renderer.py
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │  Composite onto     │
       │  Real Image         │ ◄─── visualization.py
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │   AR Result!        │
       │   (Final Output)    │
       └─────────────────────┘
```

---

## 📁 File Purposes

### Core Modules (in `src/`)

| File | Purpose | Grading Points |
|------|---------|----------------|
| **pose_estimation.py** | Calculates camera position from plane | 20 pts |
| **renderer.py** | Sets up PyTorch3D with correct camera | 25 pts |
| **object_placement.py** | Creates & positions 3D objects | 25 pts |
| **visualization.py** | Overlays objects on real image | 20 pts |
| **utils.py** | Helper functions | - |

### Main Files

| File | Purpose |
|------|---------|
| **Assignment4_AR_PyTorch3D.ipynb** | Main notebook - run this! |
| **quick_start.py** | Test if everything works |
| **requirements.txt** | List of dependencies |

### Documentation

| File | Purpose |
|------|---------|
| **STEP_BY_STEP.md** | ⭐ Start here! Step-by-step setup |
| **QUICK_REFERENCE.md** | Quick commands cheat sheet |
| **VSCODE_SETUP.md** | Detailed VS Code setup |
| **USAGE_GUIDE.md** | How to get full marks |

---

## 🧩 How Components Work Together

### 1. Camera Pose Estimation (`pose_estimation.py`)

**Input**:
- 4 corner points of plane in image (pixels)
- 4 corner points in 3D world (meters)
- Camera intrinsics (focal length, etc.)

**Process**:
- Uses homography or solvePnP
- Computes rotation matrix (R) and translation vector (t)

**Output**:
- Camera position and orientation
- Reprojection error (RMSE)

**Example**:
```python
from src.pose_estimation import PoseEstimator

estimator = PoseEstimator(camera_matrix)
R, t, rmse = estimator.estimate_pose_solvepnp(image_points, object_points)
# R = 3x3 rotation, t = 3x1 translation
```

---

### 2. PyTorch3D Renderer (`renderer.py`)

**Input**:
- Camera matrix (K)
- Rotation and translation (R, t)
- Image size

**Process**:
- Converts OpenCV coordinates to PyTorch3D
- Sets up camera, lights, rasterizer, shader
- Configures rendering pipeline

**Output**:
- Configured renderer ready to render meshes

**Example**:
```python
from src.renderer import PyTorch3DRenderer

renderer = PyTorch3DRenderer(image_size=(720, 1280), camera_matrix=K, device=device)
cameras = renderer.setup_camera(R, t)
renderer.setup_renderer(cameras)
```

---

### 3. Object Placement (`object_placement.py`)

**Input**:
- Shape type (cube, pyramid, etc.)
- Position on plane
- Size and color

**Process**:
- Creates 3D mesh (vertices + faces)
- Applies transformations (translate, rotate, scale)
- Positions on detected plane

**Output**:
- PyTorch3D mesh ready to render

**Example**:
```python
from src.object_placement import ObjectPlacer

placer = ObjectPlacer(device)
cube = placer.create_primitive_mesh('cube', size=0.05, color=[0.8, 0.2, 0.2])
positioned = placer.place_on_plane(cube, plane_center, height_offset=0.03)
```

---

### 4. Visualization (`visualization.py`)

**Input**:
- Real image (background)
- Rendered synthetic image (foreground)

**Process**:
- Uses alpha blending to composite
- Creates side-by-side comparisons
- Draws coordinate axes

**Output**:
- Final AR image
- Visualization figures

**Example**:
```python
from src.visualization import ImageCompositor

compositor = ImageCompositor()
ar_result = compositor.composite_images(real_image, rendered_image)
```

---

## 🎓 Grading Breakdown

```
Total: 100 points

├── Camera Pose Estimation (20 pts)
│   ├── Correct R and t matrices        [10 pts]
│   ├── Low reprojection error          [5 pts]
│   └── Both methods implemented        [5 pts]
│
├── Rendering Setup (25 pts)
│   ├── Correct camera parameters       [10 pts]
│   ├── Proper coordinate conversion    [8 pts]
│   └── Image size alignment            [7 pts]
│
├── Synthetic Object Integration (25 pts)
│   ├── Objects render correctly        [10 pts]
│   ├── Proper placement on plane       [10 pts]
│   └── Excellent alignment             [5 pts]
│
├── Results & Visualization (20 pts)
│   ├── Multiple result images          [8 pts]
│   ├── High-quality visualizations     [7 pts]
│   └── Discussion of limitations       [5 pts]
│
└── Code Quality & Notebook (10 pts)
    ├── Runs end-to-end                 [5 pts]
    └── Well-documented                 [5 pts]
```

---

## 🔑 Key Concepts

### 1. Camera Intrinsics (K matrix)
```
    [fx  0  cx]
K = [ 0 fy  cy]
    [ 0  0   1]

fx, fy = focal lengths
cx, cy = principal point (image center)
```

### 2. Camera Extrinsics (R, t)
```
R = 3x3 rotation matrix (camera orientation)
t = 3x1 translation vector (camera position)

Together they define where the camera is in 3D space
```

### 3. Coordinate Systems

**OpenCV** (what we use for pose):
- +X = right
- +Y = down
- +Z = forward (into scene)

**PyTorch3D** (what we use for rendering):
- +X = left
- +Y = up
- +Z = into scene

We convert between them in `renderer.py`!

### 4. Homogeneous Coordinates
```
2D point: [x, y]     → [x, y, 1]
3D point: [x, y, z]  → [x, y, z, 1]

Allows matrix operations for transformations
```

---

## 🚀 Quick Start Summary

1. **Install everything**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install torch torchvision pytorch3d
   pip install -r requirements.txt
   ```

2. **Test setup**:
   ```bash
   python quick_start.py
   ```

3. **Run notebook**:
   - Open `Assignment4_AR_PyTorch3D.ipynb` in VS Code
   - Select kernel (venv)
   - Run All

4. **Check results**:
   - Look in `results/` folder
   - See visualizations in notebook

---

## 🎨 Customization Examples

### Change Object Color
```python
red_cube = placer.create_primitive_mesh('cube', size=0.05, color=[1.0, 0.0, 0.0])
```

### Change Object Position
```python
custom_position = np.array([0.1, 0.15, 0])  # x, y, z in meters
positioned = placer.place_on_plane(mesh, custom_position, height_offset=0.02)
```

### Change Object Size
```python
big_pyramid = placer.create_primitive_mesh('pyramid', size=0.1)  # 10cm
```

### Use Custom 3D Model
```python
mesh = placer.load_mesh('models/bunny.obj')
```

---

## 📊 Data Flow Example

```
Input:
  - image.jpg (1280x720 pixels)
  - 4 corner points: [(200,150), (800,200), (850,600), (150,550)]
  - Object: 0.21m × 0.297m (A4 paper)

↓

Pose Estimation:
  - R = [[0.98, -0.15, 0.12], [...], [...]]
  - t = [0.05, 0.03, 0.5]  # 0.5m away from camera
  - RMSE = 2.3 pixels

↓

Create Object:
  - Cube: 5cm, red color
  - Position: center of paper, 3cm above

↓

Render:
  - PyTorch3D renders with camera pose
  - Output: 1280x720 RGBA image

↓

Composite:
  - Blend rendered object onto real image
  - Use alpha channel for transparency

↓

Result:
  - AR image with cube floating above paper!
```

---

## 💡 Tips for Success

1. **Start with the default example** - Get it working first
2. **Use real images** - More impressive results
3. **Check each step** - Run cells one at a time
4. **Verify pose** - RMSE should be < 5 pixels
5. **Experiment** - Try different objects, positions, colors

---

## 🔗 Learning Resources

- **PyTorch3D Tutorial**: https://pytorch3d.org/tutorials/
- **Camera Calibration**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- **Homography**: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- **3D Rendering**: https://github.com/ribeiro-computer-vision/pytorch3d_rendering

---

## 📝 Next Steps

1. ✅ **Setup**: Follow [STEP_BY_STEP.md](STEP_BY_STEP.md)
2. ✅ **Run**: Execute notebook end-to-end
3. ✅ **Customize**: Add your own images
4. ✅ **Verify**: Check grading criteria
5. ✅ **Submit**: Push to GitHub and submit

---

**Ready to dive in? Start with [STEP_BY_STEP.md](STEP_BY_STEP.md)!** 🚀

# 📷 How to Use Your Own Images

## Quick Guide: 3 Steps

### Step 1: Upload Your Image to Colab

In Colab, run this cell:

```python
from google.colab import files
uploaded = files.upload()
```

Click "Choose Files" and upload your image (JPG, PNG, etc.)

---

### Step 2: Click Corner Points

Use this interactive tool to click the 4 corners of a planar object in your image:

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load your image
image_path = list(uploaded.keys())[0]  # Gets the uploaded filename
image = cv2.imread(image_path)

print(f"Image loaded: {image.shape}")
print("Instructions:")
print("1. Note the 4 corners of your planar object (book, paper, door, etc.)")
print("2. Write down the pixel coordinates")
print("")
print("Display your image below to see coordinates:")

# Display image
cv2_imshow(image)

print("\nNow manually enter the 4 corner points you see:")
print("Format: Top-left, Top-right, Bottom-right, Bottom-left")
```

---

### Step 3: Run Pipeline with Your Image

```python
from run_ar_pipeline import run_ar_pipeline_custom
import numpy as np

# Your corner points (replace with actual coordinates from your image)
image_points_2d = np.array([
    [100, 50],    # Top-left corner
    [500, 60],    # Top-right corner
    [490, 400],   # Bottom-right corner
    [110, 390]    # Bottom-left corner
], dtype=np.float32)

# Size of your planar object in real world (in meters)
object_width = 0.21   # Example: A4 paper width (21cm)
object_height = 0.297  # Example: A4 paper height (29.7cm)

# Run pipeline with your image
run_ar_pipeline_custom(
    image_path=image_path,
    image_points_2d=image_points_2d,
    object_width=object_width,
    object_height=object_height
)
```

---

## 📖 Detailed Example

### Example: Using a Book Cover

```python
# 1. Upload your book image
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# 2. Define the 4 corners of the book
# (Look at your image and note the pixel coordinates)
image_points_2d = np.array([
    [150, 200],   # Top-left
    [650, 180],   # Top-right
    [680, 550],   # Bottom-right
    [120, 570]    # Bottom-left
], dtype=np.float32)

# 3. Measure your book (in meters)
book_width = 0.15   # 15cm
book_height = 0.23  # 23cm

# 4. Run pipeline
from run_ar_pipeline import run_ar_pipeline_custom
run_ar_pipeline_custom(image_path, image_points_2d, book_width, book_height)
```

---

## 🔧 Interactive Point Picker (Advanced)

For easier point selection, use this interactive tool:

```python
# Click points directly on image
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from IPython.display import display, HTML

# Load image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
display_image = image.copy()

points = []

def click_point(x, y):
    """Click to add points"""
    points.append([x, y])
    cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(display_image, str(len(points)), (x+10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return display_image

print("Click 4 corners in this order:")
print("1. Top-left")
print("2. Top-right")
print("3. Bottom-right")
print("4. Bottom-left")

# Display image with click coordinates shown
cv2_imshow(image)

# After viewing, manually enter coordinates
print("\nEnter the pixel coordinates you want to use:")
```

---

## 📝 Common Object Sizes

Use these for reference:

| Object | Width (m) | Height (m) |
|--------|-----------|------------|
| A4 Paper | 0.210 | 0.297 |
| Letter Paper | 0.216 | 0.279 |
| Book (typical) | 0.15 | 0.23 |
| Laptop Screen 15" | 0.33 | 0.21 |
| Door (typical) | 0.90 | 2.10 |
| Credit Card | 0.086 | 0.054 |

---

## 🎯 Tips for Good Results

### 1. **Choose a Good Planar Object**
   - ✅ Book cover
   - ✅ Paper on desk
   - ✅ Laptop keyboard area
   - ✅ Whiteboard
   - ✅ Picture frame
   - ❌ Curved surfaces

### 2. **Take a Good Photo**
   - Clear, well-lit
   - Object is visible and not too tilted
   - All 4 corners are visible
   - High resolution (at least 640x480)

### 3. **Click Points Accurately**
   - Click exactly at the corners
   - Follow the order: TL, TR, BR, BL
   - Double-check your coordinates

### 4. **Measure Your Object**
   - Use a ruler or measuring tape
   - Measure in centimeters, convert to meters
   - Be as accurate as possible

---

## 🚀 Complete Colab Example

Here's a complete notebook cell you can use:

```python
# STEP 1: Upload image
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# STEP 2: Display image to see coordinates
import cv2
from google.colab.patches import cv2_imshow
image = cv2.imread(image_path)
cv2_imshow(image)
print(f"Image size: {image.shape[1]}x{image.shape[0]} (width x height)")

# STEP 3: Define your corner points
import numpy as np
image_points_2d = np.array([
    [150, 100],   # Top-left - CHANGE THESE!
    [500, 120],   # Top-right
    [480, 400],   # Bottom-right
    [130, 380]    # Bottom-left
], dtype=np.float32)

# STEP 4: Define object size (in meters)
object_width = 0.21   # CHANGE THIS!
object_height = 0.297 # CHANGE THIS!

# STEP 5: Run AR pipeline
from run_ar_pipeline import run_ar_pipeline_custom
result = run_ar_pipeline_custom(
    image_path=image_path,
    image_points_2d=image_points_2d,
    object_width=object_width,
    object_height=object_height
)
```

---

## ❓ FAQ

**Q: How do I find pixel coordinates?**
A: After running `cv2_imshow(image)`, hover your mouse over the image. The coordinates appear at the bottom. Or use image editing software like Paint.

**Q: What if my image is too large?**
A: Resize it first:
```python
image = cv2.imread(image_path)
image = cv2.resize(image, (1280, 720))
cv2.imwrite('resized.jpg', image)
image_path = 'resized.jpg'
```

**Q: Can I use multiple objects?**
A: Yes! Just run the pipeline multiple times with different corner points.

**Q: Objects look wrong/misaligned?**
A: Check:
- Corner points are accurate
- Points are in correct order (TL, TR, BR, BL)
- Object measurements are correct
- Camera intrinsics match your image

---

Need help? See **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** for more details!

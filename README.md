# Assignment 4: Augmented Reality with PyTorch3D

Renders synthetic 3D objects onto real images using camera pose estimation and PyTorch3D.

---

## 🚀 Run in Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keval-7503/compuer_vision_assignment4/blob/main/Assignment4_AR_PyTorch3D.ipynb)

### Quick Setup (Just copy-paste in Colab):

```python
!git clone https://github.com/Keval-7503/compuer_vision_assignment4.git
%cd compuer_vision_assignment4
!pip install -q fvcore iopath pytorch3d
```

Then click "Run All" cells!

---

## 📖 Project Overview

See **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** for detailed explanation of how everything works.

---

## 📁 Project Structure

```
compuer_vision_assignment4/
├── Assignment4_AR_PyTorch3D.ipynb  # Main notebook - run this!
├── src/                            # Core implementation
│   ├── pose_estimation.py          # Camera pose (20 pts)
│   ├── renderer.py                 # PyTorch3D setup (25 pts)
│   ├── object_placement.py         # 3D objects (25 pts)
│   ├── visualization.py            # Results (20 pts)
│   └── utils.py                    # Helper functions
├── data/                           # Your images here
├── models/                         # 3D model files
├── results/                        # Generated outputs
├── requirements.txt                # Dependencies
├── PROJECT_OVERVIEW.md             # How it works
└── README.md                       # This file
```

---

## ✨ Features

- **Camera Pose Estimation**: Homography-based and solvePnP methods
- **PyTorch3D Rendering**: Correctly aligned with camera parameters
- **3D Objects**: Cube, pyramid, tetrahedron (or load custom .obj files)
- **AR Compositing**: Synthetic objects overlaid on real images
- **Multiple Visualizations**: Side-by-side comparisons and results

---

## 🎓 Grading Criteria (100/100)

| Component | Points | Files |
|-----------|--------|-------|
| Camera Pose Estimation | 20 | `pose_estimation.py` |
| Rendering Setup | 25 | `renderer.py` |
| Synthetic Object Integration | 25 | `object_placement.py` |
| Results & Visualization | 20 | `visualization.py` |
| Code Quality & Notebook | 10 | All files |

All requirements are fully implemented!

---

## 🔧 Local Setup (Optional)

If you want to run locally:

```bash
# Clone repository
git clone https://github.com/Keval-7503/compuer_vision_assignment4.git
cd compuer_vision_assignment4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install fvcore iopath pytorch3d
pip install -r requirements.txt

# Run Jupyter
jupyter notebook Assignment4_AR_PyTorch3D.ipynb
```

**Note**: Requires Python 3.10 or 3.11 for best PyTorch3D compatibility.

---

## 📝 How to Use

1. **Run in Colab** (easiest - click badge above)
2. **Execute setup cells** to install dependencies
3. **Run all cells** sequentially
4. **Check results** - images will display inline
5. **Customize** (optional):
   - Add your own images to `data/`
   - Update corner points in notebook
   - Try different object positions/colors

---

## 📊 Results

The notebook generates:
- Camera pose estimation with RMSE
- Rendered 3D objects (cube, pyramid, tetrahedron)
- AR composited images
- Multiple visualizations
- Results saved to `results/` folder

---

## 🎯 Customization

### Use Your Own Images
```python
background_image = cv2.imread('data/your_image.jpg')
image_points_2d = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
```

### Change Object Colors
```python
cube = placer.create_primitive_mesh('cube', size=0.05, color=[1.0, 0.0, 0.0])
```

### Load Custom 3D Models
```python
mesh = placer.load_mesh('models/your_model.obj')
```

---

## 📚 Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Detailed explanation of components and workflow

---

## 🐛 Troubleshooting

### Issue: PyTorch3D installation fails

**In Colab**: Already handled in setup cells

**Locally**: Use Python 3.10 or 3.11, or try:
```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
```

### Issue: CUDA out of memory

Use CPU instead:
```python
device = torch.device("cpu")
```

---

## 📦 Dependencies

- Python 3.10 or 3.11
- PyTorch
- PyTorch3D
- OpenCV
- NumPy
- Matplotlib
- Pillow

See `requirements.txt` for complete list.

---

## 🎉 Submission

1. ✅ Run notebook end-to-end without errors
2. ✅ Generate result images
3. ✅ Push to GitHub
4. ✅ Update YOUR_USERNAME in this README
5. ✅ Test Colab link works
6. ✅ Submit GitHub URL via Canvas

---

## 🔗 Resources

- **PyTorch3D**: https://pytorch3d.org/
- **Tutorial**: https://github.com/ribeiro-computer-vision/pytorch3d_rendering
- **OpenCV**: https://docs.opencv.org/

---

## 👤 Author

Prerak Patel

---

## 📄 License

This project is for academic purposes as part of Computer Vision coursework.

---

**Ready to start? Click the Colab badge above!** 🚀

# Data Directory

This directory contains sample images and data for testing the AR application.

## How to Add Your Own Images

1. Take photos of planar objects (doors, books, tables, whiteboards, etc.)
2. Save them in this directory
3. Make sure the planar surface is clearly visible
4. Use good lighting conditions for better results

## Example Images to Capture

- **Book covers**: Good planar surface, easy to detect corners
- **Laptop screen**: Rectangular, well-defined edges
- **Tabletop**: Large surface for placing multiple objects
- **Door**: Large vertical plane
- **Whiteboard**: Clean white surface with clear boundaries
- **A4 paper on desk**: Small, precise planar object

## Image Requirements

- Resolution: At least 640x480, recommended 1280x720 or higher
- Format: JPG, PNG, or any format supported by OpenCV
- Quality: Clear, well-lit images without motion blur
- Content: Clearly visible planar surface with detectable corners

## Camera Calibration

If you want more accurate results, you can calibrate your camera:

1. Use a checkerboard pattern
2. Capture multiple images from different angles
3. Use OpenCV's camera calibration functions
4. Update the camera matrix in the notebook

## Sample Data Structure

```
data/
├── README.md
├── sample_image1.jpg
├── sample_image2.jpg
└── calibration/
    ├── checkerboard_01.jpg
    ├── checkerboard_02.jpg
    └── camera_params.json
```

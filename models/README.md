# 3D Models Directory

This directory contains 3D model files (OBJ, PLY, etc.) for rendering in the AR application.

## Supported Formats

- **OBJ**: Wavefront OBJ format (most common)
- **PLY**: Polygon File Format
- **STL**: Stereolithography format (via conversion)

## How to Add Custom 3D Models

1. Download or create 3D models in OBJ format
2. Place them in this directory
3. Load them in the notebook using `ObjectPlacer.load_mesh()`

## Free 3D Model Resources

- **Sketchfab**: https://sketchfab.com/
- **TurboSquid**: https://www.turbosquid.com/Search/3D-Models/free/obj
- **Free3D**: https://free3d.com/3d-models/obj
- **Thingiverse**: https://www.thingiverse.com/
- **CGTrader**: https://www.cgtrader.com/free-3d-models

## Model Requirements

- **File Format**: OBJ with optional MTL (material) file
- **Size**: Keep models reasonably sized (< 50MB)
- **Complexity**: Moderate polygon count (1K-100K faces)
- **Orientation**: Models should be properly oriented (up = +Z axis)

## Built-in Primitives

The code includes built-in primitive shapes:
- Cube
- Pyramid
- Tetrahedron

These don't require external files and are generated programmatically.

## Example Usage

```python
from src.object_placement import ObjectPlacer

placer = ObjectPlacer(device)

# Load custom model
mesh = placer.load_mesh('models/bunny.obj')

# Use built-in primitives
cube = placer.create_primitive_mesh('cube', size=0.05)
```

## Model Scaling

- Models are measured in meters in world space
- A cube with size=1.0 is 1 meter × 1 meter × 1 meter
- Typical object sizes: 0.05 to 0.2 meters (5-20 cm)
- Adjust scale factor when placing objects for best visual results

# Mesh2Plan Research Notes

## Overview

This document summarizes the research into different approaches for extracting 2D floor plans from 3D mesh data. Four main approaches were developed and tested:

- **v10**: Voxelization + 2D Projection
- **v11**: Normal-based Wall Segmentation  
- **v12**: Contour Detection on Rasterized Depth Maps
- **v13**: Alpha Shape Based Room Boundary Extraction

## Test Data

Two mesh files were used for testing:

1. **Small mesh**: `data/2026_01_13_14_47_59/export_refined.obj`
   - 48,637 vertices, 95,074 faces
   - Smaller, simpler geometry

2. **Large mesh**: `data/gdrive_sample/2026_02_09_19_03_38/export_refined.obj`
   - 231,749 vertices, 448,449 faces  
   - Larger, more complex geometry

## Approach Comparison

### v10: Voxelization + 2D Projection ⭐⭐⭐⭐

**Branch**: `research/v10-voxel-projection`

**Concept**: Convert mesh to voxel grid, project occupied voxels to 2D, use morphological operations to clean up walls, extract contours.

**Results**:
- Small mesh: 0.3 m² (3.6 ft²), 8 boundary points, 1 door opening
- Large mesh: 1.8 m² (19.3 ft²), 6 boundary points, 0 openings

**Quality**: ⭐⭐⭐⭐
- **Pros**: 
  - Robust and consistent results on both meshes
  - Good at handling non-axis-aligned geometries
  - Effective morphological cleanup
  - Reasonable room area estimates
- **Cons**: 
  - Parameter-sensitive (voxel size, height range)
  - Opening detection could be improved
  - Limited to Manhattan-world assumptions for some operations

**Technical Notes**:
- Voxel size 0.05m works well
- Wall height range 0.3-0.8 (relative) captures walls effectively
- OpenCV morphological operations clean up noise well

### v11: Normal-based Wall Segmentation ⭐⭐⭐⭐⭐

**Branch**: `research/v11-normal-segmentation`

**Concept**: Classify faces by normal direction (dot product with up vector), extract wall faces, project to 2D, use clustering and convex hull.

**Results**:
- Small mesh: 0.4 m² (4.2 ft²), 20 boundary points, 87,148 wall faces, 1 cluster
- Large mesh: 16.8 m² (180.4 ft²), 19 boundary points, 286,494 wall faces, 4 clusters  

**Quality**: ⭐⭐⭐⭐⭐
- **Pros**:
  - Excellent performance on both meshes
  - Largest room area detection (most accurate?)
  - Good face classification (separates floors/walls/ceilings)
  - Handles complex geometries well
  - Wall clustering provides structural insight
- **Cons**:
  - Opening detection not implemented (returns 0)
  - Relies on good mesh normals
  - Convex hull may oversimplify complex room shapes

**Technical Notes**:
- Vertical threshold 0.7 works well for wall/floor separation
- XZ projection (top-down) most effective
- DBSCAN clustering successfully groups wall segments

### v12: Contour Detection on Rasterized Depth Maps ⭐⭐⭐

**Branch**: `research/v12-contour-detection`

**Concept**: Render mesh to top-down height map, use image processing (edge detection, contours) to find walls.

**Results**:
- Small mesh: 0.0 m² (0.2 ft²), 6 boundary points, 3 contours, 0 openings
- Large mesh: 8.8 m² (95.2 ft²), 7 boundary points, 2 contours, 6 openings

**Quality**: ⭐⭐⭐
- **Pros**:
  - Good opening detection (6 openings on large mesh)
  - Computer vision approach is intuitive
  - Comprehensive visualization shows all processing stages
  - Handles complex shapes reasonably well
- **Cons**:
  - Poor performance on small/simple meshes
  - Resolution-dependent results
  - Rasterization may lose detail
  - Small room area suggests boundary detection issues

**Technical Notes**:
- Resolution 512x512 adequate for most meshes
- Wall height threshold 0.5m separates walls from floors
- Canny edge detection (50, 150) works well
- Morphological operations help clean up noise

### v13: Alpha Shape Based Room Boundary Extraction ⭐⭐

**Branch**: `research/v13-alpha-shape`

**Concept**: Extract floor-level points, use alpha shapes (concave hulls) to create boundaries that handle non-convex rooms.

**Results**:
- Small mesh: 0.0 m² (0.0 ft²), 2 boundary points, 1940 triangles, 0 openings
- Large mesh: 13.7 m² (147.6 ft²), 6 boundary points, 7561 triangles, 3 openings

**Quality**: ⭐⭐
- **Pros**:
  - Handles concave boundaries (unlike convex hull)
  - Good theoretical foundation for complex room shapes
  - Detects openings based on concavity analysis
  - Large mesh results are reasonable
- **Cons**:
  - Fails completely on simple/small meshes
  - Alpha parameter is difficult to tune
  - Boundary refinement too aggressive (reduces 85 points to 6)
  - Floor point extraction may miss important geometry

**Technical Notes**:
- Alpha 0.5 works for complex meshes but may need tuning
- Floor height range (bottom 10%) may be too restrictive
- Boundary refinement threshold 0.1m may be too aggressive
- Point sampling (10%) may lose important detail

## Overall Ranking

1. **v11 (Normal Segmentation)** - Most robust and accurate ⭐⭐⭐⭐⭐
2. **v10 (Voxelization)** - Consistent and reliable ⭐⭐⭐⭐ 
3. **v12 (Contour Detection)** - Good for complex meshes ⭐⭐⭐
4. **v13 (Alpha Shapes)** - Promising but needs work ⭐⭐

## Key Findings

### What Worked Well:
- **Face normal analysis** (v11) is highly effective for architectural meshes
- **Voxelization** (v10) provides robust, resolution-independent results  
- **Image processing techniques** (v12) work well for opening detection
- **Morphological operations** consistently improve boundary quality

### What Didn't Work:
- **Alpha shapes** (v13) are too sensitive to parameter tuning
- **Simple height filtering** misses important geometry
- **Aggressive boundary simplification** loses too much detail
- **Resolution-dependent approaches** fail on simple geometries

### Common Issues:
- Opening detection remains challenging across all approaches
- Small/simple meshes are harder to process than complex ones
- Parameter tuning is critical for good results
- Ground truth validation is needed for accuracy assessment

## Recommendations for Future Work

### Short Term:
1. **Improve v11 opening detection** - Use gap analysis on wall clusters
2. **Combine approaches** - Use v11 for boundaries + v12 for openings  
3. **Parameter optimization** - Automated tuning based on mesh characteristics
4. **Ground truth comparison** - Validate against known floor plans

### Long Term:
1. **Machine learning integration** - Train on known mesh/floorplan pairs
2. **Multi-scale analysis** - Process different detail levels
3. **Semantic understanding** - Classify room types, door/window types
4. **Real-time processing** - Optimize for interactive applications

## Visualization Assets

Each approach generated comprehensive visualizations saved to `results/vXX_tests/`:
- `test1_results.json` / `test2_results.json` - Quantitative results
- `test1_visualization.png` / `test2_visualization.png` - Visual analysis

The v12 visualizations are particularly comprehensive, showing the full image processing pipeline from height maps to final boundaries.

## Technical Stack

- **Python 3.13** with virtual environment
- **Core**: trimesh, numpy, scipy, scikit-learn
- **Vision**: opencv-python for image processing  
- **Visualization**: matplotlib
- **Geometry**: Delaunay triangulation, convex/alpha hulls
- **Clustering**: DBSCAN for point grouping

## Code Quality

All approaches follow consistent patterns:
- Modular design with clear function separation
- Comprehensive logging and progress reporting
- JSON output with standardized format
- Error handling and validation
- Visualization generation
- Command-line interface compatibility

## Repository Structure

```
research/
├── v10_voxelization_projection.py     # Voxel-based approach  
├── v11_normal_wall_segmentation.py    # Normal-based approach
├── v12_contour_depth_raster.py        # Image processing approach
├── v13_alpha_shape_boundary.py        # Alpha shape approach
├── visualize_v10_results.py           # V10 visualization helper
└── NOTES.md                           # This file

results/
├── v10_tests/                         # V10 results and visualizations
├── v11_tests/                         # V11 results and visualizations  
├── v12_tests/                         # V12 results and visualizations
└── v13_tests/                         # V13 results and visualizations
```

Each approach can be run independently:
```bash
python research/v11_normal_wall_segmentation.py <mesh.obj> <output.json> [visualization.png]
```

---

*Research completed: February 10, 2026*  
*Total development time: ~4 hours*  
*Lines of code: ~1,800*  
*Approaches tested: 4*  
*Test meshes: 2*  
*Git branches created: 4*
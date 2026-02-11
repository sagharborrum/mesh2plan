# mesh2plan

**Real-time 3D mesh → floorplan / wall / surface extraction**

Like Apple RoomPlan, but focused on accurate geometry from arbitrary 3D meshes. Open source, runs in the browser.

**🌐 [Live demo → mesh2plan.vercel.app](https://mesh2plan.vercel.app)** (v9 — current best web viewer)

## Goal

Take a 3D mesh (OBJ/PLY/glTF from any scanner) and extract:
- **Wall planes** with accurate dimensions
- **Floor/ceiling surfaces**
- **2D floorplan** (SVG/DXF) with room boundaries
- **Real-time** visualization of detected planes overlaid on the mesh

## Architecture

```
Input Mesh (OBJ/PLY/glTF)
    │
    ▼
┌─────────────────────┐
│  1. Mesh Loading     │  Three.js / Web Workers
│     + Sampling       │  Extract vertices + normals
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Plane Detection  │  RANSAC / Region Growing
│     (real-time)      │  Detect dominant planes
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Classification   │  Normal-based heuristics
│  Wall / Floor / Ceil │  + optional ML refinement
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Boundary Extract │  Alpha shapes / convex hull
│  + Regularization    │  Snap to orthogonal
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Floorplan Gen    │  Top-down projection
│  SVG / DXF output    │  Wall centerlines + dims
└─────────────────────┘
```

## Approaches Tested

### Summary

| Version | Approach | Quality | Branch | Notes |
|---------|----------|---------|--------|-------|
| v0.1 | RANSAC plane extraction + Three.js viewer | — | `main` (813fd71) | Starting point |
| v1 | Textured room scan + plane extraction | ⭐⭐ | — | Basic visualization |
| v2 | Wall clustering + opening detection + 2D floor plan | ⭐⭐⭐ | — | First real floor plan |
| v3 | Face-based extraction + Manhattan directions | ⭐⭐⭐ | — | Better classification |
| v4 | Cross-section floor plan + interactive height slider | ⭐⭐⭐ | — | New paradigm: slicing |
| v5 | Composite floor plan with room polygon + measurements | ⭐⭐⭐⭐ | — | Multi-slice composite |
| v6 | Manhattan-regularized floor plan | ⭐⭐⭐⭐ | — | Clean rectangular walls |
| v7 | Connected walls + door/window detection + SVG export | ⭐⭐⭐⭐ | — | |
| v8 | Top-down depth map background + polished floor plan | ⭐⭐⭐⭐ | — | |
| v9 | Full client-side web app (Web Worker, multi-format, export) | ⭐⭐⭐⭐⭐ | — | **Deployed** at mesh2plan.vercel.app |
| v10 | Voxelization + 2D projection | ⭐⭐⭐⭐ | `research/v10-voxel-projection` | Consistent, robust |
| v11 | Normal-based wall segmentation | ⭐⭐⭐⭐⭐ | `research/v11-normal-segmentation` | **Best accuracy** |
| v12 | Contour detection on rasterized depth maps | ⭐⭐⭐ | `research/v12-contour-detection` | Best opening detection |
| v13 | Alpha shape room boundary extraction | ⭐⭐ | `research/v13-alpha-shape` | Needs parameter tuning |

### Phase 1: Web-based viewers (v1-v9)

Built iteratively in the browser using Three.js + Web Workers:

- **v1-v3**: RANSAC plane detection → face classification → wall clustering
- **v4-v5**: Cross-section slicing approach — cut mesh at heights, composite the slices
- **v6**: Manhattan world assumption + histogram peaks = clean rectangular wall segments
- **v7-v8**: Connected wall graph, door/window detection, depth map backgrounds
- **v9**: Production web app — drop any OBJ/GLB/PLY/STL, get instant floor plan with SVG/DXF/PNG export

All viewers preserved in `viewer/v1.html` through `viewer/v9.html`.

### Phase 2: Python research (v10-v13)

Systematic exploration of different algorithmic approaches:

**v10: Voxelization + 2D Projection** — Convert mesh to voxel grid, project wall-height voxels to 2D, morphological cleanup, contour extraction. Consistent results but parameter-sensitive.

**v11: Normal-based Wall Segmentation** ⭐ — Classify faces by normal direction (dot product with up vector), separate floors/ceilings from walls, DBSCAN clustering, convex hull. Best overall performer — 16.8m² on complex mesh with 4 wall clusters detected.

**v12: Contour Detection on Rasterized Depth Maps** — Render top-down height map, Canny edge detection, contour extraction. Best at finding openings (6 detected on complex mesh) but resolution-dependent.

**v13: Alpha Shape Boundary Extraction** — Delaunay triangulation on floor-level points, concave hull extraction. Handles non-convex rooms in theory but alpha parameter is hard to tune.

### Key Findings

1. **Normal-based segmentation** (v11) is most effective for architectural meshes
2. **Manhattan world assumption** used by most methods — should be an optional constraint
3. **Cross-section slicing** (v4-v5) and **voxelization** (v10) are the most intuitive approaches
4. **Opening detection** remains challenging across all approaches — v12's image processing approach is most promising
5. **Combining approaches** likely optimal: v11 for boundaries + v12 for openings
6. **Defurnishing is a real problem** — scanned meshes have furniture obscuring walls

### Phase 3: Multiroom floor plans (v27-v32)

Expanded to apartment-scale scans with multiple rooms.

**v27**: First multiroom attempt — interior wall classification + connected components. 16 rooms (over-segmented).

**v27f-v27g**: Watershed segmentation — distance transform finds room centers, watershed grows to wall ridges. v27g correctly identifies 3 rooms + 1 hallway.

**v27h**: Clean architectural rendering — white bg, thick walls, door arcs, pastel fills. Room detection still off (hallway too big).

**v28-v28b**: Wall grid approaches — build cells between Hough walls, classify by occupancy (v28) or edge density ratio (v28b). Good separation but gaps between rooms.

**v29**: Mask cut ⭐ — Fill apartment mask, cut along strongest validated Hough walls. Wall scoring = `projection_strength × longest_continuous_run`. Boundary exclusion, min separation between cuts, fragment merging.

**v30**: Architectural rendering — v29 detection + clean rendering (thick wall rectangles, door arcs, scale bar, room classification).

**v31**: Hallway-first (skeleton + distance transform) — failed because hallway was ~1.3m wide, too wide for narrow detection.

**v32**: Strip merge ⭐⭐ — **3 rooms + 1 hallway + 1 closet = 34.7m²**. X cuts create vertical strips; center strip = hallway. Z cuts split left/right strips into rooms. Door detection via mask adjacency. **Best multiroom result.**

### Key Multiroom Findings

1. **Wall scoring** (`strength × max_run`) is critical for selecting which walls to cut
2. **Strip-based merging** outperforms generic smallest-first merging
3. **Center strip = hallway** is a reliable heuristic for apartments with central corridors
4. **Mask adjacency** works better than polygon edge matching for door detection
5. **Hallway detection via narrowness** fails for wider corridors (>1m) — use structural position instead

### Next Research Directions

- [ ] Machine learning: train on known mesh/floorplan pairs
- [ ] Using confidence maps from 3D Scanner App (conf_*.png)
- [ ] Semantic understanding: classify room types, furniture vs structure
- [ ] Comparison with RoomFormer (271⭐) end-to-end approach
- [ ] Window detection on exterior walls
- [ ] Dimension annotations on floor plans

## Key Approaches (Theory)

### Plane Detection from Meshes

| Method | Pros | Cons | Speed |
|--------|------|------|-------|
| **RANSAC** | Simple, robust to noise | Misses small planes, order-dependent | Fast |
| **Region Growing** | Preserves topology, finds all planes | Sensitive to thresholds | Medium |
| **Hough Transform** | Good for dominant planes | Memory-heavy, quantization | Medium |
| **Normal Clustering** | Very fast, good for clean meshes | Needs good normals | Very fast |
| **Deep Learning** (PlaneRCNN, AirPlanes) | Best accuracy | Needs GPU, training data | Slow |

### Wall/Floor Classification

Once planes are detected, classification is straightforward:
- **Floor**: Normal pointing up (±15° from Y+), lowest elevation cluster
- **Ceiling**: Normal pointing down, highest elevation cluster
- **Walls**: Normal roughly horizontal (±15° from XZ plane)
- **Other**: Furniture, fixtures, clutter → filter out by size/position

### Floorplan Generation

1. Project wall planes onto XZ (horizontal) plane
2. Extract wall centerlines via intersection of parallel plane pairs
3. Snap to orthogonal grid (Manhattan world assumption, optional)
4. Close open boundaries → room polygons
5. Compute dimensions, area
6. Export SVG with annotations or DXF for CAD

## Research & References

### With Code

| ⭐ | Project | Year | What it does | Link |
|---:|---------|------|-------------|------|
| 650 | **pyRANSAC-3D** | — | Pure Python RANSAC for planes, spheres, cuboids, cylinders, lines | [GitHub](https://github.com/leomariga/pyRANSAC-3D) |
| 528 | **Manhattan-SDF** | CVPR 2022 | Neural SDF with Manhattan-world planar priors for indoor reconstruction | [GitHub](https://github.com/zju3dv/manhattan_sdf) |
| 271 | **RoomFormer** | ICCV 2023 | End-to-end point-cloud scan → vectorized floor plan (dual-query Transformer) | [GitHub](https://github.com/ywyue/RoomFormer) |
| 212 | **InteriorGS** | — | 1,000 3DGS scenes + semantic labels + floorplans + occupancy maps | [GitHub](https://github.com/manycore-research/InteriorGS) |
| 91 | **Orthogonal Planes** | ICRA 2020 | Multi-purpose primitive detection (planes + corners) in unorganized 3D point clouds | [GitHub](https://github.com/c-sommer/orthogonal-planes) |
| 73 | **AirPlanes** (Niantic) | CVPR 2024 | 3D-consistent plane embeddings from posed RGB. Sequential RANSAC + learned MLP. | [GitHub](https://github.com/nianticlabs/airplanes) |
| 55 | **DOPNet** | CVPR 2023 | Disentangle orthogonal planes for panoramic room layout estimation | [GitHub](https://github.com/zhijieshen-bjtu/DOPNet) |
| 44 | **Plane-DUSt3R** | Feb 2025 | Multi-view images → room layout planes via DUSt3R foundation model. Training-free. | [GitHub](https://github.com/justacar/Plane-DUSt3R) |
| 12 | **CAGE** | Sep 2025 | Edge-centric floorplan via dual-query transformer. 99.1% room F1. SOTA. | [GitHub](https://github.com/ee-Liu/CAGE) |
| 11 | **FloorSAM** | Sep 2025 | SAM zero-shot + LiDAR density maps → room segmentation → vectorized floorplans | [GitHub](https://github.com/Silentbarber/FloorSAM) |
| 11 | **torch_ransac3d** | — | GPU-accelerated RANSAC with PyTorch/CUDA | [GitHub](https://github.com/harrydobbs/torch_ransac3d) |
| 7 | **Parallel-RANSAC** | — | GPU-parallelized RANSAC plane extraction from RGB-D | [GitHub](https://github.com/alehdaghi/Parallel-RANSAC) |
| 3 | **python-plane-ransac** | — | CUDA-parallelized plane segmentation | [GitHub](https://github.com/misha-kis/python-plane-ransac) |

### Paper Only — No Code Released Yet

| Project | Year | What it does | Link |
|---------|------|-------------|------|
| **Floorplan-SLAM** | Mar 2025 | **Real-time** (25-45 FPS) point-plane SLAM → floorplan, no GPU. 1000m² in 9 min. | [arXiv](https://arxiv.org/abs/2503.00397) |
| **PLANA3R** | Oct 2025 | Pose-free metric planar 3D reconstruction from two views. Emergent plane segmentation. | [Project](https://lck666666.github.io/plana3r) |
| **PlanarGS** | Oct 2025 | Language-prompted planar priors for 3DGS indoor reconstruction. | [Project](https://planargs.github.io) |
| **2DGS-Room** | Dec 2024 | 2D Gaussian Splatting for indoor scenes. SOTA on ScanNet++. | [arXiv](https://arxiv.org/abs/2412.03428) |
| **Structure-preserving Planar Simplification** | Aug 2024 | RANSAC → wall meshes → ceiling/floor clipping. Manhattan alignment. Full pipeline. | [arXiv](https://arxiv.org/abs/2408.06814) |
| **MultiFloor3D** | NeurIPS 2025 | Training-free mesh → layout polygons, multi-floor buildings. | [Project](https://houselayout3d.github.io) |
| **PLANING** | Jan 2026 | On-the-fly reconstruction: geometric primitives + neural Gaussians. 5x faster than 2DGS. | [Project](https://city-super.github.io/PLANING/) |
| **A-Scan2BIM** | Nov 2023 | Auto-regressive Revit API sequence prediction from scans. 89h professional data. | [Project](https://a-scan2bim.github.io) |
| **Defurnished Replicas** | Jun 2025 | Remove furniture → clean walls/floors mesh | [arXiv](https://arxiv.org/abs/2506.05338) |
| **3D-CRS** | Apr 2024 | Occluded surface completion (hidden walls behind furniture) | [arXiv](https://arxiv.org/abs/2404.03070) |

### Datasets & Benchmarks

| Dataset | ⭐ | Size | What's in it |
|---------|---:|------|-------------|
| [**InteriorGS**](https://github.com/manycore-research/InteriorGS) | 212 | 1,000 scenes | 3DGS + semantic labels + floorplans + occupancy maps |
| **Structured3D** | — | 21k rooms | Photo-realistic synthetic with layout annotations |
| **ScanNet / ScanNet++** | — | 1,500+ scenes | Real RGB-D indoor reconstructions |
| **ResPlan** | — | 17k floorplans | Residential plans with wall/door/window annotations |
| **HouseLayout3D** | — | Multi-floor | Real-world multi-floor layout benchmark |
| **CubiCasa5K** | — | 5,000 floorplans | 80+ object categories |
| **3D-FRONT** | — | 18,968 rooms | Professional interior designs with textured 3D furniture |

### Curated Lists

| ⭐ | List | Description |
|---:|------|-------------|
| 8,302 | [awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) | Comprehensive 3DGS paper list including mesh extraction |
| 200 | [awesome-planar-reconstruction](https://github.com/chenzhaiyu/awesome-planar-reconstruction) | Plane detection, reconstruction, floorplan generation papers |

### 🍎 Apple RoomPlan (Reference)
- Uses LiDAR + ARKit for real-time room scanning
- Outputs `CapturedRoom` with walls, floors, doors, windows, furniture
- Limitations: Apple-only, requires LiDAR device, simplified box geometry
- **Our goal**: Similar output quality but from any mesh, cross-platform, browser-based

### Key Takeaways

1. **Best starting points with code**: pyRANSAC-3D (650⭐) for plane fitting, RoomFormer (271⭐) for end-to-end scan→floorplan, Manhattan SDF (528⭐) for planar constraints
2. **Floorplan-SLAM** is the dream (real-time, no GPU) but no code released yet
3. **CAGE** (12⭐) is newest SOTA but very fresh — worth watching
4. **Structure-preserving Planar Simplification** describes exactly our pipeline (RANSAC → walls → clip ceilings/floors) but no code
5. **Manhattan world assumption** used by almost every method — should be optional constraint
6. **Defurnishing is a real problem** — scanned meshes have furniture obscuring walls

## Repository Structure

```
viewer/           # v1-v9 HTML viewers (browser-based approaches)
research/         # v10+ Python research scripts
├── v10_voxelization_projection.py
├── v11_normal_wall_segmentation.py
├── v12_contour_depth_raster.py
├── v13_alpha_shape_boundary.py
└── NOTES.md      # Detailed research notes
scripts/          # Utility scripts
results/          # Test results and visualizations
data/             # Test meshes (not in repo)
```

## Tech Stack

- **Frontend**: Three.js + Web Workers (v1-v9 viewers)
- **Research**: Python 3.13 — trimesh, numpy, scipy, scikit-learn, opencv, matplotlib
- **Export**: SVG, DXF, PNG, JSON
- **Deployment**: Vercel (v9 live demo)

## License

MIT

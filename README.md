# mesh2plan

**Real-time 3D mesh → floorplan / wall / surface extraction**

Like Apple RoomPlan, but focused on accurate geometry from arbitrary 3D meshes. Open source, runs in the browser.

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

## Key Approaches

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

### ✅ Working Code — Sorted by GitHub Stars

Projects with released, runnable code.

| ⭐ | Project | Year | What it does | Code |
|---:|---------|------|-------------|------|
| 3,193 | **SuGaR** | 2023 | Gaussian splat → mesh extraction (surface-aligned) | [GitHub](https://github.com/Anttwo/SuGaR) |
| 650 | **pyRANSAC-3D** | — | Fit planes, cuboids, cylinders to point clouds. Pure Python. | [GitHub](https://github.com/leomariga/pyRANSAC-3D) |
| 605 | **PlaneRCNN** (NVIDIA) | CVPR 2019 | Single-image plane detection + 3D reconstruction | [GitHub](https://github.com/NVlabs/planercnn) |
| 528 | **Manhattan SDF** | CVPR 2022 | Neural SDF + Manhattan assumption → planar walls/floors. Joint geometry+semantics. | [GitHub](https://github.com/zju3dv/manhattan_sdf) |
| 374 | **Planar Reconstruction** | CVPR 2019 | Real-time (30fps) piece-wise planar 3D from single image. Arbitrary # planes. | [GitHub](https://github.com/svip-lab/PlanarReconstruction) |
| 271 | **RoomFormer** | CVPR 2023 | Transformer: 3D scan → room polygons directly. End-to-end, SOTA on Structured3D. | [GitHub](https://github.com/ywyue/RoomFormer) |
| 248 | **FloorNet** | 2018 | PointNet + CNN: RGBD streams → vectorized floorplan. 155 house dataset. | [GitHub](https://github.com/art-programmer/FloorNet) |
| 233 | **GaussianRoom** | Dec 2024 | SDF + 3DGS for indoor scenes. Solves textureless walls. | [GitHub](https://github.com/xhd0612/GaussianRoom) |
| 115 | **Multiple Planes Detection** | — | Fast iterative RANSAC multi-plane detection. Open3D. | [GitHub](https://github.com/yuecideng/Multiple_Planes_Detection) |
| 91 | **Orthogonal Planes** | ICRA 2020 | Multi-purpose primitive detection (planes + corners) in unorganized 3D point clouds | [GitHub](https://github.com/c-sommer/orthogonal-planes) |
| 73 | **AirPlanes** (Niantic) | CVPR 2024 | 3D-consistent plane embeddings from posed RGB. Sequential RANSAC + learned MLP. | [GitHub](https://github.com/nianticlabs/airplanes) |
| 55 | **DOPNet** | CVPR 2023 | Disentangle orthogonal planes for panoramic room layout estimation | [GitHub](https://github.com/zhijieshen-bjtu/DOPNet) |
| 44 | **Plane-DUSt3R** | Feb 2025 | Multi-view images → room layout planes via DUSt3R foundation model. Training-free. | [GitHub](https://github.com/justacar/Plane-DUSt3R) |
| 12 | **CAGE** | Sep 2025 | Edge-centric floorplan via dual-query transformer. 99.1% room F1. SOTA. | [GitHub](https://github.com/ee-Liu/CAGE) |
| 11 | **FloorSAM** | Sep 2025 | SAM zero-shot + LiDAR density maps → room segmentation → vectorized floorplans | [GitHub](https://github.com/Silentbarber/FloorSAM) |
| 11 | **torch_ransac3d** | — | GPU-accelerated RANSAC with PyTorch/CUDA | [GitHub](https://github.com/harrydobbs/torch_ransac3d) |
| 7 | **Parallel-RANSAC** | — | GPU-parallelized RANSAC plane extraction from RGB-D | [GitHub](https://github.com/alehdaghi/Parallel-RANSAC) |
| 3 | **python-plane-ransac** | — | CUDA-parallelized plane segmentation | [GitHub](https://github.com/misha-kis/python-plane-ransac) |

### 📄 Paper Only — No Code Released Yet

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

### 📊 Datasets & Benchmarks

| Dataset | ⭐ | Size | What's in it |
|---------|---:|------|-------------|
| [**InteriorGS**](https://github.com/manycore-research/InteriorGS) | 212 | 1,000 scenes | 3DGS + semantic labels + floorplans + occupancy maps |
| **Structured3D** | — | 21k rooms | Photo-realistic synthetic with layout annotations |
| **ScanNet / ScanNet++** | — | 1,500+ scenes | Real RGB-D indoor reconstructions |
| **ResPlan** | — | 17k floorplans | Residential plans with wall/door/window annotations |
| **HouseLayout3D** | — | Multi-floor | Real-world multi-floor layout benchmark |
| **CubiCasa5K** | — | 5,000 floorplans | 80+ object categories |
| **3D-FRONT** | — | 18,968 rooms | Professional interior designs with textured 3D furniture |

### 📚 Curated Lists

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

## Tech Stack (Planned)

- **Frontend**: SvelteKit + Three.js (mesh viewer + plane overlay)
- **Compute**: Web Workers for RANSAC / region growing (keep UI responsive)
- **Optional GPU**: WebGPU compute shaders for large meshes
- **Export**: SVG floorplans, DXF (for CAD), JSON (structured data)
- **Backend** (optional): Python + Open3D for heavy processing, WASM for portable

## Milestones

- [ ] **v0.1** — Load mesh (OBJ/PLY/glTF), extract vertices + normals, display
- [ ] **v0.2** — RANSAC plane detection, color-code detected planes
- [ ] **v0.3** — Classify walls/floors/ceilings by normal direction
- [ ] **v0.4** — Extract plane boundaries, compute wall dimensions
- [ ] **v0.5** — Generate 2D floorplan (top-down SVG)
- [ ] **v0.6** — Real-time: process incrementally as mesh streams in
- [ ] **v1.0** — Room segmentation, door/window detection, DXF export

## License

MIT

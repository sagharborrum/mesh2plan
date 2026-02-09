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

### Core Papers

| Paper | Year | Key Contribution | Code |
|-------|------|-----------------|------|
| **AirPlanes** (Niantic) | CVPR 2024 | 3D-consistent plane embeddings from posed RGB → accurate plane estimation | [GitHub](https://github.com/nianticlabs/airplanes) |
| **MultiFloor3D** | NeurIPS 2025 | Training-free mesh → layout polygons (walls/floors/ceilings), multi-floor | — |
| **FloorSAM** | Sep 2025 | SAM-guided floorplan from LiDAR point clouds, density maps → room segmentation | [GitHub](https://github.com/Silentbarber/FloorSAM) |
| **GaussianRoom** | Dec 2024 | SDF + 3DGS for indoor surface reconstruction (solves textureless walls) | [GitHub](https://github.com/xhd0612/GaussianRoom) |
| **PlaneRCNN** (NVIDIA) | CVPR 2019 | Single-image plane detection + reconstruction | [GitHub](https://github.com/NVlabs/planercnn) |
| **Efficient RANSAC** | 2007 | Foundation for point cloud shape detection | [Code](https://cg.cs.uni-bonn.de/en/publications/paper-details/schnabel-2007-efficient/) |

### Curated Lists
- [awesome-planar-reconstruction](https://github.com/chenzhaiyu/awesome-planar-reconstruction) — Comprehensive paper list for plane detection, single/multi-view reconstruction, floorplan generation
- [awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) — Splat → mesh methods

### Relevant Open Source Tools

| Tool | What it does | Language |
|------|-------------|----------|
| [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D) | Fit planes, cuboids, cylinders to point clouds | Python |
| [torch_ransac3d](https://github.com/harrydobbs/torch_ransac3d) | GPU-accelerated RANSAC with PyTorch/CUDA | Python |
| [Multiple_Planes_Detection](https://github.com/yuecideng/Multiple_Planes_Detection) | Fast iterative RANSAC multi-plane detection | Python/Open3D |
| [python-plane-ransac](https://github.com/misha-kis/python-plane-ransac) | CUDA-parallelized plane segmentation | Python/CUDA |
| [SuGaR](https://github.com/Anttwo/SuGaR) | Gaussian splat → mesh extraction | Python |
| [InteriorGS](https://github.com/manycore-research/InteriorGS) | 1000 indoor scenes with semantic labels + floorplans (dataset) | — |

### Apple RoomPlan (Reference)
- Uses LiDAR + ARKit for real-time room scanning
- Outputs `CapturedRoom` with walls, floors, doors, windows, furniture
- Limitations: Apple-only, requires LiDAR device, simplified box geometry
- **Our goal**: Similar output quality but from any mesh, cross-platform, browser-based

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

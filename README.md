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

### 🔥 Most Relevant Papers

| Paper | Year | Key Contribution | Code |
|-------|------|-----------------|------|
| **Floorplan-SLAM** | Mar 2025 | **Real-time** point-plane SLAM → floorplan at 25-45 FPS, no GPU needed. 1000m² in 9 min vs 10+ hrs. Multi-session. Stereo camera only. | [arXiv](https://arxiv.org/abs/2503.00397) |
| **CAGE** (Edge-centric) | Sep 2025 | Native edge representation for floorplans via dual-query transformer. 99.1% room F1, 91.7% corner F1. SOTA on Structured3D/SceneCAD. | [GitHub](https://github.com/ee-Liu/CAGE) |
| **RoomFormer** | CVPR 2023 | Transformer: 3D scan → room polygons directly. Two-level queries (polygon + corner). End-to-end, no heuristic pipeline. | [GitHub](https://github.com/ywyue/RoomFormer) |
| **Plane-DUSt3R** | Feb 2025 | Multi-view perspective images → room layout planes via DUSt3R foundation model. Training-free, works on cartoon/real. | [GitHub](https://github.com/justacar/Plane-DUSt3R) |
| **PLANA3R** | Oct 2025 | Pose-free metric planar 3D reconstruction from two views. No explicit plane supervision needed. Emergent plane segmentation. | [Project](https://lck666666.github.io/plana3r) |
| **Structure-preserving Planar Simplification** | Aug 2024 | Point cloud → RANSAC planes → wall meshes → ceiling/floor clipping. Manhattan alignment. Full pipeline. | [arXiv](https://arxiv.org/abs/2408.06814) |
| **Manhattan SDF** | CVPR 2022 | Neural SDF + Manhattan world assumption → planar constraints for walls/floors. Joint geometry + semantics optimization. | [GitHub](https://zju3dv.github.io/manhattan_sdf) |

### Plane Detection & Estimation

| Paper | Year | Key Contribution | Code |
|-------|------|-----------------|------|
| **AirPlanes** (Niantic) | CVPR 2024 | 3D-consistent plane embeddings from posed RGB. Sequential RANSAC + learned MLP. | [GitHub](https://github.com/nianticlabs/airplanes) |
| **PlanarGS** | Oct 2025 | Language-prompted planar priors for 3DGS indoor reconstruction. Cross-view fusion. | [Project](https://planargs.github.io) |
| **2DGS-Room** | Dec 2024 | 2D Gaussian Splatting for indoor scenes with monocular depth/normal priors. SOTA on ScanNet++. | [arXiv](https://arxiv.org/abs/2412.03428) |
| **PlaneRCNN** (NVIDIA) | CVPR 2019 | Single-image plane detection + reconstruction | [GitHub](https://github.com/NVlabs/planercnn) |
| **Planar Reconstruction** (Associative Embedding) | CVPR 2019 | Real-time (30fps) piece-wise planar 3D from single image. Arbitrary # planes. | [GitHub](https://github.com/svip-lab/PlanarReconstruction) |
| **Efficient RANSAC** | 2007 | Foundation for point cloud shape detection | [Code](https://cg.cs.uni-bonn.de/en/publications/paper-details/schnabel-2007-efficient/) |

### Scan → Floorplan / BIM

| Paper | Year | Key Contribution | Code |
|-------|------|-----------------|------|
| **FloorSAM** | Sep 2025 | SAM zero-shot + LiDAR density maps → room segmentation → vectorized floorplans | [GitHub](https://github.com/Silentbarber/FloorSAM) |
| **MultiFloor3D** | NeurIPS 2025 | Training-free mesh → layout polygons (walls/floors/ceilings), multi-floor | [Project](https://houselayout3d.github.io) |
| **FloorNet** | 2018 | PointNet + CNN: RGBD streams → floorplan. 155 house dataset. | [GitHub](https://github.com/art-programmer/FloorNet) |
| **A-Scan2BIM** | Nov 2023 | Assistive Scan-to-BIM: auto-regressive Revit API sequence prediction. 89h modeling data. | [Project](https://a-scan2bim.github.io) |
| **PLANING** | Jan 2026 | On-the-fly reconstruction: geometric primitives + neural Gaussians. 5x faster than 2DGS. | [Project](https://city-super.github.io/PLANING/) |

### Indoor Reconstruction (Supporting)

| Paper | Year | Key Contribution | Code |
|-------|------|-----------------|------|
| **GaussianRoom** | Dec 2024 | SDF + 3DGS for indoor scenes. Solves textureless walls. | [GitHub](https://github.com/xhd0612/GaussianRoom) |
| **Defurnished Replicas** | Jun 2025 | Remove furniture from mesh → simplified defurnished mesh (SDM) → clean walls/floors | [arXiv](https://arxiv.org/abs/2506.05338) |
| **3D-CRS** | Apr 2024 | Indoor 3D reconstruction with occluded surface completion (hidden walls behind furniture) | [arXiv](https://arxiv.org/abs/2404.03070) |

### Datasets & Benchmarks

| Dataset | Size | What's in it |
|---------|------|-------------|
| **Structured3D** | 21k rooms | Photo-realistic synthetic indoor scenes with layout annotations |
| **ScanNet / ScanNet++** | 1500+ scenes | Real RGB-D indoor reconstructions with semantic labels |
| **InteriorGS** | 1000 scenes | 3DGS with semantic labels + floorplans + occupancy maps |
| **ResPlan** | 17k floorplans | Detailed residential plans with wall/door/window annotations |
| **HouseLayout3D** | Multi-floor | Real-world benchmark for full building-scale layout estimation |
| **CubiCasa5K** | 5000 floorplans | Annotated into 80+ object categories |
| **3D-FRONT** | 18,968 rooms | Professional interior designs with textured 3D furniture |

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

### Key Takeaways from Research

1. **Floorplan-SLAM is the closest to our goal** — real-time, no GPU, plane-based SLAM → floorplan. But it needs stereo camera input, not a pre-existing mesh.
2. **CAGE / RoomFormer** are SOTA for scan → vectorized floorplan, but require training on Structured3D.
3. **Structure-preserving Planar Simplification** (2408.06814) is the most practical pipeline for our use case: RANSAC → wall meshes → ceiling/floor clipping. Fully geometric, no ML.
4. **Manhattan world assumption** is used by almost every method — most indoor spaces have orthogonal walls. We should support it as optional constraint.
5. **The defurnishing problem** is real — scanned meshes have furniture that obscures walls. Need to handle or at least detect clutter.
6. **Plane-DUSt3R and PLANA3R** show the trend: foundation models that understand planar structure from images, training-free.

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

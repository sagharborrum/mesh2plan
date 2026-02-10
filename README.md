# mesh2plan

**3D mesh → floor plan, entirely in your browser.**

Drop a LiDAR scan (OBJ, GLB, PLY, STL) and get an instant architectural floor plan with wall detection, measurements, and door/window identification. No backend, no uploads — everything runs client-side.

**🌐 [Try it live → mesh2plan.vercel.app](https://mesh2plan.vercel.app)**

## Features

- **Manhattan wall detection** — Hough-like angle voting finds dominant wall direction, histogram peak detection identifies wall positions from cross-section point clouds
- **Door & window detection** — identifies gaps in walls by point density analysis
- **Measurements** — automatic wall dimensions with dimension lines
- **Room area** — computed from wall intersection polygon (m² and ft²)
- **Multi-format** — OBJ, GLB/glTF, PLY (with vertex colors), STL
- **Export** — SVG (vector), DXF (for CAD/architect tools)
- **Dark/light theme** — dark mode for screen, light mode for print
- **Metric/imperial toggle** — switch between meters and feet
- **Interactive 3D viewer** — orbit controls, opacity slider, wireframe toggle
- **Cross-section slider** — visualize horizontal slices through the mesh
- **Web Worker** — analysis runs off-thread for smooth UI
- **Zero dependencies** — single HTML file + worker.js, Three.js from CDN
- **Private** — nothing leaves your browser

## How It Works

1. **Load mesh** — drag & drop or file picker
2. **Cross-section slicing** — 20 horizontal slices through the mesh via triangle-plane intersection
3. **Dominant angle** — histogram sharpness voting across 180° finds the primary wall direction
4. **Wall detection** — in the rotated (axis-aligned) coordinate frame, histogram peaks in X and Z reveal wall positions
5. **Wall merging** — parallel walls within 15cm are merged
6. **Room polygon** — wall intersections form vertices; convex hull + rectilinear snapping creates the room outline
7. **Gap detection** — gaps >30cm in wall point clouds are classified as doors (0.6-1.3m) or windows (0.3-2.0m)

## Supported Scan Sources

- **3D Scanner App** (iOS) — export as OBJ
- **Polycam** — export as GLB or OBJ
- **Scaniverse** — export as GLB
- **RealityCapture** — export as OBJ or PLY
- **COLMAP + OpenMVS** — export as PLY
- Any mesh in OBJ/GLB/PLY/STL format

## Local Development

```bash
# Clone
git clone https://github.com/sagharborrum/mesh2plan
cd mesh2plan

# Serve (no build step needed)
python3 -m http.server 3847 --bind 0.0.0.0

# Open http://localhost:3847/viewer/v9.html
```

### Python extraction scripts (optional)

For batch processing or more advanced analysis:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install trimesh numpy scipy scikit-learn pyransac3d shapely

# Run extraction
python scripts/extract_floorplan_v7.py path/to/mesh.obj output/result.json
```

## Architecture

```
viewer/
  v1.html - v9.html    # Evolution of viewers (v9 = latest)
  index.html            # Production entry (= v9)
  worker.js             # Web Worker for analysis

scripts/
  extract_planes.py       # v1: RANSAC plane extraction
  extract_planes_v2.py    # v2: wall clustering
  extract_planes_v3.py    # v3: face-based RANSAC
  extract_floorplan_v4.py # v4: cross-section slicing
  extract_floorplan_v5.py # v5: multi-slice composite
  extract_floorplan_v6.py # v6: Manhattan wall fitting
  extract_floorplan_v7.py # v7: connected walls + gaps + SVG
  extract_floorplan_v8.py # v8: top-down depth map
```

## Version History

| Version | Approach | Key Innovation |
|---------|----------|---------------|
| v1-v3 | RANSAC planes | Face-based plane extraction, wall clustering |
| v4 | Cross-sections | `trimesh.section()` — breakthrough for room outlines |
| v5 | Multi-slice | Shapely polygon union (slow but complete) |
| v6 | Manhattan | Histogram peak detection ⭐ first real floor plan |
| v7 | Connected | Wall intersection polygon, door/window gaps, SVG |
| v8 | Depth map | Top-down raycasting for floor plan background |
| v9 | Browser | Full client-side JS, multi-format, Web Worker |

## Tech Stack

- **Three.js** — 3D rendering, OBJ/GLB/PLY/STL loading
- **Canvas 2D** — floor plan rendering
- **Web Workers** — off-thread analysis
- **Python** (optional) — trimesh, pyransac3d, scipy, shapely

## License

MIT

---

Built by [@sagharborrum](https://github.com/sagharborrum)

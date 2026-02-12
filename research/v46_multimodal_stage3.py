#!/usr/bin/env python3
"""
mesh2plan v46 - Multimodal Stage 3: Photo Feature Detection → 3D Projection

Detects architectural features (wall edges, doors, windows) in photos,
projects them to 3D via depth maps, accumulates in floor plan view.

Outputs to results/v46_stage3/
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import cv2
from pathlib import Path
from scipy import ndimage
import glob
import os
import shutil

# Paths
BASE = Path(__file__).parent.parent
SCAN_DIR = BASE / "data" / "multiroom" / "2026_02_10_18_31_36"
OUT_DIR = BASE / "results" / "v46_stage3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_DIR = BASE / "results" / "v44_stage1"

RESOLUTION = 0.02  # meters per pixel

# Photo intrinsics (4032×3024)
FX_PHOTO = 2812.42
FY_PHOTO = 2812.42
CX_PHOTO = 2013.95
CY_PHOTO = 1509.67

# Depth map size
DEPTH_W, DEPTH_H = 256, 192
DEPTH_SCALE_X = DEPTH_W / 4032.0
DEPTH_SCALE_Y = DEPTH_H / 3024.0
FX_DEPTH = FX_PHOTO * DEPTH_SCALE_X
FY_DEPTH = FY_PHOTO * DEPTH_SCALE_Y
CX_DEPTH = CX_PHOTO * DEPTH_SCALE_X
CY_DEPTH = CY_PHOTO * DEPTH_SCALE_Y

# Edge detection image size
EDGE_W, EDGE_H = 1008, 756  # ~1000x750, maintaining 4:3
EDGE_SCALE_X = EDGE_W / 4032.0
EDGE_SCALE_Y = EDGE_H / 3024.0


def load_mesh():
    mesh = trimesh.load(SCAN_DIR / "export_refined.obj", process=False)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def mesh_to_density(mesh):
    verts = np.array(mesh.vertices)
    x, z = verts[:, 0], verts[:, 2]
    pad = 0.3
    x_min, x_max = x.min() - pad, x.max() + pad
    z_min, z_max = z.min() - pad, z.max() + pad
    w = int((x_max - x_min) / RESOLUTION) + 1
    h = int((z_max - z_min) / RESOLUTION) + 1
    density = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((x - x_min) / RESOLUTION).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min) / RESOLUTION).astype(int), 0, h - 1)
    np.add.at(density, (zi, xi), 1)
    transform = (x_min, z_min, RESOLUTION)
    return density, transform


def world_to_pixel(x, z, transform):
    x_min, z_min, res = transform
    px = int((x - x_min) / res)
    py = int((z - z_min) / res)
    return px, py


def load_room_frames():
    """Load room-to-frame mapping from Stage 1."""
    with open(STAGE1_DIR / "room_frames.json") as f:
        return json.load(f)


def load_frame_data(idx):
    """Load frame JSON, photo, depth, confidence."""
    json_path = SCAN_DIR / f"frame_{idx:05d}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    pose = np.array(data['cameraPoseARFrame']).reshape(4, 4, order='C')
    return {
        'index': idx,
        'pose': pose,
        'motion_quality': data.get('motionQuality', 0),
    }


def select_best_frames(room_frames_json, n_per_room=8):
    """Select best frames per room: high quality, good temporal spread."""
    selected = {}
    for rid, info in room_frames_json.items():
        indices = info['frame_indices']
        # Load motion quality for each
        frames = []
        for idx in indices:
            fd = load_frame_data(idx)
            if fd and (SCAN_DIR / f"frame_{idx:05d}.jpg").exists():
                frames.append(fd)
        if not frames:
            selected[rid] = []
            continue
        
        # Sort by quality, take top half
        frames.sort(key=lambda f: f['motion_quality'], reverse=True)
        top = frames[:max(len(frames) // 2, n_per_room * 2)]
        
        # Sort by index for temporal spread, pick evenly
        top.sort(key=lambda f: f['index'])
        n = min(n_per_room, len(top))
        if n >= len(top):
            selected[rid] = top
        else:
            idxs = np.linspace(0, len(top) - 1, n, dtype=int)
            selected[rid] = [top[i] for i in idxs]
        
        print(f"Room {rid}: selected {len(selected[rid])} frames from {len(indices)}")
    return selected


def detect_structural_lines(img_path):
    """Detect structural (H/V) lines in a photo. Returns lines in EDGE resolution coords."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None
    
    img_small = cv2.resize(img, (EDGE_W, EDGE_H))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    
    # Enhance edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=40, maxLineGap=15)
    
    structural = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 30:
                continue
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            # Keep near-horizontal (<20°) or near-vertical (>70°)
            if angle < 20 or angle > 70:
                line_type = 'H' if angle < 20 else 'V'
                structural.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'length': length, 'angle': angle, 'type': line_type
                })
    
    return structural, img_small, edges


def sample_points_along_line(line, n_samples=20):
    """Sample pixel coordinates along a line (in EDGE resolution)."""
    x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
    t = np.linspace(0, 1, n_samples)
    us = x1 + t * (x2 - x1)
    vs = y1 + t * (y2 - y1)
    return us, vs


def project_pixels_to_3d(us_edge, vs_edge, depth_map, conf_map, pose, min_conf=2):
    """
    Project pixel coords (in EDGE resolution) to 3D world coords via depth map.
    
    us_edge, vs_edge: pixel coords in edge detection resolution
    Returns: Nx3 world coords, mask of valid points
    """
    # Convert edge resolution → depth resolution
    us_depth = us_edge * (DEPTH_W / EDGE_W)
    vs_depth = vs_edge * (DEPTH_H / EDGE_H)
    
    # Bilinear sample depth (nearest neighbor for speed)
    ui = np.clip(np.round(us_depth).astype(int), 0, DEPTH_W - 1)
    vi = np.clip(np.round(vs_depth).astype(int), 0, DEPTH_H - 1)
    
    depths_mm = depth_map[vi, ui].astype(float)
    confs = conf_map[vi, ui] if conf_map is not None else np.full_like(depths_mm, 2)
    
    # Filter: valid depth + high confidence
    valid = (depths_mm > 10) & (depths_mm < 10000) & (confs >= min_conf)
    if valid.sum() == 0:
        return np.zeros((0, 3)), valid
    
    d = depths_mm[valid] / 1000.0  # to meters
    u = us_depth[valid]
    v = vs_depth[valid]
    
    # Camera coordinates
    X_c = (u - CX_DEPTH) * d / FX_DEPTH
    Y_c = (v - CY_DEPTH) * d / FY_DEPTH
    Z_c = d
    
    pts_cam = np.stack([X_c, Y_c, Z_c], axis=1)  # Nx3
    
    # World coordinates: pose is 4x4, R = pose[:3,:3], t = pose[:3,3]
    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_world = (R @ pts_cam.T).T + t  # Nx3
    
    return pts_world, valid


def process_all_frames(selected_frames):
    """Process all selected frames: detect lines, project to 3D."""
    all_points = []  # list of (x, y, z, line_type, room_id, frame_idx)
    room_examples = {}  # for visualization
    
    total_frames = sum(len(frs) for frs in selected_frames.values())
    processed = 0
    
    for rid, frames in selected_frames.items():
        room_pts = []
        for fr in frames:
            idx = fr['index']
            img_path = SCAN_DIR / f"frame_{idx:05d}.jpg"
            depth_path = SCAN_DIR / f"depth_{idx:05d}.png"
            conf_path = SCAN_DIR / f"conf_{idx:05d}.png"
            
            if not depth_path.exists():
                continue
            
            # Detect lines
            lines, img_small, edges = detect_structural_lines(img_path)
            if lines is None:
                continue
            
            # Load depth and confidence
            depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            conf_map = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED) if conf_path.exists() else None
            
            # Store first example per room for visualization
            if rid not in room_examples and len(lines) > 0:
                room_examples[rid] = {
                    'img': img_small, 'lines': lines, 'edges': edges, 'frame_idx': idx
                }
            
            # Project each line to 3D
            for line in lines:
                us, vs = sample_points_along_line(line, n_samples=25)
                pts_3d, valid = project_pixels_to_3d(us, vs, depth_map, conf_map, fr['pose'])
                
                if len(pts_3d) > 0:
                    for pt in pts_3d:
                        all_points.append({
                            'x': float(pt[0]), 'y': float(pt[1]), 'z': float(pt[2]),
                            'type': line['type'], 'room': int(rid), 'frame': idx
                        })
                    room_pts.extend(pts_3d.tolist())
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{total_frames} frames, {len(all_points)} points so far")
        
        print(f"Room {rid}: {len(room_pts)} 3D feature points")
    
    print(f"Total: {len(all_points)} 3D feature points from {processed} frames")
    return all_points, room_examples


def create_feature_density(all_points, density, transform):
    """Create feature density map in XZ plane."""
    x_min, z_min, res = transform
    h, w = density.shape
    
    feat_density = np.zeros((h, w), dtype=np.float32)
    
    pts = np.array([[p['x'], p['z']] for p in all_points])
    if len(pts) == 0:
        return feat_density
    
    xi = np.clip(((pts[:, 0] - x_min) / res).astype(int), 0, w - 1)
    zi = np.clip(((pts[:, 1] - z_min) / res).astype(int), 0, h - 1)
    np.add.at(feat_density, (zi, xi), 1)
    
    return feat_density


def extract_wall_segments_from_density(feat_density, transform):
    """Run Hough line detection on feature density to extract wall segments."""
    # Normalize and threshold
    fd = feat_density.copy()
    if fd.max() == 0:
        return []
    
    fd_norm = (fd / fd.max() * 255).astype(np.uint8)
    _, binary = cv2.threshold(fd_norm, 20, 255, cv2.THRESH_BINARY)
    
    # Slight dilation to connect nearby points
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Hough lines on the density image
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=15,
                            minLineLength=10, maxLineGap=8)
    
    x_min, z_min, res = transform
    segments = []
    if lines is not None:
        for line in lines:
            px1, py1, px2, py2 = line[0]
            # Convert pixel to world coords
            x1 = px1 * res + x_min
            z1 = py1 * res + z_min
            x2 = px2 * res + x_min
            z2 = py2 * res + z_min
            length = np.sqrt((x2 - x1)**2 + (z2 - z1)**2)
            angle = np.degrees(np.arctan2(abs(z2 - z1), abs(x2 - x1)))
            
            # Keep walls: mostly axis-aligned and > 0.3m
            if length > 0.3 and (angle < 25 or angle > 65):
                segments.append({
                    'x1': float(x1), 'z1': float(z1),
                    'x2': float(x2), 'z2': float(z2),
                    'length': float(length),
                    'angle': float(angle),
                    'orientation': 'EW' if angle < 25 else 'NS',
                    'px1': int(px1), 'py1': int(py1),
                    'px2': int(px2), 'py2': int(py2),
                })
    
    print(f"Extracted {len(segments)} wall segments from photo features")
    return segments


def detect_doors_windows(all_points, wall_segments):
    """Detect doors/windows as gaps along wall segments using 3D height info."""
    features = []
    
    for seg in wall_segments:
        # Get points near this wall segment
        sx, sz = seg['x1'], seg['z1']
        ex, ez = seg['x2'], seg['z2']
        wall_vec = np.array([ex - sx, ez - sz])
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 0.5:
            continue
        wall_dir = wall_vec / wall_len
        wall_normal = np.array([-wall_dir[1], wall_dir[0]])
        
        # Find points within 0.15m of wall line
        nearby = []
        for p in all_points:
            px, pz = p['x'], p['z']
            rel = np.array([px - sx, pz - sz])
            along = np.dot(rel, wall_dir)
            perp = abs(np.dot(rel, wall_normal))
            if perp < 0.15 and 0 <= along <= wall_len:
                nearby.append({'along': along, 'y': p['y'], 'type': p['type']})
        
        if len(nearby) < 5:
            continue
        
        # Bin points along wall
        n_bins = max(int(wall_len / 0.1), 5)
        bins = np.linspace(0, wall_len, n_bins + 1)
        bin_counts = np.zeros(n_bins)
        bin_min_y = np.full(n_bins, np.inf)
        bin_max_y = np.full(n_bins, -np.inf)
        
        for p in nearby:
            bi = min(int(p['along'] / wall_len * n_bins), n_bins - 1)
            bin_counts[bi] += 1
            bin_min_y[bi] = min(bin_min_y[bi], p['y'])
            bin_max_y[bi] = max(bin_max_y[bi], p['y'])
        
        # Find gaps (low density runs)
        median_count = np.median(bin_counts[bin_counts > 0]) if np.any(bin_counts > 0) else 0
        threshold = median_count * 0.3
        
        gap_start = None
        for i in range(n_bins):
            if bin_counts[i] <= threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_width = (i - gap_start) * wall_len / n_bins
                    gap_center = ((gap_start + i) / 2) * wall_len / n_bins
                    
                    if 0.6 < gap_width < 1.8:
                        # Check height of surrounding points to classify
                        adj_bins = list(range(max(0, gap_start - 2), gap_start)) + \
                                   list(range(i, min(n_bins, i + 2)))
                        heights = [bin_max_y[b] - bin_min_y[b] for b in adj_bins
                                   if bin_counts[b] > 0 and bin_max_y[b] > -np.inf]
                        avg_height = np.mean(heights) if heights else 0
                        
                        # Position along wall
                        cx = sx + wall_dir[0] * gap_center
                        cz = sz + wall_dir[1] * gap_center
                        
                        feat_type = 'door' if gap_width < 1.2 else 'window'
                        features.append({
                            'type': feat_type,
                            'x': float(cx), 'z': float(cz),
                            'width': float(gap_width),
                            'wall_segment_length': float(wall_len),
                        })
                    gap_start = None
        
    print(f"Detected {len(features)} door/window candidates")
    return features


def plot_feature_density(feat_density, density, transform, wall_segments, out_path):
    """Plot feature density overlaid on mesh density."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Mesh density
    ax = axes[0]
    d_vis = np.log1p(density)
    ax.imshow(d_vis, cmap='gray_r', origin='upper')
    ax.set_title("Mesh Vertex Density", fontsize=11)
    ax.set_aspect('equal')
    
    # Feature density
    ax = axes[1]
    fd_vis = np.log1p(feat_density)
    ax.imshow(fd_vis, cmap='hot', origin='upper')
    ax.set_title("Photo Feature Density", fontsize=11)
    ax.set_aspect('equal')
    
    # Overlay
    ax = axes[2]
    # RGB overlay: mesh=blue channel, features=red channel
    h, w = density.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    d_n = np.log1p(density)
    d_n = d_n / d_n.max() if d_n.max() > 0 else d_n
    f_n = np.log1p(feat_density)
    f_n = f_n / f_n.max() if f_n.max() > 0 else f_n
    overlay[:, :, 2] = d_n  # mesh = blue
    overlay[:, :, 0] = f_n  # features = red
    overlay[:, :, 1] = np.minimum(d_n, f_n) * 0.5  # overlap = yellowish
    ax.imshow(overlay, origin='upper')
    
    # Draw wall segments
    for seg in wall_segments:
        ax.plot([seg['px1'], seg['px2']], [seg['py1'], seg['py2']],
                'g-', linewidth=1.5, alpha=0.7)
    
    ax.set_title(f"Overlay + {len(wall_segments)} wall segments (green)", fontsize=11)
    ax.set_aspect('equal')
    
    fig.suptitle("Stage 3: Photo Feature Detection → 3D Projection", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_wall_comparison(feat_density, density, wall_segments, out_path):
    """Compare photo-detected walls with mesh density walls."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Mesh-based wall detection (high density = walls)
    ax = axes[0]
    d_vis = np.log1p(density)
    d_max = d_vis.max()
    wall_mask = d_vis > (d_max * 0.4)
    ax.imshow(wall_mask.astype(float), cmap='gray_r', origin='upper', alpha=0.5)
    ax.imshow(d_vis, cmap='gray_r', origin='upper', alpha=0.5)
    ax.set_title("Mesh-based walls (high vertex density)", fontsize=11)
    ax.set_aspect('equal')
    
    # Photo-based walls
    ax = axes[1]
    fd_vis = np.log1p(feat_density)
    ax.imshow(fd_vis, cmap='hot', origin='upper')
    for seg in wall_segments:
        ax.plot([seg['px1'], seg['px2']], [seg['py1'], seg['py2']],
                'lime', linewidth=2, alpha=0.8)
    ax.set_title(f"Photo-based walls ({len(wall_segments)} segments)", fontsize=11)
    ax.set_aspect('equal')
    
    fig.suptitle("Wall Comparison: Mesh vs Photo Features", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_room_examples(room_examples, out_dir):
    """Save per-room feature detection examples."""
    for rid, ex in room_examples.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original
        ax = axes[0]
        ax.imshow(cv2.cvtColor(ex['img'], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Room {rid} — Frame {ex['frame_idx']}", fontsize=10)
        ax.axis('off')
        
        # Edges
        ax = axes[1]
        ax.imshow(ex['edges'], cmap='gray')
        ax.set_title("Canny Edges", fontsize=10)
        ax.axis('off')
        
        # Lines overlay
        ax = axes[2]
        overlay = ex['img'].copy()
        for line in ex['lines']:
            color = (0, 255, 0) if line['type'] == 'H' else (0, 100, 255)
            cv2.line(overlay, (line['x1'], line['y1']), (line['x2'], line['y2']), color, 2)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Structural Lines: {len(ex['lines'])} (green=H, orange=V)", fontsize=10)
        ax.axis('off')
        
        fig.tight_layout()
        path = out_dir / f"detected_features_room_{rid}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Saved: {path}")


def main():
    print("=" * 60)
    print("mesh2plan v46 - Multimodal Stage 3")
    print("Photo Feature Detection → 3D Projection")
    print("=" * 60)
    
    # Load mesh for density reference
    mesh = load_mesh()
    density, transform = mesh_to_density(mesh)
    
    # Load room-frame mapping from Stage 1
    room_frames_json = load_room_frames()
    print(f"Loaded room mappings: {len(room_frames_json)} rooms")
    
    # Step 1: Select best frames per room
    print("\n--- Selecting best frames ---")
    selected = select_best_frames(room_frames_json, n_per_room=8)
    
    # Step 2-3: Detect features and project to 3D
    print("\n--- Detecting features and projecting to 3D ---")
    all_points, room_examples = process_all_frames(selected)
    
    # Step 3: Create feature density
    print("\n--- Creating feature density map ---")
    feat_density = create_feature_density(all_points, density, transform)
    
    # Step 4: Extract wall segments
    print("\n--- Extracting wall segments ---")
    wall_segments = extract_wall_segments_from_density(feat_density, transform)
    
    # Step 5: Detect doors/windows
    print("\n--- Detecting doors/windows ---")
    dw_features = detect_doors_windows(all_points, wall_segments)
    
    # === Outputs ===
    print("\n--- Generating outputs ---")
    
    # Feature density plot
    fd_path = OUT_DIR / "feature_density.png"
    plot_feature_density(feat_density, density, transform, wall_segments, fd_path)
    
    # Wall comparison
    plot_wall_comparison(feat_density, density, wall_segments, OUT_DIR / "wall_comparison.png")
    
    # Per-room examples
    plot_room_examples(room_examples, OUT_DIR)
    
    # Save 3D points (sampled to keep file size reasonable)
    pts_out = all_points
    if len(pts_out) > 50000:
        indices = np.random.choice(len(pts_out), 50000, replace=False)
        pts_out = [all_points[i] for i in indices]
    with open(OUT_DIR / "feature_points_3d.json", 'w') as f:
        json.dump({'n_total': len(all_points), 'points': pts_out}, f)
    print(f"Saved: {OUT_DIR / 'feature_points_3d.json'} ({len(pts_out)} points)")
    
    # Save wall segments
    wall_out = {
        'n_segments': len(wall_segments),
        'segments': wall_segments,
        'door_window_candidates': dw_features,
    }
    with open(OUT_DIR / "wall_segments.json", 'w') as f:
        json.dump(wall_out, f, indent=2)
    print(f"Saved: {OUT_DIR / 'wall_segments.json'}")
    
    # Copy to workspace
    workspace = Path(os.path.expanduser("~/.openclaw/workspace"))
    shutil.copy2(fd_path, workspace / "v46_features.png")
    print(f"Copied feature_density.png to workspace")
    
    print("\n" + "=" * 60)
    print(f"Stage 3 complete! {len(all_points)} feature points, {len(wall_segments)} wall segments, {len(dw_features)} door/window candidates")
    print(f"Outputs in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

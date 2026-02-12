#!/usr/bin/env python3
"""
mesh2plan v44 - Multimodal Stage 1

Uses ALL available scan data: mesh, camera poses, photos, depth maps.
Stage 1: understand the data, map cameras to rooms, sample photos, analyze depth.

Outputs to results/v44_stage1/
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

# Paths
BASE = Path(__file__).parent.parent
SCAN_DIR = BASE / "data" / "multiroom" / "2026_02_10_18_31_36"
OUT_DIR = BASE / "results" / "v44_stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 0.02  # meters per pixel


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


def get_apartment_mask(density):
    mask = (density >= 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 1, -1)
    return mask


def watershed_rooms(density, mask, n_target=6, min_dist=25):
    """Density ridge watershed (from v39)."""
    d = density.copy()
    d[mask == 0] = 0
    d_smooth = cv2.GaussianBlur(d, (11, 11), 3)
    
    masked_vals = d_smooth[mask > 0]
    median = np.median(masked_vals[masked_vals > 0]) if np.any(masked_vals > 0) else 1
    floor_mask = ((d_smooth < median) & (mask > 0)).astype(np.uint8)
    
    dist = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
    boundary_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    combined = dist * 0.7 + boundary_dist * 0.3
    combined[mask == 0] = 0
    
    # Find seeds
    for md in [min_dist, min_dist - 5, min_dist - 10, 10, 5]:
        if md < 5:
            md = 5
        ks = md * 2 + 1
        local_max = ndimage.maximum_filter(combined, size=ks)
        peaks = (combined == local_max) & (combined > md)
        n_labels, peak_labels = cv2.connectedComponents(peaks.astype(np.uint8))
        if n_labels - 1 >= n_target:
            break
    
    seeds = []
    for lbl in range(1, n_labels):
        ys, xs = np.where(peak_labels == lbl)
        best = np.argmax(combined[ys, xs])
        seeds.append((xs[best], ys[best], combined[ys[best], xs[best]]))
    seeds.sort(key=lambda s: s[2], reverse=True)
    if len(seeds) > n_target * 2:
        seeds = seeds[:n_target * 2]
    
    # Watershed
    d_norm = np.zeros_like(d_smooth)
    d_max = d_smooth[mask > 0].max() if np.any(mask > 0) else 1
    d_norm[mask > 0] = (d_smooth[mask > 0] / d_max * 255)
    d_uint8 = d_norm.astype(np.uint8)
    
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1
    for i, (sx, sy, _) in enumerate(seeds):
        cv2.circle(markers, (sx, sy), 3, i + 2, -1)
    
    grad_color = cv2.cvtColor(d_uint8, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(grad_color, markers.copy())
    
    # Extract rooms
    min_px = int(2.5 / (RESOLUTION * RESOLUTION))
    rooms = []
    for i in range(len(seeds)):
        lbl = i + 2
        room_mask = ((ws == lbl) & (mask > 0)).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px >= min_px:
            rooms.append({'id': len(rooms), 'mask': room_mask, 'area_m2': area_px * RESOLUTION * RESOLUTION})
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    for i, r in enumerate(rooms):
        r['id'] = i
    
    # Merge small rooms
    changed = True
    while changed:
        changed = False
        small = [r for r in rooms if r['area_m2'] < 2.5]
        if not small:
            break
        for sr in small:
            dilated = cv2.dilate(sr['mask'], np.ones((7, 7), np.uint8))
            best, best_ov = None, 0
            for r in rooms:
                if r is sr:
                    continue
                ov = (dilated & r['mask']).sum()
                if ov > best_ov:
                    best_ov = ov
                    best = r
            if best and best_ov > 0:
                best['mask'] = best['mask'] | sr['mask']
                best['area_m2'] = best['mask'].sum() * RESOLUTION * RESOLUTION
                rooms.remove(sr)
                changed = True
                break
    
    for i, r in enumerate(rooms):
        r['id'] = i
    
    print(f"Watershed: {len(rooms)} rooms")
    for r in rooms:
        print(f"  Room {r['id']}: {r['area_m2']:.1f} m²")
    
    return rooms


# ============================================================
# Task 1: Camera trajectory mapping
# ============================================================
def load_camera_poses():
    """Load all frame JSONs, extract camera poses."""
    frames = []
    json_files = sorted(glob.glob(str(SCAN_DIR / "frame_*.json")))
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        pose = np.array(data['cameraPoseARFrame']).reshape(4, 4, order='C')  # row-major
        pos = pose[:3, 3]  # translation = last column
        frames.append({
            'index': data.get('frame_index', int(Path(jf).stem.split('_')[1])),
            'pos': pos,
            'pose': pose,
            'time': data.get('time', 0),
            'motion_quality': data.get('motionQuality', 0),
            'velocity': data.get('averageVelocity', 0),
            'has_photo': (SCAN_DIR / f"frame_{data.get('frame_index', int(Path(jf).stem.split('_')[1])):05d}.jpg").exists(),
            'json_path': jf,
        })
    print(f"Loaded {len(frames)} camera frames")
    return frames


def world_to_pixel(x, z, transform):
    x_min, z_min, res = transform
    px = int((x - x_min) / res)
    py = int((z - z_min) / res)
    return px, py


def assign_cameras_to_rooms(frames, rooms, transform):
    """Assign each frame to a room based on camera XZ position."""
    room_frames = {r['id']: [] for r in rooms}
    unassigned = []
    
    for fr in frames:
        x, z = fr['pos'][0], fr['pos'][2]
        px, py = world_to_pixel(x, z, transform)
        
        assigned = False
        for r in rooms:
            m = r['mask']
            if 0 <= py < m.shape[0] and 0 <= px < m.shape[1] and m[py, px] > 0:
                fr['room_id'] = r['id']
                room_frames[r['id']].append(fr)
                assigned = True
                break
        
        if not assigned:
            # Find nearest room
            min_dist = float('inf')
            best_room = None
            for r in rooms:
                ys, xs = np.where(r['mask'] > 0)
                if len(xs) == 0:
                    continue
                dists = (xs - px)**2 + (ys - py)**2
                d = dists.min()
                if d < min_dist:
                    min_dist = d
                    best_room = r['id']
            if best_room is not None:
                fr['room_id'] = best_room
                room_frames[best_room].append(fr)
            else:
                fr['room_id'] = -1
                unassigned.append(fr)
    
    print(f"\nCamera-to-room assignment:")
    for rid, frs in room_frames.items():
        n_photos = sum(1 for f in frs if f['has_photo'])
        print(f"  Room {rid}: {len(frs)} frames ({n_photos} with photos)")
    if unassigned:
        print(f"  Unassigned: {len(unassigned)}")
    
    return room_frames


def plot_trajectory(frames, rooms, density, transform, out_path):
    """Plot camera trajectory on density map, colored by room."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # Show density
    d_vis = np.log1p(density)
    ax.imshow(d_vis, cmap='gray_r', origin='upper')
    
    # Room colors
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(rooms), 1)))
    
    # Draw room boundaries lightly
    for r in rooms:
        contours, _ = cv2.findContours(r['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.reshape(-1, 2)
            ax.plot(pts[:, 0], pts[:, 1], color=colors[r['id'] % len(colors)], alpha=0.3, linewidth=1)
    
    # Plot camera positions
    for fr in frames:
        x, z = fr['pos'][0], fr['pos'][2]
        px, py = world_to_pixel(x, z, transform)
        rid = fr.get('room_id', -1)
        c = colors[rid % len(colors)] if rid >= 0 else 'gray'
        marker = 'o' if fr['has_photo'] else '.'
        ms = 4 if fr['has_photo'] else 2
        ax.plot(px, py, marker, color=c, markersize=ms, alpha=0.6)
    
    # Connect trajectory with lines
    pxs = [world_to_pixel(fr['pos'][0], fr['pos'][2], transform) for fr in frames]
    ax.plot([p[0] for p in pxs], [p[1] for p in pxs], '-', color='blue', alpha=0.15, linewidth=0.5)
    
    # Legend
    for r in rooms:
        ax.plot([], [], 'o', color=colors[r['id'] % len(colors)], label=f"Room {r['id']} ({r['area_m2']:.1f}m²)")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"Camera Trajectory — {len(frames)} frames, {len(rooms)} rooms")
    ax.set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Task 3: Photo sampling per room
# ============================================================
def sample_photos_per_room(room_frames, n_per_room=4):
    """Pick representative photos per room: good quality, spread angles."""
    sampled = {}
    for rid, frs in room_frames.items():
        with_photo = [f for f in frs if f['has_photo']]
        if not with_photo:
            sampled[rid] = []
            continue
        
        # Sort by motion quality descending
        with_photo.sort(key=lambda f: f['motion_quality'], reverse=True)
        
        # Take top quality half, then spread by time
        top = with_photo[:max(len(with_photo) // 2, n_per_room * 2)]
        top.sort(key=lambda f: f['time'])
        
        # Evenly space
        if len(top) <= n_per_room:
            sampled[rid] = top
        else:
            indices = np.linspace(0, len(top) - 1, n_per_room, dtype=int)
            sampled[rid] = [top[i] for i in indices]
    
    return sampled


def make_contact_sheet(sampled, rid, out_path):
    """Create contact sheet of sampled photos for a room."""
    frames = sampled.get(rid, [])
    if not frames:
        return
    
    n = len(frames)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        
        img_path = SCAN_DIR / f"frame_{fr['index']:05d}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Downsample for display
            h, w = img.shape[:2]
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ax.imshow(img)
        ax.set_title(f"Frame {fr['index']} q={fr['motion_quality']:.2f}", fontsize=9)
        ax.axis('off')
    
    # Hide empty
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')
    
    fig.suptitle(f"Room {rid} — {n} sample photos", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Task 4: Depth map analysis
# ============================================================
def analyze_depth():
    """Analyze depth map format and project to 3D."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    sample_indices = [0, 100, 250]
    
    for col, idx in enumerate(sample_indices):
        depth_path = SCAN_DIR / f"depth_{idx:05d}.png"
        conf_path = SCAN_DIR / f"conf_{idx:05d}.png"
        
        if not depth_path.exists():
            continue
        
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED) if conf_path.exists() else None
        
        # Depth visualization
        ax = axes[0, col]
        d_vis = depth.astype(float)
        d_vis[depth == 0] = np.nan
        im = ax.imshow(d_vis, cmap='turbo')
        ax.set_title(f"Depth {idx}\n{depth.dtype} {depth.shape}\nmin={depth[depth>0].min()} max={depth.max()}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Confidence visualization
        ax = axes[1, col]
        if conf is not None:
            ax.imshow(conf, cmap='RdYlGn', vmin=0, vmax=2)
            ax.set_title(f"Confidence {idx}\n{conf.dtype} {conf.shape}\nvals: {np.unique(conf)}", fontsize=9)
        
        # 3D projection test (for first sample)
        if col == 0:
            print(f"\nDepth analysis (frame {idx}):")
            print(f"  dtype={depth.dtype}, shape={depth.shape}")
            print(f"  Range: {depth[depth>0].min()} - {depth.max()}")
            print(f"  Likely millimeters: {depth[depth>0].min()/1000:.3f}m - {depth.max()/1000:.3f}m")
            if conf is not None:
                print(f"  Confidence values: {np.unique(conf)} (0=low, 1=med, 2=high)")
                for v in np.unique(conf):
                    print(f"    conf={v}: {(conf==v).sum()} px ({(conf==v).sum()*100/conf.size:.1f}%)")
    
    fig.suptitle("Depth Map Analysis", fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / "depth_analysis.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")
    
    # 3D projection test
    depth = cv2.imread(str(SCAN_DIR / "depth_00000.png"), cv2.IMREAD_UNCHANGED).astype(float)
    # Assume millimeters
    depth_m = depth / 1000.0
    
    # Intrinsics from frame JSON — but depth is 256x192, photos are 4032x3024
    # Scale intrinsics: depth_w/photo_w
    fx_photo, fy_photo = 2812.42, 2812.42
    cx_photo, cy_photo = 2013.95, 1509.67
    dh, dw = depth.shape
    # Photo is 4032x3024, depth is 256x192
    scale_x = dw / 4032.0
    scale_y = dh / 3024.0
    fx = fx_photo * scale_x
    fy = fy_photo * scale_y
    cx = cx_photo * scale_x
    cy = cy_photo * scale_y
    print(f"\n  Depth intrinsics (scaled): fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Project to 3D (camera frame)
    u, v = np.meshgrid(np.arange(dw), np.arange(dh))
    valid = depth_m > 0.01
    x3d = (u[valid] - cx) * depth_m[valid] / fx
    y3d = (v[valid] - cy) * depth_m[valid] / fy
    z3d = depth_m[valid]
    print(f"  Projected {valid.sum()} points to 3D")
    print(f"  X range: {x3d.min():.2f} to {x3d.max():.2f} m")
    print(f"  Y range: {y3d.min():.2f} to {y3d.max():.2f} m")
    print(f"  Z range: {z3d.min():.2f} to {z3d.max():.2f} m")


# ============================================================
# Task 5: Feature detection
# ============================================================
def detect_features():
    """Line detection on sample photos."""
    sample_indices = [0, 50, 150]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for col, idx in enumerate(sample_indices):
        img_path = SCAN_DIR / f"frame_{idx:05d}.jpg"
        if not img_path.exists():
            # Find nearest existing
            for offset in range(20):
                for d in [0, offset, -offset]:
                    p = SCAN_DIR / f"frame_{idx+d:05d}.jpg"
                    if p.exists():
                        img_path = p
                        idx = idx + d
                        break
                if img_path.exists():
                    break
        
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        # Downsample
        scale = 1024 / max(img.shape[:2])
        img_small = cv2.resize(img, None, fx=scale, fy=scale)
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=50, maxLineGap=10)
        
        # Original
        ax = axes[0, col]
        ax.imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {idx}", fontsize=10)
        ax.axis('off')
        
        # Lines overlay
        ax = axes[1, col]
        line_img = img_small.copy()
        n_lines = 0
        if lines is not None:
            # Filter: keep long, roughly H or V lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
                # Keep lines that are roughly horizontal or vertical
                if angle < 15 or angle > 75:
                    color = (0, 255, 0) if angle < 15 else (0, 0, 255)
                    cv2.line(line_img, (x1, y1), (x2, y2), color, 2)
                    n_lines += 1
        
        ax.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Lines: {n_lines} (green=H, blue=V)", fontsize=10)
        ax.axis('off')
    
    fig.suptitle("Feature Detection — Hough Lines on Photos", fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / "feature_detection.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("mesh2plan v44 - Multimodal Stage 1")
    print("=" * 60)
    
    # Load mesh and build density
    mesh = load_mesh()
    density, transform = mesh_to_density(mesh)
    mask = get_apartment_mask(density)
    
    # Room segmentation via watershed
    rooms = watershed_rooms(density, mask)
    
    # Task 1: Load camera poses
    frames = load_camera_poses()
    
    # Task 2: Assign cameras to rooms
    room_frames = assign_cameras_to_rooms(frames, rooms, transform)
    
    # Plot trajectory
    traj_path = OUT_DIR / "camera_trajectory.png"
    plot_trajectory(frames, rooms, density, transform, traj_path)
    
    # Copy to workspace
    import shutil
    shutil.copy2(traj_path, os.path.expanduser("~/.openclaw/workspace/v44_trajectory.png"))
    print("Copied trajectory to workspace")
    
    # Save room_frames.json
    rf_out = {}
    for rid, frs in room_frames.items():
        rf_out[str(rid)] = {
            'frame_indices': [f['index'] for f in frs],
            'n_frames': len(frs),
            'n_photos': sum(1 for f in frs if f['has_photo']),
            'avg_quality': np.mean([f['motion_quality'] for f in frs]) if frs else 0,
        }
    with open(OUT_DIR / "room_frames.json", 'w') as f:
        json.dump(rf_out, f, indent=2)
    print(f"Saved: {OUT_DIR / 'room_frames.json'}")
    
    # Task 3: Photo sampling + contact sheets
    sampled = sample_photos_per_room(room_frames)
    for rid in room_frames:
        if sampled.get(rid):
            make_contact_sheet(sampled, rid, OUT_DIR / f"contact_sheet_room_{rid}.png")
    
    # Task 4: Depth analysis
    analyze_depth()
    
    # Task 5: Feature detection
    detect_features()
    
    print("\n" + "=" * 60)
    print("Stage 1 complete! Outputs in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()

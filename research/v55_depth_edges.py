#!/usr/bin/env python3
"""
mesh2plan v55 - Depth Map Wall Edge Detection

NEW APPROACH: Use depth maps to find wall edges directly.
For each depth frame:
1. Compute depth gradients (Sobel) → large gradients = depth discontinuities = wall edges
2. Also find vertical surfaces: depth changes smoothly along walls but jumps at edges
3. Project wall-edge pixels to 3D using depth + intrinsics + camera pose
4. Accumulate 3D wall-edge points across all 556 frames
5. Project to XZ (top-down) → wall-edge density image
6. Hough on edge density → wall lines → room partition (reuse v52 approach)

Key insight: Depth discontinuities in the depth map correspond to wall-wall and wall-floor
junctions. These are MORE RELIABLE than mesh density because they're direct measurements.
"""

import numpy as np
import json
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse
from scipy import ndimage


def load_frame(data_dir, idx):
    """Load frame JSON, depth map, and confidence map."""
    frame_path = data_dir / f'frame_{idx:05d}.json'
    depth_path = data_dir / f'depth_{idx:05d}.png'
    conf_path = data_dir / f'conf_{idx:05d}.png'
    
    if not frame_path.exists() or not depth_path.exists():
        return None, None, None
    
    frame = json.load(open(frame_path))
    depth = np.array(Image.open(depth_path)).astype(np.float32)  # uint16 mm
    depth = depth / 1000.0  # → meters
    
    conf = None
    if conf_path.exists():
        conf = np.array(Image.open(conf_path))
    
    return frame, depth, conf


def depth_to_3d(depth, intrinsics_flat, pose, conf=None, conf_min=1):
    """Project depth map pixels to 3D world coordinates.
    
    intrinsics_flat: [fx, 0, cx, 0, fy, cy, 0, 0, 1] (full-res)
    depth: (192, 256) in meters
    pose: 4x4 camera-to-world (row-major ARKit)
    
    Returns: Nx3 world points
    """
    H, W = depth.shape
    fx_full, _, cx_full, _, fy_full, cy_full = intrinsics_flat[:6]
    
    # Scale intrinsics from full-res (4032x3024) to depth (256x192)
    # Note: depth is 192(H) x 256(W), photo is 3024(H) x 4032(W)
    sx = W / 4032.0
    sy = H / 3024.0
    fx = fx_full * sx
    fy = fy_full * sy
    cx = cx_full * sx
    cy = cy_full * sy
    
    # Create pixel grid
    v, u = np.mgrid[0:H, 0:W]
    
    # Mask: valid depth + confidence
    mask = depth > 0.1
    if conf is not None:
        mask &= conf >= conf_min
    
    u_valid = u[mask].astype(np.float32)
    v_valid = v[mask].astype(np.float32)
    d_valid = depth[mask]
    
    # Unproject to camera space
    x_cam = (u_valid - cx) * d_valid / fx
    y_cam = (v_valid - cy) * d_valid / fy
    z_cam = d_valid
    
    # Camera coords: (N, 4) homogeneous
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    
    # Transform to world
    pts_world = (pose @ pts_cam.T).T[:, :3]
    
    return pts_world


def find_depth_edges(depth, conf=None, gradient_thresh=0.15):
    """Find wall edges in depth map using gradient analysis.
    
    Returns: binary mask of edge pixels
    """
    # Smooth slightly to reduce noise
    depth_smooth = cv2.GaussianBlur(depth, (3, 3), 0.5)
    
    # Sobel gradients
    gx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize by depth (farther walls have smaller absolute gradients)
    depth_safe = np.maximum(depth_smooth, 0.3)
    grad_normalized = grad_mag / depth_safe
    
    # Threshold
    edge_mask = grad_normalized > gradient_thresh
    
    # Only where depth is valid
    edge_mask &= depth > 0.1
    if conf is not None:
        edge_mask &= conf >= 1
    
    return edge_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Scan data directory')
    parser.add_argument('--mesh', help='Mesh file for apartment mask', default=None)
    parser.add_argument('--output-dir', default='results/v55')
    parser.add_argument('--max-frames', type=int, default=556)
    parser.add_argument('--resolution', type=float, default=0.02, help='m/pixel')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Phase 1: Accumulate 3D wall-edge points ──
    print("Phase 1: Extracting depth edges from all frames...")
    all_edge_pts = []
    all_wall_pts = []  # Also collect all depth points for wall-density comparison
    
    n_processed = 0
    for idx in range(args.max_frames):
        frame, depth, conf = load_frame(data_dir, idx)
        if frame is None:
            continue
        
        pose = np.array(frame['cameraPoseARFrame']).reshape(4, 4, order='C')
        intrinsics = frame['intrinsics']
        
        # Find depth edges
        edge_mask = find_depth_edges(depth, conf, gradient_thresh=0.12)
        
        if edge_mask.sum() < 10:
            continue
        
        # Project edge pixels to 3D
        H, W = depth.shape
        fx_full, _, cx_full, _, fy_full, cy_full = intrinsics[:6]
        sx, sy = W / 4032.0, H / 3024.0
        fx, fy = fx_full * sx, fy_full * sy
        cx, cy = cx_full * sx, cy_full * sy
        
        v, u = np.mgrid[0:H, 0:W]
        
        # Edge points
        eu, ev = u[edge_mask], v[edge_mask]
        ed = depth[edge_mask]
        x_cam = (eu.astype(np.float32) - cx) * ed / fx
        y_cam = (ev.astype(np.float32) - cy) * ed / fy
        z_cam = ed
        pts = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
        pts_world = (pose @ pts.T).T[:, :3]
        
        # Filter: only points at wall height (Y between floor and ceiling)
        # ARKit Y is up, but this scan may have different orientation
        # We'll filter after accumulation based on histogram
        all_edge_pts.append(pts_world)
        
        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  {n_processed} frames, {sum(len(p) for p in all_edge_pts)} edge points")
    
    all_edge_pts = np.vstack(all_edge_pts)
    print(f"  Total: {n_processed} frames, {len(all_edge_pts)} edge points")
    
    # ── Phase 2: Filter to wall-height points ──
    # Find the dominant Y range (wall height)
    y_vals = all_edge_pts[:, 1]
    print(f"  Y range: {y_vals.min():.2f} to {y_vals.max():.2f}")
    
    # Use middle 60% of Y range (skip floor and ceiling edges)
    y_lo, y_hi = np.percentile(y_vals, [20, 80])
    wall_mask = (y_vals >= y_lo) & (y_vals <= y_hi)
    wall_pts = all_edge_pts[wall_mask]
    print(f"  Wall-height filter ({y_lo:.2f} to {y_hi:.2f}): {len(wall_pts)} points")
    
    # ── Phase 3: Project to XZ density image ──
    x_vals = wall_pts[:, 0]
    z_vals = wall_pts[:, 2]
    
    res = args.resolution
    x_min, x_max = x_vals.min() - 0.5, x_vals.max() + 0.5
    z_min, z_max = z_vals.min() - 0.5, z_vals.max() + 0.5
    
    W_img = int((x_max - x_min) / res) + 1
    H_img = int((z_max - z_min) / res) + 1
    
    # Clamp image size
    if W_img > 1000 or H_img > 1000:
        res = max((x_max - x_min), (z_max - z_min)) / 800
        W_img = int((x_max - x_min) / res) + 1
        H_img = int((z_max - z_min) / res) + 1
    
    density = np.zeros((H_img, W_img), dtype=np.float32)
    
    xi = ((x_vals - x_min) / res).astype(int)
    zi = ((z_vals - z_min) / res).astype(int)
    xi = np.clip(xi, 0, W_img - 1)
    zi = np.clip(zi, 0, H_img - 1)
    
    np.add.at(density, (zi, xi), 1)
    
    print(f"  Edge density image: {W_img}×{H_img}, max={density.max():.0f}")
    
    # ── Phase 4: Also build mesh wall-density for comparison ──
    # Load mesh for apartment mask
    mesh_wall_density = None
    if args.mesh:
        import trimesh
        mesh = trimesh.load(args.mesh, process=False)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Wall faces by normal
        if hasattr(mesh, 'face_normals'):
            normals = mesh.face_normals
            wall_face_mask = np.abs(normals[:, 1]) < 0.5
            wall_face_idx = np.where(wall_face_mask)[0]
            centroids = verts[faces[wall_face_idx]].mean(axis=1)
            
            mesh_wall_density = np.zeros((H_img, W_img), dtype=np.float32)
            mxi = ((centroids[:, 0] - x_min) / res).astype(int)
            mzi = ((centroids[:, 2] - z_min) / res).astype(int)
            valid = (mxi >= 0) & (mxi < W_img) & (mzi >= 0) & (mzi < H_img)
            np.add.at(mesh_wall_density, (mzi[valid], mxi[valid]), 1)
    
    # ── Phase 5: Hough line detection on edge density ──
    # Normalize density to uint8
    edge_img = density.copy()
    edge_img = np.minimum(edge_img, np.percentile(edge_img[edge_img > 0], 95))
    edge_img = (edge_img / edge_img.max() * 255).astype(np.uint8) if edge_img.max() > 0 else edge_img.astype(np.uint8)
    
    # Threshold to binary wall mask
    thresh = max(1, int(np.percentile(edge_img[edge_img > 0], 70)))
    wall_binary = (edge_img >= thresh).astype(np.uint8) * 255
    
    # Hough lines
    lines = cv2.HoughLines(wall_binary, 1, np.pi/180, threshold=50)
    
    if lines is None:
        print("  No Hough lines found!")
        return
    
    lines = lines[:, 0]
    print(f"  {len(lines)} Hough lines detected")
    
    # Find dominant angles
    angles_deg = np.degrees(lines[:, 1]) % 180
    
    # Cluster angles into families (expect ~2 perpendicular families)
    from scipy.cluster.hierarchy import fcluster, linkage
    
    # Use circular distance for angles
    angles_for_cluster = angles_deg.reshape(-1, 1)
    Z = linkage(angles_for_cluster, method='complete')
    clusters = fcluster(Z, t=15, criterion='distance')
    
    families = {}
    for c in np.unique(clusters):
        mask = clusters == c
        mean_angle = np.median(angles_deg[mask])
        families[c] = {
            'angle': mean_angle,
            'lines': lines[mask],
            'count': mask.sum()
        }
    
    # Sort by count, keep top 2
    sorted_fams = sorted(families.values(), key=lambda f: f['count'], reverse=True)[:2]
    print(f"  Dominant angles: {sorted_fams[0]['angle']:.0f}° ({sorted_fams[0]['count']} lines), "
          f"{sorted_fams[1]['angle']:.0f}° ({sorted_fams[1]['count']} lines)")
    
    # ── Phase 6: Cluster walls by perpendicular offset (like v52) ──
    all_walls = []
    for fam_idx, fam in enumerate(sorted_fams):
        angle_rad = np.radians(fam['angle'])
        rhos = fam['lines'][:, 0]
        
        # Sort by rho and cluster
        sorted_idx = np.argsort(rhos)
        sorted_rhos = rhos[sorted_idx]
        
        # Merge walls within 15px (0.3m)
        min_gap_px = int(0.4 / res)
        wall_groups = []
        current = [sorted_rhos[0]]
        for r in sorted_rhos[1:]:
            if r - current[-1] < min_gap_px:
                current.append(r)
            else:
                wall_groups.append(np.mean(current))
                current = [r]
        wall_groups.append(np.mean(current))
        
        # Score each wall by density along it
        for rho in wall_groups:
            cos_t = np.cos(angle_rad)
            sin_t = np.sin(angle_rad)
            
            # Sample density along line
            score = 0
            n_samples = 0
            for t in range(-max(H_img, W_img), max(H_img, W_img), 2):
                px = int(rho * cos_t - t * sin_t)
                py = int(rho * sin_t + t * cos_t)
                if 0 <= px < W_img and 0 <= py < H_img:
                    score += density[py, px]
                    n_samples += 1
            
            offset_m = rho * res + (x_min if abs(cos_t) > 0.5 else z_min)
            all_walls.append({
                'family': fam_idx,
                'angle': fam['angle'],
                'rho': rho,
                'score': score,
                'offset_m': offset_m,
            })
    
    # Sort by score, keep top walls
    all_walls.sort(key=lambda w: w['score'], reverse=True)
    
    # Keep walls above mean score
    scores = [w['score'] for w in all_walls]
    mean_score = np.mean(scores)
    good_walls = [w for w in all_walls if w['score'] > mean_score * 0.5]
    
    # Ensure at least 3 per family
    for fam_idx in range(2):
        fam_walls = [w for w in good_walls if w['family'] == fam_idx]
        if len(fam_walls) < 3:
            extras = [w for w in all_walls if w['family'] == fam_idx and w not in good_walls]
            good_walls.extend(extras[:3 - len(fam_walls)])
    
    print(f"  {len(good_walls)} walls selected:")
    for w in sorted(good_walls, key=lambda w: (w['family'], w['rho'])):
        print(f"    Fam {w['family']} ({w['angle']:.0f}°) rho={w['rho']:.0f} score={w['score']:.0f} offset={w['offset_m']:.2f}m")
    
    # ── Phase 7: Build room partition from wall lines ──
    # Draw walls on apartment mask
    # First create apartment mask from edge density
    all_density = density.copy()
    if mesh_wall_density is not None:
        all_density = np.maximum(density, mesh_wall_density * 0.5)
    
    apt_mask = (ndimage.gaussian_filter(all_density, sigma=5) > 0.5).astype(np.uint8)
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    apt_mask = ndimage.binary_fill_holes(apt_mask).astype(np.uint8)
    
    # Draw wall lines on mask
    wall_img = np.zeros((H_img, W_img), dtype=np.uint8)
    for w in good_walls:
        angle_rad = np.radians(w['angle'])
        rho = w['rho']
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)
        
        # Line endpoints
        t_range = max(H_img, W_img)
        x0 = int(rho * cos_t - (-t_range) * sin_t)
        y0 = int(rho * sin_t + (-t_range) * cos_t)
        x1 = int(rho * cos_t - t_range * sin_t)
        y1 = int(rho * sin_t + t_range * cos_t)
        
        cv2.line(wall_img, (x0, y0), (x1, y1), 255, thickness=3)
    
    # Mask to apartment
    wall_img = wall_img & (apt_mask * 255)
    
    # Also add apartment boundary as wall
    contours, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(wall_img, contours, -1, 255, thickness=3)
    
    # Connected components of non-wall interior
    interior = apt_mask.copy()
    interior[wall_img > 0] = 0
    
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  {n_labels - 1} raw room regions")
    
    # Filter small regions and classify
    rooms = []
    for i in range(1, n_labels):
        area_px = (labels == i).sum()
        area_m2 = area_px * res * res
        if area_m2 < 1.0:
            continue
        
        # Centroid
        ys, xs = np.where(labels == i)
        cx_px, cy_px = xs.mean(), ys.mean()
        cx_m = cx_px * res + x_min
        cy_m = cy_px * res + z_min
        
        # Aspect ratio for classification
        if len(xs) > 0:
            bbox_w = (xs.max() - xs.min()) * res
            bbox_h = (ys.max() - ys.min()) * res
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 0.1)
        else:
            aspect = 1
        
        # Classify
        if area_m2 > 8:
            name = "Room"
        elif area_m2 > 5:
            name = "Room"
        elif aspect > 2.5:
            name = "Hallway"
        elif area_m2 < 3:
            name = "Closet"
        else:
            name = "Bathroom"
        
        rooms.append({
            'label': i,
            'area_m2': area_m2,
            'cx': cx_m, 'cy': cy_m,
            'cx_px': cx_px, 'cy_px': cy_px,
            'aspect': aspect,
            'name': name,
        })
    
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    
    # Deduplicate names
    name_counts = {}
    for r in rooms:
        n = r['name']
        name_counts[n] = name_counts.get(n, 0) + 1
        if name_counts[n] > 1:
            r['name'] = f"{n} {name_counts[n]}"
    
    total = sum(r['area_m2'] for r in rooms)
    print(f"\n  {len(rooms)} rooms, {total:.1f}m² total")
    for r in rooms:
        print(f"    {r['name']}: {r['area_m2']:.1f}m² (aspect={r['aspect']:.1f})")
    
    # ── Phase 8: Render ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # 1. Raw edge density
    ax = axes[0]
    ax.set_title('Depth Edge Density')
    ax.imshow(np.log1p(density), cmap='hot', origin='lower')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # 2. Wall lines on density
    ax = axes[1]
    ax.set_title(f'Wall Lines ({len(good_walls)} walls)')
    ax.imshow(np.log1p(density), cmap='gray', origin='lower', alpha=0.5)
    for w in good_walls:
        angle_rad = np.radians(w['angle'])
        rho = w['rho']
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        t = max(H_img, W_img)
        x0, y0 = rho*cos_t + t*sin_t, rho*sin_t - t*cos_t
        x1, y1 = rho*cos_t - t*sin_t, rho*sin_t + t*cos_t
        color = 'red' if w['family'] == 0 else 'blue'
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1, alpha=0.7)
    ax.set_xlim(0, W_img)
    ax.set_ylim(0, H_img)
    
    # 3. Room partition
    ax = axes[2]
    angles_str = f"{sorted_fams[0]['angle']:.0f}°, {sorted_fams[1]['angle']:.0f}°"
    ax.set_title(f'v55 Depth Edges — {len(rooms)} rooms, {total:.1f}m²\nWall angles: {angles_str}')
    
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 3)))
    room_img = np.ones((H_img, W_img, 3))
    for i, r in enumerate(rooms):
        mask = labels == r['label']
        room_img[mask] = colors[i][:3]
    
    ax.imshow(room_img, origin='lower')
    
    # Draw walls
    for w in good_walls:
        angle_rad = np.radians(w['angle'])
        rho = w['rho']
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        t = max(H_img, W_img)
        x0, y0 = rho*cos_t + t*sin_t, rho*sin_t - t*cos_t
        x1, y1 = rho*cos_t - t*sin_t, rho*sin_t + t*cos_t
        ax.plot([x0, x1], [y0, y1], 'k-', linewidth=2)
    
    # Room labels
    for r in rooms:
        ax.text(r['cx_px'], r['cy_px'], f"{r['name']}\n{r['area_m2']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Scale bar
    bar_px = 1.0 / res
    ax.plot([10, 10 + bar_px], [10, 10], 'k-', linewidth=3)
    ax.text(10 + bar_px/2, 20, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir / 'floorplan.png'}")
    
    # Also save comparison if mesh density available
    if mesh_wall_density is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
        axes2[0].set_title('Mesh Wall Density (v41b style)')
        axes2[0].imshow(np.log1p(mesh_wall_density), cmap='hot', origin='lower')
        axes2[1].set_title('Depth Edge Density (v55)')
        axes2[1].imshow(np.log1p(density), cmap='hot', origin='lower')
        plt.savefig(out_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {out_dir / 'comparison.png'}")


if __name__ == '__main__':
    main()

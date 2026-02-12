#!/usr/bin/env python3
"""
mesh2plan v56b - Cross-Section with Targeted Gap Filling

The v55c cross-section image shows nearly-complete wall outlines.
The problem: a few gaps at doorways/corners let rooms merge.

New strategy:
1. Get cross-section wall segments
2. Find wall skeleton (thin to 1px)
3. Find endpoints of skeleton lines
4. For nearby endpoint pairs (< 0.8m gap), check if they're aligned
   with a wall angle — if so, bridge the gap
5. Flood fill the now-closed regions

Also try: use connected components on the wall image itself to find
wall segments, then extend them to fill gaps.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', default='results/v56b')
    parser.add_argument('--resolution', type=float, default=0.02)
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = args.resolution
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min
    
    pad = 0.3
    x_min_w = verts[:, 0].min() - pad
    x_max_w = verts[:, 0].max() + pad
    z_min_w = verts[:, 2].min() - pad
    z_max_w = verts[:, 2].max() + pad
    W_img = int((x_max_w - x_min_w) / res) + 1
    H_img = int((z_max_w - z_min_w) / res) + 1
    print(f"  Image: {W_img}×{H_img}")
    
    # ── Phase 1: Multi-slice cross-sections ──
    slice_heights = np.linspace(y_min + 0.3*y_range, y_min + 0.8*y_range, 15)
    xsection = np.zeros((H_img, W_img), dtype=np.uint8)
    
    for y_h in slice_heights:
        try:
            section = mesh.section(plane_origin=[0, y_h, 0], plane_normal=[0, 1, 0])
            if section is None:
                continue
        except:
            continue
        for entity in section.entities:
            pts = section.vertices[entity.points]
            for i in range(len(pts) - 1):
                x0 = int((pts[i][0] - x_min_w) / res)
                z0 = int((pts[i][2] - z_min_w) / res)
                x1 = int((pts[i+1][0] - x_min_w) / res)
                z1 = int((pts[i+1][2] - z_min_w) / res)
                if (0 <= x0 < W_img and 0 <= z0 < H_img and
                    0 <= x1 < W_img and 0 <= z1 < H_img):
                    cv2.line(xsection, (x0, z0), (x1, z1), 255, thickness=1)
    
    print(f"  Cross-section wall: {(xsection > 0).mean()*100:.1f}%")
    
    # ── Phase 2: Add normal-filtered wall density ──
    normals = mesh.face_normals
    wall_mask_3d = np.abs(normals[:, 1]) < 0.5
    wall_centroids = verts[faces[wall_mask_3d]].mean(axis=1)
    
    wall_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((wall_centroids[:, 0] - x_min_w) / res).astype(int)
    zi = ((wall_centroids[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(wall_density, (zi[valid], xi[valid]), 1)
    
    # Combine: cross-section + high-density walls
    wd_thresh = np.percentile(wall_density[wall_density > 0], 85)
    wd_strong = (wall_density >= wd_thresh).astype(np.uint8) * 255
    combined = np.maximum(xsection, wd_strong)
    
    print(f"  Combined wall: {(combined > 0).mean()*100:.1f}%")
    
    # ── Phase 3: Detect dominant angles from Hough ──
    lines = cv2.HoughLines(combined, 1, np.pi/180, threshold=60)
    if lines is not None:
        angles_deg = np.degrees(lines[:, 0, 1]) % 180
        # Histogram peaks
        hist, bins = np.histogram(angles_deg, bins=180, range=(0, 180))
        hist_s = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist_s, height=max(hist_s)*0.15, distance=20)
        peak_angles = bins[peaks] + 0.5
        peak_h = hist_s[peaks]
        top2 = peak_angles[np.argsort(peak_h)[::-1][:2]]
        angle1, angle2 = top2[0], top2[1] if len(top2) > 1 else top2[0]+90
        print(f"  Dominant angles: {angle1:.0f}° and {angle2:.0f}°")
    else:
        angle1, angle2 = 28, 120
    
    # ── Phase 4: Skeleton + endpoint detection ──
    # Dilate slightly first to connect micro-gaps, then thin
    wall_dilated = cv2.dilate(combined, np.ones((2, 2), np.uint8))
    
    # Morphological thinning (Zhang-Suen via ximgproc or manual)
    # Use OpenCV's built-in thinning if available, else use simple skeleton
    try:
        skeleton = cv2.ximgproc.thinning(wall_dilated)
    except:
        # Fallback: use scipy skeletonize
        from skimage.morphology import skeletonize
        skeleton = (skeletonize(wall_dilated > 0) * 255).astype(np.uint8)
    
    # Find endpoints: pixels with exactly 1 neighbor in skeleton
    skel_binary = (skeleton > 0).astype(np.uint8)
    # Count neighbors via convolution
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, cv2.CV_32F, kernel.astype(np.float32))
    
    endpoints = (skel_binary > 0) & (neighbor_count <= 1.5) & (neighbor_count >= 0.5)
    ep_ys, ep_xs = np.where(endpoints)
    print(f"  Skeleton: {skel_binary.sum()} px, {len(ep_xs)} endpoints")
    
    # ── Phase 5: Bridge nearby endpoints ──
    # For each pair of endpoints within 0.8m, check angle alignment
    max_gap_px = int(0.8 / res)  # 40px
    min_gap_px = int(0.1 / res)  # 5px (don't bridge tiny gaps)
    
    bridged = combined.copy()
    n_bridges = 0
    
    if len(ep_xs) > 0:
        ep_pts = np.column_stack([ep_xs, ep_ys])
        
        for i in range(len(ep_pts)):
            dists = np.linalg.norm(ep_pts - ep_pts[i], axis=1)
            # Find nearby endpoints (not self)
            nearby = np.where((dists > min_gap_px) & (dists < max_gap_px))[0]
            
            for j in nearby:
                if j <= i:
                    continue
                
                # Check if the gap direction matches a wall angle
                dx = ep_pts[j][0] - ep_pts[i][0]
                dy = ep_pts[j][1] - ep_pts[i][1]
                gap_angle = np.degrees(np.arctan2(dy, dx)) % 180
                
                # Is gap aligned with either wall angle?
                for wall_angle in [angle1, angle2]:
                    diff = abs(gap_angle - wall_angle)
                    diff = min(diff, 180 - diff)
                    if diff < 20:
                        # Bridge this gap
                        cv2.line(bridged,
                                 (ep_pts[i][0], ep_pts[i][1]),
                                 (ep_pts[j][0], ep_pts[j][1]),
                                 255, thickness=1)
                        n_bridges += 1
                        break
    
    print(f"  Bridged {n_bridges} gaps")
    
    # ── Phase 6: Small directional closing for remaining gaps ──
    def make_line_kernel(angle_deg, length=8):
        k = np.zeros((length*2+1, length*2+1), dtype=np.uint8)
        dx, dy = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
        for t in range(-length, length+1):
            x, y = int(length + t*dx), int(length + t*dy)
            if 0 <= x < length*2+1 and 0 <= y < length*2+1:
                k[y, x] = 1
        return k
    
    for angle in [angle1, angle2]:
        kernel = make_line_kernel(angle, length=8)  # ~16px = 0.32m
        closed = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, kernel)
        bridged = np.maximum(bridged, closed)
    
    # Thin dilation
    wall_final = cv2.dilate(bridged, np.ones((2, 2), np.uint8))
    
    print(f"  Final wall: {(wall_final > 0).mean()*100:.1f}%")
    
    # ── Phase 7: Apartment mask + flood fill ──
    all_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((verts[:, 0] - x_min_w) / res).astype(int)
    zi = ((verts[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(all_density, (zi[valid], xi[valid]), 1)
    
    apt_mask = (ndimage.gaussian_filter(all_density, sigma=5) > 0.5).astype(np.uint8)
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    apt_mask = ndimage.binary_fill_holes(apt_mask).astype(np.uint8)
    apt_mask = cv2.erode(apt_mask, np.ones((3, 3), np.uint8))
    
    interior = apt_mask.copy()
    interior[wall_final > 0] = 0
    
    n_labels, labels = cv2.connectedComponents(interior)
    
    rooms = []
    for i in range(1, n_labels):
        area_px = (labels == i).sum()
        area_m2 = area_px * res * res
        if area_m2 < 1.5:
            continue
        
        ys, xs = np.where(labels == i)
        cx_px, cy_px = xs.mean(), ys.mean()
        bbox_w = (xs.max() - xs.min()) * res
        bbox_h = (ys.max() - ys.min()) * res
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 0.1)
        
        if area_m2 > 8: name = "Room"
        elif aspect > 2.5: name = "Hallway"
        elif area_m2 < 3: name = "Closet"
        else: name = "Bathroom"
        
        rooms.append({
            'label': i, 'area_m2': area_m2,
            'cx_px': cx_px, 'cy_px': cy_px,
            'aspect': aspect, 'name': name,
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].set_title('Raw Cross-Section')
    axes[0,0].imshow(xsection, cmap='gray_r', origin='lower')
    
    axes[0,1].set_title(f'Skeleton + {len(ep_xs)} Endpoints')
    skel_vis = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    skel_vis[skeleton > 0] = [255, 255, 255]
    skel_vis[endpoints] = [255, 0, 0]
    axes[0,1].imshow(skel_vis, origin='lower')
    
    axes[0,2].set_title(f'After Bridging ({n_bridges} bridges)')
    axes[0,2].imshow(bridged, cmap='gray_r', origin='lower')
    
    axes[1,0].set_title('Wall Density (normal-filtered)')
    axes[1,0].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    
    axes[1,1].set_title(f'Final Walls ({(wall_final > 0).mean()*100:.1f}%)')
    axes[1,1].imshow(wall_final, cmap='gray_r', origin='lower')
    
    # Room partition
    ax = axes[1,2]
    ax.set_title(f'v56b — {len(rooms)} rooms, {total:.1f}m²\nAngles: {angle1:.0f}°, {angle2:.0f}°')
    
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 3)))
    room_img = np.ones((H_img, W_img, 3))
    for i, r in enumerate(rooms):
        mask = labels == r['label']
        room_img[mask] = colors[i % len(colors)][:3]
    room_img[wall_final > 0] = [0.2, 0.2, 0.2]
    
    ax.imshow(room_img, origin='lower')
    for r in rooms:
        ax.text(r['cx_px'], r['cy_px'], f"{r['name']}\n{r['area_m2']:.1f}m²",
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    
    bar_px = 1.0 / res
    ax.plot([10, 10+bar_px], [15, 15], 'k-', linewidth=3)
    ax.text(10+bar_px/2, 30, '1m', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir / 'floorplan.png'}")


if __name__ == '__main__':
    main()

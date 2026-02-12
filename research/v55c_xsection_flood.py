#!/usr/bin/env python3
"""
mesh2plan v55c - Cross-Section Direct Flood Fill

The cross-section image from v55b shows clean wall outlines already!
No need for Hough — just close gaps in the cross-section walls and flood fill.

Pipeline:
1. Multi-height cross-sections → wall image
2. Directional morphological closing along dominant angles to bridge wall gaps
3. Flood fill from low-density interior points → rooms
4. Clean up: merge small rooms, extract polygons
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
from scipy.cluster.hierarchy import fcluster, linkage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', default='results/v55c')
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
    print(f"  {len(verts)} verts, Y: {y_min:.2f} to {y_max:.2f}")
    
    # Image setup
    x_vals, z_vals = verts[:, 0], verts[:, 2]
    pad = 0.3
    x_min_w, x_max_w = x_vals.min() - pad, x_vals.max() + pad
    z_min_w, z_max_w = z_vals.min() - pad, z_vals.max() + pad
    W_img = int((x_max_w - x_min_w) / res) + 1
    H_img = int((z_max_w - z_min_w) / res) + 1
    print(f"  Image: {W_img}×{H_img}")
    
    # ── Phase 1: Cross-section wall image ──
    slice_heights = np.linspace(y_min + 0.25*y_range, y_min + 0.85*y_range, 15)
    xsection = np.zeros((H_img, W_img), dtype=np.float32)
    
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
                    cv2.line(xsection, (x0, z0), (x1, z1), 1.0, thickness=1)
    
    print(f"  Cross-section coverage: {(xsection > 0).mean()*100:.1f}%")
    
    # ── Phase 2: Also add wall-normal density (thresholded) ──
    normals = mesh.face_normals
    wall_face_mask = np.abs(normals[:, 1]) < 0.5
    wall_centroids = verts[faces[wall_face_mask]].mean(axis=1)
    
    wall_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((wall_centroids[:, 0] - x_min_w) / res).astype(int)
    zi = ((wall_centroids[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(wall_density, (zi[valid], xi[valid]), 1)
    
    # Binary wall from density (strong walls only)
    if wall_density.max() > 0:
        wd_thresh = np.percentile(wall_density[wall_density > 0], 80)
        wd_binary = (wall_density >= wd_thresh).astype(np.float32)
    else:
        wd_binary = np.zeros_like(wall_density)
    
    # Combine: cross-section + dense walls
    wall_combined = np.maximum(xsection, wd_binary)
    print(f"  Combined wall coverage: {(wall_combined > 0).mean()*100:.1f}%")
    
    # ── Phase 3: Detect dominant angles ──
    wall_u8 = (wall_combined * 255).astype(np.uint8)
    lines = cv2.HoughLines(wall_u8, 1, np.pi/180, threshold=50)
    
    if lines is not None:
        angles_deg = np.degrees(lines[:, 0, 1]) % 180
        # Histogram-based angle detection (more robust than clustering)
        hist, bin_edges = np.histogram(angles_deg, bins=180, range=(0, 180))
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        # Find peaks
        from scipy.signal import find_peaks
        peaks, props = find_peaks(hist_smooth, height=max(hist_smooth)*0.2, distance=20)
        peak_angles = bin_edges[peaks] + 0.5
        peak_heights = hist_smooth[peaks]
        
        # Sort by height, keep top 2
        top_idx = np.argsort(peak_heights)[::-1][:2]
        peak_angles = peak_angles[top_idx]
        
        # Ensure they're roughly perpendicular (within 70-110°)
        if len(peak_angles) >= 2:
            diff = abs(peak_angles[0] - peak_angles[1])
            if diff < 70 or diff > 110:
                # Force perpendicular
                peak_angles[1] = (peak_angles[0] + 90) % 180
        
        angle1, angle2 = peak_angles[0] if len(peak_angles) > 0 else 28, peak_angles[1] if len(peak_angles) > 1 else 120
        print(f"  Dominant angles: {angle1:.0f}° and {angle2:.0f}°")
    else:
        angle1, angle2 = 28, 120
        print(f"  No Hough lines, using default angles: {angle1}° and {angle2}°")
    
    # ── Phase 4: Directional closing to bridge wall gaps ──
    # Create directional kernels along each wall angle
    def make_line_kernel(angle_deg, length=25):
        """Create a line structuring element at given angle."""
        angle_rad = np.radians(angle_deg)
        k = np.zeros((length*2+1, length*2+1), dtype=np.uint8)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        for t in range(-length, length+1):
            x = int(length + t * dx)
            y = int(length + t * dy)
            if 0 <= x < length*2+1 and 0 <= y < length*2+1:
                k[y, x] = 1
        return k
    
    # Close along both wall directions (shorter bridging to avoid blobs)
    wall_closed = wall_u8.copy()
    for angle in [angle1, angle2]:
        kernel = make_line_kernel(angle, length=10)  # ~20px = 0.4m bridging
        closed = cv2.morphologyEx(wall_u8, cv2.MORPH_CLOSE, kernel)
        wall_closed = np.maximum(wall_closed, closed)
    
    # Thin dilation only
    wall_closed = cv2.dilate(wall_closed, np.ones((2, 2), np.uint8))
    
    print(f"  After closing: {(wall_closed > 0).mean()*100:.1f}%")
    
    # ── Phase 5: Apartment mask ──
    all_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((verts[:, 0] - x_min_w) / res).astype(int)
    zi = ((verts[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(all_density, (zi[valid], xi[valid]), 1)
    
    apt_mask = (ndimage.gaussian_filter(all_density, sigma=5) > 0.5).astype(np.uint8)
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    apt_mask = ndimage.binary_fill_holes(apt_mask).astype(np.uint8)
    apt_mask = cv2.erode(apt_mask, np.ones((3, 3), np.uint8))
    
    print(f"  Apartment mask: {apt_mask.sum()} px ({apt_mask.mean()*100:.1f}%)")
    
    # ── Phase 6: Flood fill for rooms ──
    # Interior = apartment minus walls
    interior = apt_mask.copy()
    interior[wall_closed > 0] = 0
    
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  {n_labels - 1} raw regions")
    
    # Filter and classify
    rooms = []
    for i in range(1, n_labels):
        area_px = (labels == i).sum()
        area_m2 = area_px * res * res
        if area_m2 < 1.5:
            continue
        
        ys, xs = np.where(labels == i)
        cx_px, cy_px = xs.mean(), ys.mean()
        cx_m = cx_px * res + x_min_w
        cy_m = cy_px * res + z_min_w
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
            'cx_m': cx_m, 'cy_m': cy_m,
            'aspect': aspect, 'name': name,
            'bbox_w': bbox_w, 'bbox_h': bbox_h,
        })
    
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    
    # ── Phase 6b: Merge small adjacent rooms ──
    # If two rooms < 4m² are adjacent, merge them
    def are_adjacent(labels, l1, l2, gap=5):
        m1 = cv2.dilate((labels == l1).astype(np.uint8), np.ones((gap, gap), np.uint8))
        m2 = (labels == l2).astype(np.uint8)
        return (m1 & m2).any()
    
    merged = True
    while merged and len(rooms) > 5:
        merged = False
        for i in range(len(rooms) - 1, -1, -1):
            if rooms[i]['area_m2'] >= 4:
                continue
            for j in range(len(rooms)):
                if i == j:
                    continue
                if are_adjacent(labels, rooms[i]['label'], rooms[j]['label']):
                    # Merge i into j
                    labels[labels == rooms[i]['label']] = rooms[j]['label']
                    rooms[j]['area_m2'] += rooms[i]['area_m2']
                    rooms.pop(i)
                    merged = True
                    break
            if merged:
                break
    
    # Rename
    name_counts = {}
    for r in rooms:
        # Reclassify after merge
        if r['area_m2'] > 8: r['name'] = "Room"
        elif r['aspect'] > 2.5: r['name'] = "Hallway"
        elif r['area_m2'] < 3: r['name'] = "Closet"
        else: r['name'] = "Bathroom"
        
        n = r['name']
        name_counts[n] = name_counts.get(n, 0) + 1
        if name_counts[n] > 1:
            r['name'] = f"{n} {name_counts[n]}"
    
    total = sum(r['area_m2'] for r in rooms)
    print(f"\n  {len(rooms)} rooms, {total:.1f}m² total")
    for r in rooms:
        print(f"    {r['name']}: {r['area_m2']:.1f}m² (aspect={r['aspect']:.1f})")
    
    # ── Phase 7: Render ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 1. Raw cross-section
    ax = axes[0, 0]
    ax.set_title('Cross-Section Walls')
    ax.imshow(xsection, cmap='gray_r', origin='lower')
    ax.set_xlabel('X'); ax.set_ylabel('Z')
    
    # 2. Combined + closed walls
    ax = axes[0, 1]
    ax.set_title(f'Closed Walls ({(wall_closed > 0).mean()*100:.1f}%)')
    ax.imshow(wall_closed, cmap='gray_r', origin='lower')
    
    # 3. Wall density comparison
    ax = axes[1, 0]
    ax.set_title('Normal-Filtered Wall Density')
    wd_vis = np.log1p(wall_density)
    ax.imshow(wd_vis, cmap='hot', origin='lower')
    
    # 4. Room partition
    ax = axes[1, 1]
    ax.set_title(f'v55c — {len(rooms)} rooms, {total:.1f}m²\nAngles: {angle1:.0f}°, {angle2:.0f}°')
    
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 3)))
    room_img = np.ones((H_img, W_img, 3))
    for i, r in enumerate(rooms):
        mask = labels == r['label']
        room_img[mask] = colors[i % len(colors)][:3]
    
    # Draw wall closed as black overlay
    room_img[wall_closed > 0] = [0.2, 0.2, 0.2]
    
    ax.imshow(room_img, origin='lower')
    
    for r in rooms:
        ax.text(r['cx_px'], r['cy_px'], f"{r['name']}\n{r['area_m2']:.1f}m²",
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    
    # Scale bar
    bar_px = 1.0 / res
    ax.plot([10, 10+bar_px], [15, 15], 'k-', linewidth=3)
    ax.text(10+bar_px/2, 30, '1m', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir / 'floorplan.png'}")


if __name__ == '__main__':
    main()

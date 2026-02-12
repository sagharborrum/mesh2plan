#!/usr/bin/env python3
"""
mesh2plan v54c - Multi-Slice Gap Filling → Room Extraction

v54b showed single slice has gaps in walls. Solution:
1. Take 10 slices across wall height range
2. Accumulate ALL segments into one image with thick lines
3. Any pixel hit by ANY slice = wall (threshold=1, not 3)
4. Strong morphological closing to bridge remaining gaps
5. Flood fill → rooms
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import math
import cv2
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', default='results/v54c_multislice')
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    print(f"  {len(verts)} verts, Y: {verts[:,1].min():.2f} to {verts[:,1].max():.2f}")
    
    # Multiple slices
    y_values = np.linspace(-1.8, -0.3, 12)
    res = 0.02
    
    # Get bounds from first valid slice
    all_segs = []
    for y in y_values:
        segs = trimesh.intersections.mesh_plane(mesh, [0,1,0], [0, y, 0])
        if len(segs) > 0:
            all_segs.append(segs)
            print(f"  Y={y:.2f}: {len(segs)} segments")
    
    if not all_segs:
        print("No segments found!")
        return
    
    # Compute bounds from all segments
    all_pts = np.concatenate([np.concatenate([s[:, 0], s[:, 1]]) for s in all_segs])
    xmin = all_pts[:, 0].min() - 0.5
    xmax = all_pts[:, 0].max() + 0.5
    zmin = all_pts[:, 2].min() - 0.5
    zmax = all_pts[:, 2].max() + 0.5
    
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    print(f"\n  Image: {w}×{h}, {len(all_segs)} valid slices")
    
    # Rasterize ALL segments
    wall_img = np.zeros((h, w), dtype=np.uint8)
    for segs in all_segs:
        for seg in segs:
            px0 = int((seg[0][0] - xmin) / res)
            pz0 = int((seg[0][2] - zmin) / res)
            px1 = int((seg[1][0] - xmin) / res)
            pz1 = int((seg[1][2] - zmin) / res)
            px0 = max(0, min(w-1, px0))
            pz0 = max(0, min(h-1, pz0))
            px1 = max(0, min(w-1, px1))
            pz1 = max(0, min(h-1, pz1))
            cv2.line(wall_img, (px0, pz0), (px1, pz1), 255, thickness=3)
    
    wall_pct = (wall_img > 0).sum() / wall_img.size * 100
    print(f"  Raw wall: {wall_pct:.1f}%")
    
    # Save raw wall image
    cv2.imwrite(str(out / 'walls_raw.png'), wall_img)
    
    # Strong morphological closing to bridge gaps
    # Use directional kernels matching dominant wall angles (~30° and ~120°)
    # First detect angles
    angles_deg = []
    lengths = []
    for segs in all_segs:
        for seg in segs:
            dx = seg[1][0] - seg[0][0]
            dz = seg[1][2] - seg[0][2]
            l = math.sqrt(dx**2 + dz**2)
            if l > 0.05:
                angles_deg.append(math.degrees(math.atan2(dz, dx)) % 180)
                lengths.append(l)
    
    from scipy.ndimage import gaussian_filter1d
    hist, bins = np.histogram(angles_deg, bins=180, range=(0, 180), weights=lengths)
    hist_s = gaussian_filter1d(hist, sigma=2, mode='wrap')
    peaks = []
    for i in range(len(hist_s)):
        if hist_s[i] > hist_s[(i-1)%180] and hist_s[i] > hist_s[(i+1)%180]:
            peaks.append((hist_s[i], bins[i] + 0.5))
    peaks.sort(reverse=True)
    dominant = [p[1] for p in peaks[:4]]
    print(f"  Dominant angles: {', '.join(f'{a:.0f}°' for a in dominant)}")
    
    # Directional closing along dominant angles
    wall_closed = wall_img.copy()
    for angle in dominant[:2]:
        rad = math.radians(angle)
        ksize = 15  # ~0.3m at 0.02 resolution
        kx = int(round(math.cos(rad) * ksize))
        kz = int(round(math.sin(rad) * ksize))
        
        # Build directional kernel
        kw = max(abs(kx), 1) * 2 + 1
        kh = max(abs(kz), 1) * 2 + 1
        kernel = np.zeros((kh, kw), dtype=np.uint8)
        cx, cz = kw // 2, kh // 2
        cv2.line(kernel, (cx - kx, cz - kz), (cx + kx, cz + kz), 1, thickness=1)
        
        wall_closed = cv2.morphologyEx(wall_closed, cv2.MORPH_CLOSE, kernel)
    
    # Also general close
    kernel_sq = np.ones((7, 7), np.uint8)
    wall_closed = cv2.morphologyEx(wall_closed, cv2.MORPH_CLOSE, kernel_sq, iterations=2)
    
    close_pct = (wall_closed > 0).sum() / wall_closed.size * 100
    print(f"  After closing: {close_pct:.1f}%")
    
    # Interior
    interior = 255 - wall_closed
    
    # Flood fill exterior
    flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    for x in range(0, w, 3):
        for z in [0, 1, h-2, h-1]:
            if interior[z, x] > 0:
                cv2.floodFill(interior, flood_mask, (x, z), 0)
    for z in range(0, h, 3):
        for x in [0, 1, w-2, w-1]:
            if interior[z, x] > 0:
                cv2.floodFill(interior, flood_mask, (x, z), 0)
    
    int_pct = (interior > 0).sum() / interior.size * 100
    print(f"  Interior after flood: {int_pct:.1f}%")
    
    # Connected components
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  {n_labels - 1} components")
    
    rooms = []
    for lab in range(1, n_labels):
        mask = (labels == lab).astype(np.uint8) * 255
        area_m2 = (mask > 0).sum() * res * res
        if area_m2 < 1.0:
            continue
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        corners = [(pt[0][0] * res + xmin, pt[0][1] * res + zmin) for pt in approx]
        
        M = cv2.moments(contour)
        cx = M['m10'] / max(M['m00'], 1) * res + xmin
        cz = M['m01'] / max(M['m00'], 1) * res + zmin
        
        if area_m2 > 8: label_name = "Room"
        elif area_m2 > 4: label_name = "Hallway"
        elif area_m2 > 2: label_name = "Bathroom"
        else: label_name = "Closet"
        
        rooms.append({
            'corners': corners, 'area': area_m2,
            'centroid': (cx, cz), 'n_vertices': len(corners),
            'label': label_name,
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    total_area = sum(r['area'] for r in rooms)
    print(f"\n  {len(rooms)} rooms, {total_area:.1f}m²")
    for r in rooms:
        print(f"    {r['label']}: {r['area']:.1f}m² ({r['n_vertices']}v)")
    
    # Render
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Raw walls
    ax = axes[0]
    ax.set_title(f'Multi-slice walls ({len(all_segs)} slices)')
    ax.imshow(wall_img, extent=[xmin, xmax, zmin, zmax], origin='lower', cmap='gray_r')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Closed walls + rooms
    ax = axes[1]
    ax.set_title('Closed walls + room contours')
    ax.imshow(wall_closed, extent=[xmin, xmax, zmin, zmax], origin='lower', cmap='gray_r', alpha=0.5)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        c = r['corners']
        xs = [p[0] for p in c] + [c[0][0]]
        zs = [p[1] for p in c] + [c[0][1]]
        ax.fill(xs, zs, color=colors[i], alpha=0.3)
        ax.plot(xs, zs, '-', color=colors[i], linewidth=2)
        ax.text(r['centroid'][0], r['centroid'][1], f"{r['label']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Final
    ax = axes[2]
    ax.set_title(f'v54c Multi-Slice — {len(rooms)} rooms, {total_area:.1f}m²')
    
    room_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        c = r['corners']
        xs = [p[0] for p in c] + [c[0][0]]
        zs = [p[1] for p in c] + [c[0][1]]
        ax.fill(xs, zs, color=room_colors[i], alpha=0.5)
        ax.plot(xs, zs, 'k-', linewidth=2)
        
        for j in range(len(c)):
            k = (j + 1) % len(c)
            dx = c[k][0] - c[j][0]
            dz = c[k][1] - c[j][1]
            length = math.sqrt(dx**2 + dz**2)
            if length < 0.01: continue
            nx_w = -dz / length * 0.08
            nz_w = dx / length * 0.08
            ax.plot([c[j][0]+nx_w, c[k][0]+nx_w], [c[j][1]+nz_w, c[k][1]+nz_w], 'k-', linewidth=0.5)
            ax.plot([c[j][0]-nx_w, c[k][0]-nx_w], [c[j][1]-nz_w, c[k][1]-nz_w], 'k-', linewidth=0.5)
        
        ax.text(r['centroid'][0], r['centroid'][1], f"{r['label']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.plot([xmin + 0.3, xmin + 1.3], [zmin + 0.3, zmin + 0.3], 'k-', linewidth=3)
    ax.text(xmin + 0.8, zmin + 0.5, '1m', ha='center', fontsize=8)
    
    ax.text(0.02, 0.98,
            f"Angles: {', '.join(f'{a:.0f}°' for a in dominant[:2])}\n"
            f"{len(rooms)} rooms, {total_area:.1f}m²\n"
            f"{len(all_segs)} slices, res={res}m",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out / 'floorplan.png'}")
    
    summary = {
        'version': 'v54c_multislice',
        'dominant_angles': [round(a, 1) for a in dominant[:4]],
        'n_rooms': len(rooms),
        'total_area_m2': round(total_area, 1),
        'rooms': [{'label': r['label'], 'area_m2': round(r['area'], 1), 'vertices': r['n_vertices']} for r in rooms]
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

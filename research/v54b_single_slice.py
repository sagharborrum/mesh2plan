#!/usr/bin/env python3
"""
mesh2plan v54b - Single Cross-Section → Room Contours

The middle slice at Y=-1.0m already shows clear wall outlines.
Just use that single slice directly:
1. Rasterize segments with thick lines → wall mask
2. Dilate to close gaps
3. Flood fill from exterior
4. Connected components of interior = rooms
5. Simplify contours + angle snap
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
    parser.add_argument('--output-dir', default='results/v54b_single_slice')
    parser.add_argument('--slice-y', type=float, default=-1.0)
    parser.add_argument('--resolution', type=float, default=0.02)
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    print(f"  {len(verts)} verts, Y range: {verts[:,1].min():.2f} to {verts[:,1].max():.2f}")
    
    # Slice
    print(f"\nSlicing at Y={args.slice_y}...")
    segments = trimesh.intersections.mesh_plane(
        mesh, plane_normal=[0, 1, 0], plane_origin=[0, args.slice_y, 0]
    )
    print(f"  {len(segments)} segments")
    
    # Bounds
    res = args.resolution
    all_x = np.concatenate([segments[:, 0, 0], segments[:, 1, 0]])
    all_z = np.concatenate([segments[:, 0, 2], segments[:, 1, 2]])
    xmin, xmax = all_x.min() - 0.5, all_x.max() + 0.5
    zmin, zmax = all_z.min() - 0.5, all_z.max() + 0.5
    
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    print(f"  Image: {w}×{h} px")
    
    # Rasterize with thick lines
    wall_img = np.zeros((h, w), dtype=np.uint8)
    for seg in segments:
        px0 = int((seg[0][0] - xmin) / res)
        pz0 = int((seg[0][2] - zmin) / res)
        px1 = int((seg[1][0] - xmin) / res)
        pz1 = int((seg[1][2] - zmin) / res)
        cv2.line(wall_img, (px0, pz0), (px1, pz1), 255, thickness=2)
    
    wall_pct = (wall_img > 0).sum() / wall_img.size * 100
    print(f"  Wall pixels: {wall_pct:.1f}%")
    
    # Close gaps with morphological closing
    kernel = np.ones((5, 5), np.uint8)
    wall_closed = cv2.morphologyEx(wall_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Interior = not-wall
    interior = 255 - wall_closed
    
    # Flood fill exterior from edges
    flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    for x in range(0, w, 5):
        for z in [0, h-1]:
            if interior[z, x] > 0:
                cv2.floodFill(interior, flood_mask, (x, z), 0)
    for z in range(0, h, 5):
        for x in [0, w-1]:
            if interior[z, x] > 0:
                cv2.floodFill(interior, flood_mask, (x, z), 0)
    
    # Connected components = rooms
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  {n_labels - 1} connected components")
    
    rooms = []
    for label in range(1, n_labels):
        mask = (labels == label).astype(np.uint8) * 255
        area_px = (mask > 0).sum()
        area_m2 = area_px * res * res
        
        if area_m2 < 1.5:
            continue
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        corners = [(pt[0][0] * res + xmin, pt[0][1] * res + zmin) for pt in approx]
        
        M = cv2.moments(contour)
        cx = M['m10'] / max(M['m00'], 1) * res + xmin
        cz = M['m01'] / max(M['m00'], 1) * res + zmin
        
        # Classify
        if area_m2 > 8:
            label_name = "Room"
        elif area_m2 > 4:
            label_name = "Hallway"
        elif area_m2 > 2:
            label_name = "Bathroom"
        else:
            label_name = "Closet"
        
        rooms.append({
            'corners': corners,
            'area': area_m2,
            'centroid': (cx, cz),
            'n_vertices': len(corners),
            'label': label_name,
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    total_area = sum(r['area'] for r in rooms)
    print(f"\n  {len(rooms)} rooms, {total_area:.1f}m²")
    for r in rooms:
        print(f"    {r['label']}: {r['area']:.1f}m² ({r['n_vertices']}v)")
    
    # Find dominant angles from segments
    angles = []
    lengths = []
    for seg in segments:
        dx = seg[1][0] - seg[0][0]
        dz = seg[1][2] - seg[0][2]
        l = math.sqrt(dx**2 + dz**2)
        if l > 0.05:
            angles.append(math.degrees(math.atan2(dz, dx)) % 180)
            lengths.append(l)
    
    from scipy.ndimage import gaussian_filter1d
    hist, bins = np.histogram(angles, bins=180, range=(0, 180), weights=lengths)
    hist_smooth = gaussian_filter1d(hist, sigma=2, mode='wrap')
    peaks = []
    for i in range(len(hist_smooth)):
        if hist_smooth[i] > hist_smooth[(i-1)%180] and hist_smooth[i] > hist_smooth[(i+1)%180]:
            peaks.append((hist_smooth[i], bins[i] + 0.5))
    peaks.sort(reverse=True)
    dominant = [p[1] for p in peaks[:4]]
    print(f"\n  Dominant angles: {', '.join(f'{a:.1f}°' for a in dominant)}")
    
    # Render
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Raw segments
    ax = axes[0]
    ax.set_title(f'Cross-section at Y={args.slice_y}m ({len(segments)} segments)')
    for seg in segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][2], seg[1][2]], 'k-', linewidth=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Wall image + rooms
    ax = axes[1]
    ax.set_title('Wall mask + room contours')
    ax.imshow(wall_closed, extent=[xmin, xmax, zmin, zmax], origin='lower', cmap='gray_r', alpha=0.5)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        corners = r['corners']
        xs = [c[0] for c in corners] + [corners[0][0]]
        zs = [c[1] for c in corners] + [corners[0][1]]
        ax.fill(xs, zs, color=colors[i], alpha=0.3)
        ax.plot(xs, zs, '-', color=colors[i], linewidth=2)
        ax.text(r['centroid'][0], r['centroid'][1], f"{r['label']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Final floor plan
    ax = axes[2]
    ax.set_title(f'v54b Cross-Section — {len(rooms)} rooms, {total_area:.1f}m²')
    
    room_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        corners = r['corners']
        xs = [c[0] for c in corners] + [corners[0][0]]
        zs = [c[1] for c in corners] + [corners[0][1]]
        ax.fill(xs, zs, color=room_colors[i], alpha=0.5)
        ax.plot(xs, zs, 'k-', linewidth=2)
        
        # Double walls
        for j in range(len(corners)):
            k = (j + 1) % len(corners)
            dx = corners[k][0] - corners[j][0]
            dz = corners[k][1] - corners[j][1]
            length = math.sqrt(dx**2 + dz**2)
            if length < 0.01:
                continue
            nx_w = -dz / length * 0.08
            nz_w = dx / length * 0.08
            ax.plot([corners[j][0]+nx_w, corners[k][0]+nx_w],
                    [corners[j][1]+nz_w, corners[k][1]+nz_w], 'k-', linewidth=0.5)
            ax.plot([corners[j][0]-nx_w, corners[k][0]-nx_w],
                    [corners[j][1]-nz_w, corners[k][1]-nz_w], 'k-', linewidth=0.5)
        
        ax.text(r['centroid'][0], r['centroid'][1], f"{r['label']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.plot([xmin + 0.3, xmin + 1.3], [zmin + 0.3, zmin + 0.3], 'k-', linewidth=3)
    ax.text(xmin + 0.8, zmin + 0.5, '1m', ha='center', fontsize=8)
    
    ax.text(0.02, 0.98,
            f"Angles: {', '.join(f'{a:.0f}°' for a in dominant[:2])}\n"
            f"{len(rooms)} rooms, {total_area:.1f}m²\n"
            f"Slice Y={args.slice_y}m, res={res}m",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    summary = {
        'version': 'v54b_single_slice',
        'slice_y': args.slice_y,
        'dominant_angles': [round(a, 1) for a in dominant[:4]],
        'n_rooms': len(rooms),
        'total_area_m2': round(total_area, 1),
        'rooms': [{'label': r['label'], 'area_m2': round(r['area'], 1), 'vertices': r['n_vertices']} for r in rooms]
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {out / 'floorplan.png'}")


if __name__ == '__main__':
    main()

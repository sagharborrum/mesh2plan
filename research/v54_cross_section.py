#!/usr/bin/env python3
"""
mesh2plan v54 - Cross-Section Slicing

FUNDAMENTALLY DIFFERENT from all v32-v53 approaches.
Instead of density images, Hough lines, or RANSAC planes:

1. Take a horizontal slice through the mesh at wall height (Y=1.0-1.5m)
2. The cross-section contour IS the wall outline
3. Multi-slice: take 5 slices at different heights, overlay them
4. The consistent contours across heights are walls (furniture varies)
5. Vectorize the resulting contour into line segments
6. Snap to dominant angles → clean wall lines
7. Wall lines partition space into rooms

This uses trimesh.intersections.mesh_plane() which gives the exact
intersection of a plane with mesh triangles → line segments.
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
from collections import defaultdict


def multi_slice(mesh, y_values, resolution=0.02):
    """Take multiple horizontal slices and accumulate into a density image."""
    verts = np.array(mesh.vertices)
    xmin, xmax = verts[:, 0].min() - 0.5, verts[:, 0].max() + 0.5
    zmin, zmax = verts[:, 2].min() - 0.5, verts[:, 2].max() + 0.5
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    accumulator = np.zeros((h, w), dtype=np.float32)
    all_segments = []
    
    for y in y_values:
        print(f"  Slicing at Y={y:.2f}m...")
        try:
            # Slice mesh with horizontal plane
            lines = trimesh.intersections.mesh_plane(
                mesh, 
                plane_normal=[0, 1, 0],  # Y-up
                plane_origin=[0, y, 0]
            )
        except Exception as e:
            print(f"    Error: {e}")
            continue
        
        print(f"    {len(lines)} line segments")
        all_segments.append(lines)
        
        # Rasterize segments into accumulator
        for seg in lines:
            # seg is (2, 3) - two 3D endpoints
            x0, z0 = seg[0][0], seg[0][2]
            x1, z1 = seg[1][0], seg[1][2]
            
            # Convert to pixel coordinates
            px0 = int((x0 - xmin) / resolution)
            pz0 = int((z0 - zmin) / resolution)
            px1 = int((x1 - xmin) / resolution)
            pz1 = int((z1 - zmin) / resolution)
            
            # Draw line on accumulator (thickness=3 so adjacent slices overlap)
            cv2.line(accumulator, (px0, pz0), (px1, pz1), 1.0, thickness=3)
    
    return accumulator, (xmin, xmax, zmin, zmax, resolution), all_segments


def find_dominant_angles(segments_list, n_bins=180):
    """Find dominant wall angles from cross-section segments."""
    angles = []
    lengths = []
    
    for segments in segments_list:
        for seg in segments:
            dx = seg[1][0] - seg[0][0]
            dz = seg[1][2] - seg[0][2]
            length = math.sqrt(dx**2 + dz**2)
            if length < 0.05:  # skip tiny segments
                continue
            angle = math.degrees(math.atan2(dz, dx)) % 180
            angles.append(angle)
            lengths.append(length)
    
    # Weighted histogram
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 180), weights=lengths)
    
    # Smooth and find peaks
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=2, mode='wrap')
    
    # Find top 2 peaks
    peaks = []
    for i in range(len(hist_smooth)):
        prev = hist_smooth[(i-1) % len(hist_smooth)]
        next_ = hist_smooth[(i+1) % len(hist_smooth)]
        if hist_smooth[i] > prev and hist_smooth[i] > next_:
            peaks.append((hist_smooth[i], bin_edges[i] + 0.5))
    
    peaks.sort(reverse=True)
    dominant = [p[1] for p in peaks[:4]]
    
    return dominant, hist_smooth, bin_edges


def slice_to_wall_image(accumulator, bounds, threshold=3):
    """Convert multi-slice accumulator to binary wall image."""
    # Pixels present in multiple slices are walls
    wall_img = (accumulator >= threshold).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    wall_img = cv2.morphologyEx(wall_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return wall_img


def extract_rooms_from_walls(wall_img, bounds, min_area_m2=1.5):
    """Find rooms as connected components of non-wall space."""
    xmin, xmax, zmin, zmax, res = bounds
    
    # Invert: rooms are non-wall areas
    interior = 255 - wall_img
    
    # Fill from edges to remove exterior
    h, w = interior.shape
    flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    # Flood fill from corners
    for seed in [(0,0), (w-1,0), (0,h-1), (w-1,h-1)]:
        if interior[seed[1], seed[0]] > 0:
            cv2.floodFill(interior, flood_mask, seed, 0)
    
    # Label connected components
    n_labels, labels = cv2.connectedComponents(interior)
    
    rooms = []
    for label in range(1, n_labels):
        mask = (labels == label)
        area_px = mask.sum()
        area_m2 = area_px * res * res
        
        if area_m2 < min_area_m2:
            continue
        
        # Get contour
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to metric coordinates
        corners = []
        for pt in approx:
            x = pt[0][0] * res + xmin
            z = pt[0][1] * res + zmin
            corners.append((x, z))
        
        # Centroid
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00'] * res + xmin
            cz = M['m01'] / M['m00'] * res + zmin
        else:
            cx = np.mean([c[0] for c in corners])
            cz = np.mean([c[1] for c in corners])
        
        rooms.append({
            'corners': corners,
            'area': area_m2,
            'centroid': (cx, cz),
            'n_vertices': len(corners),
            'contour_px': contour,
            'mask': mask,
        })
    
    # Classify
    rooms.sort(key=lambda r: r['area'], reverse=True)
    for i, r in enumerate(rooms):
        a = r['area']
        if a > 8:
            r['label'] = f"Room {i+1}"
        elif a > 4:
            r['label'] = "Hallway"
        elif a > 2:
            r['label'] = "Bathroom"
        else:
            r['label'] = "Closet"
    
    return rooms


def snap_contour_to_angles(corners, dominant_angles, tolerance=12):
    """Snap polygon edges to dominant wall angles."""
    if len(corners) < 3 or len(dominant_angles) < 2:
        return corners
    
    angles_rad = [math.radians(a) for a in dominant_angles]
    
    snapped_edges = []
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        dx = corners[j][0] - corners[i][0]
        dz = corners[j][1] - corners[i][1]
        length = math.sqrt(dx**2 + dz**2)
        if length < 0.1:
            continue
        
        edge_angle = math.atan2(dz, dx)
        
        # Find nearest dominant angle
        best_snap = edge_angle
        best_diff = tolerance
        for da in angles_rad:
            for offset in [0, math.pi]:
                diff = abs(((edge_angle - da - offset + math.pi) % (2*math.pi)) - math.pi)
                diff_deg = math.degrees(diff)
                if diff_deg < best_diff:
                    best_diff = diff_deg
                    best_snap = da + offset if abs(((edge_angle - da - offset + math.pi) % (2*math.pi)) - math.pi) == diff else da
        
        snapped_edges.append({
            'start': corners[i],
            'angle': best_snap,
            'length': length,
        })
    
    if len(snapped_edges) < 3:
        return corners
    
    # Reconstruct polygon from snapped edges
    new_corners = [snapped_edges[0]['start']]
    for i in range(len(snapped_edges)):
        j = (i + 1) % len(snapped_edges)
        
        # Intersection of edge i and edge j
        x1, z1 = snapped_edges[i]['start']
        dx1 = math.cos(snapped_edges[i]['angle'])
        dz1 = math.sin(snapped_edges[i]['angle'])
        
        x2, z2 = snapped_edges[j]['start']
        dx2 = math.cos(snapped_edges[j]['angle'])
        dz2 = math.sin(snapped_edges[j]['angle'])
        
        det = dx1 * dz2 - dz1 * dx2
        if abs(det) < 1e-10:
            new_corners.append(snapped_edges[j]['start'])
            continue
        
        t = ((x2 - x1) * dz2 - (z2 - z1) * dx2) / det
        ix = x1 + t * dx1
        iz = z1 + t * dz1
        new_corners.append((ix, iz))
    
    # Remove duplicates and close polygon
    return new_corners[:-1] if len(new_corners) > 3 else corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', default='results/v54_cross_section')
    parser.add_argument('--n-slices', type=int, default=7)
    parser.add_argument('--y-min', type=float, default=0.8)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--resolution', type=float, default=0.02)
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    print(f"  {len(verts)} vertices, {len(mesh.faces)} faces")
    print(f"  Y range: {verts[:, 1].min():.2f} to {verts[:, 1].max():.2f}")
    
    # Slice heights
    y_values = np.linspace(args.y_min, args.y_max, args.n_slices)
    print(f"\nSlicing at Y = {', '.join(f'{y:.2f}' for y in y_values)}")
    
    # Step 1: Multi-slice
    accumulator, bounds, all_segments = multi_slice(mesh, y_values, args.resolution)
    xmin, xmax, zmin, zmax, res = bounds
    
    total_segs = sum(len(s) for s in all_segments)
    print(f"\n  Total segments across all slices: {total_segs}")
    print(f"  Accumulator shape: {accumulator.shape}, max={accumulator.max():.1f}, nonzero={np.count_nonzero(accumulator)}")
    
    # Step 2: Find dominant angles
    print("\nFinding dominant angles...")
    dominant_angles, angle_hist, angle_bins = find_dominant_angles(all_segments)
    print(f"  Dominant angles: {', '.join(f'{a:.1f}°' for a in dominant_angles)}")
    
    # Step 3: Create wall image
    print("\nCreating wall image...")
    # Threshold: present in at least 3 slices
    wall_img = slice_to_wall_image(accumulator, bounds, threshold=2)
    wall_pct = (wall_img > 0).sum() / wall_img.size * 100
    print(f"  Wall coverage: {wall_pct:.1f}%")
    
    # Step 4: Find rooms
    print("\nFinding rooms...")
    rooms = extract_rooms_from_walls(wall_img, bounds, min_area_m2=1.5)
    total_area = sum(r['area'] for r in rooms)
    print(f"  {len(rooms)} rooms, {total_area:.1f}m²")
    for r in rooms:
        print(f"    {r['label']}: {r['area']:.1f}m² ({r['n_vertices']}v)")
    
    # Step 5: Angle snap (optional)
    if len(dominant_angles) >= 2:
        print("\nSnapping to dominant angles...")
        for r in rooms:
            snapped = snap_contour_to_angles(r['corners'], dominant_angles[:2])
            if len(snapped) >= 3:
                r['corners_snapped'] = snapped
            else:
                r['corners_snapped'] = r['corners']
    
    # Step 6: Render
    print("\nRendering...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Top-left: raw accumulator
    ax = axes[0, 0]
    ax.set_title(f'Multi-slice accumulator ({args.n_slices} slices)')
    im = ax.imshow(accumulator, extent=[xmin, xmax, zmin, zmax], origin='lower', cmap='hot')
    plt.colorbar(im, ax=ax, label='Slice count')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    
    # Top-right: angle histogram
    ax = axes[0, 1]
    ax.set_title('Edge angle distribution')
    centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    ax.plot(centers, angle_hist, 'b-')
    for a in dominant_angles[:4]:
        ax.axvline(a, color='r', linestyle='--', alpha=0.7, label=f'{a:.1f}°')
    ax.set_xlabel('Angle (°)')
    ax.legend()
    
    # Bottom-left: wall image + room contours
    ax = axes[1, 0]
    ax.set_title(f'Wall image (threshold ≥3 slices)')
    ax.imshow(wall_img, extent=[xmin, xmax, zmin, zmax], origin='lower', cmap='gray_r')
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        corners = r['corners']
        xs = [c[0] for c in corners] + [corners[0][0]]
        zs = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(xs, zs, '-', color=colors[i], linewidth=2)
        ax.text(r['centroid'][0], r['centroid'][1], f"{r['label']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=7, color=colors[i], fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Bottom-right: final floor plan
    ax = axes[1, 1]
    ax.set_title(f'v54 Cross-Section — {len(rooms)} rooms, {total_area:.1f}m²')
    
    # Light mesh background
    ax.scatter(verts[::10, 0], verts[::10, 2], s=0.1, c='#eee', alpha=0.3)
    
    room_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        corners = r.get('corners_snapped', r['corners'])
        xs = [c[0] for c in corners] + [corners[0][0]]
        zs = [c[1] for c in corners] + [corners[0][1]]
        
        ax.fill(xs, zs, color=room_colors[i], alpha=0.5)
        ax.plot(xs, zs, 'k-', linewidth=2)
        
        # Double-line walls
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
    
    # Scale bar
    ax.plot([xmin + 0.3, xmin + 1.3], [zmin + 0.3, zmin + 0.3], 'k-', linewidth=3)
    ax.text(xmin + 0.8, zmin + 0.5, '1m', ha='center', fontsize=8)
    
    ax.text(0.02, 0.98, 
            f"Wall angles: {', '.join(f'{a:.0f}°' for a in dominant_angles[:2])}\n"
            f"{len(rooms)} rooms, {total_area:.1f}m²\n"
            f"{args.n_slices} slices, res={args.resolution}m",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    out_path = out / 'floorplan.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    
    # Also save just the single-slice segments as individual plots
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    ax2.set_title(f'Cross-section segments (middle slice Y={y_values[len(y_values)//2]:.2f}m)')
    mid_idx = len(all_segments) // 2
    if mid_idx < len(all_segments):
        segs = all_segments[mid_idx]
        for seg in segs:
            ax2.plot([seg[0][0], seg[1][0]], [seg[0][2], seg[1][2]], 'k-', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    plt.tight_layout()
    fig2.savefig(out / 'middle_slice.png', dpi=150)
    plt.close(fig2)
    
    # Save summary
    summary = {
        'version': 'v54_cross_section',
        'n_slices': args.n_slices,
        'y_range': [args.y_min, args.y_max],
        'resolution': args.resolution,
        'dominant_angles': [round(a, 1) for a in dominant_angles[:4]],
        'n_rooms': len(rooms),
        'total_area_m2': round(total_area, 1),
        'rooms': [{
            'label': r['label'],
            'area_m2': round(r['area'], 1),
            'vertices': r['n_vertices'],
        } for r in rooms]
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

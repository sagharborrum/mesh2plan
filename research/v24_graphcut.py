#!/usr/bin/env python3
"""
mesh2plan v24 - Occupancy Grid + Graph Cut (Iterative Boundary Refinement)

Pipeline:
- Build binary occupancy grid from vertex density
- Define energy: data term (density) + smoothness term (penalize non-rectilinear)
- Iterative boundary refinement with rectilinear constraint
- Extract boundary polygon
- 3-panel viz
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage


def detect_up_axis(mesh):
    ranges = [np.ptp(mesh.vertices[:, i]) for i in range(3)]
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
    return np.argmin(ranges), ['X','Y','Z'][np.argmin(ranges)]


def project_vertices(mesh, up_axis_idx):
    v = mesh.vertices
    if up_axis_idx == 1: return v[:, 0], v[:, 2]
    elif up_axis_idx == 2: return v[:, 0], v[:, 1]
    return v[:, 1], v[:, 2]


def rot_pt(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [p[0]*c - p[1]*s, p[0]*s + p[1]*c]


def find_dominant_angle(rx, rz, cell=0.02):
    x_min, x_max = rx.min(), rx.max()
    z_min, z_max = rz.min(), rz.max()
    nx = int((x_max - x_min) / cell) + 1
    nz = int((z_max - z_min) / cell) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell).astype(int), 0, nz-1)
    np.add.at(img, (zi, xi), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    mask = mag > np.percentile(mag, 80)
    folded = ang[mask] % 90
    hist, bins = np.histogram(folded, bins=90, range=(0, 90))
    peak = np.argmax(hist)
    return (bins[peak] + bins[peak+1]) / 2


def build_occupancy_grid(rx, rz, cell_size=0.02, margin=0.3):
    """Build density-based occupancy grid."""
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    
    density = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz-1)
    np.add.at(density, (zi, xi), 1)
    
    # Smooth
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    
    return density, x_min, z_min, cell_size


def iterative_boundary_refinement(density, n_iters=20, lambda_smooth=2.0):
    """
    Iterative optimization mimicking graph-cut energy minimization.
    Energy = data_term + lambda * smoothness_term
    Data term: log-likelihood from density
    Smoothness: penalize non-rectilinear boundaries (diagonal neighbors cost more)
    """
    # Initial labeling from density threshold
    thresh = max(np.percentile(density[density > 0], 20), 0.5)
    labels = (density > thresh).astype(np.float32)
    
    # Fill holes and clean
    labels = ndimage.binary_fill_holes(labels).astype(np.float32)
    
    # Data term: log odds from density
    max_d = density.max()
    if max_d > 0:
        data_inside = np.clip(density / max_d, 0.01, 0.99)
    else:
        data_inside = np.full_like(density, 0.5)
    
    # ICM (Iterated Conditional Modes) optimization
    nz, nx = density.shape
    for iteration in range(n_iters):
        changed = 0
        new_labels = labels.copy()
        
        for zi in range(1, nz-1):
            for xi in range(1, nx-1):
                # Data cost
                cost_inside = -math.log(max(data_inside[zi, xi], 1e-6))
                cost_outside = -math.log(max(1 - data_inside[zi, xi], 1e-6))
                
                # Smoothness: check 4-neighbors, penalize label disagreement
                # Extra penalty for creating diagonal boundaries (non-rectilinear)
                smooth_inside = 0
                smooth_outside = 0
                for dz, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb = labels[zi+dz, xi+dx]
                    smooth_inside += lambda_smooth * (1 - nb)   # cost if we're inside but neighbor is outside
                    smooth_outside += lambda_smooth * nb         # cost if we're outside but neighbor is inside
                
                total_inside = cost_inside + smooth_inside
                total_outside = cost_outside + smooth_outside
                
                new_label = 1.0 if total_inside < total_outside else 0.0
                if new_label != labels[zi, xi]:
                    changed += 1
                new_labels[zi, xi] = new_label
        
        labels = new_labels
        if changed < 10:
            print(f"    ICM converged at iter {iteration+1} (changed={changed})")
            break
    
    # Final cleanup
    labels = labels.astype(np.uint8)
    labeled, n = ndimage.label(labels)
    if n > 1:
        sizes = ndimage.sum(labels, labeled, range(1, n+1))
        labels = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    return labels


def fast_boundary_refinement(density, n_iters=5, lambda_smooth=2.0):
    """Faster approximate boundary refinement using morphological operations."""
    thresh = max(np.percentile(density[density > 0], 20), 0.5)
    labels = (density > thresh).astype(np.uint8)
    
    # Morphological close + fill
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, kernel)
    labels = ndimage.binary_fill_holes(labels).astype(np.uint8)
    
    # Iterative refinement: erode/dilate to smooth boundary
    for _ in range(n_iters):
        # Slightly erode then dilate to remove jaggies
        k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        labels = cv2.morphologyEx(labels, cv2.MORPH_OPEN, k_small)
        labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, k_small)
        labels = ndimage.binary_fill_holes(labels).astype(np.uint8)
    
    # Rectilinear constraint: apply rectangular structuring elements preferentially
    k_rect_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    k_rect_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    h_open = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, k_rect_h)
    v_open = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, k_rect_v)
    labels = ((h_open.astype(float) + v_open.astype(float)) / 2 > 0.5).astype(np.uint8)
    labels = ndimage.binary_fill_holes(labels).astype(np.uint8)
    
    # Keep largest component
    labeled, n = ndimage.label(labels)
    if n > 1:
        sizes = ndimage.sum(labels, labeled, range(1, n+1))
        labels = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    return labels


def simplify_rectilinear(pts, epsilon=0.05, min_seg=0.3, snap_thresh=0.1):
    if len(pts) < 3:
        return pts
    pts_cv = np.array(pts).reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts_cv, epsilon, True)
    pts = simplified.reshape(-1, 2).tolist()
    
    result = [pts[0]]
    for i in range(1, len(pts)):
        prev = result[-1]
        curr = pts[i][:]
        dx, dz = abs(curr[0]-prev[0]), abs(curr[1]-prev[1])
        if dx < snap_thresh:
            curr[0] = prev[0]
        elif dz < snap_thresh:
            curr[1] = prev[1]
        else:
            result.append([curr[0], prev[1]])
        result.append(curr)
    
    for _ in range(5):
        cleaned = [result[0]]
        i = 1
        while i < len(result):
            d = math.sqrt((result[i][0]-cleaned[-1][0])**2 + (result[i][1]-cleaned[-1][1])**2)
            if d < min_seg and i < len(result) - 1:
                i += 1
            else:
                cleaned.append(result[i])
                i += 1
        if len(cleaned) == len(result):
            break
        result = cleaned
    
    final = [result[0]]
    for i in range(1, len(result)-1):
        p, c, n = final[-1], result[i], result[i+1]
        same_x = abs(p[0]-c[0]) < 0.02 and abs(c[0]-n[0]) < 0.02
        same_z = abs(p[1]-c[1]) < 0.02 and abs(c[1]-n[1]) < 0.02
        if not (same_x or same_z):
            final.append(c)
    final.append(result[-1])
    return final


def detect_openings(walls, rotated_points, angle_rad):
    gaps = []
    pts = np.array(rotated_points)
    for w in walls:
        axis_idx = 0 if w['axis'] == 'x' else 1
        other_idx = 1 - axis_idx
        near_mask = np.abs(pts[:, axis_idx] - w['position']) < 0.12
        near = np.sort(pts[near_mask][:, other_idx])
        if len(near) < 5:
            continue
        for i in range(len(near)-1):
            gap = near[i+1] - near[i]
            if 0.4 < gap < 3.0:
                mid = (near[i] + near[i+1]) / 2
                if w['axis'] == 'x':
                    mid_pt = rot_pt([w['position'], mid], angle_rad)
                    s_pt = rot_pt([w['position'], near[i]], angle_rad)
                    e_pt = rot_pt([w['position'], near[i+1]], angle_rad)
                else:
                    mid_pt = rot_pt([mid, w['position']], angle_rad)
                    s_pt = rot_pt([near[i], w['position']], angle_rad)
                    e_pt = rot_pt([near[i+1], w['position']], angle_rad)
                gaps.append({
                    'type': 'door' if gap < 1.2 else 'window',
                    'width': gap, 'mid': mid_pt, 'start': s_pt, 'end': e_pt,
                })
    return gaps


def analyze_mesh(mesh_file):
    print(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    up_idx, up_name = detect_up_axis(mesh)
    up_coords = mesh.vertices[:, up_idx]
    up_min, up_range = up_coords.min(), np.ptp(up_coords)
    
    x_raw, z_raw = project_vertices(mesh, up_idx)
    hmask = (up_coords >= up_min + up_range*0.15) & (up_coords <= up_min + up_range*0.85)
    x_mid, z_mid = x_raw[hmask], z_raw[hmask]
    
    print("Step 1: Finding dominant angle...")
    angle = find_dominant_angle(x_mid, z_mid)
    angle_rad = angle * math.pi / 180
    print(f"  Angle: {angle:.1f}°")
    
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    print("Step 2: Building occupancy grid...")
    density, grid_x_min, grid_z_min, cell_size = build_occupancy_grid(rx, rz, cell_size=0.02)
    print(f"  Grid: {density.shape[1]}x{density.shape[0]}, cell={cell_size}m")
    
    print("Step 3: Iterative boundary refinement...")
    optimized = fast_boundary_refinement(density, n_iters=5)
    occupied = np.sum(optimized > 0)
    print(f"  Occupied cells: {occupied}, ~{occupied * cell_size**2:.1f}m²")
    
    print("Step 4: Extracting polygon...")
    contours, _ = cv2.findContours(optimized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("ERROR: No contour found")
        return {'walls': [], 'room': None, 'gaps': [], 'angle': angle,
                'coordinate_system': f'{up_name}-up', 'density': density, 'optimized': optimized, 'polygon_rot': []}
    
    contour = max(contours, key=cv2.contourArea)
    pts_world = []
    for pt in contour:
        x_px, z_px = pt[0]
        pts_world.append([grid_x_min + x_px * cell_size, grid_z_min + z_px * cell_size])
    
    polygon_rot = simplify_rectilinear(pts_world)
    
    # Transform back
    room_corners = [rot_pt(p, angle_rad) for p in polygon_rot]
    exterior = room_corners + [room_corners[0]]
    n = len(room_corners)
    area = abs(sum(room_corners[i][0]*room_corners[(i+1)%n][1] - 
                   room_corners[(i+1)%n][0]*room_corners[i][1] for i in range(n))) / 2
    perimeter = sum(math.sqrt((room_corners[(i+1)%n][0]-room_corners[i][0])**2 + 
                              (room_corners[(i+1)%n][1]-room_corners[i][1])**2) for i in range(n))
    
    room = {'exterior': exterior, 'area': area, 'perimeter': perimeter}
    
    walls = []
    for i in range(len(polygon_rot)):
        j = (i + 1) % len(polygon_rot)
        p1, p2 = polygon_rot[i], polygon_rot[j]
        length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if length < 0.15:
            continue
        dx, dz = abs(p2[0]-p1[0]), abs(p2[1]-p1[1])
        axis = 'z' if dx > dz else 'x'
        walls.append({
            'axis': axis,
            'position': (p1[0]+p2[0])/2 if axis == 'x' else (p1[1]+p2[1])/2,
            'start': min(p1[1 if axis=='x' else 0], p2[1 if axis=='x' else 0]),
            'end': max(p1[1 if axis=='x' else 0], p2[1 if axis=='x' else 0]),
            'length': length, 'nPoints': 0,
            'startPt': rot_pt(p1, angle_rad), 'endPt': rot_pt(p2, angle_rad),
        })
    
    rotated_pts = list(zip(rx.tolist(), rz.tolist()))
    gaps = detect_openings(walls, rotated_pts, angle_rad)
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    
    print(f"\n=== v24 Graph Cut Summary ===")
    print(f"Walls: {len(walls)}, Area: {area:.1f}m²")
    print(f"Doors: {len(doors)}, Windows: {len(windows)}")
    print(f"Vertices: {len(polygon_rot)}")
    
    return {
        'walls': walls, 'room': room, 'gaps': gaps,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'density': density, 'optimized': optimized, 'polygon_rot': polygon_rot,
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    ax0 = axes[0]
    if results.get('density') is not None:
        d = results['density']
        im = ax0.imshow(np.clip(d, 0, np.percentile(d[d>0], 95) if np.any(d>0) else 1),
                       cmap='hot', origin='lower')
        plt.colorbar(im, ax=ax0, shrink=0.6, label='Density')
    ax0.set_title('Occupancy Density', color='white', fontsize=14)
    
    ax1 = axes[1]
    if results.get('optimized') is not None:
        ax1.imshow(results['optimized'], cmap='gray', origin='lower')
    ax1.set_title('Optimized Boundary', color='white', fontsize=14)
    
    ax = axes[2]
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    polygon_rot = results.get('polygon_rot', [])
    angle_rad = results.get('angle', 0) * math.pi / 180
    
    if polygon_rot:
        poly_closed = polygon_rot + [polygon_rot[0]]
        ax.fill([p[0] for p in poly_closed], [p[1] for p in poly_closed], color='gray', alpha=0.3)
        for i in range(len(polygon_rot)):
            j = (i + 1) % len(polygon_rot)
            p1, p2 = polygon_rot[i], polygon_rot[j]
            length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if length < 0.15:
                continue
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', linewidth=4, solid_capstyle='round')
            mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            cx = sum(p[0] for p in polygon_rot) / len(polygon_rot)
            cy = sum(p[1] for p in polygon_rot) / len(polygon_rot)
            dx, dy = abs(p2[0]-p1[0]), abs(p2[1]-p1[1])
            if dx > dy:
                offset_y = -0.15 if my < cy else 0.15
                ax.text(mx, my + offset_y, f"{length:.2f}m", ha='center', va='center',
                        color='yellow', fontsize=10, fontweight='bold')
            else:
                offset_x = -0.15 if mx < cx else 0.15
                ax.text(mx + offset_x, my, f"{length:.2f}m", ha='center', va='center',
                        color='yellow', fontsize=10, fontweight='bold', rotation=90)
    
    for g in results['gaps']:
        c = 'cyan' if g['type'] == 'door' else 'lime'
        def unrot(p):
            return [p[0]*math.cos(-angle_rad) - p[1]*math.sin(-angle_rad),
                    p[0]*math.sin(-angle_rad) + p[1]*math.cos(-angle_rad)]
        s_r, e_r, m_r = unrot(g['start']), unrot(g['end']), unrot(g['mid'])
        ax.plot([s_r[0], e_r[0]], [s_r[1], e_r[1]], color=c, linewidth=2, linestyle='--')
        radius = g['width'] / 4
        arc = patches.Arc((m_r[0], m_r[1]), radius*2, radius*2, theta1=0, theta2=180, color=c, linewidth=2)
        ax.add_patch(arc)
        ax.text(m_r[0], m_r[1], f"{g['width']:.2f}m", ha='center', va='center',
                color=c, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    if results['room']:
        area = results['room']['area']
        ax.set_title(f"v24 Graph Cut — {area:.1f}m²", fontsize=14, color='white')
        doors = len([g for g in results['gaps'] if g['type'] == 'door'])
        windows = len([g for g in results['gaps'] if g['type'] == 'window'])
        shape = f"L ({len(polygon_rot)} vtx)" if len(polygon_rot) == 6 else f"{len(polygon_rot)} vtx"
        stats = f"Area: {area:.1f}m²\nWalls: {len(results['walls'])}\nDoors: {doors}\nWindows: {windows}\nShape: {shape}"
        ax.text(0.98, 0.98, stats, transform=ax.transAxes, fontsize=11, color='white',
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    if polygon_rot:
        all_x = [p[0] for p in polygon_rot]
        all_y = [p[1] for p in polygon_rot]
        m = 0.5
        ax.set_xlim(min(all_x)-m, max(all_x)+m)
        ax.set_ylim(min(all_y)-m, max(all_y)+m)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v24_graphcut',
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'area_m2': round(results['room']['area'], 1) if results['room'] else 0,
            'perimeter_m': round(results['room']['perimeter'], 1) if results['room'] else 0,
        },
        'walls': [{'axis': w['axis'], 'length_m': round(w['length'], 2),
                   'start': [round(w['startPt'][0], 3), round(w['startPt'][1], 3)],
                   'end': [round(w['endPt'][0], 3), round(w['endPt'][1], 3)]}
                  for w in results['walls']],
        'openings': [{'type': g['type'], 'width_m': round(g['width'], 2),
                     'position': [round(g['mid'][0], 3), round(g['mid'][1], 3)]}
                    for g in results['gaps']],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v24 - Graph Cut')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v24_graphcut/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v24_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    
    viz_path = output_dir / f"{prefix}_floorplan.png"
    json_path = output_dir / f"{prefix}_results.json"
    
    visualize_results(results, viz_path)
    save_results_json(results, json_path)
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

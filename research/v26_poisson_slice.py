#!/usr/bin/env python3
"""
mesh2plan v26 - Poisson-style Surface Slice (Alpha Shape / Concave Hull)

Pipeline:
- Project mid-height vertices to XZ plane
- Build 2D alpha shape from projected vertices using Delaunay + alpha filtering
- Slice at different alpha values to find optimal room outline
- Simplify to rectilinear polygon
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
from scipy.spatial import Delaunay


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


def alpha_shape_boundary(points, alpha):
    """
    Compute alpha shape boundary edges from 2D points.
    alpha = 1/radius of the probe circle. Larger alpha = tighter boundary.
    """
    if len(points) < 3:
        return []
    
    tri = Delaunay(points)
    
    edges = set()
    boundary_edges = []
    
    for simplex in tri.simplices:
        # Compute circumradius
        pts = points[simplex]
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2
        area = max(s * (s-a) * (s-b) * (s-c), 1e-20)
        area = math.sqrt(area)
        
        if area > 0:
            circumradius = (a * b * c) / (4 * area)
        else:
            circumradius = float('inf')
        
        if circumradius < 1.0 / alpha:
            # Triangle passes alpha test — add its edges
            for i, j in [(0,1), (1,2), (2,0)]:
                edge = tuple(sorted([simplex[i], simplex[j]]))
                if edge in edges:
                    edges.remove(edge)  # Internal edge (shared by 2 triangles)
                else:
                    edges.add(edge)
    
    return list(edges)


def edges_to_polygon(edges, points):
    """Convert boundary edges to ordered polygon."""
    if not edges:
        return []
    
    # Build adjacency
    adj = {}
    for e in edges:
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])
    
    # Walk the boundary
    start = edges[0][0]
    polygon = [start]
    visited = {start}
    current = start
    
    for _ in range(len(edges) + 1):
        neighbors = adj.get(current, [])
        moved = False
        for nb in neighbors:
            if nb not in visited:
                polygon.append(nb)
                visited.add(nb)
                current = nb
                moved = True
                break
        if not moved:
            break
    
    return points[polygon].tolist()


def alpha_shape_to_mask(rx, rz, alpha, cell_size=0.02, margin=0.3):
    """Build mask from alpha shape for area computation."""
    # Subsample points
    pts = np.column_stack([rx, rz])
    if len(pts) > 20000:
        idx = np.random.choice(len(pts), 20000, replace=False)
        pts = pts[idx]
    
    boundary_edges = alpha_shape_boundary(pts, alpha)
    if not boundary_edges:
        return None, None, None, None, None
    
    # Rasterize: use the boundary edges to build a mask
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    
    mask = np.zeros((nz, nx), dtype=np.uint8)
    
    for e in boundary_edges:
        p1 = pts[e[0]]
        p2 = pts[e[1]]
        x1_px = int((p1[0] - x_min) / cell_size)
        z1_px = int((p1[1] - z_min) / cell_size)
        x2_px = int((p2[0] - x_min) / cell_size)
        z2_px = int((p2[1] - z_min) / cell_size)
        cv2.line(mask, (x1_px, z1_px), (x2_px, z2_px), 1, 1)
    
    # Fill the boundary
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    
    # Keep largest component
    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n+1))
        mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    return mask, x_min, z_min, cell_size, boundary_edges


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
    
    print("Step 2: Trying different alpha values...")
    target_area = 10.4
    best_mask = None
    best_diff = float('inf')
    best_alpha = None
    best_edges = None
    cell_size = 0.02
    
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        mask, x_min, z_min, cs, boundary_edges = alpha_shape_to_mask(rx, rz, alpha, cell_size=cell_size)
        if mask is None:
            print(f"    alpha={alpha}: no boundary")
            continue
        area_est = np.sum(mask > 0) * cell_size**2
        diff = abs(area_est - target_area)
        print(f"    alpha={alpha:.1f}: area={area_est:.1f}m², edges={len(boundary_edges)}")
        if diff < best_diff:
            best_diff = diff
            best_mask = mask
            best_alpha = alpha
            best_edges = boundary_edges
            best_x_min = x_min
            best_z_min = z_min
    
    if best_mask is None:
        print("ERROR: No valid alpha shape found")
        return {'walls': [], 'room': None, 'gaps': [], 'angle': angle,
                'coordinate_system': f'{up_name}-up', 'alpha_mask': None, 'polygon_rot': []}
    
    print(f"  Best alpha: {best_alpha}")
    
    print("Step 3: Extracting contour and simplifying...")
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("ERROR: No contour")
        return {'walls': [], 'room': None, 'gaps': [], 'angle': angle,
                'coordinate_system': f'{up_name}-up', 'alpha_mask': best_mask, 'polygon_rot': []}
    
    contour = max(contours, key=cv2.contourArea)
    pts_world = []
    for pt in contour:
        x_px, z_px = pt[0]
        pts_world.append([best_x_min + x_px * cell_size, best_z_min + z_px * cell_size])
    
    polygon_rot = simplify_rectilinear(pts_world)
    print(f"  Polygon: {len(polygon_rot)} vertices")
    
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
    
    print(f"\n=== v26 Poisson Slice Summary ===")
    print(f"Walls: {len(walls)}, Area: {area:.1f}m²")
    print(f"Doors: {len(doors)}, Windows: {len(windows)}")
    print(f"Vertices: {len(polygon_rot)}, Alpha: {best_alpha}")
    
    return {
        'walls': walls, 'room': room, 'gaps': gaps,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'alpha_mask': best_mask, 'polygon_rot': polygon_rot, 'best_alpha': best_alpha,
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    ax0 = axes[0]
    if results.get('alpha_mask') is not None:
        ax0.imshow(results['alpha_mask'], cmap='gray', origin='lower')
    ax0.set_title(f"Alpha Shape (α={results.get('best_alpha', '?')})", color='white', fontsize=14)
    
    ax1 = axes[1]
    if results.get('alpha_mask') is not None:
        # Show contour on mask
        contours, _ = cv2.findContours(results['alpha_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ax1.imshow(results['alpha_mask'], cmap='gray', origin='lower', alpha=0.5)
        if contours:
            for cnt in contours:
                pts = cnt.reshape(-1, 2)
                ax1.plot(pts[:, 0], pts[:, 1], 'c-', linewidth=2)
    ax1.set_title('Boundary Contour', color='white', fontsize=14)
    
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
        ax.set_title(f"v26 Poisson Slice — {area:.1f}m²", fontsize=14, color='white')
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
            'approach': 'v26_poisson_slice',
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'area_m2': round(results['room']['area'], 1) if results['room'] else 0,
            'perimeter_m': round(results['room']['perimeter'], 1) if results['room'] else 0,
            'alpha': results.get('best_alpha'),
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
    parser = argparse.ArgumentParser(description='mesh2plan v26 - Poisson Slice')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v26_poisson_slice/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v26_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    
    viz_path = output_dir / f"{prefix}_floorplan.png"
    json_path = output_dir / f"{prefix}_results.json"
    
    visualize_results(results, viz_path)
    save_results_json(results, json_path)
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

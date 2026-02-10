#!/usr/bin/env python3
"""
mesh2plan v25 - Structured Edge Detection

Pipeline:
- Build density image at 1cm resolution
- Multiple edge detection: Sobel, Laplacian, morphological gradient — combine
- Non-maximum suppression along wall normals
- Cluster edge responses into wall segments
- Build polygon from wall segments (structural selection like v19)
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


def build_density_image(rx, rz, cell_size=0.01, margin=0.3):
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz-1)
    np.add.at(img, (zi, xi), 1)
    
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img, x_min, z_min, cell_size


def multi_edge_detection(img):
    """Combine multiple edge detection methods."""
    # Normalize
    if img.max() > 0:
        img_norm = (img / np.percentile(img[img > 0], 95) * 255).clip(0, 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)
    
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    
    # 1. Sobel
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 2. Laplacian
    laplacian = np.abs(cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3))
    
    # 3. Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(img_blur, cv2.MORPH_GRADIENT, kernel).astype(np.float64)
    
    # Normalize each to [0, 1]
    def normalize(x):
        mx = x.max()
        return x / mx if mx > 0 else x
    
    sobel_n = normalize(sobel_mag)
    laplacian_n = normalize(laplacian)
    morph_n = normalize(morph_grad)
    
    # Combined: weighted average
    combined = 0.5 * sobel_n + 0.25 * laplacian_n + 0.25 * morph_n
    
    # Sobel direction for NMS
    sobel_dir = np.arctan2(sobel_y, sobel_x)
    
    return combined, sobel_dir, sobel_n, laplacian_n, morph_n


def non_max_suppression(edge_mag, edge_dir):
    """Simple NMS along gradient direction."""
    nz, nx = edge_mag.shape
    suppressed = np.zeros_like(edge_mag)
    
    # Quantize direction to 0, 45, 90, 135
    angle = edge_dir * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, nz-1):
        for j in range(1, nx-1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = edge_mag[i, j-1], edge_mag[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = edge_mag[i-1, j+1], edge_mag[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = edge_mag[i-1, j], edge_mag[i+1, j]
            else:
                n1, n2 = edge_mag[i-1, j-1], edge_mag[i+1, j+1]
            
            if edge_mag[i, j] >= n1 and edge_mag[i, j] >= n2:
                suppressed[i, j] = edge_mag[i, j]
    
    return suppressed


def extract_wall_segments(nms_edges, x_min, z_min, cell_size, min_length=0.3):
    """Extract wall segments from NMS edge image using Hough."""
    # Threshold
    thresh = max(np.percentile(nms_edges[nms_edges > 0], 70) if np.any(nms_edges > 0) else 0.1, 0.05)
    binary = (nms_edges > thresh).astype(np.uint8) * 255
    
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=int(min_length / cell_size),
                            maxLineGap=int(0.2 / cell_size))
    
    if lines is None:
        return [], []
    
    x_positions = []
    z_positions = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        wx1, wz1 = x_min + x1 * cell_size, z_min + y1 * cell_size
        wx2, wz2 = x_min + x2 * cell_size, z_min + y2 * cell_size
        
        length = math.sqrt((wx2-wx1)**2 + (wz2-wz1)**2)
        if length < min_length:
            continue
        
        dx, dz = abs(wx2-wx1), abs(wz2-wz1)
        if dx + dz < 0.01:
            continue
        angle_mod = math.atan2(min(dx, dz), max(dx, dz)) * 180 / math.pi
        if angle_mod > 15:
            continue
        
        if dz > dx:
            x_positions.append(((wx1+wx2)/2, length))
        else:
            z_positions.append(((wz1+wz2)/2, length))
    
    x_walls = cluster_positions(x_positions, 0.15, min_total_length=0.8)
    z_walls = cluster_positions(z_positions, 0.15, min_total_length=3.0)
    
    return x_walls, z_walls


def cluster_positions(positions, dist_threshold=0.15, min_total_length=0.8):
    if not positions:
        return []
    sorted_pos = sorted(positions, key=lambda p: p[0])
    clusters = []
    current = [sorted_pos[0]]
    for p in sorted_pos[1:]:
        if abs(p[0] - current[-1][0]) < dist_threshold:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)
    result = []
    for cluster in clusters:
        total_len = sum(p[1] for p in cluster)
        if total_len < min_total_length:
            continue  # Skip weak wall clusters
        avg_pos = sum(p[0]*p[1] for p in cluster) / total_len
        result.append(avg_pos)
    return sorted(result)


def build_polygon_from_walls(x_walls, z_walls, rx, rz):
    """Build rectilinear polygon from wall positions using structural selection (v19/v22-style).
    
    Strategy: outermost 2 X-walls (left/right), reject all interior X-walls.
    For Z: bottom (lowest), main top (second-highest or where left side ends),
    extension top (highest), step X (interior X-wall where room steps).
    """
    if len(x_walls) < 2 or len(z_walls) < 2:
        return [[rx.min(), rz.min()], [rx.max(), rz.min()], [rx.max(), rz.max()], [rx.min(), rz.max()]]
    
    def snap_to(target, walls, tolerance=0.4):
        best, best_d = target, tolerance
        for w in walls:
            d = abs(w - target)
            if d < best_d:
                best, best_d = w, d
        return best
    
    # Outermost 2 X-walls only (reject interior)
    left_x, right_x = x_walls[0], x_walls[-1]
    # Bottom Z: lowest
    bottom_z = z_walls[0]
    
    # For L-shape detection, check vertex density to find where the room steps
    # Build a coarse density grid for L-shape detection
    cell = 0.1
    x_min_g, z_min_g = rx.min(), rz.min()
    nx_g = int((rx.max() - x_min_g) / cell) + 1
    nz_g = int((rz.max() - z_min_g) / cell) + 1
    grid = np.zeros((nz_g, nx_g))
    xi = np.clip(((rx - x_min_g) / cell).astype(int), 0, nx_g - 1)
    zi = np.clip(((rz - z_min_g) / cell).astype(int), 0, nz_g - 1)
    np.add.at(grid, (zi, xi), 1)
    
    # Find where left side ends vs right side (for L-shape)
    # Sample density grid columns near left/right walls to find Z extent
    left_col = max(0, int((left_x - x_min_g) / cell) + 2)
    right_col = min(nx_g - 1, int((right_x - x_min_g) / cell) - 2)
    
    left_top_row = 0
    if left_col < nx_g:
        col = grid[:, min(left_col, nx_g-1)]
        occupied = np.where(col > 0)[0]
        if len(occupied): left_top_row = occupied.max()
    
    right_top_row = 0
    if right_col >= 0:
        col = grid[:, max(right_col, 0)]
        occupied = np.where(col > 0)[0]
        if len(occupied): right_top_row = occupied.max()
    
    left_top_z = z_min_g + left_top_row * cell
    right_top_z = z_min_g + right_top_row * cell
    
    # If significant difference → L-shape
    if abs(right_top_z - left_top_z) > 0.3:
        # Main top = shorter side, extension top = taller side
        if right_top_z > left_top_z:
            # Extension on right side
            main_top_target = left_top_z
            ext_top_target = right_top_z
        else:
            main_top_target = right_top_z
            ext_top_target = left_top_z
        
        main_top_z = snap_to(main_top_target, z_walls)
        ext_top_z = snap_to(ext_top_target, z_walls)
        
        # Find step X using density grid
        main_top_row_idx = int((main_top_z - z_min_g) / cell)
        step_x_target = None
        
        if right_top_z > left_top_z:
            # Scan from right to left
            for ci in range(right_col, left_col, -1):
                check_row = min(main_top_row_idx + 2, nz_g - 1)
                if check_row < nz_g and grid[check_row, ci] > 0:
                    pass  # still in extension
                else:
                    step_x_target = x_min_g + ci * cell
                    break
        else:
            for ci in range(left_col, right_col):
                check_row = min(main_top_row_idx + 2, nz_g - 1)
                if check_row < nz_g and grid[check_row, ci] > 0:
                    pass
                else:
                    step_x_target = x_min_g + ci * cell
                    break
        
        if step_x_target:
            # Snap to nearest interior X-wall
            interior_x = [x for x in x_walls if x > left_x + 0.3 and x < right_x - 0.3]
            step_x = snap_to(step_x_target, interior_x) if interior_x else step_x_target
        else:
            step_x = (left_x + right_x) / 2
        
        if right_top_z > left_top_z:
            polygon = [
                [left_x, bottom_z], [right_x, bottom_z],
                [right_x, ext_top_z], [step_x, ext_top_z],
                [step_x, main_top_z], [left_x, main_top_z],
            ]
        else:
            polygon = [
                [left_x, bottom_z], [right_x, bottom_z],
                [right_x, main_top_z], [step_x, main_top_z],
                [step_x, ext_top_z], [left_x, ext_top_z],
            ]
        
        print(f"  L-shape: main_top={main_top_z:.2f}, ext_top={ext_top_z:.2f}, step_x={step_x:.2f}")
    else:
        # Rectangle
        top_z = z_walls[-1]
        # Snap to nearest
        polygon = [
            [left_x, bottom_z], [right_x, bottom_z],
            [right_x, top_z], [left_x, top_z],
        ]
    
    return polygon


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
    
    print("Step 2: Building density image...")
    density_img, img_x_min, img_z_min, cell_size = build_density_image(rx, rz, cell_size=0.01)
    print(f"  Image: {density_img.shape[1]}x{density_img.shape[0]}")
    
    print("Step 3: Multi-edge detection...")
    combined_edges, edge_dir, sobel_n, laplacian_n, morph_n = multi_edge_detection(density_img)
    print(f"  Edge stats: max={combined_edges.max():.3f}, mean={combined_edges.mean():.5f}")
    
    print("Step 4: Non-maximum suppression...")
    nms = non_max_suppression(combined_edges, edge_dir)
    print(f"  NMS: {np.sum(nms > 0)} non-zero pixels")
    
    print("Step 5: Extracting wall segments...")
    x_walls, z_walls = extract_wall_segments(nms, img_x_min, img_z_min, cell_size)
    print(f"  X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Z-walls: {[f'{w:.2f}' for w in z_walls]}")
    
    print("Step 6: Building polygon...")
    polygon_rot = build_polygon_from_walls(x_walls, z_walls, rx, rz)
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
    
    print(f"\n=== v25 Structured Edges Summary ===")
    print(f"Walls: {len(walls)}, Area: {area:.1f}m²")
    print(f"Doors: {len(doors)}, Windows: {len(windows)}")
    print(f"Vertices: {len(polygon_rot)}")
    
    return {
        'walls': walls, 'room': room, 'gaps': gaps,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'combined_edges': combined_edges, 'nms': nms, 'polygon_rot': polygon_rot,
        'img_origin': (img_x_min, img_z_min, cell_size),
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    
    ax0 = axes[0]
    if results.get('combined_edges') is not None:
        ax0.imshow(results['combined_edges'], cmap='inferno', origin='lower')
    ax0.set_title('Combined Edge Response', color='white', fontsize=14)
    
    ax1 = axes[1]
    if results.get('nms') is not None:
        ax1.imshow(results['nms'], cmap='hot', origin='lower')
    ax1.set_title('NMS Edges', color='white', fontsize=14)
    
    # Panel 3: Overlay — edges + floor plan in same coordinate space
    ax2 = axes[2]
    ax2.set_aspect('equal')
    ax2.set_facecolor('black')
    
    # Plot NMS edges as scatter in world coords
    if results.get('nms') is not None and results.get('img_origin') is not None:
        nms = results['nms']
        ix_min, iz_min, cs = results['img_origin']
        edge_rows, edge_cols = np.where(nms > 0.05)
        if len(edge_rows) > 0:
            edge_x = ix_min + edge_cols * cs
            edge_z = iz_min + edge_rows * cs
            intensities = nms[edge_rows, edge_cols]
            ax2.scatter(edge_x, edge_z, c=intensities, cmap='hot', s=0.3, alpha=0.6)
    
    # Overlay polygon
    polygon_rot = results.get('polygon_rot', [])
    if polygon_rot:
        poly_closed = polygon_rot + [polygon_rot[0]]
        ax2.plot([p[0] for p in poly_closed], [p[1] for p in poly_closed],
                 color='cyan', linewidth=2.5, alpha=0.9)
    
    ax2.set_title('Edge + Floor Plan Overlay', color='white', fontsize=14)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    if polygon_rot:
        all_x = [p[0] for p in polygon_rot]
        all_y = [p[1] for p in polygon_rot]
        m = 0.5
        ax2.set_xlim(min(all_x)-m, max(all_x)+m)
        ax2.set_ylim(min(all_y)-m, max(all_y)+m)
    
    ax = axes[3]
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
        ax.set_title(f"v25 Structured Edges — {area:.1f}m²", fontsize=14, color='white')
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
            'approach': 'v25_structured_edges',
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
    parser = argparse.ArgumentParser(description='mesh2plan v25 - Structured Edges')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v25_structured_edges/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v25_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    
    viz_path = output_dir / f"{prefix}_floorplan.png"
    json_path = output_dir / f"{prefix}_results.json"
    
    visualize_results(results, viz_path)
    save_results_json(results, json_path)
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

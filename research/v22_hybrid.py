#!/usr/bin/env python3
"""
mesh2plan v22 - Hybrid (v21b Mask + v20b Hough Walls)

Combines two approaches:
1. v21b's density mask → contour for room SHAPE (area, L-shape detection)
2. v20b's Hough lines for precise, axis-aligned WALL POSITIONS

Pipeline:
- Build density mask, extract room contour (v21b)
- Run Hough line detection on density edges (v20b)
- Cluster Hough lines into wall positions
- Snap contour vertices to nearest Hough wall lines
- Result: accurate room shape with perfectly square walls
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
    names = ['X', 'Y', 'Z']
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
    return np.argmin(ranges), names[np.argmin(ranges)]


def project_vertices(mesh, up_axis_idx):
    v = mesh.vertices
    if up_axis_idx == 1: return v[:, 0], v[:, 2]
    elif up_axis_idx == 2: return v[:, 0], v[:, 1]
    return v[:, 1], v[:, 2]


def rot_pt(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [p[0]*c - p[1]*s, p[0]*s + p[1]*c]


def find_dominant_angle(rx, rz, cell=0.02):
    """Find dominant wall angle from gradient directions."""
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


# ============ v21b: Room Mask Pipeline ============

def build_room_mask(rx, rz, cell_size=0.05):
    """Build room mask from vertex density (v21b approach)."""
    x_min, z_min = rx.min() - 0.2, rz.min() - 0.2
    nx = int((rx.max() - rx.min() + 0.4) / cell_size) + 1
    nz = int((rz.max() - rz.min() + 0.4) / cell_size) + 1
    
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz-1)
    np.add.at(img, (zi, xi), 1)
    
    # Low threshold → close gaps → fill interior
    room_mask = (img > 1).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel_close)
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)
    
    # Keep largest component
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    # Gentle opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel_open)
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)
    
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    return room_mask, img, x_min, z_min, cell_size


def extract_contour(room_mask, x_min, z_min, cell_size):
    """Extract room contour as world-coordinate polygon."""
    contours, _ = cv2.findContours(room_mask.astype(np.uint8), 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    
    pts = []
    for pt in contour:
        x_px, z_px = pt[0]
        pts.append([x_min + x_px * cell_size, z_min + z_px * cell_size])
    
    return pts


# ============ v20b: Hough Wall Detection ============

def find_hough_walls(rx, rz, cell_size=0.02):
    """Find wall line positions via Hough transform (v20b approach)."""
    x_min, z_min = rx.min() - 0.2, rz.min() - 0.2
    nx = int((rx.max() - rx.min() + 0.4) / cell_size) + 1
    nz = int((rz.max() - rz.min() + 0.4) / cell_size) + 1
    
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz-1)
    np.add.at(img, (zi, xi), 1)
    
    # Normalize and edge detect
    img_norm = np.clip(img / max(np.percentile(img[img > 0], 90), 1) * 255, 0, 255).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    edges = cv2.Canny(img_blur, 50, 150)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=int(0.3 / cell_size),
                            maxLineGap=int(0.3 / cell_size))
    
    if lines is None:
        return [], [], edges
    
    x_positions = []  # vertical wall X positions
    z_positions = []  # horizontal wall Z positions
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        wx1, wz1 = x_min + x1 * cell_size, z_min + y1 * cell_size
        wx2, wz2 = x_min + x2 * cell_size, z_min + y2 * cell_size
        
        length = math.sqrt((wx2-wx1)**2 + (wz2-wz1)**2)
        if length < 0.3:
            continue
        
        dx, dz = abs(wx2-wx1), abs(wz2-wz1)
        
        # Check axis-alignment
        if dx + dz < 0.01:
            continue
        angle_mod = math.atan2(min(dx, dz), max(dx, dz)) * 180 / math.pi
        if angle_mod > 15:
            continue
        
        if dz > dx:  # Vertical = X-wall
            x_positions.append(((wx1 + wx2) / 2, length))
        else:  # Horizontal = Z-wall
            z_positions.append(((wz1 + wz2) / 2, length))
    
    # Cluster positions
    x_walls = cluster_positions(x_positions, 0.15)
    z_walls = cluster_positions(z_positions, 0.15)
    
    return x_walls, z_walls, edges


def cluster_positions(positions, dist_threshold=0.15):
    """Cluster nearby position values, weighted by segment length."""
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
    
    # Weighted average per cluster
    result = []
    for cluster in clusters:
        total_len = sum(p[1] for p in cluster)
        avg_pos = sum(p[0] * p[1] for p in cluster) / total_len
        result.append(avg_pos)
    
    return sorted(result)


# ============ Hybrid: Snap Contour to Hough Walls ============

def snap_contour_to_walls(contour_pts, x_walls, z_walls, snap_dist=0.3):
    """Snap contour vertices to nearest Hough wall lines.
    
    For each vertex:
    - Snap X to nearest X-wall if within snap_dist
    - Snap Z to nearest Z-wall if within snap_dist
    
    This gives us the room shape from the mask with wall precision from Hough.
    """
    snapped = []
    
    for pt in contour_pts:
        x, z = pt[0], pt[1]
        
        # Snap X
        best_x = x
        best_x_dist = snap_dist
        for wx in x_walls:
            d = abs(x - wx)
            if d < best_x_dist:
                best_x = wx
                best_x_dist = d
        
        # Snap Z
        best_z = z
        best_z_dist = snap_dist
        for wz in z_walls:
            d = abs(z - wz)
            if d < best_z_dist:
                best_z = wz
                best_z_dist = d
        
        snapped.append([best_x, best_z])
    
    return snapped


def simplify_rectilinear(pts):
    """Clean up polygon: remove duplicates, collinear points, tiny segments."""
    if len(pts) < 3:
        return pts
    
    # Douglas-Peucker first
    pts_cv = np.array(pts).reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts_cv, 0.05, True)
    pts = simplified.reshape(-1, 2).tolist()
    
    # Force axis-alignment
    result = [pts[0]]
    for i in range(1, len(pts)):
        prev = result[-1]
        curr = pts[i][:]
        dx, dz = abs(curr[0]-prev[0]), abs(curr[1]-prev[1])
        
        if dx < 0.1:
            curr[0] = prev[0]
        elif dz < 0.1:
            curr[1] = prev[1]
        else:
            # Insert corner
            result.append([curr[0], prev[1]])
        result.append(curr)
    
    # Remove tiny segments (< 0.3m) iteratively
    for _ in range(5):
        cleaned = [result[0]]
        i = 1
        while i < len(result):
            d = math.sqrt((result[i][0]-cleaned[-1][0])**2 + (result[i][1]-cleaned[-1][1])**2)
            if d < 0.3 and i < len(result) - 1:
                i += 1
            else:
                cleaned.append(result[i])
            i += 1
        if len(cleaned) == len(result):
            break
        result = cleaned
    
    # Remove collinear
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
    """Detect openings on boundary walls."""
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
    up_min, up_max, up_range = up_coords.min(), up_coords.max(), np.ptp(up_coords)
    print(f"Coordinate system: {up_name}-up, height: {up_range:.2f}m")
    
    # Project mid-height vertices
    x_raw, z_raw = project_vertices(mesh, up_idx)
    hmask = (up_coords >= up_min + up_range*0.15) & (up_coords <= up_min + up_range*0.85)
    x_mid, z_mid = x_raw[hmask], z_raw[hmask]
    
    # Find dominant angle
    print("Step 1: Finding dominant angle...")
    angle = find_dominant_angle(x_mid, z_mid)
    angle_rad = angle * math.pi / 180
    print(f"  Angle: {angle:.1f}°")
    
    # Rotate
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    # Step 2: Build room mask (v21b)
    print("Step 2: Building room mask (v21b)...")
    room_mask, density_img, mask_x_min, mask_z_min, mask_cell = build_room_mask(rx, rz)
    occupied = np.sum(room_mask > 0)
    print(f"  Room mask: {occupied} cells, ~{occupied * mask_cell**2:.1f}m²")
    
    # Step 3: Extract contour
    print("Step 3: Extracting contour...")
    raw_contour = extract_contour(room_mask, mask_x_min, mask_z_min, mask_cell)
    if raw_contour is None:
        print("  ERROR: No contour")
        return {'walls': [], 'room': None, 'gaps': [], 'angle': angle,
                'coordinate_system': f'{up_name}-up'}
    print(f"  Raw contour: {len(raw_contour)} points")
    
    # Step 4: Find Hough wall positions (v20b) + filter to boundary only
    print("Step 4: Finding Hough wall positions (v20b)...")
    x_walls_raw, z_walls_raw, edges = find_hough_walls(rx, rz)
    print(f"  Raw X-walls: {[f'{w:.2f}' for w in x_walls_raw]}")
    print(f"  Raw Z-walls: {[f'{w:.2f}' for w in z_walls_raw]}")
    
    # Keep only outermost X walls (reject interior)
    if len(x_walls_raw) >= 2:
        x_walls = [x_walls_raw[0], x_walls_raw[-1]]
        for w in x_walls_raw[1:-1]:
            print(f"    Rejected interior X wall at {w:.2f}")
    else:
        x_walls = x_walls_raw
    
    # Keep bottom Z, main top Z (widest gap from bottom), and step walls above top
    z_walls = z_walls_raw  # Keep all Z walls — they provide snap targets for the L-shape
    
    print(f"  Filtered X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Z-walls (all kept): {[f'{w:.2f}' for w in z_walls]}")
    
    # Step 5: HYBRID — use mask to classify Hough walls, build polygon from walls
    print("Step 5: Building hybrid polygon...")
    
    # Use the room mask contour to determine the room's bounding box
    contour_arr = np.array(raw_contour)
    mask_left = contour_arr[:, 0].min()
    mask_right = contour_arr[:, 0].max()
    mask_bottom = contour_arr[:, 1].min()
    mask_top = contour_arr[:, 1].max()
    print(f"  Mask bounds: X=[{mask_left:.2f}, {mask_right:.2f}], Z=[{mask_bottom:.2f}, {mask_top:.2f}]")
    
    # Find best matching Hough walls for each boundary
    def best_wall(positions, target, tolerance=0.3):
        """Snap target to nearest Hough wall, but only if close enough."""
        best = target
        best_dist = tolerance
        for p in positions:
            d = abs(p - target)
            if d < best_dist:
                best = p
                best_dist = d
        return best
    
    # For outer walls, prefer Hough but constrained by mask
    # Outer walls should be INSIDE the mask bounds (Hough finds wall centers)
    left_x = best_wall(x_walls, mask_left)
    right_x = best_wall(x_walls, mask_right)
    bottom_z = best_wall(z_walls, mask_bottom)
    
    # Sanity: if Hough wall is far from mask, use mask
    if abs(left_x - mask_left) > 0.5: left_x = mask_left
    if abs(right_x - mask_right) > 0.5: right_x = mask_right
    if abs(bottom_z - mask_bottom) > 0.5: bottom_z = mask_bottom
    
    # Main top: find the Z-wall closest to the mask's top on the LEFT side
    # (for L-shape, left side is shorter than right)
    # Sample mask at left-center to find where it ends vertically
    left_col = int((left_x - mask_x_min) / mask_cell) + 2
    left_col = max(0, min(left_col, room_mask.shape[1]-1))
    col_slice = room_mask[:, left_col]
    if np.any(col_slice):
        top_row_left = np.max(np.where(col_slice)) 
        main_top_from_mask = mask_z_min + top_row_left * mask_cell
    else:
        main_top_from_mask = mask_top
    
    main_top_z = best_wall(z_walls, main_top_from_mask, 0.3)
    
    # Extension top: if mask extends higher on the right
    right_col = int((right_x - mask_x_min) / mask_cell) - 2
    right_col = max(0, min(right_col, room_mask.shape[1]-1))
    col_slice_r = room_mask[:, right_col]
    if np.any(col_slice_r):
        top_row_right = np.max(np.where(col_slice_r))
        ext_top_from_mask = mask_z_min + top_row_right * mask_cell
    else:
        ext_top_from_mask = main_top_from_mask
    
    # For extension top, prefer Hough Z-wall if one exists above main_top
    ext_z_candidates = [z for z in z_walls if z > main_top_z + 0.15 and z < main_top_z + 2.0]
    if ext_z_candidates and ext_top_from_mask > main_top_from_mask + 0.2:
        ext_top_z = min(ext_z_candidates)  # Lowest Z-wall above main top
        print(f"  Extension top from Hough: Z={ext_top_z:.2f}")
    elif ext_top_from_mask > main_top_from_mask + 0.2:
        ext_top_z = ext_top_from_mask
    else:
        ext_top_z = None
    
    # Find step X from mask (where does the top edge step down?)
    if ext_top_z and ext_top_z > main_top_z + 0.1:
        # Scan columns to find where room height drops
        step_x = None
        top_row = int((main_top_z - mask_z_min) / mask_cell)
        for ci in range(room_mask.shape[1]-1, 0, -1):
            col_x = mask_x_min + ci * mask_cell
            if col_x > right_x + 0.1:
                continue
            if col_x < left_x - 0.1:
                break
            # Check if this column extends above main_top
            if top_row < room_mask.shape[0] and room_mask[min(top_row+2, room_mask.shape[0]-1), ci]:
                if step_x is None:
                    step_x = col_x  # rightmost column above main top
            else:
                if step_x is not None:
                    step_x = col_x  # leftmost empty column = step position
                    break
        
        # Snap step_x to nearest Hough X wall
        if step_x:
            step_x = best_wall(x_walls_raw, step_x, 0.3)
    else:
        step_x = None
    
    print(f"  Walls: left_x={left_x:.2f}, right_x={right_x:.2f}, bottom_z={bottom_z:.2f}")
    print(f"  main_top_z={main_top_z:.2f}, ext_top_z={ext_top_z}, step_x={step_x}")
    
    # Build polygon
    if ext_top_z and step_x and ext_top_z > main_top_z + 0.1:
        polygon_rot = [
            [left_x, bottom_z],
            [right_x, bottom_z],
            [right_x, ext_top_z],
            [step_x, ext_top_z],
            [step_x, main_top_z],
            [left_x, main_top_z],
        ]
        print(f"  L-shape: 6 vertices")
    else:
        top_z = main_top_z
        polygon_rot = [
            [left_x, bottom_z],
            [right_x, bottom_z],
            [right_x, top_z],
            [left_x, top_z],
        ]
        print(f"  Rectangle: 4 vertices")
    
    print(f"  Vertices: {[[f'{p[0]:.2f}', f'{p[1]:.2f}'] for p in polygon_rot]}")
    
    # Transform back
    room_corners = [rot_pt(p, angle_rad) for p in polygon_rot]
    exterior = room_corners + [room_corners[0]]
    
    n = len(room_corners)
    area = abs(sum(room_corners[i][0]*room_corners[(i+1)%n][1] - 
                   room_corners[(i+1)%n][0]*room_corners[i][1] for i in range(n))) / 2
    perimeter = sum(math.sqrt((room_corners[(i+1)%n][0]-room_corners[i][0])**2 + 
                              (room_corners[(i+1)%n][1]-room_corners[i][1])**2) 
                   for i in range(n))
    
    room = {'exterior': exterior, 'area': area, 'perimeter': perimeter}
    print(f"  Area: {area:.1f}m², Perimeter: {perimeter:.1f}m")
    
    # Build wall list from polygon edges
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
            'length': length,
            'nPoints': 0,
            'startPt': rot_pt(p1, angle_rad),
            'endPt': rot_pt(p2, angle_rad),
        })
    
    # Step 7: Detect openings
    print("Step 7: Detecting openings...")
    rotated_pts = list(zip(rx.tolist(), rz.tolist()))
    gaps = detect_openings(walls, rotated_pts, angle_rad)
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    print(f"  Openings: {len(doors)} doors, {len(windows)} windows")
    
    results = {
        'walls': walls, 'room': room, 'gaps': gaps,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'room_mask': room_mask, 'edges': edges,
        'x_walls': x_walls, 'z_walls': z_walls,
        'raw_contour': raw_contour, 'polygon_rot': polygon_rot,
    }
    
    print(f"\n=== v22 Hybrid Summary ===")
    print(f"Walls: {len(walls)}, Area: {area:.1f}m²")
    print(f"Doors: {len(doors)}, Windows: {len(windows)}")
    
    return results


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Left: Room mask
    ax0 = axes[0]
    if results.get('room_mask') is not None:
        ax0.imshow(results['room_mask'], cmap='gray', origin='lower')
    ax0.set_title('Room Mask (v21b)', color='white', fontsize=14)
    
    # Middle: Canny edges with Hough wall positions
    ax1 = axes[1]
    if results.get('edges') is not None:
        ax1.imshow(results['edges'], cmap='gray', origin='lower')
    ax1.set_title('Canny Edges + Hough Walls (v20b)', color='white', fontsize=14)
    
    # Right: Final floor plan
    ax = axes[2]
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    if results['room']:
        poly = results['room']['exterior']
        ax.fill([p[0] for p in poly], [p[1] for p in poly], color='gray', alpha=0.3)
        ax.plot([p[0] for p in poly], [p[1] for p in poly], color='gray', linewidth=1, alpha=0.7)
    
    for w in results['walls']:
        s, e = w['startPt'], w['endPt']
        ax.plot([s[0], e[0]], [s[1], e[1]], color='white', linewidth=4, solid_capstyle='round')
        mx, my = (s[0]+e[0])/2, (s[1]+e[1])/2
        
        dx = e[0] - s[0]
        dy = e[1] - s[1]
        ang = math.degrees(math.atan2(dy, dx))
        if ang > 90 or ang < -90:
            ang += 180
        
        ax.text(mx, my, f"{w['length']:.2f}m", ha='center', va='bottom',
                color='yellow', fontsize=10, fontweight='bold',
                rotation=ang, rotation_mode='anchor')
    
    for g in results['gaps']:
        c = 'cyan' if g['type'] == 'door' else 'lime'
        ax.plot([g['start'][0], g['end'][0]], [g['start'][1], g['end'][1]],
                color=c, linewidth=2, linestyle='--')
        
        radius = g['width'] / 4
        arc = patches.Arc((g['mid'][0], g['mid'][1]), radius*2, radius*2,
                          theta1=0, theta2=180, color=c, linewidth=2)
        ax.add_patch(arc)
        
        ax.text(g['mid'][0], g['mid'][1], f"{g['width']:.2f}m", ha='center', va='center',
                color=c, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    if results['room']:
        area = results['room']['area']
        ax.set_title(f"v22 Hybrid — {area:.1f}m²", fontsize=14, color='white')
        
        doors = len([g for g in results['gaps'] if g['type'] == 'door'])
        windows = len([g for g in results['gaps'] if g['type'] == 'window'])
        stats = f"Area: {area:.1f}m²\nWalls: {len(results['walls'])}\nDoors: {doors}\nWindows: {windows}"
        ax.text(0.98, 0.98, stats, transform=ax.transAxes, fontsize=11, color='white',
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    all_x = [w['startPt'][0] for w in results['walls']] + [w['endPt'][0] for w in results['walls']]
    all_y = [w['startPt'][1] for w in results['walls']] + [w['endPt'][1] for w in results['walls']]
    if all_x:
        m = 0.5
        ax.set_xlim(min(all_x)-m, max(all_x)+m)
        ax.set_ylim(min(all_y)-m, max(all_y)+m)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v22_hybrid_mask_hough',
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'area_m2': round(results['room']['area'], 1) if results['room'] else 0,
            'perimeter_m': round(results['room']['perimeter'], 1) if results['room'] else 0,
            'hough_x_walls': [round(w, 2) for w in results.get('x_walls', [])],
            'hough_z_walls': [round(w, 2) for w in results.get('z_walls', [])],
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
    parser = argparse.ArgumentParser(description='mesh2plan v22 - Hybrid')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v22_hybrid/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v22_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    
    viz_path = output_dir / f"{prefix}_floorplan.png"
    json_path = output_dir / f"{prefix}_results.json"
    
    visualize_results(results, viz_path)
    save_results_json(results, json_path)
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

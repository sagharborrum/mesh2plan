#!/usr/bin/env python3
"""
mesh2plan v27 - Multiroom Floor Plan Detection

Pipeline:
1. Project mid-height vertices to XZ plane (Y-up, height filter 15%-85%)
2. Find dominant angle, rotate to axis-align
3. Build high-res density image (1cm cells)
4. Multi-edge detection (Sobel + Laplacian + morph gradient, combine, NMS)
5. Extract Hough wall segments from NMS edges
6. Build room mask (threshold, morph close, fill holes, largest component)
7. Split rooms using interior walls
8. Per-room polygon extraction with wall snapping
9. Opening detection between adjacent rooms
10. 4-panel visualization
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


# ─── Utility functions (from v25) ───

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
    if img.max() > 0:
        img_norm = (img / np.percentile(img[img > 0], 95) * 255).clip(0, 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian = np.abs(cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(img_blur, cv2.MORPH_GRADIENT, kernel).astype(np.float64)
    def normalize(x):
        mx = x.max()
        return x / mx if mx > 0 else x
    combined = 0.5 * normalize(sobel_mag) + 0.25 * normalize(laplacian) + 0.25 * normalize(morph_grad)
    sobel_dir = np.arctan2(sobel_y, sobel_x)
    return combined, sobel_dir


def non_max_suppression(edge_mag, edge_dir):
    nz, nx = edge_mag.shape
    suppressed = np.zeros_like(edge_mag)
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
            continue
        avg_pos = sum(p[0]*p[1] for p in cluster) / total_len
        result.append(avg_pos)
    return sorted(result)


# ─── v27 New: Room mask + interior wall splitting ───

def build_room_mask(density_img):
    """Build binary room mask from density image (v21b-style)."""
    # Use >0 threshold to catch all occupied pixels
    mask = (density_img > 0).astype(np.uint8)
    # Very aggressive closing to bridge gaps and fill interior
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Dilate then erode to connect nearby blobs
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.dilate(mask, kernel2)
    mask = cv2.erode(mask, kernel2)
    # Fill holes
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    # Largest component
    labeled, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, labeled, range(1, n+1))
        largest = np.argmax(sizes) + 1
        filled = (labeled == largest).astype(np.uint8)
    return filled


def classify_walls(x_walls, z_walls, room_mask, x_min, z_min, cell_size):
    """Classify walls as boundary or interior based on room mask."""
    nz_img, nx_img = room_mask.shape
    
    interior_x = []
    boundary_x = []
    for xw in x_walls:
        col = int((xw - x_min) / cell_size)
        if col < 0 or col >= nx_img:
            boundary_x.append(xw)
            continue
        # Check how much of this column within mask is interior vs boundary
        col_mask = room_mask[:, max(0, col-2):min(nx_img, col+3)]
        if col_mask.size == 0:
            boundary_x.append(xw)
            continue
        col_any = np.any(col_mask, axis=1)
        occupied_rows = np.where(col_any)[0]
        if len(occupied_rows) < 10:
            boundary_x.append(xw)
            continue
        # Check if mask exists on BOTH sides of this wall (check 30cm away)
        offset = max(10, int(0.30 / cell_size))
        left_col = max(0, col - offset)
        right_col = min(nx_img - 1, col + offset)
        left_mask = room_mask[occupied_rows[0]:occupied_rows[-1]+1, left_col]
        right_mask = room_mask[occupied_rows[0]:occupied_rows[-1]+1, right_col]
        left_frac = np.mean(left_mask) if len(left_mask) > 0 else 0
        right_frac = np.mean(right_mask) if len(right_mask) > 0 else 0
        if left_frac > 0.3 and right_frac > 0.3:
            interior_x.append(xw)
        else:
            boundary_x.append(xw)
    
    interior_z = []
    boundary_z = []
    for zw in z_walls:
        row = int((zw - z_min) / cell_size)
        if row < 0 or row >= nz_img:
            boundary_z.append(zw)
            continue
        row_mask = room_mask[max(0, row-2):min(nz_img, row+3), :]
        if row_mask.size == 0:
            boundary_z.append(zw)
            continue
        row_any = np.any(row_mask, axis=0)
        occupied_cols = np.where(row_any)[0]
        if len(occupied_cols) < 10:
            boundary_z.append(zw)
            continue
        offset = max(10, int(0.30 / cell_size))
        top_row = min(nz_img - 1, row + offset)
        bot_row = max(0, row - offset)
        top_mask = room_mask[top_row, occupied_cols[0]:occupied_cols[-1]+1]
        bot_mask = room_mask[bot_row, occupied_cols[0]:occupied_cols[-1]+1]
        top_frac = np.mean(top_mask) if len(top_mask) > 0 else 0
        bot_frac = np.mean(bot_mask) if len(bot_mask) > 0 else 0
        if top_frac > 0.3 and bot_frac > 0.3:
            interior_z.append(zw)
        else:
            boundary_z.append(zw)
    
    return interior_x, interior_z, boundary_x, boundary_z


def split_rooms(room_mask, interior_x, interior_z, x_min, z_min, cell_size, wall_thickness=3):
    """Draw interior walls as separators and find connected components."""
    split_mask = room_mask.copy()
    nz_img, nx_img = split_mask.shape
    
    for xw in interior_x:
        col = int((xw - x_min) / cell_size)
        for dc in range(-wall_thickness, wall_thickness+1):
            c = col + dc
            if 0 <= c < nx_img:
                # Only draw where mask is 1
                split_mask[:, c] = split_mask[:, c] & 0  # Clear column through mask
                # Actually, set to 0 only where mask was 1
        # Redo: clear wall line only within the mask
    
    # Redo properly
    split_mask = room_mask.copy()
    for xw in interior_x:
        col = int((xw - x_min) / cell_size)
        for dc in range(-wall_thickness, wall_thickness+1):
            c = col + dc
            if 0 <= c < nx_img:
                split_mask[:, c] = 0
    
    for zw in interior_z:
        row = int((zw - z_min) / cell_size)
        for dr in range(-wall_thickness, wall_thickness+1):
            r = row + dr
            if 0 <= r < nz_img:
                split_mask[r, :] = 0
    
    # Only keep pixels that were in original mask
    split_mask = split_mask & room_mask
    
    labeled, n_rooms = ndimage.label(split_mask)
    
    # Filter tiny components (< 0.5 m² = 0.5 / cell_size² pixels)
    min_pixels = int(0.5 / (cell_size * cell_size))
    room_labels = []
    for i in range(1, n_rooms + 1):
        if np.sum(labeled == i) >= min_pixels:
            room_labels.append(i)
    
    print(f"  Found {len(room_labels)} rooms (from {n_rooms} components)")
    return labeled, room_labels


def extract_room_polygon(room_component, x_walls, z_walls, x_min, z_min, cell_size):
    """Extract a rectilinear polygon for a single room component."""
    # Find contour
    mask_u8 = (room_component > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    
    # Convert to world coords
    world_pts = []
    for pt in contour[:, 0]:
        wx = x_min + pt[0] * cell_size
        wz = z_min + pt[1] * cell_size
        world_pts.append([wx, wz])
    
    if len(world_pts) < 3:
        return world_pts
    
    # Douglas-Peucker simplification
    epsilon = 0.15  # 15cm tolerance
    pts_arr = np.array(world_pts, dtype=np.float32)
    simplified = cv2.approxPolyDP(pts_arr, epsilon, True)
    poly = simplified[:, 0].tolist()
    
    # Snap to nearest wall positions
    all_x = sorted(x_walls)
    all_z = sorted(z_walls)
    
    snapped = []
    for p in poly:
        sx = snap_to_nearest(p[0], all_x, 0.25)
        sz = snap_to_nearest(p[1], all_z, 0.25)
        snapped.append([sx, sz])
    
    # Axis-snap: force each segment to be horizontal or vertical
    snapped = axis_snap_polygon(snapped)
    
    # Remove duplicate consecutive points
    cleaned = [snapped[0]]
    for p in snapped[1:]:
        if abs(p[0] - cleaned[-1][0]) > 0.01 or abs(p[1] - cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    
    return cleaned


def snap_to_nearest(val, positions, tolerance=0.25):
    best = val
    best_d = tolerance
    for p in positions:
        d = abs(p - val)
        if d < best_d:
            best, best_d = p, d
    return best


def axis_snap_polygon(poly):
    """Force polygon segments to be axis-aligned."""
    if len(poly) < 3:
        return poly
    result = [poly[0]]
    for i in range(1, len(poly)):
        prev = result[-1]
        cur = poly[i]
        dx = abs(cur[0] - prev[0])
        dz = abs(cur[1] - prev[1])
        if dx < dz:
            # More vertical → snap x
            result.append([prev[0], cur[1]])
        else:
            # More horizontal → snap z
            result.append([cur[0], prev[1]])
    return result


def compute_polygon_area(poly):
    """Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2


def detect_openings_between_rooms(labeled, room_labels, x_walls, z_walls,
                                   interior_x, interior_z, x_min, z_min, cell_size):
    """Find door openings as gaps in interior walls between adjacent rooms."""
    openings = []
    
    # For each interior X wall
    for xw in interior_x:
        col = int((xw - x_min) / cell_size)
        if col < 5 or col >= labeled.shape[1] - 5:
            continue
        # Scan along this wall (rows) looking for gaps
        left_labels = labeled[:, max(0, col-8)]
        right_labels = labeled[:, min(labeled.shape[1]-1, col+8)]
        
        # Find runs where both sides have room labels
        in_gap = False
        gap_start = None
        for row in range(labeled.shape[0]):
            ll = left_labels[row]
            rl = right_labels[row]
            wall_pixel = labeled[row, col] if 0 <= col < labeled.shape[1] else 0
            
            if ll > 0 and rl > 0 and ll != rl and wall_pixel == 0:
                if not in_gap:
                    in_gap = True
                    gap_start = row
            else:
                if in_gap:
                    gap_end = row
                    gap_len = (gap_end - gap_start) * cell_size
                    if 0.4 < gap_len < 2.0:
                        mid_row = (gap_start + gap_end) / 2
                        mid_z = z_min + mid_row * cell_size
                        openings.append({
                            'type': 'door',
                            'width': round(gap_len, 2),
                            'position_rot': [xw, mid_z],
                            'axis': 'x',
                            'rooms': [int(left_labels[int(mid_row)]), int(right_labels[int(mid_row)])]
                        })
                    in_gap = False
        if in_gap:
            gap_end = labeled.shape[0]
            gap_len = (gap_end - gap_start) * cell_size
            if 0.4 < gap_len < 2.0:
                mid_row = (gap_start + gap_end) / 2
                mid_z = z_min + mid_row * cell_size
                openings.append({
                    'type': 'door', 'width': round(gap_len, 2),
                    'position_rot': [xw, mid_z], 'axis': 'x',
                    'rooms': [int(left_labels[int(mid_row)]), int(right_labels[int(mid_row)])]
                })
    
    # For each interior Z wall
    for zw in interior_z:
        row = int((zw - z_min) / cell_size)
        if row < 5 or row >= labeled.shape[0] - 5:
            continue
        top_labels = labeled[min(labeled.shape[0]-1, row+8), :]
        bot_labels = labeled[max(0, row-8), :]
        
        in_gap = False
        gap_start = None
        for col in range(labeled.shape[1]):
            tl = top_labels[col]
            bl = bot_labels[col]
            wall_pixel = labeled[row, col] if 0 <= row < labeled.shape[0] else 0
            
            if tl > 0 and bl > 0 and tl != bl and wall_pixel == 0:
                if not in_gap:
                    in_gap = True
                    gap_start = col
            else:
                if in_gap:
                    gap_end = col
                    gap_len = (gap_end - gap_start) * cell_size
                    if 0.4 < gap_len < 2.0:
                        mid_col = (gap_start + gap_end) / 2
                        mid_x = x_min + mid_col * cell_size
                        openings.append({
                            'type': 'door', 'width': round(gap_len, 2),
                            'position_rot': [mid_x, zw], 'axis': 'z',
                            'rooms': [int(bot_labels[int(mid_col)]), int(top_labels[int(mid_col)])]
                        })
                    in_gap = False
    
    return openings


# ─── Main analysis ───

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
    combined_edges, edge_dir = multi_edge_detection(density_img)
    print(f"  Edge stats: max={combined_edges.max():.3f}, mean={combined_edges.mean():.5f}")
    
    print("Step 4: Non-maximum suppression...")
    nms = non_max_suppression(combined_edges, edge_dir)
    print(f"  NMS: {np.sum(nms > 0)} non-zero pixels")
    
    print("Step 5: Extracting wall segments...")
    x_walls, z_walls = extract_wall_segments(nms, img_x_min, img_z_min, cell_size)
    print(f"  X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Z-walls: {[f'{w:.2f}' for w in z_walls]}")
    
    print("Step 6: Building room mask...")
    room_mask = build_room_mask(density_img)
    mask_area = np.sum(room_mask) * cell_size * cell_size
    print(f"  Mask area: {mask_area:.1f} m²")
    
    print("Step 7: Classifying walls...")
    interior_x, interior_z, boundary_x, boundary_z = classify_walls(
        x_walls, z_walls, room_mask, img_x_min, img_z_min, cell_size)
    print(f"  Interior X-walls: {[f'{w:.2f}' for w in interior_x]}")
    print(f"  Interior Z-walls: {[f'{w:.2f}' for w in interior_z]}")
    print(f"  Boundary X-walls: {[f'{w:.2f}' for w in boundary_x]}")
    print(f"  Boundary Z-walls: {[f'{w:.2f}' for w in boundary_z]}")
    
    print("Step 8: Splitting rooms...")
    labeled, room_labels = split_rooms(room_mask, interior_x, interior_z,
                                        img_x_min, img_z_min, cell_size)
    
    print("Step 9: Extracting room polygons...")
    rooms = []
    total_area = 0
    for idx, label in enumerate(room_labels):
        component = (labeled == label).astype(np.uint8)
        pixel_area = np.sum(component) * cell_size * cell_size
        poly = extract_room_polygon(component, x_walls, z_walls,
                                     img_x_min, img_z_min, cell_size)
        poly_area = compute_polygon_area(poly)
        # Use pixel area if polygon area is way off
        area = poly_area if poly_area > 0.5 else pixel_area
        rooms.append({
            'label': label,
            'polygon_rot': poly,
            'area': area,
            'pixel_area': pixel_area,
            'name': f"Room {idx+1}",
        })
        total_area += area
        print(f"  {rooms[-1]['name']}: {area:.1f} m² ({len(poly)} vertices)")
    
    print("Step 10: Detecting openings...")
    openings = detect_openings_between_rooms(
        labeled, room_labels, x_walls, z_walls,
        interior_x, interior_z, img_x_min, img_z_min, cell_size)
    print(f"  Found {len(openings)} openings")
    for o in openings:
        print(f"    {o['type']}: {o['width']:.2f}m between rooms {o['rooms']}")
    
    print(f"\n=== v27 Multiroom Summary ===")
    print(f"Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    print(f"Interior walls: {len(interior_x)} X + {len(interior_z)} Z")
    print(f"Openings: {len(openings)}")
    
    return {
        'rooms': rooms,
        'total_area': total_area,
        'x_walls': x_walls, 'z_walls': z_walls,
        'interior_x': interior_x, 'interior_z': interior_z,
        'boundary_x': boundary_x, 'boundary_z': boundary_z,
        'openings': openings,
        'angle': angle,
        'coordinate_system': f'{up_name}-up',
        'combined_edges': combined_edges,
        'nms': nms,
        'room_mask': room_mask,
        'labeled': labeled,
        'room_labels': room_labels,
        'img_origin': (img_x_min, img_z_min, cell_size),
    }


# ─── Visualization ───

ROOM_COLORS = [
    '#4A90D9', '#E8834A', '#67B868', '#C75B8F', '#8B6CC1',
    '#D4A843', '#4ABFBF', '#D96060', '#7B8FD4', '#A0C75B',
]


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(36, 9))
    
    img_origin = results['img_origin']
    ix_min, iz_min, cs = img_origin
    
    # Panel 1: Combined edge response
    ax0 = axes[0]
    ax0.imshow(results['combined_edges'], cmap='inferno', origin='lower')
    ax0.set_title('Combined Edge Response', color='white', fontsize=14)
    ax0.axis('off')
    
    # Panel 2: NMS edges with Hough walls
    ax1 = axes[1]
    nms = results['nms']
    nz_img, nx_img = nms.shape
    # Show NMS
    ax1.imshow(nms, cmap='hot', origin='lower', alpha=0.8)
    # Overlay Hough walls
    for xw in results['x_walls']:
        col = (xw - ix_min) / cs
        color = 'cyan' if xw in results['interior_x'] else 'lime'
        ax1.axvline(x=col, color=color, linewidth=1, alpha=0.7)
    for zw in results['z_walls']:
        row = (zw - iz_min) / cs
        color = 'cyan' if zw in results['interior_z'] else 'lime'
        ax1.axhline(y=row, color=color, linewidth=1, alpha=0.7)
    ax1.set_title('NMS + Hough Walls (cyan=interior, green=boundary)', color='white', fontsize=12)
    ax1.set_xlim(0, nx_img)
    ax1.set_ylim(0, nz_img)
    ax1.axis('off')
    
    # Panel 3: Room segmentation
    ax2 = axes[2]
    labeled = results['labeled']
    room_labels = results['room_labels']
    seg_img = np.zeros((*labeled.shape, 3), dtype=np.float32)
    for i, label in enumerate(room_labels):
        color_hex = ROOM_COLORS[i % len(ROOM_COLORS)]
        r = int(color_hex[1:3], 16) / 255
        g = int(color_hex[3:5], 16) / 255
        b = int(color_hex[5:7], 16) / 255
        mask = labeled == label
        seg_img[mask] = [r, g, b]
    # Interior walls in white
    wall_mask = (results['room_mask'] > 0) & (labeled == 0)
    seg_img[wall_mask] = [1, 1, 1]
    ax2.imshow(seg_img, origin='lower')
    # Label rooms
    for i, room in enumerate(results['rooms']):
        label = room['label']
        rows, cols = np.where(labeled == label)
        if len(rows) > 0:
            cy, cx = np.mean(rows), np.mean(cols)
            ax2.text(cx, cy, f"{room['name']}\n{room['area']:.1f}m²",
                    ha='center', va='center', color='white', fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ax2.set_title(f'Room Segmentation ({len(results["rooms"])} rooms)', color='white', fontsize=14)
    ax2.axis('off')
    
    # Panel 4: Floor plan
    ax3 = axes[3]
    ax3.set_aspect('equal')
    ax3.set_facecolor('#1a1a2e')
    
    all_pts_x = []
    all_pts_z = []
    
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3:
            continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        poly_closed = poly + [poly[0]]
        xs = [p[0] for p in poly_closed]
        zs = [p[1] for p in poly_closed]
        ax3.fill(xs, zs, color=color, alpha=0.3)
        all_pts_x.extend(xs)
        all_pts_z.extend(zs)
        
        # Draw walls
        for j in range(len(poly)):
            k = (j + 1) % len(poly)
            p1, p2 = poly[j], poly[k]
            length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if length < 0.1:
                continue
            ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', linewidth=3, solid_capstyle='round')
            # Dimension label
            mx, mz = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            if length > 0.5:
                dx = abs(p2[0]-p1[0])
                dz = abs(p2[1]-p1[1])
                if dx > dz:
                    ax3.text(mx, mz + 0.12, f"{length:.2f}m", ha='center', va='bottom',
                            color='yellow', fontsize=7, fontweight='bold')
                else:
                    ax3.text(mx + 0.12, mz, f"{length:.2f}m", ha='left', va='center',
                            color='yellow', fontsize=7, fontweight='bold', rotation=90)
        
        # Room label
        cx = sum(p[0] for p in poly) / len(poly)
        cz = sum(p[1] for p in poly) / len(poly)
        ax3.text(cx, cz, f"{room['name']}\n{room['area']:.1f}m²",
                ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))
    
    # Draw interior walls as thick lines
    angle_rad = results['angle'] * math.pi / 180
    for xw in results['interior_x']:
        # Find extent from room mask
        col = int((xw - ix_min) / cs)
        if 0 <= col < results['room_mask'].shape[1]:
            col_data = results['room_mask'][:, col]
            rows = np.where(col_data > 0)[0]
            if len(rows) > 0:
                z1 = iz_min + rows[0] * cs
                z2 = iz_min + rows[-1] * cs
                ax3.plot([xw, xw], [z1, z2], color='white', linewidth=5, solid_capstyle='round')
    
    for zw in results['interior_z']:
        row = int((zw - iz_min) / cs)
        if 0 <= row < results['room_mask'].shape[0]:
            row_data = results['room_mask'][row, :]
            cols = np.where(row_data > 0)[0]
            if len(cols) > 0:
                x1 = ix_min + cols[0] * cs
                x2 = ix_min + cols[-1] * cs
                ax3.plot([x1, x2], [zw, zw], color='white', linewidth=5, solid_capstyle='round')
    
    # Draw openings (doors)
    for opening in results['openings']:
        pos = opening['position_rot']
        w = opening['width']
        if opening['axis'] == 'x':
            z1 = pos[1] - w/2
            z2 = pos[1] + w/2
            ax3.plot([pos[0], pos[0]], [z1, z2], color='cyan', linewidth=3, linestyle='--')
            arc = patches.Arc((pos[0], pos[1]), w/2, w/2, theta1=0, theta2=180,
                            color='cyan', linewidth=2)
            ax3.add_patch(arc)
        else:
            x1 = pos[0] - w/2
            x2 = pos[0] + w/2
            ax3.plot([x1, x2], [pos[1], pos[1]], color='cyan', linewidth=3, linestyle='--')
            arc = patches.Arc((pos[0], pos[1]), w/2, w/2, theta1=90, theta2=270,
                            color='cyan', linewidth=2)
            ax3.add_patch(arc)
    
    total_area = results['total_area']
    ax3.set_title(f'v27 Multiroom Floor Plan — {len(results["rooms"])} rooms, {total_area:.1f}m² total',
                  color='white', fontsize=14)
    ax3.grid(True, alpha=0.2, color='gray')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Z (meters)')
    
    if all_pts_x:
        m = 0.5
        ax3.set_xlim(min(all_pts_x)-m, max(all_pts_x)+m)
        ax3.set_ylim(min(all_pts_z)-m, max(all_pts_z)+m)
    
    # Stats box
    stats = (f"Rooms: {len(results['rooms'])}\n"
             f"Total: {total_area:.1f}m²\n"
             f"Interior walls: {len(results['interior_x'])}X + {len(results['interior_z'])}Z\n"
             f"Openings: {len(results['openings'])}")
    ax3.text(0.98, 0.98, stats, transform=ax3.transAxes, fontsize=10, color='white',
             va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27_multiroom',
            'num_rooms': len(results['rooms']),
            'total_area_m2': round(results['total_area'], 1),
            'interior_walls_x': len(results['interior_x']),
            'interior_walls_z': len(results['interior_z']),
            'openings': len(results['openings']),
            'angle_deg': round(results['angle'], 1),
            'coordinate_system': results['coordinate_system'],
        },
        'rooms': [{
            'name': r['name'],
            'area_m2': round(r['area'], 1),
            'pixel_area_m2': round(r['pixel_area'], 1),
            'vertices': len(r['polygon_rot']),
            'polygon': [[round(p[0], 3), round(p[1], 3)] for p in r['polygon_rot']],
        } for r in results['rooms']],
        'walls': {
            'x_walls': [round(w, 3) for w in results['x_walls']],
            'z_walls': [round(w, 3) for w in results['z_walls']],
            'interior_x': [round(w, 3) for w in results['interior_x']],
            'interior_z': [round(w, 3) for w in results['interior_z']],
        },
        'openings': [{
            'type': o['type'],
            'width_m': o['width'],
            'position': o['position_rot'],
            'axis': o['axis'],
        } for o in results['openings']],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27 - Multiroom Floor Plan Detection')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27_multiroom/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v27_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    
    viz_path = output_dir / f"{prefix}_floorplan.png"
    json_path = output_dir / f"{prefix}_results.json"
    
    visualize_results(results, viz_path)
    save_results_json(results, json_path)
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

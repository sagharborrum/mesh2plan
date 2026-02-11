#!/usr/bin/env python3
"""
mesh2plan v27b - Wall Evidence Filtering + Room Merging

Improvements over v27:
- Wall evidence scoring (segment length, edge intensity, continuity)
- Minimum wall evidence threshold
- Wall span validation (reject walls <40% of room dimension)
- Minimum room size filter (merge <2m² into largest neighbor)
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


# ─── Base functions (from v27) ───

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
        return [], [], []
    x_positions = []
    z_positions = []
    all_segments = []  # Store raw segments for evidence scoring
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
        seg = {'wx1': wx1, 'wz1': wz1, 'wx2': wx2, 'wz2': wz2, 'length': length,
               'px1': x1, 'py1': y1, 'px2': x2, 'py2': y2}
        all_segments.append(seg)
        if dz > dx:
            x_positions.append(((wx1+wx2)/2, length))
        else:
            z_positions.append(((wz1+wz2)/2, length))
    x_walls = cluster_positions(x_positions, 0.15, min_total_length=0.8)
    z_walls = cluster_positions(z_positions, 0.15, min_total_length=3.0)
    return x_walls, z_walls, all_segments

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

def build_room_mask(density_img):
    mask = (density_img > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.dilate(mask, kernel2)
    mask = cv2.erode(mask, kernel2)
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, labeled, range(1, n+1))
        largest = np.argmax(sizes) + 1
        filled = (labeled == largest).astype(np.uint8)
    return filled


# ─── v27b: Wall evidence scoring ───

def score_wall_evidence(wall_pos, axis, all_segments, nms_edges, room_mask,
                        x_min, z_min, cell_size):
    """Score a wall by segment length, edge intensity, and continuity."""
    nz_img, nx_img = nms_edges.shape
    
    # 1. Total segment length along this wall
    total_seg_length = 0
    tolerance = 0.15  # 15cm
    for seg in all_segments:
        if axis == 'x':
            # X-wall: vertical segments where x ≈ wall_pos
            mid_x = (seg['wx1'] + seg['wx2']) / 2
            if abs(mid_x - wall_pos) < tolerance and abs(seg['wz2'] - seg['wz1']) > abs(seg['wx2'] - seg['wx1']):
                total_seg_length += seg['length']
        else:
            # Z-wall: horizontal segments where z ≈ wall_pos
            mid_z = (seg['wz1'] + seg['wz2']) / 2
            if abs(mid_z - wall_pos) < tolerance and abs(seg['wx2'] - seg['wx1']) > abs(seg['wz2'] - seg['wz1']):
                total_seg_length += seg['length']
    
    # 2. Edge intensity along wall line + 3. Continuity
    if axis == 'x':
        col = int((wall_pos - x_min) / cell_size)
        if col < 0 or col >= nx_img:
            return 0, 0, 0, 0
        # Find mask extent along this column
        col_mask = room_mask[:, max(0, col-2):min(nx_img, col+3)]
        col_any = np.any(col_mask, axis=1)
        rows = np.where(col_any)[0]
        if len(rows) < 10:
            return 0, 0, 0, 0
        span_start, span_end = rows[0], rows[-1]
        span_length = (span_end - span_start) * cell_size
        
        # Sample NMS edge values along wall
        edge_vals = []
        for r in range(span_start, span_end + 1):
            val = max(nms_edges[r, max(0, col-1):min(nx_img, col+2)].max(), 0)
            edge_vals.append(val)
    else:
        row = int((wall_pos - z_min) / cell_size)
        if row < 0 or row >= nz_img:
            return 0, 0, 0, 0
        row_mask = room_mask[max(0, row-2):min(nz_img, row+3), :]
        row_any = np.any(row_mask, axis=0)
        cols = np.where(row_any)[0]
        if len(cols) < 10:
            return 0, 0, 0, 0
        span_start, span_end = cols[0], cols[-1]
        span_length = (span_end - span_start) * cell_size
        
        edge_vals = []
        for c in range(span_start, span_end + 1):
            val = max(nms_edges[max(0, row-1):min(nz_img, row+2), c].max(), 0)
            edge_vals.append(val)
    
    if not edge_vals:
        return 0, 0, 0, 0
    
    edge_vals = np.array(edge_vals)
    avg_intensity = float(np.mean(edge_vals))
    
    # Continuity: fraction of span with edge evidence above threshold
    edge_thresh = 0.05
    continuity = float(np.mean(edge_vals > edge_thresh))
    
    return total_seg_length, avg_intensity, continuity, span_length


def filter_walls_by_evidence(x_walls, z_walls, all_segments, nms_edges, room_mask,
                              x_min, z_min, cell_size):
    """Filter interior walls by evidence scoring."""
    nz_img, nx_img = room_mask.shape
    room_width = 0
    room_height = 0
    
    # Compute room dimensions from mask
    rows_any = np.any(room_mask, axis=1)
    cols_any = np.any(room_mask, axis=0)
    if np.any(rows_any):
        r = np.where(rows_any)[0]
        room_height = (r[-1] - r[0]) * cell_size
    if np.any(cols_any):
        c = np.where(cols_any)[0]
        room_width = (c[-1] - c[0]) * cell_size
    
    print(f"  Room extents: {room_width:.1f}m x {room_height:.1f}m")
    
    # First classify as interior/boundary (same as v27)
    interior_x, interior_z, boundary_x, boundary_z = classify_walls_basic(
        x_walls, z_walls, room_mask, x_min, z_min, cell_size)
    
    # Now score and filter interior walls
    strong_x = []
    weak_x = []
    for xw in interior_x:
        seg_len, intensity, continuity, span = score_wall_evidence(
            xw, 'x', all_segments, nms_edges, room_mask, x_min, z_min, cell_size)
        
        # Wall span validation: must cover >40% of room height at that position
        span_frac = span / room_height if room_height > 0 else 0
        
        # Combined score
        score = seg_len * (0.3 + 0.7 * continuity)
        
        print(f"    X-wall {xw:.2f}: seg_len={seg_len:.2f}m, intensity={intensity:.3f}, "
              f"continuity={continuity:.1%}, span={span:.2f}m ({span_frac:.0%}), score={score:.2f}")
        
        # Thresholds
        if score >= 3.5 and continuity >= 0.20 and span_frac >= 0.40:
            strong_x.append(xw)
        else:
            weak_x.append(xw)
    
    strong_z = []
    weak_z = []
    for zw in interior_z:
        seg_len, intensity, continuity, span = score_wall_evidence(
            zw, 'z', all_segments, nms_edges, room_mask, x_min, z_min, cell_size)
        
        span_frac = span / room_width if room_width > 0 else 0
        score = seg_len * (0.3 + 0.7 * continuity)
        
        print(f"    Z-wall {zw:.2f}: seg_len={seg_len:.2f}m, intensity={intensity:.3f}, "
              f"continuity={continuity:.1%}, span={span:.2f}m ({span_frac:.0%}), score={score:.2f}")
        
        if score >= 3.5 and continuity >= 0.20 and span_frac >= 0.40:
            strong_z.append(zw)
        else:
            weak_z.append(zw)
    
    print(f"  Strong interior walls: {len(strong_x)} X + {len(strong_z)} Z")
    print(f"  Weak (rejected) walls: {len(weak_x)} X + {len(weak_z)} Z")
    
    return strong_x, strong_z, boundary_x, boundary_z


def classify_walls_basic(x_walls, z_walls, room_mask, x_min, z_min, cell_size):
    """Basic interior/boundary classification (from v27)."""
    nz_img, nx_img = room_mask.shape
    interior_x, boundary_x = [], []
    for xw in x_walls:
        col = int((xw - x_min) / cell_size)
        if col < 0 or col >= nx_img:
            boundary_x.append(xw); continue
        col_mask = room_mask[:, max(0, col-2):min(nx_img, col+3)]
        if col_mask.size == 0:
            boundary_x.append(xw); continue
        col_any = np.any(col_mask, axis=1)
        occupied_rows = np.where(col_any)[0]
        if len(occupied_rows) < 10:
            boundary_x.append(xw); continue
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
    
    interior_z, boundary_z = [], []
    for zw in z_walls:
        row = int((zw - z_min) / cell_size)
        if row < 0 or row >= nz_img:
            boundary_z.append(zw); continue
        row_mask = room_mask[max(0, row-2):min(nz_img, row+3), :]
        if row_mask.size == 0:
            boundary_z.append(zw); continue
        row_any = np.any(row_mask, axis=0)
        occupied_cols = np.where(row_any)[0]
        if len(occupied_cols) < 10:
            boundary_z.append(zw); continue
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
    split_mask = room_mask.copy()
    nz_img, nx_img = split_mask.shape
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
    split_mask = split_mask & room_mask
    labeled, n_rooms = ndimage.label(split_mask)
    min_pixels = int(0.5 / (cell_size * cell_size))
    room_labels = []
    for i in range(1, n_rooms + 1):
        if np.sum(labeled == i) >= min_pixels:
            room_labels.append(i)
    return labeled, room_labels


def merge_small_rooms(labeled, room_labels, cell_size, min_area=2.0):
    """Merge rooms smaller than min_area into their largest neighbor."""
    if len(room_labels) <= 1:
        return labeled, room_labels
    
    # Compute areas
    areas = {}
    for label in room_labels:
        areas[label] = np.sum(labeled == label) * cell_size * cell_size
    
    changed = True
    while changed:
        changed = False
        small = [l for l in room_labels if areas.get(l, 0) < min_area]
        if not small:
            break
        for sl in small:
            # Find neighbor with most shared boundary
            mask = labeled == sl
            dilated = cv2.dilate(mask.astype(np.uint8), np.ones((25,25), np.uint8))
            border = dilated.astype(bool) & ~mask
            neighbor_labels = labeled[border]
            neighbor_labels = neighbor_labels[neighbor_labels > 0]
            neighbor_labels = neighbor_labels[neighbor_labels != sl]
            if len(neighbor_labels) == 0:
                continue
            # Find most common neighbor
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            # Pick largest neighbor by area (not boundary count)
            best_neighbor = max(unique, key=lambda u: areas.get(u, 0))
            
            # Merge
            old_area = areas[sl]
            labeled[labeled == sl] = best_neighbor
            areas[best_neighbor] = areas.get(best_neighbor, 0) + old_area
            del areas[sl]
            room_labels.remove(sl)
            changed = True
            print(f"    Merged room (label {sl}, {old_area:.1f}m²) into label {best_neighbor}")
            break  # Restart loop
    
    return labeled, room_labels


# ─── Polygon extraction (from v27) ───

def extract_room_polygon(room_component, x_walls, z_walls, x_min, z_min, cell_size):
    mask_u8 = (room_component > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    world_pts = []
    for pt in contour[:, 0]:
        wx = x_min + pt[0] * cell_size
        wz = z_min + pt[1] * cell_size
        world_pts.append([wx, wz])
    if len(world_pts) < 3:
        return world_pts
    epsilon = 0.15
    pts_arr = np.array(world_pts, dtype=np.float32)
    simplified = cv2.approxPolyDP(pts_arr, epsilon, True)
    poly = simplified[:, 0].tolist()
    all_x = sorted(x_walls)
    all_z = sorted(z_walls)
    snapped = []
    for p in poly:
        sx = snap_to_nearest(p[0], all_x, 0.25)
        sz = snap_to_nearest(p[1], all_z, 0.25)
        snapped.append([sx, sz])
    snapped = axis_snap_polygon(snapped)
    cleaned = [snapped[0]]
    for p in snapped[1:]:
        if abs(p[0] - cleaned[-1][0]) > 0.01 or abs(p[1] - cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    return cleaned

def snap_to_nearest(val, positions, tolerance=0.25):
    best, best_d = val, tolerance
    for p in positions:
        d = abs(p - val)
        if d < best_d:
            best, best_d = p, d
    return best

def axis_snap_polygon(poly):
    if len(poly) < 3:
        return poly
    result = [poly[0]]
    for i in range(1, len(poly)):
        prev = result[-1]
        cur = poly[i]
        dx = abs(cur[0] - prev[0])
        dz = abs(cur[1] - prev[1])
        if dx < dz:
            result.append([prev[0], cur[1]])
        else:
            result.append([cur[0], prev[1]])
    return result

def compute_polygon_area(poly):
    n = len(poly)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2


# ─── Main analysis ───

ROOM_COLORS = [
    '#4A90D9', '#E8834A', '#67B868', '#C75B8F', '#8B6CC1',
    '#D4A843', '#4ABFBF', '#D96060', '#7B8FD4', '#A0C75B',
]

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
    
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    print("Step 2: Building density image...")
    density_img, img_x_min, img_z_min, cell_size = build_density_image(rx, rz, cell_size=0.01)
    
    print("Step 3: Multi-edge detection + NMS...")
    combined_edges, edge_dir = multi_edge_detection(density_img)
    nms = non_max_suppression(combined_edges, edge_dir)
    
    print("Step 4: Extracting wall segments...")
    x_walls, z_walls, all_segments = extract_wall_segments(nms, img_x_min, img_z_min, cell_size)
    print(f"  X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Z-walls: {[f'{w:.2f}' for w in z_walls]}")
    
    print("Step 5: Building room mask...")
    room_mask = build_room_mask(density_img)
    mask_area = np.sum(room_mask) * cell_size * cell_size
    print(f"  Mask area: {mask_area:.1f} m²")
    
    print("Step 6: Wall evidence filtering...")
    interior_x, interior_z, boundary_x, boundary_z = filter_walls_by_evidence(
        x_walls, z_walls, all_segments, nms, room_mask, img_x_min, img_z_min, cell_size)
    print(f"  Strong interior: X={[f'{w:.2f}' for w in interior_x]}, Z={[f'{w:.2f}' for w in interior_z]}")
    
    print("Step 7: Splitting rooms...")
    labeled, room_labels = split_rooms(room_mask, interior_x, interior_z,
                                        img_x_min, img_z_min, cell_size)
    print(f"  Initial: {len(room_labels)} rooms")
    
    print("Step 8: Merging small rooms (<2m²)...")
    labeled, room_labels = merge_small_rooms(labeled, room_labels, cell_size, min_area=3.0)
    print(f"  After merge: {len(room_labels)} rooms")
    
    print("Step 9: Extracting room polygons...")
    rooms = []
    total_area = 0
    for idx, label in enumerate(room_labels):
        component = (labeled == label).astype(np.uint8)
        pixel_area = np.sum(component) * cell_size * cell_size
        poly = extract_room_polygon(component, x_walls, z_walls,
                                     img_x_min, img_z_min, cell_size)
        poly_area = compute_polygon_area(poly)
        area = poly_area if poly_area > 0.5 else pixel_area
        rooms.append({
            'label': label, 'polygon_rot': poly, 'area': area,
            'pixel_area': pixel_area, 'name': f"Room {idx+1}",
        })
        total_area += area
        print(f"  {rooms[-1]['name']}: {area:.1f} m²")
    
    print(f"\n=== v27b Summary ===")
    print(f"Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    
    return {
        'rooms': rooms, 'total_area': total_area,
        'x_walls': x_walls, 'z_walls': z_walls,
        'interior_x': interior_x, 'interior_z': interior_z,
        'boundary_x': boundary_x, 'boundary_z': boundary_z,
        'openings': [], 'angle': angle, 'coordinate_system': f'{up_name}-up',
        'combined_edges': combined_edges, 'nms': nms,
        'room_mask': room_mask, 'labeled': labeled, 'room_labels': room_labels,
        'img_origin': (img_x_min, img_z_min, cell_size),
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(36, 9))
    ix_min, iz_min, cs = results['img_origin']
    
    # Panel 1: Combined edges
    axes[0].imshow(results['combined_edges'], cmap='inferno', origin='lower')
    axes[0].set_title('Combined Edge Response', color='white', fontsize=14)
    axes[0].axis('off')
    
    # Panel 2: NMS + walls
    ax1 = axes[1]
    nms = results['nms']
    nz_img, nx_img = nms.shape
    ax1.imshow(nms, cmap='hot', origin='lower', alpha=0.8)
    for xw in results['x_walls']:
        col = (xw - ix_min) / cs
        color = 'cyan' if xw in results['interior_x'] else ('yellow' if xw in results.get('boundary_x', []) else 'lime')
        lw = 2 if xw in results['interior_x'] else 1
        ax1.axvline(x=col, color=color, linewidth=lw, alpha=0.7)
    for zw in results['z_walls']:
        row = (zw - iz_min) / cs
        color = 'cyan' if zw in results['interior_z'] else ('yellow' if zw in results.get('boundary_z', []) else 'lime')
        lw = 2 if zw in results['interior_z'] else 1
        ax1.axhline(y=row, color=color, linewidth=lw, alpha=0.7)
    ax1.set_title('NMS + Walls (cyan=strong interior, yellow=boundary/weak)', color='white', fontsize=11)
    ax1.set_xlim(0, nx_img); ax1.set_ylim(0, nz_img); ax1.axis('off')
    
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
        seg_img[labeled == label] = [r, g, b]
    wall_mask = (results['room_mask'] > 0) & (labeled == 0)
    seg_img[wall_mask] = [1, 1, 1]
    ax2.imshow(seg_img, origin='lower')
    for room in results['rooms']:
        rows, cols = np.where(labeled == room['label'])
        if len(rows) > 0:
            cy, cx = np.mean(rows), np.mean(cols)
            ax2.text(cx, cy, f"{room['name']}\n{room['area']:.1f}m²",
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ax2.set_title(f'v27b Room Segmentation ({len(results["rooms"])} rooms)', color='white', fontsize=14)
    ax2.axis('off')
    
    # Panel 4: Floor plan
    ax3 = axes[3]
    ax3.set_aspect('equal'); ax3.set_facecolor('#1a1a2e')
    all_pts_x, all_pts_z = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3: continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        poly_closed = poly + [poly[0]]
        xs = [p[0] for p in poly_closed]
        zs = [p[1] for p in poly_closed]
        ax3.fill(xs, zs, color=color, alpha=0.3)
        all_pts_x.extend(xs); all_pts_z.extend(zs)
        for j in range(len(poly)):
            k = (j + 1) % len(poly)
            p1, p2 = poly[j], poly[k]
            ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', linewidth=3, solid_capstyle='round')
        cx = sum(p[0] for p in poly) / len(poly)
        cz = sum(p[1] for p in poly) / len(poly)
        ax3.text(cx, cz, f"{room['name']}\n{room['area']:.1f}m²",
                ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))
    
    total_area = results['total_area']
    ax3.set_title(f'v27b Floor Plan — {len(results["rooms"])} rooms, {total_area:.1f}m² total',
                  color='white', fontsize=14)
    ax3.grid(True, alpha=0.2, color='gray')
    if all_pts_x:
        m = 0.5
        ax3.set_xlim(min(all_pts_x)-m, max(all_pts_x)+m)
        ax3.set_ylim(min(all_pts_z)-m, max(all_pts_z)+m)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27b_wall_evidence',
            'num_rooms': len(results['rooms']),
            'total_area_m2': round(results['total_area'], 1),
            'interior_walls_x': len(results['interior_x']),
            'interior_walls_z': len(results['interior_z']),
        },
        'rooms': [{'name': r['name'], 'area_m2': round(r['area'], 1),
                    'polygon': [[round(p[0], 3), round(p[1], 3)] for p in r['polygon_rot']]}
                  for r in results['rooms']],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27b - Wall Evidence Filtering')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27b/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v27b_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

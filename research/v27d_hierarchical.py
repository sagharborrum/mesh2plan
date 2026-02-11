#!/usr/bin/env python3
"""
mesh2plan v27d - Hierarchical Room Splitting

Strategy: Start with full room mask as one room, then greedily apply the
highest-scoring wall split. Stop when no good splits remain or rooms are small enough.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage


# ─── Base functions (same as v27) ───

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
    x_positions, z_positions, all_segments = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        wx1, wz1 = x_min + x1 * cell_size, z_min + y1 * cell_size
        wx2, wz2 = x_min + x2 * cell_size, z_min + y2 * cell_size
        length = math.sqrt((wx2-wx1)**2 + (wz2-wz1)**2)
        if length < min_length: continue
        dx, dz = abs(wx2-wx1), abs(wz2-wz1)
        if dx + dz < 0.01: continue
        angle_mod = math.atan2(min(dx, dz), max(dx, dz)) * 180 / math.pi
        if angle_mod > 15: continue
        seg = {'wx1': wx1, 'wz1': wz1, 'wx2': wx2, 'wz2': wz2, 'length': length}
        all_segments.append(seg)
        if dz > dx:
            x_positions.append(((wx1+wx2)/2, length))
        else:
            z_positions.append(((wz1+wz2)/2, length))
    x_walls = cluster_positions(x_positions, 0.15, min_total_length=0.5)
    z_walls = cluster_positions(z_positions, 0.15, min_total_length=0.5)
    return x_walls, z_walls, all_segments

def cluster_positions(positions, dist_threshold=0.15, min_total_length=0.8):
    if not positions: return []
    sorted_pos = sorted(positions, key=lambda p: p[0])
    clusters, current = [], [sorted_pos[0]]
    for p in sorted_pos[1:]:
        if abs(p[0] - current[-1][0]) < dist_threshold:
            current.append(p)
        else:
            clusters.append(current); current = [p]
    clusters.append(current)
    result = []
    for cluster in clusters:
        total_len = sum(p[1] for p in cluster)
        if total_len < min_total_length: continue
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


# ─── v27d: Hierarchical splitting ───

def score_split(room_mask_region, split_pos, axis, nms_edges, x_min, z_min, cell_size,
                wall_thickness=3):
    """Score a potential wall split on a room region.
    
    Returns (score, area1, area2) or (0, 0, 0) if split is bad.
    """
    nz_img, nx_img = room_mask_region.shape
    
    # Apply the split
    split_mask = room_mask_region.copy()
    if axis == 'x':
        col = int((split_pos - x_min) / cell_size)
        if col < 5 or col >= nx_img - 5:
            return 0, 0, 0
        for dc in range(-wall_thickness, wall_thickness + 1):
            c = col + dc
            if 0 <= c < nx_img:
                split_mask[:, c] = 0
    else:
        row = int((split_pos - z_min) / cell_size)
        if row < 5 or row >= nz_img - 5:
            return 0, 0, 0
        for dr in range(-wall_thickness, wall_thickness + 1):
            r = row + dr
            if 0 <= r < nz_img:
                split_mask[r, :] = 0
    
    split_mask = split_mask & room_mask_region
    
    # Check if split actually divides the region
    labeled, n_parts = ndimage.label(split_mask)
    if n_parts < 2:
        return 0, 0, 0
    
    # Get the two largest parts
    sizes = []
    for i in range(1, n_parts + 1):
        sizes.append((np.sum(labeled == i) * cell_size * cell_size, i))
    sizes.sort(reverse=True)
    
    area1 = sizes[0][0]
    area2 = sizes[1][0] if len(sizes) > 1 else 0
    
    # Reject if either part is too small
    if area1 < 2.0 or area2 < 2.0:
        return 0, area1, area2
    
    # Check aspect ratios of resulting rooms
    for area_val, label_val in sizes[:2]:
        comp = (labeled == label_val)
        rows, cols = np.where(comp)
        if len(rows) == 0:
            continue
        h = (rows.max() - rows.min()) * cell_size
        w = (cols.max() - cols.min()) * cell_size
        if h < 0.01 or w < 0.01:
            return 0, area1, area2
        aspect = min(h, w) / max(h, w)
        if aspect < 0.15:  # Very elongated → bad split
            return 0, area1, area2
    
    # Edge evidence along the split line
    edge_evidence = 0
    if axis == 'x':
        col = int((split_pos - x_min) / cell_size)
        # Only count within room region
        region_rows = np.where(np.any(room_mask_region[:, max(0,col-5):min(nx_img,col+5)], axis=1))[0]
        if len(region_rows) > 0:
            edge_vals = []
            for r in range(region_rows[0], region_rows[-1] + 1):
                v = nms_edges[r, max(0,col-2):min(nx_img,col+3)].max()
                edge_vals.append(v)
            edge_vals = np.array(edge_vals)
            edge_evidence = float(np.mean(edge_vals > 0.05))  # Continuity fraction
    else:
        row = int((split_pos - z_min) / cell_size)
        region_cols = np.where(np.any(room_mask_region[max(0,row-5):min(nz_img,row+5), :], axis=0))[0]
        if len(region_cols) > 0:
            edge_vals = []
            for c in range(region_cols[0], region_cols[-1] + 1):
                v = nms_edges[max(0,row-2):min(nz_img,row+3), c].max()
                edge_vals.append(v)
            edge_vals = np.array(edge_vals)
            edge_evidence = float(np.mean(edge_vals > 0.05))
    
    # Score = balance * edge_evidence
    # Balance: prefer splits that create similarly-sized rooms
    total = area1 + area2
    balance = min(area1, area2) / total  # 0.5 = perfect split
    
    # Edge evidence is king — a wall with strong edges should always score high
    # Balance matters less (real rooms can be very different sizes)
    score = (0.1 + 0.9 * edge_evidence) * (0.5 + 0.5 * balance) * min(area2, 20)
    
    return score, area1, area2


def hierarchical_split(room_mask, x_walls, z_walls, nms_edges, x_min, z_min, cell_size,
                        min_room_area=3.0, min_split_score=1.0, max_rooms=10):
    """Hierarchically split rooms using wall candidates."""
    nz_img, nx_img = room_mask.shape
    
    # Start with whole mask as one room
    # Use a labeled array where each room has a unique label
    labeled = room_mask.astype(np.int32)
    next_label = 2  # Label 1 = initial room
    
    # Candidate walls: only interior ones (both sides have mask)
    candidate_x = []
    for xw in x_walls:
        col = int((xw - x_min) / cell_size)
        if col < 10 or col >= nx_img - 10:
            continue
        # Check both sides
        left = room_mask[:, max(0, col-30):col]
        right = room_mask[:, col:min(nx_img, col+30)]
        if np.sum(left) > 100 and np.sum(right) > 100:
            candidate_x.append(xw)
    
    candidate_z = []
    for zw in z_walls:
        row = int((zw - z_min) / cell_size)
        if row < 10 or row >= nz_img - 10:
            continue
        top = room_mask[row:min(nz_img, row+30), :]
        bot = room_mask[max(0, row-30):row, :]
        if np.sum(top) > 100 and np.sum(bot) > 100:
            candidate_z.append(zw)
    
    print(f"  Candidate walls: {len(candidate_x)} X + {len(candidate_z)} Z")
    
    # Greedy splitting
    iteration = 0
    while iteration < 20:
        iteration += 1
        
        # Find current rooms
        current_labels = np.unique(labeled)
        current_labels = current_labels[current_labels > 0]
        
        if len(current_labels) >= max_rooms:
            print(f"  Reached max rooms ({max_rooms})")
            break
        
        best_score = 0
        best_split = None
        
        for room_label in current_labels:
            room_region = (labeled == room_label).astype(np.uint8)
            room_area = np.sum(room_region) * cell_size * cell_size
            
            if room_area < min_room_area:
                continue  # Don't split small rooms
            
            # Try each candidate wall
            for xw in candidate_x:
                score, a1, a2 = score_split(room_region, xw, 'x', nms_edges,
                                             x_min, z_min, cell_size)
                if score > best_score:
                    best_score = score
                    best_split = (room_label, xw, 'x', a1, a2)
            
            for zw in candidate_z:
                score, a1, a2 = score_split(room_region, zw, 'z', nms_edges,
                                             x_min, z_min, cell_size)
                if score > best_score:
                    best_score = score
                    best_split = (room_label, zw, 'z', a1, a2)
        
        if best_score < min_split_score or best_split is None:
            print(f"  No more good splits (best score: {best_score:.2f})")
            break
        
        # Apply the best split
        room_label, wall_pos, axis, a1, a2 = best_split
        print(f"  Split {iteration}: room {room_label} at {axis}={wall_pos:.2f} "
              f"(score={best_score:.2f}, areas={a1:.1f}+{a2:.1f}m²)")
        
        room_region = (labeled == room_label).astype(np.uint8)
        split_mask = room_region.copy()
        
        if axis == 'x':
            col = int((wall_pos - x_min) / cell_size)
            for dc in range(-3, 4):
                c = col + dc
                if 0 <= c < nx_img:
                    split_mask[:, c] = 0
        else:
            row = int((wall_pos - z_min) / cell_size)
            for dr in range(-3, 4):
                r = row + dr
                if 0 <= r < nz_img:
                    split_mask[r, :] = 0
        
        split_mask = split_mask & room_region
        sub_labeled, n_parts = ndimage.label(split_mask)
        
        if n_parts < 2:
            # Remove this wall from candidates
            if axis == 'x' and wall_pos in candidate_x:
                candidate_x.remove(wall_pos)
            elif axis == 'z' and wall_pos in candidate_z:
                candidate_z.remove(wall_pos)
            continue
        
        # Relabel: keep largest as original label, assign new labels to others
        sizes = []
        for i in range(1, n_parts + 1):
            sizes.append((np.sum(sub_labeled == i), i))
        sizes.sort(reverse=True)
        
        # Clear old label
        labeled[labeled == room_label] = 0
        
        # Assign largest part to original label
        labeled[sub_labeled == sizes[0][1]] = room_label
        
        # Assign new labels to other parts
        for _, sub_label in sizes[1:]:
            labeled[sub_labeled == sub_label] = next_label
            next_label += 1
        
        # Remove used wall from candidates
        if axis == 'x' and wall_pos in candidate_x:
            candidate_x.remove(wall_pos)
        elif axis == 'z' and wall_pos in candidate_z:
            candidate_z.remove(wall_pos)
    
    # Get final room labels
    final_labels = np.unique(labeled)
    final_labels = final_labels[final_labels > 0]
    
    # Filter tiny rooms
    min_pixels = int(1.0 / (cell_size * cell_size))
    room_labels = [l for l in final_labels if np.sum(labeled == l) >= min_pixels]
    
    print(f"  Final: {len(room_labels)} rooms")
    return labeled, list(room_labels)


# ─── Polygon extraction ───

def extract_room_polygon(room_component, x_walls, z_walls, x_min, z_min, cell_size):
    """Extract a clean rectilinear polygon from a room mask component."""
    mask_u8 = (room_component > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    contour = max(contours, key=cv2.contourArea)
    
    # Convert to world coordinates
    world_pts = [[x_min + pt[0][0] * cell_size, z_min + pt[0][1] * cell_size] for pt in contour]
    if len(world_pts) < 3: return world_pts
    
    # Get bounding box from the component
    rows, cols = np.where(room_component > 0)
    if len(rows) == 0: return []
    bbox_left = x_min + cols.min() * cell_size
    bbox_right = x_min + cols.max() * cell_size
    bbox_bottom = z_min + rows.min() * cell_size
    bbox_top = z_min + rows.max() * cell_size
    
    # Snap bbox to nearest Hough walls
    all_x, all_z = sorted(x_walls), sorted(z_walls)
    left = snap_to_nearest(bbox_left, all_x, 0.3)
    right = snap_to_nearest(bbox_right, all_x, 0.3)
    bottom = snap_to_nearest(bbox_bottom, all_z, 0.3)
    top = snap_to_nearest(bbox_top, all_z, 0.3)
    
    # Check for L-shape or step: sample columns to see if room height varies
    # Sample at 25% and 75% of width
    nz_img, nx_img = room_component.shape
    col_25 = int((left + (right - left) * 0.25 - x_min) / cell_size)
    col_75 = int((left + (right - left) * 0.75 - x_min) / cell_size)
    col_25 = max(0, min(col_25, nx_img - 1))
    col_75 = max(0, min(col_75, nx_img - 1))
    
    def col_extent(c):
        col_data = room_component[:, max(0, c-2):min(nx_img, c+3)]
        occ = np.where(np.any(col_data > 0, axis=1))[0]
        if len(occ) == 0: return None, None
        return z_min + occ.min() * cell_size, z_min + occ.max() * cell_size
    
    bot25, top25 = col_extent(col_25)
    bot75, top75 = col_extent(col_75)
    
    # If heights differ significantly, try L-shape
    if top25 is not None and top75 is not None:
        top25_snap = snap_to_nearest(top25, all_z, 0.3)
        top75_snap = snap_to_nearest(top75, all_z, 0.3)
        
        if abs(top25_snap - top75_snap) > 0.4:
            # L-shape! Find step position by scanning columns
            main_top = min(top25_snap, top75_snap)
            ext_top = max(top25_snap, top75_snap)
            main_top_row = int((main_top - z_min) / cell_size)
            
            # Find step X
            step_x = None
            if top75_snap > top25_snap:
                # Extension on right
                for ci in range(col_75, col_25, -1):
                    check_row = min(main_top_row + 3, nz_img - 1)
                    if room_component[check_row, ci] > 0:
                        pass
                    else:
                        step_x = x_min + ci * cell_size
                        break
            else:
                for ci in range(col_25, col_75):
                    check_row = min(main_top_row + 3, nz_img - 1)
                    if room_component[check_row, ci] > 0:
                        pass
                    else:
                        step_x = x_min + ci * cell_size
                        break
            
            if step_x:
                step_x = snap_to_nearest(step_x, all_x, 0.3)
                if top75_snap > top25_snap:
                    return [[left, bottom], [right, bottom], [right, ext_top],
                            [step_x, ext_top], [step_x, main_top], [left, main_top]]
                else:
                    return [[left, bottom], [right, bottom], [right, main_top],
                            [step_x, main_top], [step_x, ext_top], [left, ext_top]]
    
    # Default: simple rectangle snapped to walls
    return [[left, bottom], [right, bottom], [right, top], [left, top]]

def snap_to_nearest(val, positions, tolerance=0.25):
    best, best_d = val, tolerance
    for p in positions:
        d = abs(p - val)
        if d < best_d: best, best_d = p, d
    return best

def axis_snap_polygon(poly):
    if len(poly) < 3: return poly
    result = [poly[0]]
    for i in range(1, len(poly)):
        prev, cur = result[-1], poly[i]
        if abs(cur[0] - prev[0]) < abs(cur[1] - prev[1]):
            result.append([prev[0], cur[1]])
        else:
            result.append([cur[0], prev[1]])
    return result

def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    area = sum(poly[i][0] * poly[(i+1)%n][1] - poly[(i+1)%n][0] * poly[i][1] for i in range(n))
    return abs(area) / 2


# ─── Main ───

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
    
    angle = find_dominant_angle(x_mid, z_mid)
    angle_rad = angle * math.pi / 180
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    print("Building density image...")
    density_img, img_x_min, img_z_min, cell_size = build_density_image(rx, rz, cell_size=0.01)
    
    print("Edge detection + NMS...")
    combined_edges, edge_dir = multi_edge_detection(density_img)
    nms = non_max_suppression(combined_edges, edge_dir)
    
    print("Extracting wall segments...")
    x_walls, z_walls, all_segments = extract_wall_segments(nms, img_x_min, img_z_min, cell_size)
    print(f"  X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Z-walls: {[f'{w:.2f}' for w in z_walls]}")
    
    print("Building room mask...")
    room_mask = build_room_mask(density_img)
    mask_area = np.sum(room_mask) * cell_size * cell_size
    print(f"  Mask area: {mask_area:.1f} m²")
    
    print("Hierarchical splitting...")
    labeled, room_labels = hierarchical_split(
        room_mask, x_walls, z_walls, nms, img_x_min, img_z_min, cell_size)
    
    print("Extracting room polygons...")
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
    
    print(f"\n=== v27d Summary ===")
    print(f"Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    
    return {
        'rooms': rooms, 'total_area': total_area,
        'x_walls': x_walls, 'z_walls': z_walls,
        'interior_x': [], 'interior_z': [],
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'combined_edges': combined_edges, 'nms': nms,
        'room_mask': room_mask, 'labeled': labeled, 'room_labels': room_labels,
        'img_origin': (img_x_min, img_z_min, cell_size),
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(45, 9))
    ix_min, iz_min, cs = results['img_origin']
    
    axes[0].imshow(results['combined_edges'], cmap='inferno', origin='lower')
    axes[0].set_title('Combined Edge Response', color='white', fontsize=14)
    axes[0].axis('off')
    
    ax1 = axes[1]
    nms = results['nms']
    ax1.imshow(nms, cmap='hot', origin='lower', alpha=0.8)
    for xw in results['x_walls']:
        ax1.axvline(x=(xw - ix_min) / cs, color='lime', linewidth=1, alpha=0.7)
    for zw in results['z_walls']:
        ax1.axhline(y=(zw - iz_min) / cs, color='lime', linewidth=1, alpha=0.7)
    ax1.set_title('NMS + Hough Walls', color='white', fontsize=14)
    ax1.axis('off')
    
    ax2 = axes[2]
    labeled = results['labeled']
    seg_img = np.zeros((*labeled.shape, 3), dtype=np.float32)
    for i, label in enumerate(results['room_labels']):
        ch = ROOM_COLORS[i % len(ROOM_COLORS)]
        r, g, b = int(ch[1:3], 16)/255, int(ch[3:5], 16)/255, int(ch[5:7], 16)/255
        seg_img[labeled == label] = [r, g, b]
    wall_mask = (results['room_mask'] > 0) & (labeled == 0)
    seg_img[wall_mask] = [1, 1, 1]
    ax2.imshow(seg_img, origin='lower')
    for room in results['rooms']:
        rows, cols = np.where(labeled == room['label'])
        if len(rows) > 0:
            ax2.text(np.mean(cols), np.mean(rows), f"{room['name']}\n{room['area']:.1f}m²",
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ax2.set_title(f'v27d Hierarchical ({len(results["rooms"])} rooms)', color='white', fontsize=14)
    ax2.axis('off')
    
    # Panel 4: Edge + Floor Plan Overlay
    ax_overlay = axes[3]
    ax_overlay.set_aspect('equal'); ax_overlay.set_facecolor('black')
    
    # Plot NMS edges as scatter in world coords
    nms_data = results['nms']
    edge_rows, edge_cols = np.where(nms_data > 0.05)
    if len(edge_rows) > 0:
        edge_x = ix_min + edge_cols * cs
        edge_z = iz_min + edge_rows * cs
        intensities = nms_data[edge_rows, edge_cols]
        ax_overlay.scatter(edge_x, edge_z, c=intensities, cmap='hot', s=0.3, alpha=0.5)
    
    # Overlay all room polygons
    overlay_xs, overlay_zs = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3: continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs, zs = [p[0] for p in pc], [p[1] for p in pc]
        ax_overlay.plot(xs, zs, color=color, linewidth=2.5, alpha=0.9)
        overlay_xs.extend(xs); overlay_zs.extend(zs)
    
    ax_overlay.set_title('Edge + Floor Plan Overlay', color='white', fontsize=14)
    ax_overlay.grid(True, alpha=0.2)
    ax_overlay.set_xlabel('X (meters)')
    ax_overlay.set_ylabel('Z (meters)')
    if overlay_xs:
        m = 0.5
        ax_overlay.set_xlim(min(overlay_xs)-m, max(overlay_xs)+m)
        ax_overlay.set_ylim(min(overlay_zs)-m, max(overlay_zs)+m)
    
    # Panel 5: Final floor plan
    ax3 = axes[4]
    ax3.set_aspect('equal'); ax3.set_facecolor('#1a1a2e')
    all_x, all_z = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3: continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs, zs = [p[0] for p in pc], [p[1] for p in pc]
        ax3.fill(xs, zs, color=color, alpha=0.3)
        all_x.extend(xs); all_z.extend(zs)
        for j in range(len(poly)):
            k = (j+1) % len(poly)
            ax3.plot([poly[j][0], poly[k][0]], [poly[j][1], poly[k][1]],
                    color='white', linewidth=3, solid_capstyle='round')
        cx = sum(p[0] for p in poly) / len(poly)
        cz = sum(p[1] for p in poly) / len(poly)
        ax3.text(cx, cz, f"{room['name']}\n{room['area']:.1f}m²",
                ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))
    ax3.set_title(f'v27d Floor Plan — {len(results["rooms"])} rooms, {results["total_area"]:.1f}m²',
                  color='white', fontsize=14)
    ax3.grid(True, alpha=0.2, color='gray')
    if all_x:
        m = 0.5
        ax3.set_xlim(min(all_x)-m, max(all_x)+m)
        ax3.set_ylim(min(all_z)-m, max(all_z)+m)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27d_hierarchical',
            'num_rooms': len(results['rooms']),
            'total_area_m2': round(results['total_area'], 1),
        },
        'rooms': [{'name': r['name'], 'area_m2': round(r['area'], 1),
                    'polygon': [[round(p[0], 3), round(p[1], 3)] for p in r['polygon_rot']]}
                  for r in results['rooms']],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27d - Hierarchical Splitting')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27d/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27d_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

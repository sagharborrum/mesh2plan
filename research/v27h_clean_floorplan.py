#!/usr/bin/env python3
"""
mesh2plan v27h - Clean Architectural Floor Plan

Strategy:
1. v27g pipeline for room detection (density → distance transform → seeds → watershed → hallways)
2. For polygon extraction: snap basin bounding boxes to Hough wall positions
3. Build clean rectilinear polygons per room (4-6 vertices)
4. Detect doors (gaps between adjacent rooms) and windows (gaps on exterior walls)
5. Produce two outputs:
   - 5-panel research view (debug)
   - Clean architectural floor plan (white bg, thick walls, door arcs, window marks)
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyArrowPatch
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndlabel


# ─── Shared utilities (from v27g) ───

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
    xi = np.clip(((rx - x_min) / cell).astype(int), 0, nx - 1)
    zi = np.clip(((rz - z_min) / cell).astype(int), 0, nz - 1)
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
    return (bins[peak] + bins[peak + 1]) / 2


def build_density_image(rx, rz, cell_size=0.02, margin=0.3):
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx - 1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz - 1)
    np.add.at(img, (zi, xi), 1)
    return img, x_min, z_min, cell_size


def hough_wall_positions(density_img, x_min, z_min, cell_size, nms_dist=0.15):
    """Detect wall positions from density projection histograms.
    Uses smaller NMS distance to find more walls including interior walls."""
    nz_img, nx_img = density_img.shape
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)
    proj_x = smoothed.sum(axis=0)
    proj_z = smoothed.sum(axis=1)

    def find_peaks_nms(profile, origin, cs, min_dist_cells):
        from scipy.ndimage import uniform_filter1d
        prof = uniform_filter1d(profile.astype(float), size=5)
        local_max = maximum_filter(prof, size=max(3, min_dist_cells)) == prof
        # Lower threshold to catch interior walls
        threshold = prof.mean() + 0.15 * prof.std()
        peaks = np.where(local_max & (prof > threshold))[0]
        positions = origin + peaks * cs
        if len(positions) == 0:
            return np.array([])
        strengths = prof[peaks]
        order = np.argsort(-strengths)
        kept = []
        for i in order:
            pos = positions[i]
            if any(abs(pos - k) < nms_dist for k in kept):
                continue
            kept.append(pos)
        return np.sort(kept)

    min_dist = int(nms_dist / cell_size)
    x_walls = find_peaks_nms(proj_x, x_min, cell_size, min_dist)
    z_walls = find_peaks_nms(proj_z, z_min, cell_size, min_dist)
    return x_walls, z_walls


def build_room_mask(density_img, cell_size):
    occupied = (density_img > 1).astype(np.uint8)
    k_size = max(3, int(0.15 / cell_size)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    lbl, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, lbl, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        filled = (lbl == largest).astype(np.uint8)
    return filled


def find_room_seeds(dist_transform, cell_size, min_dist_m=0.5, min_sep_m=1.0):
    min_dist_px = int(min_dist_m / cell_size)
    min_sep_px = max(3, int(min_sep_m / cell_size))
    local_max = maximum_filter(dist_transform, size=min_sep_px)
    is_max = (dist_transform == local_max) & (dist_transform >= min_dist_px)
    lbl, n = ndimage.label(is_max)
    seeds = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        cy, cx = int(ys.mean()), int(xs.mean())
        val = dist_transform[cy, cx]
        seeds.append((cy, cx, val))
    seeds.sort(key=lambda s: -s[2])
    print(f"  Found {len(seeds)} seed candidates")
    for i, (sy, sx, sv) in enumerate(seeds):
        print(f"    Seed {i}: pixel ({sx},{sy}), dist={sv * cell_size:.2f}m")
    return seeds


def watershed_from_seeds(dist_transform, room_mask, seeds, cell_size):
    nz, nx = dist_transform.shape
    markers = np.zeros((nz, nx), dtype=np.int32)
    seed_radius = max(2, int(0.1 / cell_size))
    for i, (sy, sx, _) in enumerate(seeds):
        cv2.circle(markers, (sx, sy), seed_radius, int(i + 1), -1)
    bg_label = len(seeds) + 1
    markers[room_mask == 0] = bg_label
    if dist_transform.max() > 0:
        inv_dist = dist_transform.max() - dist_transform
        inv_norm = (inv_dist / inv_dist.max() * 255).clip(0, 255).astype(np.uint8)
    else:
        inv_norm = np.zeros_like(dist_transform, dtype=np.uint8)
    img_color = cv2.cvtColor(inv_norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)
    result = markers.copy()
    result[result == -1] = 0
    result[result == bg_label] = 0
    return result


def split_hallways_from_rooms(labeled, dist_transform, room_mask, seeds, cell_size,
                               hallway_max_half_width_m=0.65):
    from skimage.morphology import skeletonize
    n_seeds = len(seeds)
    thresh_px = hallway_max_half_width_m / cell_size
    skel = skeletonize(room_mask > 0).astype(np.uint8)
    narrow_skel = (skel > 0) & (dist_transform < thresh_px) & (dist_transform > 0)
    narrow_skel = narrow_skel.astype(np.uint8)
    k_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    narrow_connected = cv2.dilate(narrow_skel, k_connect)
    narrow_connected = cv2.morphologyEx(narrow_connected, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    grow_dist = cv2.distanceTransform(1 - narrow_connected, cv2.DIST_L2, 5)
    hallway_mask = ((grow_dist < dist_transform) & (room_mask > 0) &
                    (dist_transform < thresh_px)).astype(np.uint8)
    ridge = ((labeled == 0) & (room_mask > 0)).astype(np.uint8)
    ridge_grown = cv2.dilate(ridge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    hallway_mask = hallway_mask | (ridge_grown & (dist_transform < thresh_px).astype(np.uint8) & room_mask)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hallway_mask = cv2.morphologyEx(hallway_mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    hallway_mask = cv2.morphologyEx(hallway_mask, cv2.MORPH_CLOSE, k_close)

    corr_lbl, n_corr = ndimage.label(hallway_mask)
    new_labeled = labeled.copy()
    next_label = n_seeds + 1
    hallway_info = []
    for i in range(1, n_corr + 1):
        region = (corr_lbl == i)
        area_m2 = np.sum(region) * cell_size * cell_size
        if area_m2 < 1.0:
            continue
        touching = set()
        dilated = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8))
        for lbl in np.unique(labeled[dilated > 0]):
            if 0 < lbl <= n_seeds:
                touching.add(int(lbl))
        region_dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        max_width = region_dist.max() * 2 * cell_size
        if len(touching) >= 2 and area_m2 >= 1.5:
            new_labeled[region] = next_label
            hallway_info.append({
                'label': next_label, 'area_m2': area_m2,
                'max_width': max_width, 'connects': touching,
            })
            next_label += 1
    return new_labeled, hallway_info


# ─── v27h: Clean polygon extraction ───

def snap_to_nearest(val, positions, max_snap=0.4):
    """Snap a value to the nearest position in the list."""
    if len(positions) == 0:
        return val
    dists = np.abs(np.array(positions) - val)
    idx = np.argmin(dists)
    if dists[idx] < max_snap:
        return float(positions[idx])
    return val


def extract_room_bbox(labeled, label, img_x_min, img_z_min, cell_size):
    """Get bounding box of a labeled region in world coordinates."""
    mask = (labeled == label)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    x_min_w = img_x_min + cols.min() * cell_size
    x_max_w = img_x_min + cols.max() * cell_size
    z_min_w = img_z_min + rows.min() * cell_size
    z_max_w = img_z_min + rows.max() * cell_size
    return x_min_w, x_max_w, z_min_w, z_max_w


def extract_basin_polygon(labeled, label, img_x_min, img_z_min, cell_size, x_walls, z_walls):
    """Extract a clean rectilinear polygon from a basin mask.
    
    Strategy: Use minimum area bounding rectangle, then snap to walls.
    For L-shapes, detect the step and produce 6-vertex polygon.
    Uses contour percentile clipping for robustness.
    """
    mask = (labeled == label).astype(np.uint8)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    
    # Use percentile-based extent (ignore outlier pixels)
    c_lo = np.percentile(cols, 2)
    c_hi = np.percentile(cols, 98)
    r_lo = np.percentile(rows, 2)
    r_hi = np.percentile(rows, 98)
    
    wx_min = img_x_min + c_lo * cell_size
    wx_max = img_x_min + c_hi * cell_size
    wz_min = img_z_min + r_lo * cell_size
    wz_max = img_z_min + r_hi * cell_size
    
    # Check fill ratio for L-shape detection
    r_min_i, r_max_i = int(r_lo), int(r_hi)
    c_min_i, c_max_i = int(c_lo), int(c_hi)
    bbox_area = max(1, (r_max_i - r_min_i + 1) * (c_max_i - c_min_i + 1))
    # Count pixels within trimmed bbox
    trimmed = mask[r_min_i:r_max_i+1, c_min_i:c_max_i+1]
    actual_area = np.sum(trimmed)
    fill_ratio = actual_area / bbox_area
    
    # Snap to nearest wall with moderate tolerance
    snap = 0.35
    x0 = snap_to_nearest(wx_min, x_walls, snap)
    x1 = snap_to_nearest(wx_max, x_walls, snap)
    z0 = snap_to_nearest(wz_min, z_walls, snap)
    z1 = snap_to_nearest(wz_max, z_walls, snap)
    
    if fill_ratio > 0.80:
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    # L-shaped: scan rows to find the step
    best_split_r = None
    best_score = 0
    step = max(1, (r_max_i - r_min_i) // 40)
    prev_w = None
    
    for r in range(r_min_i + 3, r_max_i - 3, step):
        rc = np.where(mask[r, c_min_i:c_max_i+1] > 0)[0]
        if len(rc) < 2:
            continue
        w = rc[-1] - rc[0]
        if prev_w is not None:
            d = abs(w - prev_w)
            if d > best_score and d > (c_max_i - c_min_i) * 0.15:
                best_score = d
                best_split_r = r
        prev_w = w
    
    if best_split_r is not None:
        # Get extents of top and bottom halves
        top_cols = np.where(mask[r_min_i:best_split_r, :].any(axis=0))[0]
        bot_cols = np.where(mask[best_split_r:r_max_i+1, :].any(axis=0))[0]
        
        if len(top_cols) > 1 and len(bot_cols) > 1:
            split_z = snap_to_nearest(img_z_min + best_split_r * cell_size, z_walls, snap)
            
            tx0 = snap_to_nearest(img_x_min + np.percentile(top_cols, 2) * cell_size, x_walls, snap)
            tx1 = snap_to_nearest(img_x_min + np.percentile(top_cols, 98) * cell_size, x_walls, snap)
            bx0 = snap_to_nearest(img_x_min + np.percentile(bot_cols, 2) * cell_size, x_walls, snap)
            bx1 = snap_to_nearest(img_x_min + np.percentile(bot_cols, 98) * cell_size, x_walls, snap)
            
            # Build L-shape: figure out which side has the notch
            left_same = abs(tx0 - bx0) < 0.2
            right_same = abs(tx1 - bx1) < 0.2
            
            if left_same:
                # Notch on right side
                x_left = min(tx0, bx0)
                if tx1 > bx1:
                    poly = [[x_left, z0], [tx1, z0], [tx1, split_z],
                            [bx1, split_z], [bx1, z1], [x_left, z1]]
                else:
                    poly = [[x_left, z0], [bx1, z0], [bx1, split_z],
                            [tx1, split_z], [tx1, z1], [x_left, z1]]
            elif right_same:
                # Notch on left side
                x_right = max(tx1, bx1)
                if tx0 < bx0:
                    poly = [[tx0, z0], [x_right, z0], [x_right, z1],
                            [bx0, z1], [bx0, split_z], [tx0, split_z]]
                else:
                    poly = [[bx0, z0], [x_right, z0], [x_right, z1],
                            [tx0, z1], [tx0, split_z], [bx0, split_z]]
            else:
                # Both sides differ - use larger bbox
                poly = [[min(tx0,bx0), z0], [max(tx1,bx1), z0],
                        [max(tx1,bx1), z1], [min(tx0,bx0), z1]]
            
            # Remove degenerate edges
            cleaned = [poly[0]]
            for p in poly[1:]:
                if abs(p[0] - cleaned[-1][0]) + abs(p[1] - cleaned[-1][1]) > 0.05:
                    cleaned.append(p)
            if len(cleaned) >= 3:
                return cleaned
    
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3:
        return 0
    area = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    return abs(area) / 2


def polygon_centroid(poly):
    n = len(poly)
    if n == 0:
        return 0, 0
    cx = sum(p[0] for p in poly) / n
    cz = sum(p[1] for p in poly) / n
    return cx, cz


# ─── Door and Window detection ───

def find_shared_wall_segments(poly1, poly2, tolerance=0.15):
    """Find wall segments shared between two polygons."""
    segments = []
    n1, n2 = len(poly1), len(poly2)
    
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i+1) % n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j+1) % n2]
            
            # Check if edges are collinear and overlapping
            # Vertical edges
            if abs(a1[0] - a2[0]) < 0.05 and abs(b1[0] - b2[0]) < 0.05:
                if abs(a1[0] - b1[0]) < tolerance:
                    # Same X, check Z overlap
                    a_lo, a_hi = min(a1[1], a2[1]), max(a1[1], a2[1])
                    b_lo, b_hi = min(b1[1], b2[1]), max(b1[1], b2[1])
                    overlap_lo = max(a_lo, b_lo)
                    overlap_hi = min(a_hi, b_hi)
                    if overlap_hi - overlap_lo > 0.2:
                        x = (a1[0] + b1[0]) / 2
                        segments.append({
                            'type': 'vertical',
                            'x': x,
                            'z_min': overlap_lo,
                            'z_max': overlap_hi,
                        })
            
            # Horizontal edges
            if abs(a1[1] - a2[1]) < 0.05 and abs(b1[1] - b2[1]) < 0.05:
                if abs(a1[1] - b1[1]) < tolerance:
                    a_lo, a_hi = min(a1[0], a2[0]), max(a1[0], a2[0])
                    b_lo, b_hi = min(b1[0], b2[0]), max(b1[0], b2[0])
                    overlap_lo = max(a_lo, b_lo)
                    overlap_hi = min(a_hi, b_hi)
                    if overlap_hi - overlap_lo > 0.2:
                        z = (a1[1] + b1[1]) / 2
                        segments.append({
                            'type': 'horizontal',
                            'z': z,
                            'x_min': overlap_lo,
                            'x_max': overlap_hi,
                        })
    
    return segments


def detect_doors_from_density(density_img, x_min, z_min, cell_size, wall_segments):
    """Detect door positions by finding gaps in walls (low density on shared walls)."""
    doors = []
    
    for seg in wall_segments:
        if seg['type'] == 'vertical':
            x_px = int((seg['x'] - x_min) / cell_size)
            z_lo_px = int((seg['z_min'] - z_min) / cell_size)
            z_hi_px = int((seg['z_max'] - z_min) / cell_size)
            
            if x_px < 0 or x_px >= density_img.shape[1]:
                continue
            
            # Sample density along this wall
            strip_w = max(1, int(0.1 / cell_size))
            x_lo = max(0, x_px - strip_w)
            x_hi = min(density_img.shape[1], x_px + strip_w + 1)
            
            wall_profile = density_img[z_lo_px:z_hi_px, x_lo:x_hi].sum(axis=1)
            if len(wall_profile) == 0:
                continue
            
            # Find gaps (low density regions)
            threshold = np.percentile(wall_profile[wall_profile > 0], 30) if np.any(wall_profile > 0) else 0
            is_gap = wall_profile < max(threshold, 1)
            
            # Find connected gap regions
            gap_lbl, n_gaps = ndimage.label(is_gap)
            for g in range(1, n_gaps + 1):
                gap_rows = np.where(gap_lbl == g)[0]
                gap_len = len(gap_rows) * cell_size
                if 0.6 < gap_len < 1.5:  # Door-sized gap
                    gap_center_z = z_min + (z_lo_px + gap_rows.mean()) * cell_size
                    doors.append({
                        'x': seg['x'], 'z': gap_center_z,
                        'width': gap_len,
                        'orientation': 'vertical',  # wall is vertical, door opens horizontally
                    })
        
        elif seg['type'] == 'horizontal':
            z_px = int((seg['z'] - z_min) / cell_size)
            x_lo_px = int((seg['x_min'] - x_min) / cell_size)
            x_hi_px = int((seg['x_max'] - x_min) / cell_size)
            
            if z_px < 0 or z_px >= density_img.shape[0]:
                continue
            
            strip_w = max(1, int(0.1 / cell_size))
            z_lo = max(0, z_px - strip_w)
            z_hi = min(density_img.shape[0], z_px + strip_w + 1)
            
            wall_profile = density_img[z_lo:z_hi, x_lo_px:x_hi_px].sum(axis=0)
            if len(wall_profile) == 0:
                continue
            
            threshold = np.percentile(wall_profile[wall_profile > 0], 30) if np.any(wall_profile > 0) else 0
            is_gap = wall_profile < max(threshold, 1)
            
            gap_lbl, n_gaps = ndimage.label(is_gap)
            for g in range(1, n_gaps + 1):
                gap_cols = np.where(gap_lbl == g)[0]
                gap_len = len(gap_cols) * cell_size
                if 0.6 < gap_len < 1.5:
                    gap_center_x = x_min + (x_lo_px + gap_cols.mean()) * cell_size
                    doors.append({
                        'x': gap_center_x, 'z': seg['z'],
                        'width': gap_len,
                        'orientation': 'horizontal',
                    })
    
    return doors


def detect_windows_on_exterior(density_img, x_min, z_min, cell_size, rooms, exterior_bbox):
    """Detect windows as gaps on exterior walls."""
    windows = []
    ext_x_min, ext_x_max, ext_z_min, ext_z_max = exterior_bbox
    
    for room in rooms:
        poly = room['polygon']
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i+1) % n]
            
            # Check if this edge is on exterior
            is_exterior = False
            edge_type = None
            
            if abs(p1[0] - p2[0]) < 0.1:  # Vertical edge
                if abs(p1[0] - ext_x_min) < 0.2 or abs(p1[0] - ext_x_max) < 0.2:
                    is_exterior = True
                    edge_type = 'vertical'
            elif abs(p1[1] - p2[1]) < 0.1:  # Horizontal edge
                if abs(p1[1] - ext_z_min) < 0.2 or abs(p1[1] - ext_z_max) < 0.2:
                    is_exterior = True
                    edge_type = 'horizontal'
            
            if not is_exterior:
                continue
            
            # Sample density along this edge and find gaps
            if edge_type == 'vertical':
                x_px = int((p1[0] - x_min) / cell_size)
                z_lo = min(p1[1], p2[1])
                z_hi = max(p1[1], p2[1])
                z_lo_px = int((z_lo - z_min) / cell_size)
                z_hi_px = int((z_hi - z_min) / cell_size)
                
                strip_w = max(1, int(0.08 / cell_size))
                x_lo_s = max(0, x_px - strip_w)
                x_hi_s = min(density_img.shape[1], x_px + strip_w + 1)
                
                if z_hi_px <= z_lo_px or x_hi_s <= x_lo_s:
                    continue
                wall_profile = density_img[z_lo_px:z_hi_px, x_lo_s:x_hi_s].sum(axis=1)
                if len(wall_profile) == 0:
                    continue
                
                threshold = np.percentile(wall_profile[wall_profile > 0], 40) if np.any(wall_profile > 0) else 0
                is_gap = wall_profile < max(threshold, 1)
                gap_lbl, n_gaps = ndimage.label(is_gap)
                for g in range(1, n_gaps + 1):
                    gap_rows = np.where(gap_lbl == g)[0]
                    gap_len = len(gap_rows) * cell_size
                    if 0.5 < gap_len < 2.5:
                        center_z = z_min + (z_lo_px + gap_rows.mean()) * cell_size
                        windows.append({
                            'x': p1[0],
                            'z': center_z,
                            'length': gap_len,
                            'orientation': 'vertical',
                            'room': room['name'],
                        })
            
            elif edge_type == 'horizontal':
                z_px = int((p1[1] - z_min) / cell_size)
                x_lo = min(p1[0], p2[0])
                x_hi = max(p1[0], p2[0])
                x_lo_px = int((x_lo - x_min) / cell_size)
                x_hi_px = int((x_hi - x_min) / cell_size)
                
                strip_w = max(1, int(0.08 / cell_size))
                z_lo_s = max(0, z_px - strip_w)
                z_hi_s = min(density_img.shape[0], z_px + strip_w + 1)
                
                if x_hi_px <= x_lo_px or z_hi_s <= z_lo_s:
                    continue
                wall_profile = density_img[z_lo_s:z_hi_s, x_lo_px:x_hi_px].sum(axis=0)
                if len(wall_profile) == 0:
                    continue
                
                threshold = np.percentile(wall_profile[wall_profile > 0], 40) if np.any(wall_profile > 0) else 0
                is_gap = wall_profile < max(threshold, 1)
                gap_lbl, n_gaps = ndimage.label(is_gap)
                for g in range(1, n_gaps + 1):
                    gap_cols = np.where(gap_lbl == g)[0]
                    gap_len = len(gap_cols) * cell_size
                    if 0.5 < gap_len < 2.5:
                        center_x = x_min + (x_lo_px + gap_cols.mean()) * cell_size
                        windows.append({
                            'x': center_x,
                            'z': p1[1],
                            'length': gap_len,
                            'orientation': 'horizontal',
                            'room': room['name'],
                        })
    
    return windows


# ─── Room colors ───

ROOM_COLORS = [
    '#4A90D9', '#E8834A', '#67B868', '#C75B8F', '#8B6CC1',
    '#D4A843', '#4ABFBF', '#D96060', '#7B8FD4', '#A0C75B',
]

PASTEL_FILLS = [
    '#E8F0FE',  # light blue
    '#FFF3E0',  # light orange
    '#E8F5E9',  # light green
    '#FCE4EC',  # light pink
    '#EDE7F6',  # light purple
    '#FFF8E1',  # light yellow
    '#E0F7FA',  # light cyan
    '#FFEBEE',  # light red
]


# ─── Main analysis ───

def analyze_mesh(mesh_file):
    print(f"\n{'='*60}")
    print(f"v27h: Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    up_idx, up_name = detect_up_axis(mesh)
    up_coords = mesh.vertices[:, up_idx]
    up_min, up_range = up_coords.min(), np.ptp(up_coords)
    x_raw, z_raw = project_vertices(mesh, up_idx)
    hmask = (up_coords >= up_min + up_range * 0.15) & (up_coords <= up_min + up_range * 0.85)
    x_mid, z_mid = x_raw[hmask], z_raw[hmask]

    angle = find_dominant_angle(x_mid, z_mid)
    angle_rad = angle * math.pi / 180
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    print(f"  Rotation: {angle:.1f}°, {up_name}-up")

    cell_size = 0.02
    density_img, img_x_min, img_z_min, cs = build_density_image(rx, rz, cell_size=cell_size)

    print("  Building room mask...")
    room_mask = build_room_mask(density_img, cs)

    print("  Computing distance transform...")
    dist_transform = cv2.distanceTransform(room_mask, cv2.DIST_L2, 5)

    print("  Finding room seeds...")
    seeds = find_room_seeds(dist_transform, cs, min_dist_m=0.4, min_sep_m=1.2)
    if len(seeds) < 4 and dist_transform.max() * cs > 1.0:
        print("  Retrying with lower thresholds...")
        seeds = find_room_seeds(dist_transform, cs, min_dist_m=0.3, min_sep_m=0.8)
    if len(seeds) == 0:
        ys, xs = np.where(room_mask > 0)
        if len(ys) > 0:
            seeds = [(int(ys.mean()), int(xs.mean()), dist_transform[int(ys.mean()), int(xs.mean())])]

    print("  Running watershed...")
    labeled = watershed_from_seeds(dist_transform, room_mask, seeds, cs)

    hallway_info = []
    if len(seeds) >= 2:
        print("  Splitting hallways...")
        labeled, hallway_info = split_hallways_from_rooms(labeled, dist_transform, room_mask, seeds, cs)

    print("  Detecting wall positions...")
    x_walls, z_walls = hough_wall_positions(density_img, img_x_min, img_z_min, cs)
    print(f"    X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"    Z-walls: {[f'{w:.2f}' for w in z_walls]}")

    # Extract clean polygons for each basin
    print("  Extracting clean room polygons...")
    hallway_label_set = {h['label'] for h in hallway_info}
    all_labels = sorted(set(np.unique(labeled)) - {0, -1})
    
    rooms = []
    room_idx = 0
    for lbl in all_labels:
        mask = (labeled == lbl).astype(np.uint8)
        area_px = np.sum(mask)
        area_m2 = area_px * cs * cs
        if area_m2 < 0.5:
            continue
        
        is_hallway = lbl in hallway_label_set
        
        # Extract clean polygon from basin contour
        poly = extract_basin_polygon(labeled, lbl, img_x_min, img_z_min, cs, x_walls, z_walls)
        poly_area = compute_polygon_area(poly) if len(poly) >= 3 else 0
        
        if is_hallway:
            name = "Hall"
        else:
            room_idx += 1
            name = f"Room {room_idx}"
        
        # Compute basin distance for width info
        basin_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_width = basin_dist.max() * 2 * cs
        
        rooms.append({
            'label': lbl,
            'polygon': poly,
            'area_m2': area_m2,
            'poly_area': poly_area,
            'name': name,
            'is_hallway': is_hallway,
            'max_width': max_width,
        })
        print(f"    {name}: basin={area_m2:.1f}m² poly={poly_area:.1f}m² verts={len(poly)}")

    total_area = sum(r['poly_area'] for r in rooms if r['poly_area'] > 0)

    # Detect doors between adjacent rooms
    print("  Detecting doors...")
    all_wall_segments = []
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            segs = find_shared_wall_segments(r1['polygon'], r2['polygon'])
            for seg in segs:
                seg['room1'] = r1['name']
                seg['room2'] = r2['name']
            all_wall_segments.extend(segs)
    
    doors = detect_doors_from_density(density_img, img_x_min, img_z_min, cs, all_wall_segments)
    print(f"    Found {len(doors)} doors")
    for d in doors:
        print(f"      Door at ({d['x']:.2f}, {d['z']:.2f}) width={d['width']:.2f}m {d['orientation']}")

    # Detect windows
    print("  Detecting windows...")
    # Get exterior bbox
    all_x = [p[0] for r in rooms for p in r['polygon']]
    all_z = [p[1] for r in rooms for p in r['polygon']]
    if all_x:
        ext_bbox = (min(all_x), max(all_x), min(all_z), max(all_z))
    else:
        ext_bbox = (0, 0, 0, 0)
    
    windows = detect_windows_on_exterior(density_img, img_x_min, img_z_min, cs, rooms, ext_bbox)
    print(f"    Found {len(windows)} windows")
    for w in windows:
        print(f"      Window at ({w['x']:.2f}, {w['z']:.2f}) len={w['length']:.2f}m {w['orientation']}")

    # Fine density for overlay
    density_fine, fx_min, fz_min, fcs = build_density_image(rx, rz, cell_size=0.01)

    print(f"\n=== v27h Summary ===")
    print(f"  Spaces: {len(rooms)}")
    print(f"  Total area: {total_area:.1f} m²")
    for r in rooms:
        print(f"    {r['name']}: {r['poly_area']:.1f} m²")

    return {
        'rooms': rooms, 'total_area': total_area,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'density_img': density_img, 'dist_transform': dist_transform,
        'room_mask': room_mask, 'labeled': labeled,
        'seeds': seeds, 'doors': doors, 'windows': windows,
        'x_walls': x_walls, 'z_walls': z_walls,
        'img_origin': (img_x_min, img_z_min, cs),
        'fine_density': density_fine,
        'fine_origin': (fx_min, fz_min, fcs),
    }


# ─── 5-panel research view ───

def visualize_research(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    ix_min, iz_min, cs = results['img_origin']
    fx_min, fz_min, fcs = results['fine_origin']
    density = results['density_img']
    extent = [ix_min, ix_min + density.shape[1] * cs,
              iz_min, iz_min + density.shape[0] * cs]

    # 1. Density
    d_display = density.copy()
    if d_display.max() > 0:
        d_display = (d_display / np.percentile(d_display[d_display > 0], 95)).clip(0, 1)
    axes[0].imshow(d_display, cmap='hot', origin='lower', extent=extent)
    axes[0].set_title('1. Density Image (2cm)', color='white', fontsize=14)

    # 2. Distance transform
    dist = results['dist_transform'] * cs
    axes[1].imshow(dist, cmap='magma', origin='lower', extent=extent)
    for sy, sx, sv in results['seeds']:
        wx = ix_min + sx * cs
        wz = iz_min + sy * cs
        axes[1].plot(wx, wz, 'c*', markersize=12, markeredgecolor='white')
    axes[1].set_title('2. Distance Transform + Seeds', color='white', fontsize=14)

    # 3. Watershed
    labeled = results['labeled']
    seg_img = np.zeros((*labeled.shape, 3), dtype=np.float32)
    for i, room in enumerate(results['rooms']):
        ch = ROOM_COLORS[i % len(ROOM_COLORS)]
        r, g, b = int(ch[1:3], 16)/255, int(ch[3:5], 16)/255, int(ch[5:7], 16)/255
        seg_img[labeled == room['label']] = [r, g, b]
    axes[2].imshow(seg_img, origin='lower', extent=extent)
    for room in results['rooms']:
        mask = labeled == room['label']
        rows, cols = np.where(mask)
        if len(rows) > 0:
            cx = ix_min + np.mean(cols) * cs
            cz = iz_min + np.mean(rows) * cs
            axes[2].text(cx, cz, f"{room['name']}\n{room['area_m2']:.1f}m²",
                        ha='center', va='center', color='white', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    axes[2].set_title('3. Watershed Segmentation', color='white', fontsize=14)

    # 4. Edge + polygon overlay
    ax3 = axes[3]
    ax3.set_aspect('equal')
    ax3.set_facecolor('black')
    fine = results['fine_density']
    if fine.max() > 0:
        f_display = (fine / np.percentile(fine[fine > 0], 95)).clip(0, 1)
    else:
        f_display = fine
    fine_extent = [fx_min, fx_min + fine.shape[1] * fcs,
                   fz_min, fz_min + fine.shape[0] * fcs]
    ax3.imshow(f_display, cmap='gray', origin='lower', alpha=0.4, extent=fine_extent)
    for xw in results['x_walls']:
        ax3.axvline(xw, color='yellow', alpha=0.3, linewidth=0.5)
    for zw in results['z_walls']:
        ax3.axhline(zw, color='yellow', alpha=0.3, linewidth=0.5)
    for i, room in enumerate(results['rooms']):
        poly = room['polygon']
        if len(poly) < 3:
            continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs = [p[0] for p in pc]
        zs = [p[1] for p in pc]
        ax3.fill(xs, zs, color=color, alpha=0.15)
        ax3.plot(xs, zs, color=color, linewidth=2.5)
    ax3.set_title('4. Walls + Polygon Overlay', color='white', fontsize=14)

    # 5. Clean floor plan preview
    ax4 = axes[4]
    ax4.set_aspect('equal')
    ax4.set_facecolor('#1a1a2e')
    all_x, all_z = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon']
        if len(poly) < 3:
            continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs = [p[0] for p in pc]
        zs = [p[1] for p in pc]
        ax4.fill(xs, zs, color=color, alpha=0.3)
        all_x.extend(xs)
        all_z.extend(zs)
        for j in range(len(poly)):
            k = (j + 1) % len(poly)
            ax4.plot([poly[j][0], poly[k][0]], [poly[j][1], poly[k][1]],
                    color='white', linewidth=3, solid_capstyle='round')
        cx, cz = polygon_centroid(poly)
        ax4.text(cx, cz, f"{room['name']}\n{room['poly_area']:.1f}m²",
                ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))
    ax4.set_title(f'5. Floor Plan — {len(results["rooms"])} spaces', color='white', fontsize=14)
    if all_x:
        m = 0.5
        ax4.set_xlim(min(all_x) - m, max(all_x) + m)
        ax4.set_ylim(min(all_z) - m, max(all_z) + m)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved research view: {output_path}")


# ─── Clean architectural floor plan ───

def draw_clean_floorplan(results, output_path):
    """Draw a clean, professional architectural floor plan on white background."""
    rooms = results['rooms']
    doors = results['doors']
    windows = results['windows']
    
    if not rooms:
        print("  No rooms to draw")
        return
    
    # Collect all polygon points
    all_x = [p[0] for r in rooms for p in r['polygon'] if len(r['polygon']) >= 3]
    all_z = [p[1] for r in rooms for p in r['polygon'] if len(r['polygon']) >= 3]
    if not all_x:
        return
    
    x_range = max(all_x) - min(all_x)
    z_range = max(all_z) - min(all_z)
    
    # Figure size proportional to floor plan
    scale = 2.0  # inches per meter
    fig_w = max(8, x_range * scale + 3)
    fig_h = max(6, z_range * scale + 3)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    WALL_LW = 7  # wall line width
    WALL_COLOR = '#1a1a1a'
    
    # Determine exterior bbox
    ext_x_min, ext_x_max = min(all_x), max(all_x)
    ext_z_min, ext_z_max = min(all_z), max(all_z)
    
    # 1. Fill rooms with pastel colors
    for i, room in enumerate(rooms):
        poly = room['polygon']
        if len(poly) < 3:
            continue
        fill_color = PASTEL_FILLS[i % len(PASTEL_FILLS)]
        pc = poly + [poly[0]]
        xs = [p[0] for p in pc]
        zs = [p[1] for p in pc]
        ax.fill(xs, zs, color=fill_color, zorder=1)
    
    # 2. Collect all wall edges with their properties
    # Build a set of all wall segments from all rooms
    all_edges = []
    for room in rooms:
        poly = room['polygon']
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i+1) % n]
            # Normalize edge direction
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                p1, p2 = p2, p1
            all_edges.append((p1[0], p1[1], p2[0], p2[1]))
    
    # Draw walls - check if edge is on exterior or interior
    drawn_edges = set()
    for room in rooms:
        poly = room['polygon']
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i+1) % n]
            # Normalize
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                key = (round(p2[0],3), round(p2[1],3), round(p1[0],3), round(p1[1],3))
            else:
                key = (round(p1[0],3), round(p1[1],3), round(p2[0],3), round(p2[1],3))
            
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            
            # Check if exterior
            is_exterior = False
            if abs(p1[0] - p2[0]) < 0.05:  # vertical
                if abs(p1[0] - ext_x_min) < 0.15 or abs(p1[0] - ext_x_max) < 0.15:
                    is_exterior = True
            elif abs(p1[1] - p2[1]) < 0.05:  # horizontal
                if abs(p1[1] - ext_z_min) < 0.15 or abs(p1[1] - ext_z_max) < 0.15:
                    is_exterior = True
            
            lw = WALL_LW if is_exterior else WALL_LW - 1
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=WALL_COLOR, linewidth=lw, solid_capstyle='round', zorder=3)
    
    # 3. Draw doors as quarter-circle arcs
    door_radius = 0.4  # meters
    for door in doors:
        x, z = door['x'], door['z']
        w = min(door['width'], 0.9)
        r = w * 0.9
        
        if door['orientation'] == 'vertical':
            # Door on vertical wall - opens to the side
            # Draw a gap in the wall
            ax.plot([x - 0.02, x + 0.02], [z - w/2, z - w/2], color='white', linewidth=WALL_LW + 2, zorder=4)
            ax.plot([x - 0.02, x + 0.02], [z + w/2, z + w/2], color='white', linewidth=WALL_LW + 2, zorder=4)
            # White rectangle to erase wall
            rect = patches.Rectangle((x - 0.06, z - w/2), 0.12, w, 
                                    facecolor='white', edgecolor='none', zorder=4)
            ax.add_patch(rect)
            # Quarter circle arc
            arc = Arc((x, z - w/2), 2*r, 2*r, angle=0, theta1=0, theta2=90,
                      color='#555555', linewidth=1.5, zorder=5)
            ax.add_patch(arc)
            # Door leaf line
            ax.plot([x, x + r], [z - w/2, z - w/2], color='#555555', linewidth=1.5, zorder=5)
            ax.plot([x, x], [z - w/2, z - w/2 + r], color='#555555', linewidth=1.5, zorder=5)
        
        else:
            # Door on horizontal wall
            rect = patches.Rectangle((x - w/2, z - 0.06), w, 0.12,
                                    facecolor='white', edgecolor='none', zorder=4)
            ax.add_patch(rect)
            arc = Arc((x - w/2, z), 2*r, 2*r, angle=0, theta1=0, theta2=90,
                      color='#555555', linewidth=1.5, zorder=5)
            ax.add_patch(arc)
            ax.plot([x - w/2, x - w/2 + r], [z, z], color='#555555', linewidth=1.5, zorder=5)
            ax.plot([x - w/2, x - w/2], [z, z + r], color='#555555', linewidth=1.5, zorder=5)
    
    # 4. Draw windows as double lines on exterior walls
    for window in windows:
        x, z = window['x'], window['z']
        wlen = window['length']
        gap = 0.06  # gap between double lines
        
        if window['orientation'] == 'vertical':
            # Window on vertical exterior wall
            # Erase wall segment
            rect = patches.Rectangle((x - 0.08, z - wlen/2), 0.16, wlen,
                                    facecolor='white', edgecolor='none', zorder=4)
            ax.add_patch(rect)
            # Double line
            ax.plot([x - gap, x - gap], [z - wlen/2, z + wlen/2],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
            ax.plot([x + gap, x + gap], [z - wlen/2, z + wlen/2],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
            # End caps
            ax.plot([x - gap, x + gap], [z - wlen/2, z - wlen/2],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
            ax.plot([x - gap, x + gap], [z + wlen/2, z + wlen/2],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
        
        elif window['orientation'] == 'horizontal':
            rect = patches.Rectangle((x - wlen/2, z - 0.08), wlen, 0.16,
                                    facecolor='white', edgecolor='none', zorder=4)
            ax.add_patch(rect)
            ax.plot([x - wlen/2, x + wlen/2], [z - gap, z - gap],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
            ax.plot([x - wlen/2, x + wlen/2], [z + gap, z + gap],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
            ax.plot([x - wlen/2, x - wlen/2], [z - gap, z + gap],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
            ax.plot([x + wlen/2, x + wlen/2], [z - gap, z + gap],
                   color=WALL_COLOR, linewidth=1.5, zorder=5)
    
    # 5. Room labels
    for i, room in enumerate(rooms):
        poly = room['polygon']
        if len(poly) < 3:
            continue
        cx, cz = polygon_centroid(poly)
        area = room['poly_area']
        
        label = f"{room['name']}\n{area:.1f}m²"
        ax.text(cx, cz, label,
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='#333333', zorder=6)
    
    # Clean up axes
    margin = 0.3
    ax.set_xlim(ext_x_min - margin, ext_x_max + margin)
    ax.set_ylim(ext_z_min - margin, ext_z_max + margin)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                pad_inches=0.3)
    plt.close()
    print(f"  Saved clean floor plan: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27h_clean_floorplan',
            'num_rooms': sum(1 for r in results['rooms'] if not r['is_hallway']),
            'num_hallways': sum(1 for r in results['rooms'] if r['is_hallway']),
            'total_area_m2': round(results['total_area'], 1),
        },
        'rooms': [{
            'name': r['name'],
            'area_m2': round(r['poly_area'], 1),
            'is_hallway': bool(r['is_hallway']),
            'polygon': [[round(p[0], 3), round(p[1], 3)] for p in r['polygon']],
        } for r in results['rooms']],
        'doors': [{
            'x': round(d['x'], 3), 'z': round(d['z'], 3),
            'width': round(d['width'], 2),
            'orientation': d['orientation'],
        } for d in results['doors']],
        'windows': [{
            'x': round(w['x'], 3), 'z': round(w['z'], 3),
            'length': round(w['length'], 2),
            'orientation': w['orientation'],
        } for w in results['windows']],
        'walls': {
            'x_positions': [round(w, 3) for w in results['x_walls']],
            'z_positions': [round(w, 3) for w in results['z_walls']],
        }
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27h - Clean Architectural Floor Plan')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27h/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27h_{Path(args.mesh_file).stem}"
    
    results = analyze_mesh(args.mesh_file)
    visualize_research(results, output_dir / f"{prefix}_floorplan.png")
    draw_clean_floorplan(results, output_dir / f"{prefix}_clean.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

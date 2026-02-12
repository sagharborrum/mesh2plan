#!/usr/bin/env python3
"""
mesh2plan v35 - Per-Room v22 Hybrid Geometry

Combines v32's multiroom detection with v22's per-room geometry extraction.

Pipeline:
1. Run v32's pipeline: density → room mask → wall cuts → strip merge → individual room masks
2. For EACH detected room mask:
   a. Extract density data within/near that room (with ~0.5m padding)
   b. Run v22's technique: local mask → contour → local Hough walls → snap contour to walls
   c. Result: accurate per-room polygon (L-shapes, alcoves, etc.)
3. Combine all room polygons into final floor plan
4. Door detection between adjacent rooms
5. Render architectural style (thick walls, white rooms, door arcs)
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndlabel, uniform_filter1d


# ─── Shared Utilities (from v32) ───

def detect_up_axis(mesh):
    ranges = [np.ptp(mesh.vertices[:, i]) for i in range(3)]
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges): return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges): return 2, 'Z'
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


# ─── v32 Room Mask + Wall Cutting ───

def build_room_mask(density_img, cell_size):
    occupied = (density_img > 0).astype(np.uint8)
    k_size = max(3, int(0.15 / cell_size)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    lbl, n = ndlabel(filled)
    if n > 1:
        sizes = ndimage.sum(filled, lbl, range(1, n+1))
        largest = np.argmax(sizes) + 1
        filled = (lbl == largest).astype(np.uint8)
    return filled

def hough_wall_positions(density_img, x_min, z_min, cell_size, nms_dist=0.15):
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)
    proj_x = smoothed.sum(axis=0)
    proj_z = smoothed.sum(axis=1)
    def find_peaks_nms(profile, origin, cs, min_dist_cells):
        prof = uniform_filter1d(profile.astype(float), size=5)
        local_max = maximum_filter(prof, size=max(3, min_dist_cells)) == prof
        threshold = prof.mean() + 0.15 * prof.std()
        peaks = np.where(local_max & (prof > threshold))[0]
        positions = origin + peaks * cs
        if len(positions) == 0: return np.array([]), np.array([])
        strengths = prof[peaks]
        order = np.argsort(-strengths)
        kept, kept_str = [], []
        for i in order:
            pos = positions[i]
            if any(abs(pos - k) < nms_dist for k in kept): continue
            kept.append(pos)
            kept_str.append(strengths[i])
        si = np.argsort(kept)
        return np.array(kept)[si], np.array(kept_str)[si]
    min_dist = int(nms_dist / cell_size)
    x_walls, x_str = find_peaks_nms(proj_x, x_min, cell_size, min_dist)
    z_walls, z_str = find_peaks_nms(proj_z, z_min, cell_size, min_dist)
    return x_walls, z_walls, x_str, z_str

def wall_has_evidence(density_img, x_min, z_min, cell_size, pos, axis, room_mask, min_fraction=0.15):
    strip_half = max(2, int(0.06 / cell_size))
    if axis == 'x':
        px = int((pos - x_min) / cell_size)
        if px < 0 or px >= density_img.shape[1]: return False, 0, 0
        lo = max(0, px - strip_half); hi = min(density_img.shape[1], px + strip_half + 1)
        mask_col = room_mask[:, px]; density_strip = density_img[:, lo:hi].max(axis=1)
        inside = mask_col > 0
    else:
        px = int((pos - z_min) / cell_size)
        if px < 0 or px >= density_img.shape[0]: return False, 0, 0
        lo = max(0, px - strip_half); hi = min(density_img.shape[0], px + strip_half + 1)
        mask_col = room_mask[px, :]; density_strip = density_img[lo:hi, :].max(axis=0)
        inside = mask_col > 0
    if not np.any(inside): return False, 0, 0
    inside_idx = np.where(inside)[0]
    total_extent = (inside_idx[-1] - inside_idx[0] + 1) * cell_size
    threshold = max(2, np.percentile(density_strip[density_strip > 0], 30) if np.any(density_strip > 0) else 2)
    is_wall = (density_strip > threshold) & inside
    wall_runs = []; cur = 0
    for v in is_wall:
        if v: cur += 1
        else:
            if cur > 0: wall_runs.append(cur)
            cur = 0
    if cur > 0: wall_runs.append(cur)
    max_run = max(wall_runs) * cell_size if wall_runs else 0
    total_wall = sum(wall_runs) * cell_size
    return total_wall / max(total_extent, 0.01) > min_fraction, total_extent, max_run

def cut_mask_with_walls(room_mask, density_img, x_min, z_min, cell_size,
                         x_walls, z_walls, x_str, z_str, min_wall_run=0.5):
    cut_mask = room_mask.copy()
    wall_half_px = max(1, int(0.04 / cell_size))
    valid_x, valid_z = [], []
    
    for i, xw in enumerate(x_walls):
        has_ev, extent, max_run = wall_has_evidence(density_img, x_min, z_min, cell_size, xw, 'x', room_mask)
        if has_ev and max_run >= min_wall_run:
            valid_x.append(xw)
            px = int((xw - x_min) / cell_size)
            strip_lo = max(0, px - wall_half_px)
            strip_hi = min(cut_mask.shape[1], px + wall_half_px + 1)
            density_col = density_img[:, max(0,px-2):min(density_img.shape[1],px+3)].max(axis=1)
            threshold = max(1, np.percentile(density_col[density_col > 0], 20) if np.any(density_col > 0) else 1)
            wall_rows = np.where((density_col > threshold) & (room_mask[:, px] > 0))[0]
            if len(wall_rows) > 0:
                segments = []; seg_start = wall_rows[0]
                for k in range(1, len(wall_rows)):
                    if wall_rows[k] - wall_rows[k-1] > int(1.5 / cell_size):
                        segments.append((seg_start, wall_rows[k-1])); seg_start = wall_rows[k]
                segments.append((seg_start, wall_rows[-1]))
                extend = int(0.5 / cell_size)
                for seg_s, seg_e in segments:
                    if (seg_e - seg_s) * cell_size >= min_wall_run:
                        r_lo = max(0, seg_s - extend)
                        r_hi = min(cut_mask.shape[0], seg_e + extend + 1)
                        cut_mask[r_lo:r_hi, strip_lo:strip_hi] = 0
    
    for i, zw in enumerate(z_walls):
        has_ev, extent, max_run = wall_has_evidence(density_img, x_min, z_min, cell_size, zw, 'z', room_mask)
        if has_ev and max_run >= min_wall_run:
            valid_z.append(zw)
            px = int((zw - z_min) / cell_size)
            strip_lo = max(0, px - wall_half_px)
            strip_hi = min(cut_mask.shape[0], px + wall_half_px + 1)
            density_row = density_img[max(0,px-2):min(density_img.shape[0],px+3), :].max(axis=0)
            threshold = max(1, np.percentile(density_row[density_row > 0], 20) if np.any(density_row > 0) else 1)
            wall_cols = np.where((density_row > threshold) & (room_mask[px, :] > 0))[0]
            if len(wall_cols) > 0:
                segments = []; seg_start = wall_cols[0]
                for k in range(1, len(wall_cols)):
                    if wall_cols[k] - wall_cols[k-1] > int(1.5 / cell_size):
                        segments.append((seg_start, wall_cols[k-1])); seg_start = wall_cols[k]
                segments.append((seg_start, wall_cols[-1]))
                extend = int(0.5 / cell_size)
                for seg_s, seg_e in segments:
                    if (seg_e - seg_s) * cell_size >= min_wall_run:
                        c_lo = max(0, seg_s - extend)
                        c_hi = min(cut_mask.shape[1], seg_e + extend + 1)
                        cut_mask[strip_lo:strip_hi, c_lo:c_hi] = 0
    
    return cut_mask, valid_x, valid_z


def extract_room_masks_v32(cut_mask, density_img, x_min, z_min, cell_size,
                            x_walls, z_walls, x_cuts, z_cuts):
    """Extract individual room masks from cut mask using v32's strip merge logic.
    Returns list of dicts with 'mask', 'area_m2', 'cx', 'cz', 'type', 'name'."""
    lbl, n = ndlabel(cut_mask)
    raw_rooms = []
    for i in range(1, n + 1):
        mask = (lbl == i)
        area = np.sum(mask) * cell_size * cell_size
        if area < 0.5: continue
        rows, cols = np.where(mask)
        cx = x_min + cols.mean() * cell_size
        cz = z_min + rows.mean() * cell_size
        raw_rooms.append({'mask': mask, 'area_m2': area, 'cx': cx, 'cz': cz})

    if not raw_rooms:
        return []

    x_cuts_sorted = sorted(x_cuts)

    def get_strip(cx):
        for i, cut in enumerate(x_cuts_sorted):
            if cx < cut: return i
        return len(x_cuts_sorted)

    for r in raw_rooms:
        r['strip'] = get_strip(r['cx'])

    # Strip merge
    strips = {}
    for r in raw_rooms:
        s = r['strip']
        if s not in strips: strips[s] = []
        strips[s].append(r)

    merged_rooms = []
    for strip_idx, rooms_in_strip in sorted(strips.items()):
        if len(rooms_in_strip) == 1:
            merged_rooms.append(rooms_in_strip[0])
        else:
            big_rooms = [r for r in rooms_in_strip if r['area_m2'] >= 4.0]
            small_rooms = [r for r in rooms_in_strip if r['area_m2'] < 4.0]
            for r in big_rooms:
                merged_rooms.append(r)
            for sr in small_rooms:
                if big_rooms:
                    best = min(big_rooms, key=lambda br: abs(br['cz'] - sr['cz']))
                    best['mask'] = best['mask'] | sr['mask']
                    best['area_m2'] += sr['area_m2']
                    rows, cols = np.where(best['mask'])
                    best['cx'] = x_min + cols.mean() * cell_size
                    best['cz'] = z_min + rows.mean() * cell_size
                else:
                    merged_rooms.append(sr)

    print(f"  Strips: {len(strips)} → {len(merged_rooms)} merged rooms")

    # Classify rooms
    for r in merged_rooms:
        rows, cols = np.where(r['mask'])
        x0 = x_min + cols.min() * cell_size
        x1 = x_min + cols.max() * cell_size
        z0 = z_min + rows.min() * cell_size
        z1 = z_min + rows.max() * cell_size
        w = x1 - x0; h = z1 - z0
        min_dim = min(w, h)

        dist = cv2.distanceTransform(r['mask'].astype(np.uint8), cv2.DIST_L2, 5)
        max_half_width = dist.max() * cell_size

        if len(x_cuts) >= 2:
            x_lo, x_hi = min(x_cuts), max(x_cuts)
            is_center = x_lo < r['cx'] < x_hi
        else:
            is_center = False

        if is_center and min_dim < 2.0:
            r['type'] = 'hallway'
        elif max_half_width < 0.8 and max(w,h)/max(min_dim,0.01) > 2.5:
            r['type'] = 'hallway'
        elif r['area_m2'] < 4.5:
            r['type'] = 'closet'
        else:
            r['type'] = 'room'

    # Name
    rn, hn, cn = 1, 1, 1
    merged_rooms.sort(key=lambda r: -r['area_m2'])
    for r in merged_rooms:
        if r['type'] == 'hallway':
            r['name'] = "Hallway" if hn == 1 else f"Hallway {hn}"; hn += 1
        elif r['type'] == 'closet':
            r['name'] = "Closet" if cn == 1 else f"Closet {cn}"; cn += 1
        else:
            r['name'] = f"Room {rn}"; rn += 1

    for r in merged_rooms:
        print(f"    {r['name']}: {r['area_m2']:.1f}m² ({r['type']})")

    return merged_rooms


# ─── v22 Per-Room Geometry Extraction ───

def extract_local_density(density_img, x_min, z_min, cell_size, room_mask, padding_m=0.5):
    """Extract density data in/near a room mask region with padding."""
    rows, cols = np.where(room_mask)
    if len(rows) == 0:
        return None, 0, 0, 0, 0

    pad_px = int(padding_m / cell_size)
    r_lo = max(0, rows.min() - pad_px)
    r_hi = min(density_img.shape[0], rows.max() + pad_px + 1)
    c_lo = max(0, cols.min() - pad_px)
    c_hi = min(density_img.shape[1], cols.max() + pad_px + 1)

    local_density = density_img[r_lo:r_hi, c_lo:c_hi].copy()
    local_mask = room_mask[r_lo:r_hi, c_lo:c_hi].copy()

    # Also include density near the mask (walls are at boundary)
    dilated = cv2.dilate(local_mask.astype(np.uint8), 
                         np.ones((pad_px*2+1, pad_px*2+1), np.uint8))
    local_density = local_density * dilated

    local_x_min = x_min + c_lo * cell_size
    local_z_min = z_min + r_lo * cell_size

    return local_density, local_x_min, local_z_min, local_mask, dilated


def local_hough_walls(local_density, local_x_min, local_z_min, cell_size):
    """Find Hough wall positions in a local density region (v22/v20b style)."""
    if local_density is None or local_density.size == 0:
        return [], []

    # Normalize and edge detect
    img_max = np.percentile(local_density[local_density > 0], 90) if np.any(local_density > 0) else 1
    img_norm = np.clip(local_density / max(img_max, 1) * 255, 0, 255).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    edges = cv2.Canny(img_blur, 50, 150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=int(0.3 / cell_size),
                            maxLineGap=int(0.3 / cell_size))

    if lines is None:
        return [], []

    x_positions = []
    z_positions = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        wx1 = local_x_min + x1 * cell_size
        wz1 = local_z_min + y1 * cell_size
        wx2 = local_x_min + x2 * cell_size
        wz2 = local_z_min + y2 * cell_size

        length = math.sqrt((wx2-wx1)**2 + (wz2-wz1)**2)
        if length < 0.3: continue

        dx, dz = abs(wx2-wx1), abs(wz2-wz1)
        if dx + dz < 0.01: continue
        angle_mod = math.atan2(min(dx, dz), max(dx, dz)) * 180 / math.pi
        if angle_mod > 15: continue

        if dz > dx:
            x_positions.append(((wx1 + wx2) / 2, length))
        else:
            z_positions.append(((wz1 + wz2) / 2, length))

    x_walls = cluster_positions(x_positions, 0.15)
    z_walls = cluster_positions(z_positions, 0.15)

    return x_walls, z_walls


def cluster_positions(positions, dist_threshold=0.15):
    if not positions: return []
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
        avg_pos = sum(p[0] * p[1] for p in cluster) / total_len
        result.append(avg_pos)
    return sorted(result)


def build_local_room_mask_v22(local_density, cell_size):
    """Build room mask from local density (v22/v21b style)."""
    room_mask = (local_density > 1).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel_close)
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel_open)
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    return room_mask


def v22_extract_room_polygon(density_img, x_min, z_min, cell_size, room_mask_v32,
                              global_x_walls, global_z_walls, padding_m=0.5):
    """Apply v22's hybrid technique to a single room.
    
    1. Extract local density with padding
    2. Build local room mask
    3. Find local Hough walls
    4. Extract contour from local mask
    5. Snap contour to local Hough walls
    6. Clean up polygon
    
    Returns polygon in world coordinates.
    """
    # Extract local region
    local_density, local_x_min, local_z_min, local_mask_crop, dilated = \
        extract_local_density(density_img, x_min, z_min, cell_size, room_mask_v32, padding_m)

    if local_density is None:
        return None

    # Build local room mask (v22 style) from the local density
    local_room_mask = build_local_room_mask_v22(local_density, cell_size)

    # Constrain to the v32 room mask region (dilated slightly for wall capture)
    # This prevents bleeding into adjacent rooms
    constrain_mask = cv2.dilate(local_mask_crop.astype(np.uint8),
                                np.ones((int(0.3/cell_size), int(0.3/cell_size)), np.uint8))
    local_room_mask = local_room_mask & constrain_mask

    if np.sum(local_room_mask) < 10:
        return None

    # Find local Hough walls
    local_x_walls, local_z_walls = local_hough_walls(local_density, local_x_min, local_z_min, cell_size)

    # Also include nearby global walls as candidates
    rows, cols = np.where(room_mask_v32)
    room_x_min = x_min + cols.min() * cell_size - padding_m
    room_x_max = x_min + cols.max() * cell_size + padding_m
    room_z_min = z_min + rows.min() * cell_size - padding_m
    room_z_max = z_min + rows.max() * cell_size + padding_m

    for gxw in global_x_walls:
        if room_x_min <= gxw <= room_x_max:
            if not any(abs(gxw - lw) < 0.15 for lw in local_x_walls):
                local_x_walls.append(gxw)
    for gzw in global_z_walls:
        if room_z_min <= gzw <= room_z_max:
            if not any(abs(gzw - lw) < 0.15 for lw in local_z_walls):
                local_z_walls.append(gzw)

    local_x_walls = sorted(local_x_walls)
    local_z_walls = sorted(local_z_walls)

    # Extract contour
    contours, _ = cv2.findContours(local_room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Convert contour to world coords
    raw_pts = []
    for pt in contour.reshape(-1, 2):
        wx = local_x_min + pt[0] * cell_size
        wz = local_z_min + pt[1] * cell_size
        raw_pts.append([wx, wz])

    # Snap to local Hough walls (v22 style)
    snap_dist = 0.3
    snapped = []
    for pt in raw_pts:
        x, z = pt
        best_x, best_x_dist = x, snap_dist
        for wx in local_x_walls:
            d = abs(x - wx)
            if d < best_x_dist:
                best_x = wx; best_x_dist = d
        best_z, best_z_dist = z, snap_dist
        for wz in local_z_walls:
            d = abs(z - wz)
            if d < best_z_dist:
                best_z = wz; best_z_dist = d
        snapped.append([best_x, best_z])

    # Simplify: Douglas-Peucker + force rectilinear
    pts_cv = np.array(snapped).reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts_cv, 0.08, True)
    pts = simplified.reshape(-1, 2).tolist()

    # Force axis-alignment
    result = [pts[0]]
    for i in range(1, len(pts)):
        prev = result[-1]; curr = pts[i][:]
        dx, dz = abs(curr[0]-prev[0]), abs(curr[1]-prev[1])
        if dx < 0.12:
            curr[0] = prev[0]
        elif dz < 0.12:
            curr[1] = prev[1]
        elif dx > 0.05 and dz > 0.05:
            # Insert corner to make rectilinear
            if dx < dz:
                result.append([curr[0], prev[1]])
            else:
                result.append([prev[0], curr[1]])
        result.append(curr)

    # Re-snap after simplification
    final = []
    for pt in result:
        x, z = pt
        best_x, best_x_dist = x, snap_dist
        for wx in local_x_walls:
            d = abs(x - wx)
            if d < best_x_dist:
                best_x = wx; best_x_dist = d
        best_z, best_z_dist = z, snap_dist
        for wz in local_z_walls:
            d = abs(z - wz)
            if d < best_z_dist:
                best_z = wz; best_z_dist = d
        final.append([best_x, best_z])

    # Remove tiny segments iteratively
    for _ in range(5):
        cleaned = [final[0]]
        i = 1
        while i < len(final):
            d = math.sqrt((final[i][0]-cleaned[-1][0])**2 + (final[i][1]-cleaned[-1][1])**2)
            if d < 0.3 and len(final) - (i - len(cleaned)) > 3:
                pass  # skip
            else:
                cleaned.append(final[i])
            i += 1
        if len(cleaned) == len(final): break
        final = cleaned

    # Remove collinear
    final = remove_collinear(final)

    # Remove duplicates
    deduped = [final[0]]
    for p in final[1:]:
        if abs(p[0]-deduped[-1][0]) > 0.01 or abs(p[1]-deduped[-1][1]) > 0.01:
            deduped.append(p)
    final = deduped

    if len(final) < 3:
        return None

    return final


def remove_collinear(poly):
    if len(poly) < 3: return poly
    result = []
    n = len(poly)
    for i in range(n):
        prev = poly[(i-1)%n]; curr = poly[i]; nxt = poly[(i+1)%n]
        cross = (curr[0]-prev[0])*(nxt[1]-curr[1]) - (curr[1]-prev[1])*(nxt[0]-curr[0])
        if abs(cross) > 0.001:
            result.append(curr)
    return result if len(result) >= 3 else poly

def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    return abs(sum(poly[i][0]*poly[(i+1)%n][1] - poly[(i+1)%n][0]*poly[i][1] for i in range(n))) / 2

def polygon_centroid(poly):
    n = len(poly)
    if n == 0: return 0, 0
    return sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n


# ─── Door Detection ───

def detect_doors(density_img, x_min, z_min, cs, rooms_data):
    doors = []
    for i in range(len(rooms_data)):
        pi = rooms_data[i]['polygon']
        ni = rooms_data[i]['name']
        for j in range(i+1, len(rooms_data)):
            pj = rooms_data[j]['polygon']
            nj = rooms_data[j]['name']
            for ei in range(len(pi)):
                a1, a2 = pi[ei], pi[(ei+1)%len(pi)]
                for ej in range(len(pj)):
                    b1, b2 = pj[ej], pj[(ej+1)%len(pj)]
                    door = check_shared_edge_for_door(density_img, x_min, z_min, cs, a1, a2, b1, b2)
                    if door:
                        doors.append(door)
                        print(f"    Door: {ni} ↔ {nj} at ({door['x']:.2f}, {door['z']:.2f})")

    if not doors:
        print("    No polygon-edge doors — trying mask adjacency")
        for i in range(len(rooms_data)):
            if 'mask' not in rooms_data[i]: continue
            mi = rooms_data[i]['mask']
            ni = rooms_data[i]['name']
            di = cv2.dilate(mi.astype(np.uint8), np.ones((15,15), np.uint8))
            for j in range(i+1, len(rooms_data)):
                if 'mask' not in rooms_data[j]: continue
                mj = rooms_data[j]['mask']
                nj = rooms_data[j]['name']
                overlap = np.sum(di & mj)
                if overlap > 0:
                    boundary = di & cv2.dilate(mj.astype(np.uint8), np.ones((3,3), np.uint8))
                    brows, bcols = np.where(boundary)
                    if len(brows) > 0:
                        bx = x_min + bcols.mean() * cs
                        bz = z_min + brows.mean() * cs
                        x_spread = (bcols.max() - bcols.min()) * cs
                        z_spread = (brows.max() - brows.min()) * cs
                        orientation = 'vertical' if x_spread < z_spread else 'horizontal'
                        doors.append({'x': bx, 'z': bz, 'width': 0.8, 'orientation': orientation})
                        print(f"    Door (mask): {ni} ↔ {nj} at ({bx:.2f}, {bz:.2f})")
    return doors

def check_shared_edge_for_door(density_img, x_min, z_min, cs, a1, a2, b1, b2, tol=0.35):
    sw = max(1, int(0.08/cs))
    if abs(a1[0]-a2[0]) < 0.05 and abs(b1[0]-b2[0]) < 0.05 and abs(a1[0]-b1[0]) < tol:
        alo, ahi = min(a1[1],a2[1]), max(a1[1],a2[1])
        blo, bhi = min(b1[1],b2[1]), max(b1[1],b2[1])
        lo, hi = max(alo,blo), min(ahi,bhi)
        if hi - lo < 0.5: return None
        x_px = int(((a1[0]+b1[0])/2 - x_min)/cs)
        xl = max(0, x_px-sw); xh = min(density_img.shape[1], x_px+sw+1)
        lp = max(0, int((lo-z_min)/cs)); hp = min(density_img.shape[0], int((hi-z_min)/cs))
        if xl >= xh or lp >= hp: return None
        prof = density_img[lp:hp, xl:xh].sum(axis=1)
        return find_door_in_profile(prof, cs, (a1[0]+b1[0])/2, z_min + lp*cs, 'vertical')
    if abs(a1[1]-a2[1]) < 0.05 and abs(b1[1]-b2[1]) < 0.05 and abs(a1[1]-b1[1]) < tol:
        alo, ahi = min(a1[0],a2[0]), max(a1[0],a2[0])
        blo, bhi = min(b1[0],b2[0]), max(b1[0],b2[0])
        lo, hi = max(alo,blo), min(ahi,bhi)
        if hi - lo < 0.5: return None
        z_px = int(((a1[1]+b1[1])/2 - z_min)/cs)
        zl = max(0, z_px-sw); zh = min(density_img.shape[0], z_px+sw+1)
        lp = max(0, int((lo-x_min)/cs)); hp = min(density_img.shape[1], int((hi-x_min)/cs))
        if zl >= zh or lp >= hp: return None
        prof = density_img[zl:zh, lp:hp].sum(axis=0)
        return find_door_in_profile(prof, cs, x_min + lp*cs, (a1[1]+b1[1])/2, 'horizontal')
    return None

def find_door_in_profile(prof, cs, origin_x, origin_z, orientation):
    if len(prof) == 0 or not np.any(prof > 0): return None
    thr = np.percentile(prof[prof>0], 30) if np.any(prof>0) else 1
    is_gap = prof < max(thr, 1)
    lbl, ng = ndlabel(is_gap)
    for g in range(1, ng+1):
        gi = np.where(lbl==g)[0]
        gl = len(gi) * cs
        if 0.6 < gl < 1.5:
            center = gi.mean() * cs
            if orientation == 'vertical':
                return {'x': origin_x, 'z': origin_z + center, 'width': gl, 'orientation': orientation}
            else:
                return {'x': origin_x + center, 'z': origin_z, 'width': gl, 'orientation': orientation}
    return None


# ─── Architectural Rendering ───

ROOM_COLORS = {
    'room': ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#F3E5F5', '#FFFDE7', '#E0F7FA'],
    'hallway': '#F5F5F5',
    'closet': '#EFEBE9',
}

def render_architectural(rooms_data, doors, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    all_x, all_z = [], []
    for rd in rooms_data:
        for p in rd['polygon']:
            all_x.append(p[0]); all_z.append(p[1])
    if not all_x:
        plt.savefig(output_path, dpi=200); plt.close(); return

    margin = 0.8
    wall_w = 0.08

    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        if rd['type'] == 'hallway': color = ROOM_COLORS['hallway']
        elif rd['type'] == 'closet': color = ROOM_COLORS['closet']
        else:
            color = ROOM_COLORS['room'][ci % len(ROOM_COLORS['room'])]
            ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.9, zorder=1)

    # Walls
    for rd in rooms_data:
        poly = rd['polygon']
        n = len(poly)
        for i in range(n):
            p1 = poly[i]; p2 = poly[(i+1)%n]
            if abs(p1[0]-p2[0]) < 0.05:
                x = p1[0]
                z_lo, z_hi = min(p1[1],p2[1]), max(p1[1],p2[1])
                rect = patches.Rectangle((x - wall_w/2, z_lo), wall_w, z_hi-z_lo,
                                          facecolor='black', edgecolor='none', zorder=2)
                ax.add_patch(rect)
            elif abs(p1[1]-p2[1]) < 0.05:
                z = p1[1]
                x_lo, x_hi = min(p1[0],p2[0]), max(p1[0],p2[0])
                rect = patches.Rectangle((x_lo, z - wall_w/2), x_hi-x_lo, wall_w,
                                          facecolor='black', edgecolor='none', zorder=2)
                ax.add_patch(rect)
            else:
                ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=3, zorder=2)

    # Doors
    for door in doors:
        x, z = door['x'], door['z']
        w = door.get('width', 0.8)
        if door['orientation'] == 'vertical':
            rect = patches.Rectangle((x - wall_w, z - w/2), wall_w*2, w,
                                      facecolor='white', edgecolor='none', zorder=3)
            ax.add_patch(rect)
            arc = Arc((x, z - w/2), w, w, angle=0, theta1=0, theta2=90,
                      color='#666', linewidth=1.5, zorder=4)
            ax.add_patch(arc)
            ax.plot([x, x + w/2], [z - w/2, z - w/2], color='#666', lw=1.5, zorder=4)
        else:
            rect = patches.Rectangle((x - w/2, z - wall_w), w, wall_w*2,
                                      facecolor='white', edgecolor='none', zorder=3)
            ax.add_patch(rect)
            arc = Arc((x - w/2, z), w, w, angle=0, theta1=0, theta2=90,
                      color='#666', linewidth=1.5, zorder=4)
            ax.add_patch(arc)
            ax.plot([x - w/2, x - w/2], [z, z + w/2], color='#666', lw=1.5, zorder=4)

    # Labels
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        cx, cz = polygon_centroid(poly)
        area = rd['area']
        name = rd['name']
        fs = 9 if area > 5 else 7
        ax.text(cx, cz + 0.15, name, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#333', zorder=5)
        ax.text(cx, cz - 0.15, f"{area:.1f} m²", ha='center', va='center',
                fontsize=fs-1, color='#666', zorder=5)
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        w = max(xs) - min(xs); h = max(zs) - min(zs)
        ax.text(cx, cz - 0.35, f"{w:.1f} × {h:.1f} m", ha='center', va='center',
                fontsize=fs-2, color='#999', style='italic', zorder=5)

    # Scale bar
    sx = min(all_x) + 0.3; sy = min(all_z) - 0.4
    ax.plot([sx, sx + 1], [sy, sy], 'k-', lw=2)
    ax.plot([sx, sx], [sy - 0.05, sy + 0.05], 'k-', lw=2)
    ax.plot([sx + 1, sx + 1], [sy - 0.05, sy + 0.05], 'k-', lw=2)
    ax.text(sx + 0.5, sy - 0.15, '1 m', ha='center', va='top', fontsize=8)

    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if rd['type'] == 'room')
    nh = sum(1 for rd in rooms_data if rd['type'] == 'hallway')
    nc = sum(1 for rd in rooms_data if rd['type'] == 'closet')
    parts = [f"{nr} room{'s' if nr != 1 else ''}"]
    if nh: parts.append(f"{nh} hallway{'s' if nh != 1 else ''}")
    if nc: parts.append(f"{nc} closet{'s' if nc != 1 else ''}")
    subtitle = ' · '.join(parts) + f" · {total:.1f} m²"

    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    ax.set_aspect('equal')
    ax.set_title(f"Floor Plan (v35)\n{subtitle}", fontsize=13, fontweight='bold', pad=15, color='#333')
    ax.axis('off')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug_overlay(density_img, x_min, z_min, cs, rooms_data, x_walls, z_walls,
                          valid_x, valid_z, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    extent = [x_min, x_min + density_img.shape[1]*cs, z_min, z_min + density_img.shape[0]*cs]

    ax = axes[0]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal')
    for xw in x_walls: ax.axvline(xw, color='cyan', alpha=0.3, lw=0.5)
    for zw in z_walls: ax.axhline(zw, color='lime', alpha=0.3, lw=0.5)
    for xw in valid_x: ax.axvline(xw, color='cyan', alpha=0.9, lw=2)
    for zw in valid_z: ax.axhline(zw, color='lime', alpha=0.9, lw=2)
    ax.set_title('Density + Walls')

    ax = axes[1]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal', alpha=0.4)
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        if rd['type'] == 'room':
            color = ROOM_COLORS['room'][ci % len(ROOM_COLORS['room'])]
            ci += 1
        elif rd['type'] == 'hallway': color = ROOM_COLORS['hallway']
        else: color = ROOM_COLORS['closet']
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.4)
        n = len(poly)
        for k in range(n):
            p1, p2 = poly[k], poly[(k+1)%n]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'w-', lw=2)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²\n{len(poly)}v", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
    ax.set_title('Per-Room v22 Polygons')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ───

def process_mesh(mesh_path, output_dir, cell_size=0.02, nms_dist=0.3):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = Path(mesh_path).stem

    print(f"Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    up_idx, up_name = detect_up_axis(mesh)
    rx, rz = project_vertices(mesh, up_idx)
    angle = find_dominant_angle(rx, rz, cell=cell_size)
    print(f"  Rotation: {angle:.1f}°")

    cos_a, sin_a = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a

    density_img, x_min, z_min, cs = build_density_image(rx2, rz2, cell_size=cell_size)
    room_mask = build_room_mask(density_img, cs)

    x_walls, z_walls, x_str, z_str = hough_wall_positions(density_img, x_min, z_min, cs, nms_dist=nms_dist)
    print(f"  Hough walls: {len(x_walls)}X, {len(z_walls)}Z")

    # Score and select walls (from v32)
    mask_rows = np.where(room_mask.any(axis=1))[0]
    mask_cols = np.where(room_mask.any(axis=0))[0]
    bx_min = x_min + mask_cols[0] * cs
    bx_max = x_min + mask_cols[-1] * cs
    bz_min = z_min + mask_rows[0] * cs
    bz_max = z_min + mask_rows[-1] * cs
    bmargin = 0.3

    def score_walls(walls, strengths, axis, bound_lo, bound_hi):
        scored = []
        for i, w in enumerate(walls):
            has_ev, extent, max_run = wall_has_evidence(density_img, x_min, z_min, cs, w, axis, room_mask)
            strength = strengths[i] if i < len(strengths) else 0
            score = float(strength * max_run)
            is_boundary = abs(w - bound_lo) < bmargin or abs(w - bound_hi) < bmargin
            if not is_boundary and max_run >= 0.8:
                scored.append((w, score, max_run))
        scored.sort(key=lambda t: -t[1])
        return scored

    x_scored = score_walls(x_walls, x_str, 'x', bx_min, bx_max)
    z_scored = score_walls(z_walls, z_str, 'z', bz_min, bz_max)

    min_sep = 1.0
    def select_top(scored, max_n):
        sel = []
        for pos, score, run in scored:
            if any(abs(pos - s) < min_sep for s in sel): continue
            sel.append(pos)
            if len(sel) >= max_n: break
        return sel

    sel_x = select_top(x_scored, 2)
    sel_z = select_top(z_scored, 2)

    mask_area = np.sum(room_mask) * cs * cs
    is_single_room = mask_area < 20.0
    print(f"  Mask area: {mask_area:.1f}m² → {'single room' if is_single_room else 'multiroom'}")

    if is_single_room:
        print("  Single room mode — skipping cuts, using v22 directly")
        sel_x = []; sel_z = []

    print(f"  Selected cuts: X={[f'{w:.2f}' for w in sel_x]}, Z={[f'{w:.2f}' for w in sel_z]}")

    # Phase 1: Get room masks from v32
    if sel_x or sel_z:
        sel_x_arr = np.array(sorted(sel_x))
        sel_z_arr = np.array(sorted(sel_z))
        sel_x_str = np.array([next(s for w,s,_ in x_scored if w==xw) for xw in sel_x_arr]) if len(sel_x_arr) else np.array([])
        sel_z_str = np.array([next(s for w,s,_ in z_scored if w==zw) for zw in sel_z_arr]) if len(sel_z_arr) else np.array([])

        cut_mask, valid_x, valid_z = cut_mask_with_walls(
            room_mask, density_img, x_min, z_min, cs,
            sel_x_arr, sel_z_arr, sel_x_str, sel_z_str, min_wall_run=0.5)

        print("Phase 1: Extracting room masks (v32 strip merge)...")
        room_masks = extract_room_masks_v32(cut_mask, density_img, x_min, z_min, cs,
                                             x_walls, z_walls, valid_x, valid_z)
    else:
        valid_x, valid_z = [], []
        room_masks = [{
            'mask': room_mask, 'area_m2': mask_area,
            'type': 'room', 'name': 'Room',
            'cx': x_min + mask_cols.mean() * cs,
            'cz': z_min + mask_rows.mean() * cs,
        }]

    # Phase 2: For each room, run v22's hybrid technique
    print(f"\nPhase 2: Per-room v22 geometry extraction...")
    rooms_data = []
    for rm in room_masks:
        print(f"  Processing {rm['name']}...")
        poly = v22_extract_room_polygon(density_img, x_min, z_min, cs, rm['mask'],
                                         x_walls, z_walls, padding_m=0.5)

        if poly is None or len(poly) < 3:
            # Fallback: bounding box from mask
            rows, cols = np.where(rm['mask'])
            def snap(val, positions, max_snap=0.25):
                if len(positions) == 0: return val
                dists = np.abs(np.array(positions) - val)
                idx = np.argmin(dists)
                return float(positions[idx]) if dists[idx] < max_snap else val
            x0 = snap(x_min + cols.min() * cs, x_walls)
            x1 = snap(x_min + cols.max() * cs, x_walls)
            z0 = snap(z_min + rows.min() * cs, z_walls)
            z1 = snap(z_min + rows.max() * cs, z_walls)
            poly = [[x0,z0],[x1,z0],[x1,z1],[x0,z1]]
            print(f"    Fallback: rectangle {len(poly)}v")
        else:
            print(f"    v22 polygon: {len(poly)}v")

        area = compute_polygon_area(poly)
        rooms_data.append({
            'name': rm['name'],
            'type': rm['type'],
            'polygon': poly,
            'area': area,
            'mask': rm['mask'],
        })
        print(f"    {rm['name']}: {area:.1f}m² ({len(poly)} vertices)")

    # Phase 3: Door detection
    print(f"\nPhase 3: Door detection...")
    doors = detect_doors(density_img, x_min, z_min, cs, rooms_data)
    print(f"  Doors: {len(doors)}")

    # Phase 4: Render
    print(f"\nPhase 4: Rendering...")
    plan_path = out_dir / f"v35_{mesh_name}_plan.png"
    render_architectural(rooms_data, doors, plan_path)

    debug_path = out_dir / f"v35_{mesh_name}_debug.png"
    render_debug_overlay(density_img, x_min, z_min, cs, rooms_data, x_walls, z_walls,
                          valid_x, valid_z, debug_path)

    # JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)

    results = {
        'approach': 'v35_per_room_v22',
        'rooms': [{
            'name': rd['name'], 'area_m2': round(rd['area'],1),
            'type': rd['type'],
            'polygon': [[round(p[0],3), round(p[1],3)] for p in rd['polygon']],
            'vertices': len(rd['polygon']),
        } for rd in rooms_data],
        'doors': [{k: v for k, v in d.items()} for d in doors],
        'total_area_m2': round(sum(rd['area'] for rd in rooms_data), 1),
    }
    json_path = out_dir / f"v35_{mesh_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    print(f"  Saved: {json_path}")

    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if rd['type'] == 'room')
    nh = sum(1 for rd in rooms_data if rd['type'] == 'hallway')
    nc = sum(1 for rd in rooms_data if rd['type'] == 'closet')
    print(f"\n=== v35 RESULTS ===")
    print(f"  {nr} rooms, {nh} hallways, {nc} closets — {total:.1f}m²")
    for rd in rooms_data:
        print(f"  {rd['name']}: {rd['area']:.1f}m² ({len(rd['polygon'])}v, {rd['type']})")
    print("Done!")

    return plan_path


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v35 - Per-Room v22 Hybrid')
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--nms', type=float, default=0.3)
    parser.add_argument('-o', '--output', default='results/v35_per_room')
    args = parser.parse_args()

    process_mesh(args.mesh, args.output, cell_size=args.cell, nms_dist=args.nms)


if __name__ == '__main__':
    main()

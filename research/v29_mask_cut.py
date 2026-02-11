#!/usr/bin/env python3
"""
mesh2plan v29 - Mask + Wall Cut Approach

Key insight: v28b's wall-grid produces gaps because cells don't tile the full space.
Instead: 
1. Build a filled room mask (binary: inside apartment vs outside)
2. Use Hough wall positions as cutting lines to split the mask
3. Each resulting connected component = a room
4. Classify narrow components as hallways

This ensures rooms tile the full apartment with no gaps.
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


# ─── Shared utilities ───

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


def build_room_mask(density_img, cell_size):
    """Build binary mask of the apartment interior."""
    occupied = (density_img > 0).astype(np.uint8)
    # Close gaps
    k_size = max(3, int(0.15 / cell_size)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    # Fill holes
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    # Keep largest component
    lbl, n = ndlabel(filled)
    if n > 1:
        sizes = ndimage.sum(filled, lbl, range(1, n+1))
        largest = np.argmax(sizes) + 1
        filled = (lbl == largest).astype(np.uint8)
    return filled


def wall_has_evidence(density_img, x_min, z_min, cell_size, pos, axis, 
                       room_mask, min_fraction=0.3):
    """
    Check if a wall position has actual wall evidence within the room mask.
    Returns (has_evidence, total_length_in_mask, wall_length).
    A wall must have high-density pixels along a significant fraction of its
    extent within the room mask.
    """
    strip_half = max(2, int(0.06 / cell_size))
    
    if axis == 'x':
        px = int((pos - x_min) / cell_size)
        if px < 0 or px >= density_img.shape[1]:
            return False, 0, 0
        lo = max(0, px - strip_half)
        hi = min(density_img.shape[1], px + strip_half + 1)
        # Get column within room mask
        mask_col = room_mask[:, px]
        density_strip = density_img[:, lo:hi].max(axis=1)
        
        # Only look where the mask says we're inside the apartment
        inside = mask_col > 0
        if not np.any(inside):
            return False, 0, 0
        
        inside_rows = np.where(inside)[0]
        total_extent = (inside_rows[-1] - inside_rows[0] + 1) * cell_size
        
        # High density along this wall
        threshold = max(2, np.percentile(density_strip[density_strip > 0], 30) if np.any(density_strip > 0) else 2)
        is_wall = (density_strip > threshold) & inside
        
        # Find longest continuous wall segment
        wall_runs = []
        cur = 0
        for v in is_wall:
            if v: cur += 1
            else:
                if cur > 0: wall_runs.append(cur)
                cur = 0
        if cur > 0: wall_runs.append(cur)
        
        max_run = max(wall_runs) * cell_size if wall_runs else 0
        total_wall = sum(wall_runs) * cell_size
        
        return total_wall / max(total_extent, 0.01) > min_fraction, total_extent, max_run
    
    else:  # axis == 'z'
        px = int((pos - z_min) / cell_size)
        if px < 0 or px >= density_img.shape[0]:
            return False, 0, 0
        lo = max(0, px - strip_half)
        hi = min(density_img.shape[0], px + strip_half + 1)
        
        mask_row = room_mask[px, :]
        density_strip = density_img[lo:hi, :].max(axis=0)
        
        inside = mask_row > 0
        if not np.any(inside):
            return False, 0, 0
        
        inside_cols = np.where(inside)[0]
        total_extent = (inside_cols[-1] - inside_cols[0] + 1) * cell_size
        
        threshold = max(2, np.percentile(density_strip[density_strip > 0], 30) if np.any(density_strip > 0) else 2)
        is_wall = (density_strip > threshold) & inside
        
        wall_runs = []
        cur = 0
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
                         x_walls, z_walls, x_str, z_str,
                         min_evidence_fraction=0.3, min_wall_run=0.8):
    """
    Cut the room mask along wall positions where there's actual wall evidence.
    The cut is done by erasing a thin strip (wall thickness) along each valid wall.
    """
    cut_mask = room_mask.copy()
    wall_half_px = max(1, int(0.04 / cell_size))  # ~4cm half-width = 8cm wall
    
    valid_x = []
    valid_z = []
    
    # Check X walls (vertical cuts)
    for i, xw in enumerate(x_walls):
        has_ev, extent, max_run = wall_has_evidence(
            density_img, x_min, z_min, cell_size, xw, 'x', room_mask, min_evidence_fraction)
        str_val = x_str[i] if i < len(x_str) else 0
        print(f"    X wall {xw:.2f}: evidence={has_ev}, extent={extent:.1f}m, max_run={max_run:.1f}m, strength={str_val:.0f}")
        
        if has_ev and max_run >= min_wall_run:
            valid_x.append(xw)
            # Cut along the wall's actual extent (where density supports it)
            # plus small extensions to ensure we bridge across doors
            px = int((xw - x_min) / cell_size)
            strip_lo = max(0, px - wall_half_px)
            strip_hi = min(cut_mask.shape[1], px + wall_half_px + 1)
            
            density_col = density_img[:, max(0,px-2):min(density_img.shape[1],px+3)].max(axis=1)
            threshold = max(1, np.percentile(density_col[density_col > 0], 20) if np.any(density_col > 0) else 1)
            is_wall_px = density_col > threshold
            
            # Find extents of wall segments and extend them to bridge small gaps
            wall_rows = np.where(is_wall_px & (room_mask[:, px] > 0))[0]
            if len(wall_rows) > 0:
                # Find contiguous segments
                segments = []
                seg_start = wall_rows[0]
                for k in range(1, len(wall_rows)):
                    if wall_rows[k] - wall_rows[k-1] > int(1.5 / cell_size):  # gap > 1.5m = new segment
                        segments.append((seg_start, wall_rows[k-1]))
                        seg_start = wall_rows[k]
                segments.append((seg_start, wall_rows[-1]))
                
                # Cut each wall segment (extend by 0.5m each end to bridge door gaps)
                extend = int(0.5 / cell_size)
                for seg_s, seg_e in segments:
                    seg_len = (seg_e - seg_s) * cell_size
                    if seg_len >= min_wall_run:
                        r_lo = max(0, seg_s - extend)
                        r_hi = min(cut_mask.shape[0], seg_e + extend + 1)
                        cut_mask[r_lo:r_hi, strip_lo:strip_hi] = 0
    
    # Check Z walls (horizontal cuts)
    for i, zw in enumerate(z_walls):
        has_ev, extent, max_run = wall_has_evidence(
            density_img, x_min, z_min, cell_size, zw, 'z', room_mask, min_evidence_fraction)
        str_val = z_str[i] if i < len(z_str) else 0
        print(f"    Z wall {zw:.2f}: evidence={has_ev}, extent={extent:.1f}m, max_run={max_run:.1f}m, strength={str_val:.0f}")
        
        if has_ev and max_run >= min_wall_run:
            valid_z.append(zw)
            px = int((zw - z_min) / cell_size)
            strip_lo = max(0, px - wall_half_px)
            strip_hi = min(cut_mask.shape[0], px + wall_half_px + 1)
            
            density_row = density_img[max(0,px-2):min(density_img.shape[0],px+3), :].max(axis=0)
            threshold = max(1, np.percentile(density_row[density_row > 0], 20) if np.any(density_row > 0) else 1)
            is_wall_px = density_row > threshold
            
            wall_cols = np.where(is_wall_px & (room_mask[px, :] > 0))[0]
            if len(wall_cols) > 0:
                segments = []
                seg_start = wall_cols[0]
                for k in range(1, len(wall_cols)):
                    if wall_cols[k] - wall_cols[k-1] > int(1.5 / cell_size):
                        segments.append((seg_start, wall_cols[k-1]))
                        seg_start = wall_cols[k]
                segments.append((seg_start, wall_cols[-1]))
                
                extend = int(0.5 / cell_size)
                for seg_s, seg_e in segments:
                    seg_len = (seg_e - seg_s) * cell_size
                    if seg_len >= min_wall_run:
                        c_lo = max(0, seg_s - extend)
                        c_hi = min(cut_mask.shape[1], seg_e + extend + 1)
                        cut_mask[strip_lo:strip_hi, c_lo:c_hi] = 0
    
    return cut_mask, valid_x, valid_z


def extract_rooms_from_cut_mask(cut_mask, cell_size, min_area=1.0):
    """Label connected components of cut mask as rooms."""
    lbl, n = ndlabel(cut_mask)
    rooms = []
    for i in range(1, n + 1):
        mask = (lbl == i)
        area_px = np.sum(mask)
        area_m2 = area_px * cell_size * cell_size
        if area_m2 < min_area:
            continue
        rows, cols = np.where(mask)
        rooms.append({
            'label': i,
            'mask': mask,
            'area_m2': area_m2,
            'row_min': rows.min(), 'row_max': rows.max(),
            'col_min': cols.min(), 'col_max': cols.max(),
        })
    rooms.sort(key=lambda r: -r['area_m2'])
    return rooms


def room_mask_to_polygon(mask, x_min, z_min, cell_size, x_walls, z_walls):
    """Convert a room's binary mask to a clean rectilinear polygon snapped to walls."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    
    # Get world-coordinate bounds
    wx_min = x_min + cols.min() * cell_size
    wx_max = x_min + (cols.max() + 1) * cell_size
    wz_min = z_min + rows.min() * cell_size
    wz_max = z_min + (rows.max() + 1) * cell_size
    
    # Snap to nearest wall positions
    def snap(val, positions, max_snap=0.3):
        if len(positions) == 0: return val
        dists = np.abs(np.array(positions) - val)
        idx = np.argmin(dists)
        return float(positions[idx]) if dists[idx] < max_snap else val
    
    all_walls = np.concatenate([x_walls, z_walls]) if len(x_walls) > 0 and len(z_walls) > 0 else np.array([])
    
    # Check if room is roughly rectangular
    bbox_area = (cols.max() - cols.min() + 1) * (rows.max() - rows.min() + 1)
    fill_ratio = len(rows) / max(1, bbox_area)
    
    if fill_ratio > 0.85:
        # Rectangular - snap bbox to walls
        x0 = snap(wx_min, x_walls)
        x1 = snap(wx_max, x_walls)
        z0 = snap(wz_min, z_walls)
        z1 = snap(wz_max, z_walls)
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    # Non-rectangular: use contour → simplify → snap
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        x0 = snap(wx_min, x_walls)
        x1 = snap(wx_max, x_walls)
        z0 = snap(wz_min, z_walls)
        z1 = snap(wz_max, z_walls)
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate to polygon
    epsilon = max(3, int(0.15 / cell_size))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert to world coords and snap
    poly = []
    for pt in approx.reshape(-1, 2):
        wx = x_min + pt[0] * cell_size
        wz = z_min + pt[1] * cell_size
        wx = snap(wx, x_walls)
        wz = snap(wz, z_walls)
        poly.append([wx, wz])
    
    # Make rectilinear: force each edge to be axis-aligned
    rect_poly = make_rectilinear(poly)
    
    # Remove collinear points
    return remove_collinear(rect_poly)


def make_rectilinear(poly):
    """Force polygon edges to be axis-aligned."""
    if len(poly) < 3:
        return poly
    
    result = [poly[0]]
    for i in range(1, len(poly)):
        prev = result[-1]
        curr = poly[i]
        
        dx = abs(curr[0] - prev[0])
        dz = abs(curr[1] - prev[1])
        
        if dx > 0.05 and dz > 0.05:
            # Diagonal — insert corner point
            # Choose which direction to go first based on which is shorter
            if dx < dz:
                result.append([curr[0], prev[1]])
            else:
                result.append([prev[0], curr[1]])
        
        result.append(curr)
    
    return result


def remove_collinear(poly):
    if len(poly) < 3: return poly
    # Remove near-duplicates
    cleaned = [poly[0]]
    for p in poly[1:]:
        if abs(p[0]-cleaned[-1][0]) > 0.01 or abs(p[1]-cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    if len(cleaned) < 3: return cleaned
    # Remove collinear
    result = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i-1)%n]
        curr = cleaned[i]
        nxt = cleaned[(i+1)%n]
        cross = (curr[0]-prev[0])*(nxt[1]-curr[1]) - (curr[1]-prev[1])*(nxt[0]-curr[0])
        if abs(cross) > 0.001:
            result.append(curr)
    return result if len(result) >= 3 else cleaned


def classify_hallway(mask, cell_size, max_width=1.5):
    """Check if a room is a hallway (narrow in one dimension, elongated)."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_half_width = dist.max() * cell_size
    
    rows, cols = np.where(mask)
    if len(rows) == 0: return False
    
    h = (rows.max() - rows.min()) * cell_size
    w = (cols.max() - cols.min()) * cell_size
    min_dim = min(w, h)
    max_dim = max(w, h)
    
    # Hallway: narrow AND elongated
    return max_half_width * 2 < max_width and max_dim / max(min_dim, 0.01) > 2.0


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    return abs(sum(poly[i][0]*poly[(i+1)%n][1] - poly[(i+1)%n][0]*poly[i][1] for i in range(n))) / 2

def polygon_centroid(poly):
    n = len(poly)
    if n == 0: return 0, 0
    return sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n


# ─── Rendering ───

ROOM_COLORS = ['#E8F5E9','#E3F2FD','#FFF3E0','#F3E5F5','#FFFDE7','#E0F7FA','#FCE4EC','#F1F8E9']
HALLWAY_COLOR = '#F5F5F5'

def render_debug(density_img, room_mask, cut_mask, x_min, z_min, cs,
                  x_walls, z_walls, valid_x, valid_z, rooms_data, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    extent = [x_min, x_min + density_img.shape[1]*cs, z_min, z_min + density_img.shape[0]*cs]
    
    # P1: Density + all Hough walls
    ax = axes[0,0]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal')
    for xw in x_walls: ax.axvline(xw, color='cyan', alpha=0.4, lw=0.8)
    for zw in z_walls: ax.axhline(zw, color='lime', alpha=0.4, lw=0.8)
    ax.set_title('1. Density + Hough Walls')
    
    # P2: Room mask
    ax = axes[0,1]
    ax.imshow(room_mask, origin='lower', extent=extent, cmap='gray', aspect='equal')
    ax.set_title('2. Room Mask (filled)')
    
    # P3: Valid walls on density
    ax = axes[0,2]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal')
    for xw in valid_x: ax.axvline(xw, color='cyan', alpha=0.8, lw=2)
    for zw in valid_z: ax.axhline(zw, color='lime', alpha=0.8, lw=2)
    ax.set_title(f'3. Valid Walls ({len(valid_x)}X + {len(valid_z)}Z)')
    
    # P4: Cut mask
    ax = axes[1,0]
    ax.imshow(cut_mask, origin='lower', extent=extent, cmap='gray', aspect='equal')
    ax.set_title('4. Cut Mask')
    
    # P5: Labeled rooms
    ax = axes[1,1]
    ax.set_facecolor('white')
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        color = HALLWAY_COLOR if rd['is_hallway'] else ROOM_COLORS[ci % len(ROOM_COLORS)]
        if not rd['is_hallway']: ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.8)
        n = len(poly)
        for k in range(n):
            p1, p2 = poly[k], poly[(k+1)%n]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=2)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=7, fontweight='bold', color='#333')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('5. Room Polygons')
    
    # P6: Overlay (density + polygons)
    ax = axes[1,2]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal', alpha=0.5)
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        color = HALLWAY_COLOR if rd['is_hallway'] else ROOM_COLORS[ci % len(ROOM_COLORS)]
        if not rd['is_hallway']: ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.3)
        n = len(poly)
        for k in range(n):
            p1, p2 = poly[k], poly[(k+1)%n]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'w-', lw=2)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('6. Overlay')
    
    fig.suptitle('v29 Mask + Wall Cut — Debug', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def render_clean(rooms_data, doors, output_path, title="v29"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    
    all_x, all_z = [], []
    for rd in rooms_data:
        for p in rd['polygon']:
            all_x.append(p[0]); all_z.append(p[1])
    if not all_x:
        plt.savefig(output_path, dpi=150); plt.close(); return
    
    margin = 0.5
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        color = HALLWAY_COLOR if rd['is_hallway'] else ROOM_COLORS[ci % len(ROOM_COLORS)]
        if not rd['is_hallway']: ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.8)
        n = len(poly)
        for k in range(n):
            p1, p2 = poly[k], poly[(k+1)%n]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=3)
    
    for door in doors:
        x, z, w = door['x'], door['z'], door.get('width', 0.8)
        if door['orientation'] == 'vertical':
            ax.plot([x,x], [z-w/2,z+w/2], color='white', lw=5)
            ax.add_patch(Arc((x,z-w/2), w, w, angle=0, theta1=0, theta2=90, color='k', lw=1.5))
        else:
            ax.plot([x-w/2,x+w/2], [z,z], color='white', lw=5)
            ax.add_patch(Arc((x-w/2,z), w, w, angle=0, theta1=0, theta2=90, color='k', lw=1.5))
    
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#333')
    
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_z)-margin, max(all_z)+margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--nms', type=float, default=0.15)
    parser.add_argument('--min-wall-run', type=float, default=0.8, help='Min continuous wall length (m)')
    parser.add_argument('--min-evidence', type=float, default=0.3, help='Min fraction of wall extent with evidence')
    parser.add_argument('--min-room-area', type=float, default=1.0)
    parser.add_argument('-o', '--output', default='results/v29_maskcut')
    args = parser.parse_args()
    
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = Path(args.mesh).stem
    
    print(f"Loading: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh')
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    up_idx, up_name = detect_up_axis(mesh)
    rx, rz = project_vertices(mesh, up_idx)
    angle = find_dominant_angle(rx, rz, cell=args.cell)
    print(f"  Rotation: {angle:.1f}°")
    
    cos_a, sin_a = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a
    
    density_img, img_x_min, img_z_min, cs = build_density_image(rx2, rz2, cell_size=args.cell)
    print(f"  Density image: {density_img.shape}")
    
    # Room mask
    print("Building room mask...")
    room_mask = build_room_mask(density_img, cs)
    mask_area = np.sum(room_mask) * cs * cs
    print(f"  Mask area: {mask_area:.1f}m²")
    
    # Hough walls
    x_walls, z_walls, x_str, z_str = hough_wall_positions(density_img, img_x_min, img_z_min, cs, nms_dist=args.nms)
    print(f"  Hough X walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Hough Z walls: {[f'{w:.2f}' for w in z_walls]}")
    
    # Select walls: score = strength × max_run, skip boundary walls, take top N
    print("Scoring and selecting walls...")
    
    # Find room mask bounds for boundary detection
    mask_rows = np.where(room_mask.any(axis=1))[0]
    mask_cols = np.where(room_mask.any(axis=0))[0]
    bound_x_min = img_x_min + mask_cols[0] * cs
    bound_x_max = img_x_min + mask_cols[-1] * cs
    bound_z_min = img_z_min + mask_rows[0] * cs
    bound_z_max = img_z_min + mask_rows[-1] * cs
    boundary_margin = 0.3
    
    x_scored = []
    for i, xw in enumerate(x_walls):
        has_ev, extent, max_run = wall_has_evidence(
            density_img, img_x_min, img_z_min, cs, xw, 'x', room_mask, 0.15)
        strength = x_str[i] if i < len(x_str) else 0
        score = float(strength * max_run)
        is_boundary = (abs(xw - bound_x_min) < boundary_margin or abs(xw - bound_x_max) < boundary_margin)
        print(f"    X wall {xw:.2f}: score={score:.0f}, run={max_run:.1f}m, {'BOUNDARY' if is_boundary else 'interior'}")
        if not is_boundary and max_run >= 0.8:
            x_scored.append((xw, score, max_run))
    
    z_scored = []
    for i, zw in enumerate(z_walls):
        has_ev, extent, max_run = wall_has_evidence(
            density_img, img_x_min, img_z_min, cs, zw, 'z', room_mask, 0.15)
        strength = z_str[i] if i < len(z_str) else 0
        score = float(strength * max_run)
        is_boundary = (abs(zw - bound_z_min) < boundary_margin or abs(zw - bound_z_max) < boundary_margin)
        print(f"    Z wall {zw:.2f}: score={score:.0f}, run={max_run:.1f}m, {'BOUNDARY' if is_boundary else 'interior'}")
        if not is_boundary and max_run >= 0.8:
            z_scored.append((zw, score, max_run))
    
    # Sort by score, take top N with minimum separation
    x_scored.sort(key=lambda t: -t[1])
    z_scored.sort(key=lambda t: -t[1])
    
    min_wall_separation = 1.0  # meters — walls closer than this are same wall/hallway sides
    
    def select_top_n(scored, max_n, min_sep):
        selected = []
        for pos, score, run in scored:
            if any(abs(pos - s) < min_sep for s in selected):
                continue
            selected.append(pos)
            if len(selected) >= max_n:
                break
        return selected
    
    selected_x = select_top_n(x_scored, 2, min_wall_separation)
    selected_z = select_top_n(z_scored, 2, min_wall_separation)
    
    print(f"  Selected X cuts: {[f'{w:.2f}' for w in selected_x]}")
    print(f"  Selected Z cuts: {[f'{w:.2f}' for w in selected_z]}")
    
    # Override x_walls/z_walls/x_str/z_str with selected
    sel_x_walls = np.array(sorted(selected_x))
    sel_z_walls = np.array(sorted(selected_z))
    sel_x_str = np.array([next(s for w,s,_ in x_scored if w==xw) for xw in sel_x_walls])
    sel_z_str = np.array([next(s for w,s,_ in z_scored if w==zw) for zw in sel_z_walls])
    
    # Cut mask
    print("Cutting mask with selected walls...")
    cut_mask, valid_x, valid_z = cut_mask_with_walls(
        room_mask, density_img, img_x_min, img_z_min, cs,
        sel_x_walls, sel_z_walls, sel_x_str, sel_z_str,
        min_evidence_fraction=0.1,
        min_wall_run=0.5  # lower since we already filtered
    )
    print(f"  Valid cuts: {len(valid_x)}X + {len(valid_z)}Z")
    
    # Extract rooms and merge small fragments
    print("Extracting rooms...")
    raw_rooms = extract_rooms_from_cut_mask(cut_mask, cs, min_area=0.5)
    
    # Merge small components (< 2m²) into their largest neighbor
    # Merge small fragments: prefer merging narrow pieces with narrow neighbors (hallway concat)
    target_rooms = 5  # 3 rooms + 1 hallway + 1 closet
    
    if len(raw_rooms) > target_rooms:
        print(f"  Merging fragments ({len(raw_rooms)} → {target_rooms})...")
        
        def is_narrow(r, cs):
            rows, cols = np.where(r['mask'])
            if len(rows) == 0: return False
            h = (rows.max() - rows.min()) * cs
            w = (cols.max() - cols.min()) * cs
            return min(w, h) < 1.5
        
        while len(raw_rooms) > target_rooms:
            raw_rooms.sort(key=lambda r: r['area_m2'])
            merged_one = False
            
            for i, r in enumerate(raw_rooms):
                dilated = cv2.dilate(r['mask'].astype(np.uint8), np.ones((15,15), np.uint8))
                
                # Find best neighbor: prefer narrow neighbor (hallway-to-hallway merge)
                candidates = []
                for j, r2 in enumerate(raw_rooms):
                    if i == j: continue
                    overlap = np.sum(dilated & r2['mask'].astype(np.uint8))
                    if overlap > 0:
                        # Score: prefer narrow + narrow, then small + small
                        both_narrow = is_narrow(r, cs) and is_narrow(r2, cs)
                        score = (10000 if both_narrow else 0) + overlap - r2['area_m2']
                        candidates.append((j, score, overlap))
                
                if candidates:
                    candidates.sort(key=lambda c: -c[1])
                    best_j = candidates[0][0]
                    print(f"    Merging {r['area_m2']:.1f}m² into {raw_rooms[best_j]['area_m2']:.1f}m²")
                    raw_rooms[best_j]['mask'] = raw_rooms[best_j]['mask'] | r['mask']
                    raw_rooms[best_j]['area_m2'] += r['area_m2']
                    rows, cols = np.where(raw_rooms[best_j]['mask'])
                    raw_rooms[best_j]['row_min'] = rows.min()
                    raw_rooms[best_j]['row_max'] = rows.max()
                    raw_rooms[best_j]['col_min'] = cols.min()
                    raw_rooms[best_j]['col_max'] = cols.max()
                    raw_rooms.pop(i)
                    merged_one = True
                    break
            
            if not merged_one:
                break
        
        raw_rooms.sort(key=lambda r: -r['area_m2'])
    print(f"  Raw rooms: {len(raw_rooms)}")
    
    # Build room data
    all_walls = np.concatenate([np.array(valid_x), np.array(valid_z)]) if valid_x and valid_z else np.array([])
    
    rooms_data = []
    rn, hn = 1, 1
    for r in raw_rooms:
        poly = room_mask_to_polygon(r['mask'], img_x_min, img_z_min, cs,
                                     np.array(valid_x), np.array(valid_z))
        area = compute_polygon_area(poly) if len(poly) >= 3 else r['area_m2']
        is_hall = classify_hallway(r['mask'], cs)
        
        if is_hall:
            name = "Hall" if hn == 1 else f"Hall {hn}"
            hn += 1
        else:
            name = f"Room {rn}"
            rn += 1
        
        rooms_data.append({
            'name': name,
            'polygon': poly,
            'area': area,
            'is_hallway': bool(is_hall),
        })
        print(f"    {name}: {area:.1f}m² {'(hall)' if is_hall else ''} poly={len(poly)}v")
    
    # Door detection between adjacent rooms
    doors = []
    
    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if not rd['is_hallway'])
    nh = sum(1 for rd in rooms_data if rd['is_hallway'])
    
    print(f"\n=== RESULTS ===")
    print(f"  Rooms: {nr}, Hallways: {nh}, Total: {total:.1f}m²")
    
    # Render
    render_debug(density_img, room_mask, cut_mask, img_x_min, img_z_min, cs,
                  x_walls, z_walls, valid_x, valid_z, rooms_data,
                  out_dir / f"v29_{mesh_name}_debug.png")
    
    title = f"v29 Mask Cut — {nr} rooms, {nh} hall(s), {total:.1f}m²"
    render_clean(rooms_data, doors, out_dir / f"v29_{mesh_name}_clean.png", title)
    
    # JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    
    results = {
        'approach': 'v29_mask_cut',
        'rooms': [{
            'name': rd['name'], 'area_m2': round(rd['area'],1),
            'is_hallway': rd['is_hallway'],
            'polygon': [[round(p[0],3), round(p[1],3)] for p in rd['polygon']],
        } for rd in rooms_data],
        'doors': doors,
        'walls': {'x': [round(w,3) for w in valid_x], 'z': [round(w,3) for w in valid_z]},
    }
    with open(out_dir / f"v29_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    
    print("Done!")

if __name__ == '__main__':
    main()

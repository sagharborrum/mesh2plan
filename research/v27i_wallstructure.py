#!/usr/bin/env python3
"""
mesh2plan v27i - Wall-Structure Room Detection

Strategy: Instead of watershed, identify rooms from wall structure directly.
1. Standard pipeline: project vertices, rotate, build density, Hough walls
2. Build room mask
3. Wall-structure room detection:
   a. Wall evidence: sample density along each Hough wall line, score how "real" each wall is
   b. Hallway detection: find parallel wall pairs (0.8-1.5m apart) with strong evidence
   c. Room detection: hallway walls divide space; cross-walls split into rooms
   d. Closet detection: small enclosed regions (<4m²)
4. Clean rectilinear polygons from wall positions
5. Opening detection: gaps in wall density = doors

Two outputs:
  - 5-panel research view (density, wall evidence, segmentation, polygons, floor plan)
  - Clean architectural plan (white bg, thick walls, door arcs, pastel fills)
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
from scipy.ndimage import maximum_filter, label as ndlabel


# ─── Shared utilities ───

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
    nz_img, nx_img = density_img.shape
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)
    proj_x = smoothed.sum(axis=0)
    proj_z = smoothed.sum(axis=1)

    def find_peaks_nms(profile, origin, cs, min_dist_cells):
        from scipy.ndimage import uniform_filter1d
        prof = uniform_filter1d(profile.astype(float), size=5)
        local_max = maximum_filter(prof, size=max(3, min_dist_cells)) == prof
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


# ─── Wall evidence scoring ───

def compute_wall_evidence(density_img, x_min, z_min, cell_size, x_walls, z_walls, room_mask):
    """For each wall, compute how much density evidence exists along its length within the room mask.
    Returns dict with wall position -> (evidence_score, extent_lo, extent_hi) in world coords."""
    nz, nx = density_img.shape
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)
    strip_half = max(2, int(0.06 / cell_size))  # ±6cm strip
    
    x_evidence = {}
    for xw in x_walls:
        col = int((xw - x_min) / cell_size)
        if col < 0 or col >= nx:
            x_evidence[xw] = (0, 0, 0, np.array([]))
            continue
        c_lo = max(0, col - strip_half)
        c_hi = min(nx, col + strip_half + 1)
        strip = smoothed[:, c_lo:c_hi].sum(axis=1)  # profile along Z
        # Only consider rows inside room mask
        mask_col = room_mask[:, c_lo:c_hi].max(axis=1)
        strip_masked = strip * mask_col
        
        # Find extent where wall has density
        nonzero = np.where(strip_masked > 0)[0]
        if len(nonzero) == 0:
            x_evidence[xw] = (0, 0, 0, strip_masked)
            continue
        z_lo_w = z_min + nonzero[0] * cell_size
        z_hi_w = z_min + nonzero[-1] * cell_size
        extent = z_hi_w - z_lo_w
        # Evidence = fraction of extent that has significant density
        in_range = strip_masked[nonzero[0]:nonzero[-1]+1]
        if len(in_range) == 0:
            x_evidence[xw] = (0, z_lo_w, z_hi_w, strip_masked)
            continue
        thresh = np.percentile(in_range[in_range > 0], 20) if np.any(in_range > 0) else 0
        frac = np.sum(in_range > thresh) / len(in_range) if len(in_range) > 0 else 0
        score = frac * extent  # longer walls with more coverage score higher
        x_evidence[xw] = (score, z_lo_w, z_hi_w, strip_masked)
    
    z_evidence = {}
    for zw in z_walls:
        row = int((zw - z_min) / cell_size)
        if row < 0 or row >= nz:
            z_evidence[zw] = (0, 0, 0, np.array([]))
            continue
        r_lo = max(0, row - strip_half)
        r_hi = min(nz, row + strip_half + 1)
        strip = smoothed[r_lo:r_hi, :].sum(axis=0)  # profile along X
        mask_row = room_mask[r_lo:r_hi, :].max(axis=0)
        strip_masked = strip * mask_row
        
        nonzero = np.where(strip_masked > 0)[0]
        if len(nonzero) == 0:
            z_evidence[zw] = (0, 0, 0, strip_masked)
            continue
        x_lo_w = x_min + nonzero[0] * cell_size
        x_hi_w = x_min + nonzero[-1] * cell_size
        extent = x_hi_w - x_lo_w
        in_range = strip_masked[nonzero[0]:nonzero[-1]+1]
        if len(in_range) == 0:
            z_evidence[zw] = (0, x_lo_w, x_hi_w, strip_masked)
            continue
        thresh = np.percentile(in_range[in_range > 0], 20) if np.any(in_range > 0) else 0
        frac = np.sum(in_range > thresh) / len(in_range) if len(in_range) > 0 else 0
        score = frac * extent
        z_evidence[zw] = (score, x_lo_w, x_hi_w, strip_masked)
    
    return x_evidence, z_evidence


# ─── Hallway detection ───

def detect_hallway(x_walls, z_walls, x_evidence, z_evidence, room_mask, x_min, z_min, cell_size,
                   min_gap=0.8, max_gap=1.8, min_evidence=1.0):
    """Find the hallway by looking for pairs of parallel walls that are close together
    and both have strong evidence. The hallway is the narrow corridor between them."""
    
    candidates = []
    
    # Check X-wall pairs (vertical hallway = corridor running in Z direction)
    for i, xw1 in enumerate(x_walls):
        ev1 = x_evidence.get(xw1, (0,))[0]
        if ev1 < min_evidence:
            continue
        for xw2 in x_walls[i+1:]:
            gap = xw2 - xw1
            if gap < min_gap or gap > max_gap:
                continue
            ev2 = x_evidence.get(xw2, (0,))[0]
            if ev2 < min_evidence:
                continue
            # Both walls are strong and close together — this is a hallway candidate
            score = (ev1 + ev2) * gap  # prefer wider, better-evidenced hallways
            
            # Hallway must be elongated: length >> width
            _, z_lo1, z_hi1, _ = x_evidence[xw1]
            _, z_lo2, z_hi2, _ = x_evidence[xw2]
            z_lo = max(z_lo1, z_lo2)
            z_hi = min(z_hi1, z_hi2)
            length = z_hi - z_lo
            aspect = length / gap if gap > 0 else 0
            if aspect < 2.0:  # Must be at least 2x longer than wide
                continue
            score *= aspect  # Bonus for elongated corridors
            
            candidates.append({
                    'type': 'vertical',
                    'x_min': xw1, 'x_max': xw2,
                    'z_min': z_lo, 'z_max': z_hi,
                    'score': score,
                })
    
    # Check Z-wall pairs (horizontal hallway = corridor running in X direction)
    for i, zw1 in enumerate(z_walls):
        ev1 = z_evidence.get(zw1, (0,))[0]
        if ev1 < min_evidence:
            continue
        for zw2 in z_walls[i+1:]:
            gap = zw2 - zw1
            if gap < min_gap or gap > max_gap:
                continue
            ev2 = z_evidence.get(zw2, (0,))[0]
            if ev2 < min_evidence:
                continue
            score = (ev1 + ev2) * gap
            r1 = int((zw1 - z_min) / cell_size)
            r2 = int((zw2 - z_min) / cell_size)
            if r1 >= 0 and r2 < room_mask.shape[0]:
                corridor_mask = room_mask[max(0,r1):min(room_mask.shape[0],r2+1), :]
                coverage = np.sum(corridor_mask) / max(1, corridor_mask.size)
                if coverage < 0.15:
                    continue
                score *= coverage
            
            # Hallway must be elongated
            _, x_lo1, x_hi1, _ = z_evidence[zw1]
            _, x_lo2, x_hi2, _ = z_evidence[zw2]
            x_lo = max(x_lo1, x_lo2)
            x_hi = min(x_hi1, x_hi2)
            length = x_hi - x_lo
            aspect = length / gap if gap > 0 else 0
            if aspect < 2.0:
                continue
            score *= aspect
            
            candidates.append({
                    'type': 'horizontal',
                    'x_min': x_lo, 'x_max': x_hi,
                    'z_min': zw1, 'z_max': zw2,
                    'score': score,
                })
    
    # Sort candidates by score descending, validate each
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    best_hallway = None
    for hw in candidates:
        rows_m, cols_m = np.where(room_mask > 0)
        if len(rows_m) > 0:
            mask_x_min = x_min + cols_m.min() * cell_size
            mask_x_max = x_min + cols_m.max() * cell_size
            mask_z_min = z_min + rows_m.min() * cell_size
            mask_z_max = z_min + rows_m.max() * cell_size
            
            # Hallway should leave significant room on both sides
            if hw['type'] == 'vertical':
                left_space = hw['x_min'] - mask_x_min
                right_space = mask_x_max - hw['x_max']
                if left_space < 1.0 or right_space < 1.0:
                    print(f"    Vertical hallway rejected: not enough room on sides ({left_space:.1f}m left, {right_space:.1f}m right)")
                    continue
            else:
                below_space = hw['z_min'] - mask_z_min
                above_space = mask_z_max - hw['z_max']
                if below_space < 1.0 or above_space < 1.0:
                    print(f"    Horizontal hallway rejected: not enough room on sides ({below_space:.1f}m below, {above_space:.1f}m above)")
                    continue
    
        # Passed validation
        best_hallway = hw
        break
    
    return best_hallway


# ─── Room detection from wall structure ───

def snap_to_nearest(val, positions, max_snap=0.4):
    if len(positions) == 0:
        return val
    dists = np.abs(np.array(positions) - val)
    idx = np.argmin(dists)
    if dists[idx] < max_snap:
        return float(positions[idx])
    return val


def detect_rooms_from_walls(x_walls, z_walls, x_evidence, z_evidence, hallway,
                            room_mask, x_min, z_min, cell_size):
    """Detect rooms by using the hallway as organizing feature and walls as dividers."""
    rooms = []
    nz, nx = room_mask.shape
    
    # Get overall extent from room mask
    rows, cols = np.where(room_mask > 0)
    if len(rows) == 0:
        return rooms
    
    mask_x_min = x_min + cols.min() * cell_size
    mask_x_max = x_min + cols.max() * cell_size
    mask_z_min = z_min + rows.min() * cell_size
    mask_z_max = z_min + rows.max() * cell_size
    
    # Snap mask extents to nearest walls
    ext_x_min = snap_to_nearest(mask_x_min, x_walls, 0.3)
    ext_x_max = snap_to_nearest(mask_x_max, x_walls, 0.3)
    ext_z_min = snap_to_nearest(mask_z_min, z_walls, 0.3)
    ext_z_max = snap_to_nearest(mask_z_max, z_walls, 0.3)
    
    if hallway is None:
        # No hallway found — single room
        rooms.append({
            'polygon': [[ext_x_min, ext_z_min], [ext_x_max, ext_z_min],
                        [ext_x_max, ext_z_max], [ext_x_min, ext_z_max]],
            'name': 'Room 1',
            'is_hallway': False,
        })
        return rooms
    
    # Add hallway
    hw = hallway
    rooms.append({
        'polygon': [[hw['x_min'], hw['z_min']], [hw['x_max'], hw['z_min']],
                    [hw['x_max'], hw['z_max']], [hw['x_min'], hw['z_max']]],
        'name': 'Hallway',
        'is_hallway': True,
    })
    
    if hallway['type'] == 'vertical':
        # Hallway runs vertically. Rooms are to the left and right.
        hall_x_min = hw['x_min']
        hall_x_max = hw['x_max']
        
        # LEFT SIDE: rooms from ext_x_min to hall_x_min
        # Find Z-walls that have evidence in the left region
        left_z_dividers = _find_dividing_walls(
            z_walls, z_evidence, x_evidence,
            x_range=(ext_x_min, hall_x_min),
            z_range=(ext_z_min, ext_z_max),
            axis='z', x_min=x_min, z_min=z_min, cell_size=cell_size,
            room_mask=room_mask, min_evidence=3.5
        )
        left_z_dividers = sorted(set([ext_z_min] + left_z_dividers + [ext_z_max]))
        
        # Also check for X-wall dividers within left side (for closets etc)
        left_x_dividers = _find_dividing_walls(
            x_walls, x_evidence, z_evidence,
            x_range=(ext_x_min, hall_x_min),
            z_range=(ext_z_min, ext_z_max),
            axis='x', x_min=x_min, z_min=z_min, cell_size=cell_size,
            room_mask=room_mask, min_evidence=3.5
        )
        # Only keep x-dividers that are interior (not the exterior walls or hallway walls)
        left_x_dividers = [xw for xw in left_x_dividers 
                          if abs(xw - ext_x_min) > 0.3 and abs(xw - hall_x_min) > 0.3]
        
        # Generate left rooms
        for i in range(len(left_z_dividers) - 1):
            z0, z1 = left_z_dividers[i], left_z_dividers[i+1]
            if z1 - z0 < 0.3:
                continue
            
            # Check if there's an x-divider that splits this row
            active_x_divs = [xd for xd in left_x_dividers 
                           if _wall_has_evidence_in_range(xd, z0, z1, x_evidence, z_min, cell_size)]
            
            if active_x_divs:
                # Split by x-divider
                x_bounds = sorted([ext_x_min] + active_x_divs + [hall_x_min])
                for j in range(len(x_bounds) - 1):
                    x0, x1 = x_bounds[j], x_bounds[j+1]
                    if _region_has_mask_coverage(x0, x1, z0, z1, room_mask, x_min, z_min, cell_size, 0.2):
                        rooms.append({
                            'polygon': [[x0, z0], [x1, z0], [x1, z1], [x0, z1]],
                            'name': '',
                            'is_hallway': False,
                        })
            else:
                if _region_has_mask_coverage(ext_x_min, hall_x_min, z0, z1, room_mask, x_min, z_min, cell_size, 0.2):
                    rooms.append({
                        'polygon': [[ext_x_min, z0], [hall_x_min, z0],
                                    [hall_x_min, z1], [ext_x_min, z1]],
                        'name': '',
                        'is_hallway': False,
                    })
        
        # RIGHT SIDE: rooms from hall_x_max to ext_x_max
        right_z_dividers = _find_dividing_walls(
            z_walls, z_evidence, x_evidence,
            x_range=(hall_x_max, ext_x_max),
            z_range=(ext_z_min, ext_z_max),
            axis='z', x_min=x_min, z_min=z_min, cell_size=cell_size,
            room_mask=room_mask, min_evidence=3.5
        )
        right_z_dividers = sorted(set([ext_z_min] + right_z_dividers + [ext_z_max]))
        
        for i in range(len(right_z_dividers) - 1):
            z0, z1 = right_z_dividers[i], right_z_dividers[i+1]
            if z1 - z0 < 0.3:
                continue
            if _region_has_mask_coverage(hall_x_max, ext_x_max, z0, z1, room_mask, x_min, z_min, cell_size, 0.2):
                rooms.append({
                    'polygon': [[hall_x_max, z0], [ext_x_max, z0],
                                [ext_x_max, z1], [hall_x_max, z1]],
                    'name': '',
                    'is_hallway': False,
                })
    
    else:
        # Horizontal hallway — rooms above and below
        hall_z_min = hw['z_min']
        hall_z_max = hw['z_max']
        
        # BELOW
        below_x_dividers = _find_dividing_walls(
            x_walls, x_evidence, z_evidence,
            x_range=(ext_x_min, ext_x_max),
            z_range=(ext_z_min, hall_z_min),
            axis='x', x_min=x_min, z_min=z_min, cell_size=cell_size,
            room_mask=room_mask, min_evidence=3.5
        )
        below_x_dividers = sorted(set([ext_x_min] + below_x_dividers + [ext_x_max]))
        
        for i in range(len(below_x_dividers) - 1):
            x0, x1 = below_x_dividers[i], below_x_dividers[i+1]
            if x1 - x0 < 0.3:
                continue
            if _region_has_mask_coverage(x0, x1, ext_z_min, hall_z_min, room_mask, x_min, z_min, cell_size, 0.2):
                rooms.append({
                    'polygon': [[x0, ext_z_min], [x1, ext_z_min],
                                [x1, hall_z_min], [x0, hall_z_min]],
                    'name': '',
                    'is_hallway': False,
                })
        
        # ABOVE
        above_x_dividers = _find_dividing_walls(
            x_walls, x_evidence, z_evidence,
            x_range=(ext_x_min, ext_x_max),
            z_range=(hall_z_max, ext_z_max),
            axis='x', x_min=x_min, z_min=z_min, cell_size=cell_size,
            room_mask=room_mask, min_evidence=3.5
        )
        above_x_dividers = sorted(set([ext_x_min] + above_x_dividers + [ext_x_max]))
        
        for i in range(len(above_x_dividers) - 1):
            x0, x1 = above_x_dividers[i], above_x_dividers[i+1]
            if x1 - x0 < 0.3:
                continue
            if _region_has_mask_coverage(x0, x1, hall_z_max, ext_z_max, room_mask, x_min, z_min, cell_size, 0.2):
                rooms.append({
                    'polygon': [[x0, hall_z_max], [x1, hall_z_max],
                                [x1, ext_z_max], [x0, ext_z_max]],
                    'name': '',
                    'is_hallway': False,
                })
    
    # Name rooms by size and detect closets
    non_hall = [r for r in rooms if not r['is_hallway']]
    non_hall.sort(key=lambda r: compute_polygon_area(r['polygon']), reverse=True)
    
    room_num = 0
    for r in non_hall:
        area = compute_polygon_area(r['polygon'])
        if area < 4.0:
            r['name'] = 'Closet'
        else:
            room_num += 1
            r['name'] = f'Room {room_num}'
    
    return rooms


def _wall_has_evidence_in_range(x_wall_pos, z_lo, z_hi, x_evidence, z_min, cell_size):
    """Check if an X-wall has density evidence in a specific Z range."""
    ev = x_evidence.get(x_wall_pos)
    if ev is None or ev[0] < 0.3:
        return False
    profile = ev[3]
    if len(profile) == 0:
        return False
    r_lo = int((z_lo - z_min) / cell_size)
    r_hi = int((z_hi - z_min) / cell_size)
    r_lo = max(0, r_lo)
    r_hi = min(len(profile), r_hi)
    if r_hi <= r_lo:
        return False
    segment = profile[r_lo:r_hi]
    return np.sum(segment > 0) / len(segment) > 0.2


def _find_dividing_walls(walls, wall_evidence, cross_evidence,
                         x_range, z_range, axis,
                         x_min, z_min, cell_size, room_mask, min_evidence=3.5):
    """Find walls that act as dividers within a given region.
    axis='z' means we're looking for Z-walls that divide a region horizontally.
    axis='x' means we're looking for X-walls that divide a region vertically."""
    dividers = []
    
    x_lo, x_hi = x_range
    z_lo, z_hi = z_range
    
    for w in walls:
        ev = wall_evidence.get(w, (0, 0, 0, np.array([])))
        score = ev[0]
        
        if axis == 'z':
            # Z-wall: check if it's within z_range and spans across x_range
            if w <= z_lo + 0.2 or w >= z_hi - 0.2:
                continue
            if score < min_evidence:
                continue
            # Check profile in x_range
            profile = ev[3]
            if len(profile) == 0:
                continue
            c_lo = int((x_lo - x_min) / cell_size)
            c_hi = int((x_hi - x_min) / cell_size)
            c_lo = max(0, c_lo)
            c_hi = min(len(profile), c_hi)
            if c_hi <= c_lo:
                continue
            segment = profile[c_lo:c_hi]
            coverage = np.sum(segment > 0) / len(segment) if len(segment) > 0 else 0
            if coverage > 0.5:
                dividers.append(w)
        
        elif axis == 'x':
            if w <= x_lo + 0.2 or w >= x_hi - 0.2:
                continue
            if score < min_evidence:
                continue
            profile = ev[3]
            if len(profile) == 0:
                continue
            r_lo = int((z_lo - z_min) / cell_size)
            r_hi = int((z_hi - z_min) / cell_size)
            r_lo = max(0, r_lo)
            r_hi = min(len(profile), r_hi)
            if r_hi <= r_lo:
                continue
            segment = profile[r_lo:r_hi]
            coverage = np.sum(segment > 0) / len(segment) if len(segment) > 0 else 0
            if coverage > 0.5:
                dividers.append(w)
    
    return dividers


def _region_has_mask_coverage(x0, x1, z0, z1, room_mask, x_min, z_min, cell_size, min_coverage=0.2):
    """Check if a rectangular region has sufficient room mask coverage."""
    c0 = int((x0 - x_min) / cell_size)
    c1 = int((x1 - x_min) / cell_size)
    r0 = int((z0 - z_min) / cell_size)
    r1 = int((z1 - z_min) / cell_size)
    nz, nx = room_mask.shape
    c0, c1 = max(0, c0), min(nx, c1)
    r0, r1 = max(0, r0), min(nz, r1)
    if c1 <= c0 or r1 <= r0:
        return False
    region = room_mask[r0:r1, c0:c1]
    return np.sum(region) / max(1, region.size) > min_coverage


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3:
        return 0
    area = sum(poly[i][0] * poly[(i+1)%n][1] - poly[(i+1)%n][0] * poly[i][1] for i in range(n))
    return abs(area) / 2


def polygon_centroid(poly):
    n = len(poly)
    if n == 0:
        return 0, 0
    return sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n


# ─── Door detection ───

def find_shared_wall_segments(poly1, poly2, tolerance=0.15):
    segments = []
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i+1)%n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j+1)%n2]
            if abs(a1[0]-a2[0]) < 0.05 and abs(b1[0]-b2[0]) < 0.05:
                if abs(a1[0]-b1[0]) < tolerance:
                    a_lo, a_hi = min(a1[1],a2[1]), max(a1[1],a2[1])
                    b_lo, b_hi = min(b1[1],b2[1]), max(b1[1],b2[1])
                    olo, ohi = max(a_lo,b_lo), min(a_hi,b_hi)
                    if ohi - olo > 0.2:
                        segments.append({'type':'vertical','x':(a1[0]+b1[0])/2,'z_min':olo,'z_max':ohi})
            if abs(a1[1]-a2[1]) < 0.05 and abs(b1[1]-b2[1]) < 0.05:
                if abs(a1[1]-b1[1]) < tolerance:
                    a_lo, a_hi = min(a1[0],a2[0]), max(a1[0],a2[0])
                    b_lo, b_hi = min(b1[0],b2[0]), max(b1[0],b2[0])
                    olo, ohi = max(a_lo,b_lo), min(a_hi,b_hi)
                    if ohi - olo > 0.2:
                        segments.append({'type':'horizontal','z':(a1[1]+b1[1])/2,'x_min':olo,'x_max':ohi})
    return segments


def detect_doors_from_density(density_img, x_min, z_min, cell_size, wall_segments):
    doors = []
    for seg in wall_segments:
        if seg['type'] == 'vertical':
            x_px = int((seg['x'] - x_min) / cell_size)
            z_lo_px = int((seg['z_min'] - z_min) / cell_size)
            z_hi_px = int((seg['z_max'] - z_min) / cell_size)
            if x_px < 0 or x_px >= density_img.shape[1]:
                continue
            sw = max(1, int(0.1 / cell_size))
            x_lo = max(0, x_px - sw)
            x_hi = min(density_img.shape[1], x_px + sw + 1)
            wp = density_img[z_lo_px:z_hi_px, x_lo:x_hi].sum(axis=1)
            if len(wp) == 0:
                continue
            threshold = np.percentile(wp[wp>0], 30) if np.any(wp>0) else 0
            is_gap = wp < max(threshold, 1)
            gl, ng = ndimage.label(is_gap)
            for g in range(1, ng+1):
                gr = np.where(gl==g)[0]
                glen = len(gr) * cell_size
                if 0.6 < glen < 1.5:
                    doors.append({'x':seg['x'],'z':z_min+(z_lo_px+gr.mean())*cell_size,
                                 'width':glen,'orientation':'vertical'})
        elif seg['type'] == 'horizontal':
            z_px = int((seg['z'] - z_min) / cell_size)
            x_lo_px = int((seg['x_min'] - x_min) / cell_size)
            x_hi_px = int((seg['x_max'] - x_min) / cell_size)
            if z_px < 0 or z_px >= density_img.shape[0]:
                continue
            sw = max(1, int(0.1 / cell_size))
            z_lo = max(0, z_px - sw)
            z_hi = min(density_img.shape[0], z_px + sw + 1)
            wp = density_img[z_lo:z_hi, x_lo_px:x_hi_px].sum(axis=0)
            if len(wp) == 0:
                continue
            threshold = np.percentile(wp[wp>0], 30) if np.any(wp>0) else 0
            is_gap = wp < max(threshold, 1)
            gl, ng = ndimage.label(is_gap)
            for g in range(1, ng+1):
                gc = np.where(gl==g)[0]
                glen = len(gc) * cell_size
                if 0.6 < glen < 1.5:
                    doors.append({'x':x_min+(x_lo_px+gc.mean())*cell_size,'z':seg['z'],
                                 'width':glen,'orientation':'horizontal'})
    return doors


# ─── Colors ───

ROOM_COLORS = ['#4A90D9','#E8834A','#67B868','#C75B8F','#8B6CC1',
               '#D4A843','#4ABFBF','#D96060','#7B8FD4','#A0C75B']
PASTEL_FILLS = ['#E8F0FE','#FFF3E0','#E8F5E9','#FCE4EC','#EDE7F6',
                '#FFF8E1','#E0F7FA','#FFEBEE']


# ─── Main analysis ───

def analyze_mesh(mesh_file):
    print(f"\n{'='*60}")
    print(f"v27i: Loading mesh: {mesh_file}")
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

    print("  Detecting wall positions...")
    x_walls, z_walls = hough_wall_positions(density_img, img_x_min, img_z_min, cs)
    print(f"    X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"    Z-walls: {[f'{w:.2f}' for w in z_walls]}")

    print("  Computing wall evidence...")
    x_evidence, z_evidence = compute_wall_evidence(
        density_img, img_x_min, img_z_min, cs, x_walls, z_walls, room_mask)
    
    for xw in x_walls:
        ev = x_evidence[xw]
        print(f"    X={xw:.2f}: score={ev[0]:.2f}, extent=[{ev[1]:.2f}, {ev[2]:.2f}]")
    for zw in z_walls:
        ev = z_evidence[zw]
        print(f"    Z={zw:.2f}: score={ev[0]:.2f}, extent=[{ev[1]:.2f}, {ev[2]:.2f}]")

    print("  Detecting hallway...")
    hallway = detect_hallway(x_walls, z_walls, x_evidence, z_evidence,
                            room_mask, img_x_min, img_z_min, cs)
    if hallway:
        print(f"    Hallway: {hallway['type']}, X=[{hallway['x_min']:.2f}, {hallway['x_max']:.2f}], "
              f"Z=[{hallway['z_min']:.2f}, {hallway['z_max']:.2f}], score={hallway['score']:.2f}")
    else:
        print("    No hallway detected (single room?)")

    print("  Detecting rooms from wall structure...")
    rooms = detect_rooms_from_walls(x_walls, z_walls, x_evidence, z_evidence, hallway,
                                    room_mask, img_x_min, img_z_min, cs)
    
    # Compute areas
    for r in rooms:
        r['poly_area'] = compute_polygon_area(r['polygon'])
        r['area_m2'] = r['poly_area']
    
    total_area = sum(r['poly_area'] for r in rooms)
    
    for r in rooms:
        print(f"    {r['name']}: {r['poly_area']:.1f}m² {'[hallway]' if r['is_hallway'] else ''}")

    # Detect doors
    print("  Detecting doors...")
    all_wall_segments = []
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            segs = find_shared_wall_segments(r1['polygon'], r2['polygon'])
            all_wall_segments.extend(segs)
    
    doors = detect_doors_from_density(density_img, img_x_min, img_z_min, cs, all_wall_segments)
    print(f"    Found {len(doors)} doors")

    # Fine density
    density_fine, fx_min, fz_min, fcs = build_density_image(rx, rz, cell_size=0.01)

    print(f"\n=== v27i Summary ===")
    print(f"  Spaces: {len(rooms)} ({sum(1 for r in rooms if not r['is_hallway'])} rooms"
          f" + {sum(1 for r in rooms if r['is_hallway'])} hallways)")
    print(f"  Total area: {total_area:.1f} m²")

    return {
        'rooms': rooms, 'total_area': total_area,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'density_img': density_img, 'room_mask': room_mask,
        'doors': doors,
        'x_walls': x_walls, 'z_walls': z_walls,
        'x_evidence': x_evidence, 'z_evidence': z_evidence,
        'hallway': hallway,
        'img_origin': (img_x_min, img_z_min, cs),
        'fine_density': density_fine,
        'fine_origin': (fx_min, fz_min, fcs),
    }


# ─── 5-panel research view ───

def visualize_research(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    ix_min, iz_min, cs = results['img_origin']
    density = results['density_img']
    nz, nx = density.shape
    extent = [ix_min, ix_min + nx * cs, iz_min, iz_min + nz * cs]

    # 1. Density + Hough walls
    d_display = density.copy()
    if d_display.max() > 0:
        d_display = (d_display / np.percentile(d_display[d_display > 0], 95)).clip(0, 1)
    axes[0].imshow(d_display, cmap='hot', origin='lower', extent=extent)
    for xw in results['x_walls']:
        axes[0].axvline(xw, color='cyan', alpha=0.5, linewidth=1)
    for zw in results['z_walls']:
        axes[0].axhline(zw, color='lime', alpha=0.5, linewidth=1)
    axes[0].set_title('1. Density + Hough Walls', color='white', fontsize=14)

    # 2. Wall evidence heatmap
    ax1 = axes[1]
    ax1.set_facecolor('black')
    ax1.set_aspect('equal')
    # Draw room mask as faint background
    ax1.imshow(results['room_mask'], cmap='gray', origin='lower', alpha=0.2, extent=extent)
    
    x_ev = results['x_evidence']
    z_ev = results['z_evidence']
    max_score = max(
        max((ev[0] for ev in x_ev.values()), default=1),
        max((ev[0] for ev in z_ev.values()), default=1),
    )
    if max_score == 0:
        max_score = 1
    
    for xw, ev in x_ev.items():
        score, z_lo, z_hi, profile = ev
        if score < 0.1:
            continue
        alpha = min(1.0, score / max_score)
        color = plt.cm.plasma(alpha)
        ax1.plot([xw, xw], [z_lo, z_hi], color=color, linewidth=3, alpha=0.8)
        ax1.text(xw, z_hi + 0.05, f'{score:.1f}', color='white', fontsize=7, ha='center')
    
    for zw, ev in z_ev.items():
        score, x_lo, x_hi, profile = ev
        if score < 0.1:
            continue
        alpha = min(1.0, score / max_score)
        color = plt.cm.plasma(alpha)
        ax1.plot([x_lo, x_hi], [zw, zw], color=color, linewidth=3, alpha=0.8)
        ax1.text(x_hi + 0.05, zw, f'{score:.1f}', color='white', fontsize=7, va='center')
    
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.set_title('2. Wall Evidence Scores', color='white', fontsize=14)

    # 3. Room segmentation (colored regions)
    ax2 = axes[2]
    ax2.set_facecolor('black')
    ax2.set_aspect('equal')
    ax2.imshow(results['room_mask'], cmap='gray', origin='lower', alpha=0.15, extent=extent)
    
    # Highlight hallway
    hw = results['hallway']
    if hw:
        rect = patches.Rectangle((hw['x_min'], hw['z_min']),
                                hw['x_max']-hw['x_min'], hw['z_max']-hw['z_min'],
                                facecolor='yellow', alpha=0.2, edgecolor='yellow', linewidth=2)
        ax2.add_patch(rect)
    
    for i, room in enumerate(results['rooms']):
        poly = room['polygon']
        if len(poly) < 3:
            continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs = [p[0] for p in pc]
        zs = [p[1] for p in pc]
        ax2.fill(xs, zs, color=color, alpha=0.35)
        ax2.plot(xs, zs, color=color, linewidth=2)
        cx, cz = polygon_centroid(poly)
        ax2.text(cx, cz, f"{room['name']}\n{room['poly_area']:.1f}m²",
                ha='center', va='center', color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_title('3. Room Segmentation', color='white', fontsize=14)

    # 4. Edge + polygon overlay on density
    ax3 = axes[3]
    ax3.set_aspect('equal')
    fine = results['fine_density']
    fx_min, fz_min, fcs = results['fine_origin']
    if fine.max() > 0:
        f_display = (fine / np.percentile(fine[fine > 0], 95)).clip(0, 1)
    else:
        f_display = fine
    fine_extent = [fx_min, fx_min + fine.shape[1]*fcs, fz_min, fz_min + fine.shape[0]*fcs]
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

    # 5. Floor plan
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
        all_x.extend(xs); all_z.extend(zs)
        for j in range(len(poly)):
            k = (j+1) % len(poly)
            ax4.plot([poly[j][0], poly[k][0]], [poly[j][1], poly[k][1]],
                    color='white', linewidth=3, solid_capstyle='round')
        cx, cz = polygon_centroid(poly)
        # Dimensions
        xs_p = [p[0] for p in poly]
        zs_p = [p[1] for p in poly]
        w = max(xs_p) - min(xs_p)
        h = max(zs_p) - min(zs_p)
        ax4.text(cx, cz, f"{room['name']}\n{room['poly_area']:.1f}m²\n{w:.1f}×{h:.1f}m",
                ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))
    
    if all_x:
        m = 0.5
        ax4.set_xlim(min(all_x)-m, max(all_x)+m)
        ax4.set_ylim(min(all_z)-m, max(all_z)+m)
    ax4.set_title(f'5. Floor Plan — {len(results["rooms"])} spaces, {results["total_area"]:.0f}m²',
                 color='white', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved research view: {output_path}")


# ─── Clean architectural floor plan ───

def draw_clean_floorplan(results, output_path):
    rooms = results['rooms']
    doors = results['doors']
    if not rooms:
        return
    
    all_x = [p[0] for r in rooms for p in r['polygon'] if len(r['polygon']) >= 3]
    all_z = [p[1] for r in rooms for p in r['polygon'] if len(r['polygon']) >= 3]
    if not all_x:
        return
    
    x_range = max(all_x) - min(all_x)
    z_range = max(all_z) - min(all_z)
    scale = 2.0
    fig_w = max(8, x_range * scale + 3)
    fig_h = max(6, z_range * scale + 3)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    WALL_LW = 8
    WALL_COLOR = '#1a1a1a'
    ext_x_min, ext_x_max = min(all_x), max(all_x)
    ext_z_min, ext_z_max = min(all_z), max(all_z)
    
    # 1. Pastel fills
    for i, room in enumerate(rooms):
        poly = room['polygon']
        if len(poly) < 3: continue
        fill = PASTEL_FILLS[i % len(PASTEL_FILLS)]
        pc = poly + [poly[0]]
        ax.fill([p[0] for p in pc], [p[1] for p in pc], color=fill, zorder=1)
    
    # 2. Walls
    drawn_edges = set()
    for room in rooms:
        poly = room['polygon']
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i+1)%n]
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                key = (round(p2[0],3), round(p2[1],3), round(p1[0],3), round(p1[1],3))
            else:
                key = (round(p1[0],3), round(p1[1],3), round(p2[0],3), round(p2[1],3))
            if key in drawn_edges: continue
            drawn_edges.add(key)
            
            is_ext = False
            if abs(p1[0]-p2[0]) < 0.05:
                if abs(p1[0]-ext_x_min) < 0.15 or abs(p1[0]-ext_x_max) < 0.15:
                    is_ext = True
            elif abs(p1[1]-p2[1]) < 0.05:
                if abs(p1[1]-ext_z_min) < 0.15 or abs(p1[1]-ext_z_max) < 0.15:
                    is_ext = True
            
            lw = WALL_LW if is_ext else WALL_LW - 2
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]],
                   color=WALL_COLOR, linewidth=lw, solid_capstyle='round', zorder=3)
    
    # 3. Doors
    for door in doors:
        x, z = door['x'], door['z']
        w = min(door['width'], 0.9)
        r = w * 0.9
        if door['orientation'] == 'vertical':
            rect = patches.Rectangle((x-0.06, z-w/2), 0.12, w,
                                    facecolor='white', edgecolor='none', zorder=4)
            ax.add_patch(rect)
            arc = Arc((x, z-w/2), 2*r, 2*r, angle=0, theta1=0, theta2=90,
                      color='#555555', linewidth=1.5, zorder=5)
            ax.add_patch(arc)
            ax.plot([x, x+r], [z-w/2, z-w/2], color='#555555', linewidth=1.5, zorder=5)
            ax.plot([x, x], [z-w/2, z-w/2+r], color='#555555', linewidth=1.5, zorder=5)
        else:
            rect = patches.Rectangle((x-w/2, z-0.06), w, 0.12,
                                    facecolor='white', edgecolor='none', zorder=4)
            ax.add_patch(rect)
            arc = Arc((x-w/2, z), 2*r, 2*r, angle=0, theta1=0, theta2=90,
                      color='#555555', linewidth=1.5, zorder=5)
            ax.add_patch(arc)
            ax.plot([x-w/2, x-w/2+r], [z, z], color='#555555', linewidth=1.5, zorder=5)
            ax.plot([x-w/2, x-w/2], [z, z+r], color='#555555', linewidth=1.5, zorder=5)
    
    # 4. Window double-lines on exterior walls
    # Sample exterior walls for gaps
    for room in rooms:
        poly = room['polygon']
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i+1)%n]
            is_ext = False
            if abs(p1[0]-p2[0]) < 0.05:
                if abs(p1[0]-ext_x_min) < 0.15 or abs(p1[0]-ext_x_max) < 0.15:
                    is_ext = True
            elif abs(p1[1]-p2[1]) < 0.05:
                if abs(p1[1]-ext_z_min) < 0.15 or abs(p1[1]-ext_z_max) < 0.15:
                    is_ext = True
            # Skip window detection for now — density sampling is unreliable for this
    
    # 5. Room labels
    for i, room in enumerate(rooms):
        poly = room['polygon']
        if len(poly) < 3: continue
        cx, cz = polygon_centroid(poly)
        area = room['poly_area']
        label = f"{room['name']}\n{area:.1f}m²"
        ax.text(cx, cz, label, ha='center', va='center', fontsize=12,
               fontweight='bold', color='#333333', zorder=6)
    
    margin = 0.3
    ax.set_xlim(ext_x_min - margin, ext_x_max + margin)
    ax.set_ylim(ext_z_min - margin, ext_z_max + margin)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"  Saved clean floor plan: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27i_wallstructure',
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
            'width': round(d['width'], 2), 'orientation': d['orientation'],
        } for d in results['doors']],
        'hallway': {
            'type': results['hallway']['type'],
            'x_min': round(results['hallway']['x_min'], 3),
            'x_max': round(results['hallway']['x_max'], 3),
            'z_min': round(results['hallway']['z_min'], 3),
            'z_max': round(results['hallway']['z_max'], 3),
        } if results['hallway'] else None,
        'walls': {
            'x_positions': [round(w, 3) for w in results['x_walls']],
            'z_positions': [round(w, 3) for w in results['z_walls']],
        }
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27i - Wall-Structure Room Detection')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27i/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27i_{Path(args.mesh_file).stem}"
    
    results = analyze_mesh(args.mesh_file)
    visualize_research(results, output_dir / f"{prefix}_floorplan.png")
    draw_clean_floorplan(results, output_dir / f"{prefix}_clean.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

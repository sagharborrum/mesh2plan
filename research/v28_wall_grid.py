#!/usr/bin/env python3
"""
mesh2plan v28 - Wall Grid Approach

Strategy: Instead of watershed/distance-transform for room detection,
use Hough wall positions to define a grid of rectangular cells.
Classify cells as room interior vs empty, then merge adjacent cells
into rooms. Hallways emerge naturally as narrow cell groups.

Key insight: Walls define a grid. Rooms are rectangles (or L-shapes)
formed by merging grid cells. This avoids the watershed "hallway eats
everything" problem.

Steps:
1. Build density image from rotated vertices
2. Detect Hough wall positions (X and Z)
3. Create grid cells from wall intersections
4. For each cell, measure interior density (low = room, high = wall)
5. Merge adjacent room cells into rooms
6. Classify narrow groups as hallways
7. Snap to clean rectilinear polygons
8. Render clean architectural floor plan
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
from collections import deque


# ─── Utilities ───

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
        prof = uniform_filter1d(profile.astype(float), size=5)
        local_max = maximum_filter(prof, size=max(3, min_dist_cells)) == prof
        threshold = prof.mean() + 0.15 * prof.std()
        peaks = np.where(local_max & (prof > threshold))[0]
        positions = origin + peaks * cs
        if len(positions) == 0:
            return np.array([]), np.array([])
        strengths = prof[peaks]
        order = np.argsort(-strengths)
        kept = []
        kept_str = []
        for i in order:
            pos = positions[i]
            if any(abs(pos - k) < nms_dist for k in kept):
                continue
            kept.append(pos)
            kept_str.append(strengths[i])
        sort_idx = np.argsort(kept)
        return np.array(kept)[sort_idx], np.array(kept_str)[sort_idx]

    min_dist = int(nms_dist / cell_size)
    x_walls, x_str = find_peaks_nms(proj_x, x_min, cell_size, min_dist)
    z_walls, z_str = find_peaks_nms(proj_z, z_min, cell_size, min_dist)
    return x_walls, z_walls, x_str, z_str


def filter_walls_by_evidence(density_img, x_min, z_min, cell_size, 
                               x_walls, z_walls, x_str, z_str,
                               min_length_m=1.0):
    """Filter walls by checking actual wall evidence (continuous high-density segments)."""
    
    def check_wall_evidence(density_img, pos, axis, origin, cs, min_len):
        """Check if a wall position has continuous evidence of min_len meters."""
        if axis == 'x':
            px = int((pos - origin) / cs)
            if px < 0 or px >= density_img.shape[1]:
                return False, 0
            strip_w = max(1, int(0.08 / cs))
            lo = max(0, px - strip_w)
            hi = min(density_img.shape[1], px + strip_w + 1)
            profile = density_img[:, lo:hi].max(axis=1)
        else:
            px = int((pos - origin) / cs)
            if px < 0 or px >= density_img.shape[0]:
                return False, 0
            strip_w = max(1, int(0.08 / cs))
            lo = max(0, px - strip_w)
            hi = min(density_img.shape[0], px + strip_w + 1)
            profile = density_img[lo:hi, :].max(axis=0)
        
        # Find longest continuous high-density segment
        threshold = max(2, np.percentile(profile[profile > 0], 25) if np.any(profile > 0) else 2)
        is_wall = profile > threshold
        
        # Find runs
        max_run = 0
        current_run = 0
        for v in is_wall:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        max_run_m = max_run * cs
        return max_run_m >= min_len, max_run_m
    
    x_keep = []
    for i, xw in enumerate(x_walls):
        ok, length = check_wall_evidence(density_img, xw, 'x', x_min, cell_size, min_length_m)
        print(f"    X wall {xw:.2f}: evidence length={length:.1f}m {'✓' if ok else '✗'}")
        if ok:
            x_keep.append(xw)
    
    z_keep = []
    for i, zw in enumerate(z_walls):
        ok, length = check_wall_evidence(density_img, zw, 'z', z_min, cell_size, min_length_m)
        print(f"    Z wall {zw:.2f}: evidence length={length:.1f}m {'✓' if ok else '✗'}")
        if ok:
            z_keep.append(zw)
    
    return np.array(x_keep), np.array(z_keep)


# ─── Grid-based room detection ───

def classify_grid_cells(density_img, x_min, z_min, cell_size, x_walls, z_walls):
    """
    Create grid cells from wall positions and classify each as room/empty.
    A cell is "room" if its interior has low density (scanned floor with sparse points)
    vs walls which have very high density.
    
    Key insight: Room interiors in LiDAR scans have SOME points (floor/ceiling reflections)
    but much less than walls. Empty/unscanned areas have ZERO points.
    """
    cells = []
    nz_img, nx_img = density_img.shape
    
    for i in range(len(x_walls) - 1):
        for j in range(len(z_walls) - 1):
            x0, x1 = x_walls[i], x_walls[i+1]
            z0, z1 = z_walls[j], z_walls[j+1]
            
            # Cell size in meters
            w = x1 - x0
            h = z1 - z0
            
            # Skip tiny cells (wall thickness)
            if w < 0.2 or h < 0.2:
                continue
            
            # Get pixel bounds (inset slightly to avoid wall pixels)
            inset = int(0.1 / cell_size)
            px0 = int((x0 - x_min) / cell_size) + inset
            px1 = int((x1 - x_min) / cell_size) - inset
            pz0 = int((z0 - z_min) / cell_size) + inset
            pz1 = int((z1 - z_min) / cell_size) - inset
            
            px0 = max(0, min(px0, nx_img - 1))
            px1 = max(px0 + 1, min(px1, nx_img))
            pz0 = max(0, min(pz0, nz_img - 1))
            pz1 = max(pz0 + 1, min(pz1, nz_img))
            
            region = density_img[pz0:pz1, px0:px1]
            
            if region.size == 0:
                continue
            
            # Metrics
            total_pixels = region.size
            nonzero_pixels = np.count_nonzero(region)
            occupancy = nonzero_pixels / total_pixels if total_pixels > 0 else 0
            mean_density = region.mean()
            
            # A room cell should have:
            # - Some occupancy (it was scanned) but not super high density everywhere
            # - Moderate density (floor points exist)
            # vs empty (unscanned) which has ~0 occupancy
            # vs wall which has very high mean density
            
            area_m2 = w * h
            
            cells.append({
                'i': i, 'j': j,
                'x0': x0, 'x1': x1, 'z0': z0, 'z1': z1,
                'w': w, 'h': h, 'area': area_m2,
                'occupancy': occupancy,
                'mean_density': mean_density,
                'is_room': False,  # classified later
            })
    
    if not cells:
        return cells
    
    # Classify: room cells have moderate occupancy (scanned but not all wall)
    occupancies = [c['occupancy'] for c in cells]
    densities = [c['mean_density'] for c in cells]
    
    # Room = occupancy > 0.15 (floor was scanned) 
    # We want cells with some data but not pure wall
    occ_threshold = 0.15
    
    for c in cells:
        c['is_room'] = c['occupancy'] > occ_threshold
    
    return cells


def merge_cells_into_rooms(cells, x_walls, z_walls, min_room_area=1.5):
    """
    Merge adjacent room cells into rooms using flood fill.
    Two cells are adjacent if they share a wall edge (same i±1 or j±1).
    """
    room_cells = [c for c in cells if c['is_room']]
    if not room_cells:
        return []
    
    # Build adjacency: cells indexed by (i, j)
    cell_map = {}
    for c in room_cells:
        cell_map[(c['i'], c['j'])] = c
    
    visited = set()
    rooms = []
    
    for c in room_cells:
        key = (c['i'], c['j'])
        if key in visited:
            continue
        
        # BFS flood fill
        group = []
        queue = deque([key])
        visited.add(key)
        
        while queue:
            ci, cj = queue.popleft()
            group.append(cell_map[(ci, cj)])
            
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                nk = (ci+di, cj+dj)
                if nk in cell_map and nk not in visited:
                    visited.add(nk)
                    queue.append(nk)
        
        # Compute total area
        total_area = sum(c['area'] for c in group)
        if total_area < min_room_area:
            continue
        
        rooms.append(group)
    
    return rooms


def room_to_polygon(cells):
    """
    Convert a group of grid cells into a clean rectilinear polygon.
    For a single rectangle, returns 4 vertices.
    For L-shapes or more complex, returns the rectilinear outline.
    """
    if not cells:
        return []
    
    # Simple approach: compute the rectilinear hull
    # Create a binary grid of the cells
    i_vals = [c['i'] for c in cells]
    j_vals = [c['j'] for c in cells]
    i_min, i_max = min(i_vals), max(i_vals)
    j_min, j_max = min(j_vals), max(j_vals)
    
    # If it's a single rectangle (all cells present in grid), just use bbox
    all_keys = set((c['i'], c['j']) for c in cells)
    expected = set((i, j) for i in range(i_min, i_max+1) for j in range(j_min, j_max+1))
    
    if all_keys == expected:
        # Perfect rectangle
        x0 = min(c['x0'] for c in cells)
        x1 = max(c['x1'] for c in cells)
        z0 = min(c['z0'] for c in cells)
        z1 = max(c['z1'] for c in cells)
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    # Non-rectangular: compute rectilinear outline
    # Build a pixel-level mask at cell resolution
    ni = i_max - i_min + 1
    nj = j_max - j_min + 1
    grid = np.zeros((nj, ni), dtype=np.uint8)
    
    # Map cells to grid coords
    x_coords = {}  # i -> (x0, x1)
    z_coords = {}  # j -> (z0, z1)
    for c in cells:
        gi = c['i'] - i_min
        gj = c['j'] - j_min
        grid[gj, gi] = 1
        x_coords[c['i']] = (c['x0'], c['x1'])
        z_coords[c['j']] = (c['z0'], c['z1'])
    
    # Find contour of the binary grid
    # Scale up for better contour detection
    scale = 10
    big = cv2.resize(grid, (ni * scale, nj * scale), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to bbox
        x0 = min(c['x0'] for c in cells)
        x1 = max(c['x1'] for c in cells)
        z0 = min(c['z0'] for c in cells)
        z1 = max(c['z1'] for c in cells)
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    contour = max(contours, key=cv2.contourArea)
    
    # Convert contour pixels back to world coordinates
    # Each pixel in the big grid maps to a cell boundary
    all_x = sorted(set(x_coords.keys()))
    all_j = sorted(set(z_coords.keys()))
    
    # Build coordinate mapping: pixel position -> world coordinate
    x_boundaries = []
    for ii in range(i_min, i_max + 2):
        if ii in x_coords:
            x_boundaries.append(x_coords[ii][0])
        elif ii - 1 in x_coords:
            x_boundaries.append(x_coords[ii-1][1])
    
    z_boundaries = []
    for jj in range(j_min, j_max + 2):
        if jj in z_coords:
            z_boundaries.append(z_coords[jj][0])
        elif jj - 1 in z_coords:
            z_boundaries.append(z_coords[jj-1][1])
    
    poly = []
    for pt in contour.reshape(-1, 2):
        px, py = pt
        # Map back: px/scale -> i index, py/scale -> j index
        fi = px / scale
        fj = py / scale
        
        # Interpolate world coordinates
        if len(x_boundaries) > 1:
            xi = fi / ni * (len(x_boundaries) - 1)
            xi_lo = int(xi)
            xi_hi = min(xi_lo + 1, len(x_boundaries) - 1)
            frac = xi - xi_lo
            wx = x_boundaries[xi_lo] * (1 - frac) + x_boundaries[xi_hi] * frac
        else:
            wx = x_boundaries[0] if x_boundaries else 0
        
        if len(z_boundaries) > 1:
            zj = fj / nj * (len(z_boundaries) - 1)
            zj_lo = int(zj)
            zj_hi = min(zj_lo + 1, len(z_boundaries) - 1)
            frac = zj - zj_lo
            wz = z_boundaries[zj_lo] * (1 - frac) + z_boundaries[zj_hi] * frac
        else:
            wz = z_boundaries[0] if z_boundaries else 0
        
        poly.append([wx, wz])
    
    # Simplify: snap to cell boundaries and remove collinear points
    return simplify_rectilinear(poly, cells)


def simplify_rectilinear(poly, cells):
    """Simplify polygon to rectilinear (axis-aligned edges only)."""
    if len(poly) < 3:
        return poly
    
    # Get all unique X and Z values from cells
    all_x = sorted(set([c['x0'] for c in cells] + [c['x1'] for c in cells]))
    all_z = sorted(set([c['z0'] for c in cells] + [c['z1'] for c in cells]))
    
    def snap(val, positions):
        if not positions:
            return val
        dists = [abs(val - p) for p in positions]
        idx = np.argmin(dists)
        return positions[idx]
    
    # Snap all points
    snapped = []
    for p in poly:
        sx = snap(p[0], all_x)
        sz = snap(p[1], all_z)
        snapped.append([sx, sz])
    
    # Remove duplicates
    cleaned = [snapped[0]]
    for p in snapped[1:]:
        if abs(p[0] - cleaned[-1][0]) > 0.01 or abs(p[1] - cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    
    # Remove collinear points
    if len(cleaned) < 3:
        return cleaned
    
    result = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i - 1) % n]
        curr = cleaned[i]
        next_p = cleaned[(i + 1) % n]
        
        # Keep if direction changes
        dx1 = curr[0] - prev[0]
        dz1 = curr[1] - prev[1]
        dx2 = next_p[0] - curr[0]
        dz2 = next_p[1] - curr[1]
        
        if abs(dx1 * dz2 - dz1 * dx2) > 0.001:
            result.append(curr)
    
    return result if len(result) >= 3 else cleaned


def classify_rooms_vs_hallways(rooms, max_narrow_dim=1.5):
    """Classify room groups as rooms or hallways based on aspect ratio."""
    classified = []
    for room in rooms:
        x0 = min(c['x0'] for c in room)
        x1 = max(c['x1'] for c in room)
        z0 = min(c['z0'] for c in room)
        z1 = max(c['z1'] for c in room)
        w = x1 - x0
        h = z1 - z0
        area = sum(c['area'] for c in room)
        
        # Hallway = narrow in one dimension AND connects to multiple rooms
        min_dim = min(w, h)
        aspect = max(w, h) / min_dim if min_dim > 0 else 999
        
        is_hallway = min_dim < max_narrow_dim and aspect > 2.0
        
        classified.append({
            'cells': room,
            'is_hallway': is_hallway,
            'area': area,
            'bounds': (x0, x1, z0, z1),
            'min_dim': min_dim,
            'aspect': aspect,
        })
    
    return classified


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
    return sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n


# ─── Door detection ───

def detect_doors_between_rooms(density_img, x_min, z_min, cell_size, rooms_data):
    """Detect doors by finding low-density gaps on shared walls between rooms."""
    doors = []
    
    for i in range(len(rooms_data)):
        for j in range(i+1, len(rooms_data)):
            ri = rooms_data[i]
            rj = rooms_data[j]
            pi = ri['polygon']
            pj = rj['polygon']
            
            if len(pi) < 3 or len(pj) < 3:
                continue
            
            # Find shared wall segments
            shared = find_shared_edges(pi, pj, tolerance=0.2)
            
            for seg in shared:
                door = find_gap_in_wall(density_img, x_min, z_min, cell_size, seg)
                if door:
                    doors.append(door)
    
    return doors


def find_shared_edges(poly1, poly2, tolerance=0.2):
    """Find wall segments shared between two polygons."""
    segments = []
    n1, n2 = len(poly1), len(poly2)
    
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i+1) % n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j+1) % n2]
            
            # Vertical edges (same X)
            if abs(a1[0] - a2[0]) < 0.05 and abs(b1[0] - b2[0]) < 0.05:
                if abs(a1[0] - b1[0]) < tolerance:
                    a_lo, a_hi = min(a1[1], a2[1]), max(a1[1], a2[1])
                    b_lo, b_hi = min(b1[1], b2[1]), max(b1[1], b2[1])
                    overlap_lo = max(a_lo, b_lo)
                    overlap_hi = min(a_hi, b_hi)
                    if overlap_hi - overlap_lo > 0.3:
                        segments.append({
                            'type': 'vertical',
                            'x': (a1[0] + b1[0]) / 2,
                            'lo': overlap_lo, 'hi': overlap_hi,
                        })
            
            # Horizontal edges (same Z)
            if abs(a1[1] - a2[1]) < 0.05 and abs(b1[1] - b2[1]) < 0.05:
                if abs(a1[1] - b1[1]) < tolerance:
                    a_lo, a_hi = min(a1[0], a2[0]), max(a1[0], a2[0])
                    b_lo, b_hi = min(b1[0], b2[0]), max(b1[0], b2[0])
                    overlap_lo = max(a_lo, b_lo)
                    overlap_hi = min(a_hi, b_hi)
                    if overlap_hi - overlap_lo > 0.3:
                        segments.append({
                            'type': 'horizontal',
                            'z': (a1[1] + b1[1]) / 2,
                            'lo': overlap_lo, 'hi': overlap_hi,
                        })
    
    return segments


def find_gap_in_wall(density_img, x_min, z_min, cell_size, seg):
    """Find a door-sized gap in a wall segment."""
    strip_w = max(1, int(0.08 / cell_size))
    
    if seg['type'] == 'vertical':
        x_px = int((seg['x'] - x_min) / cell_size)
        lo_px = int((seg['lo'] - z_min) / cell_size)
        hi_px = int((seg['hi'] - z_min) / cell_size)
        
        x_lo = max(0, x_px - strip_w)
        x_hi = min(density_img.shape[1], x_px + strip_w + 1)
        
        if x_lo >= x_hi or lo_px >= hi_px:
            return None
        
        lo_px = max(0, lo_px)
        hi_px = min(density_img.shape[0], hi_px)
        
        profile = density_img[lo_px:hi_px, x_lo:x_hi].sum(axis=1)
    else:
        z_px = int((seg['z'] - z_min) / cell_size)
        lo_px = int((seg['lo'] - x_min) / cell_size)
        hi_px = int((seg['hi'] - x_min) / cell_size)
        
        z_lo = max(0, z_px - strip_w)
        z_hi = min(density_img.shape[0], z_px + strip_w + 1)
        
        if z_lo >= z_hi or lo_px >= hi_px:
            return None
        
        lo_px = max(0, lo_px)
        hi_px = min(density_img.shape[1], hi_px)
        
        profile = density_img[z_lo:z_hi, lo_px:hi_px].sum(axis=0)
    
    if len(profile) == 0 or not np.any(profile > 0):
        return None
    
    threshold = np.percentile(profile[profile > 0], 30) if np.any(profile > 0) else 1
    is_gap = profile < max(threshold, 1)
    
    gap_lbl, n_gaps = ndlabel(is_gap)
    for g in range(1, n_gaps + 1):
        gap_idx = np.where(gap_lbl == g)[0]
        gap_len = len(gap_idx) * cell_size
        if 0.6 < gap_len < 1.5:
            center = gap_idx.mean()
            if seg['type'] == 'vertical':
                return {
                    'x': seg['x'],
                    'z': z_min + (lo_px + center) * cell_size,
                    'width': gap_len,
                    'orientation': 'vertical',
                }
            else:
                return {
                    'x': x_min + (lo_px + center) * cell_size,
                    'z': seg['z'],
                    'width': gap_len,
                    'orientation': 'horizontal',
                }
    
    return None


# ─── Window detection ───

def detect_windows_on_exterior(density_img, x_min, z_min, cell_size, rooms_data):
    """Detect windows as gaps on exterior walls."""
    windows = []
    
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3:
            continue
        
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1) % n]
            
            # Check if this edge is on the convex hull exterior (not shared with another room)
            is_exterior = True
            mid_x = (p1[0] + p2[0]) / 2
            mid_z = (p1[1] + p2[1]) / 2
            
            for rd2 in rooms_data:
                if rd2 is rd:
                    continue
                poly2 = rd2['polygon']
                # Check if midpoint is near any edge of poly2
                for j in range(len(poly2)):
                    q1 = poly2[j]
                    q2 = poly2[(j+1) % len(poly2)]
                    # Close enough to be shared
                    mid2_x = (q1[0] + q2[0]) / 2
                    mid2_z = (q1[1] + q2[1]) / 2
                    if abs(mid_x - mid2_x) < 0.3 and abs(mid_z - mid2_z) < 0.3:
                        is_exterior = False
                        break
                if not is_exterior:
                    break
            
            if not is_exterior:
                continue
            
            # Check for window-sized gaps on this exterior edge
            edge_len = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if edge_len < 0.5:
                continue
            
            seg = None
            if abs(p1[0] - p2[0]) < 0.05:  # Vertical edge
                seg = {'type': 'vertical', 'x': p1[0],
                       'lo': min(p1[1], p2[1]), 'hi': max(p1[1], p2[1])}
            elif abs(p1[1] - p2[1]) < 0.05:  # Horizontal edge
                seg = {'type': 'horizontal', 'z': p1[1],
                       'lo': min(p1[0], p2[0]), 'hi': max(p1[0], p2[0])}
            
            if seg:
                gap = find_gap_in_wall(density_img, x_min, z_min, cell_size, seg)
                if gap and gap['width'] > 0.8:
                    windows.append({
                        'x': gap['x'], 'z': gap['z'],
                        'length': gap['width'],
                        'orientation': gap['orientation'],
                    })
    
    return windows


# ─── Rendering ───

ROOM_COLORS = [
    '#E8F5E9',  # light green
    '#E3F2FD',  # light blue  
    '#FFF3E0',  # light orange
    '#F3E5F5',  # light purple
    '#FFFDE7',  # light yellow
    '#E0F7FA',  # light cyan
    '#FCE4EC',  # light pink
    '#F1F8E9',  # light lime
]

HALLWAY_COLOR = '#F5F5F5'  # light gray


def render_clean_floorplan(rooms_data, doors, windows, output_path, title="v28 Wall Grid"):
    """Render clean architectural floor plan."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Compute bounds
    all_x = []
    all_z = []
    for rd in rooms_data:
        for p in rd['polygon']:
            all_x.append(p[0])
            all_z.append(p[1])
    
    if not all_x:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    margin = 0.5
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    z_min, z_max = min(all_z) - margin, max(all_z) + margin
    
    wall_thickness = 0.08  # meters
    
    # Draw room fills
    color_idx = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3:
            continue
        
        color = HALLWAY_COLOR if rd['is_hallway'] else ROOM_COLORS[color_idx % len(ROOM_COLORS)]
        if not rd['is_hallway']:
            color_idx += 1
        
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.8)
    
    # Draw walls (thick black lines on polygon edges)
    for rd in rooms_data:
        poly = rd['polygon']
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1) % n]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3)
    
    # Draw doors (gaps in walls with arc)
    for door in doors:
        x, z = door['x'], door['z']
        w = door.get('width', 0.8)
        
        if door['orientation'] == 'vertical':
            # Gap on vertical wall
            ax.plot([x, x], [z - w/2, z + w/2], color='white', linewidth=5)
            # Door arc
            arc = Arc((x, z - w/2), w, w, angle=0, theta1=0, theta2=90, color='black', linewidth=1.5)
            ax.add_patch(arc)
        else:
            # Gap on horizontal wall
            ax.plot([x - w/2, x + w/2], [z, z], color='white', linewidth=5)
            arc = Arc((x - w/2, z), w, w, angle=0, theta1=0, theta2=90, color='black', linewidth=1.5)
            ax.add_patch(arc)
    
    # Draw windows (double lines)
    for win in windows:
        x, z = win['x'], win['z']
        length = win.get('length', 1.0)
        
        if win['orientation'] == 'vertical':
            ax.plot([x, x], [z - length/2, z + length/2], color='white', linewidth=5)
            ax.plot([x - 0.03, x - 0.03], [z - length/2, z + length/2], 'k-', linewidth=1.5)
            ax.plot([x + 0.03, x + 0.03], [z - length/2, z + length/2], 'k-', linewidth=1.5)
        else:
            ax.plot([x - length/2, x + length/2], [z, z], color='white', linewidth=5)
            ax.plot([x - length/2, x + length/2], [z - 0.03, z - 0.03], 'k-', linewidth=1.5)
            ax.plot([x - length/2, x + length/2], [z + 0.03, z + 0.03], 'k-', linewidth=1.5)
    
    # Room labels
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3:
            continue
        cx, cz = polygon_centroid(poly)
        area = rd['area']
        name = rd['name']
        
        ax.text(cx, cz, f"{name}\n{area:.1f}m²",
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='#333333')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug_panels(density_img, x_min, z_min, cell_size, 
                         x_walls, z_walls, cells, rooms_classified, 
                         rooms_data, output_path):
    """Render 4-panel debug view."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    extent = [x_min, x_min + density_img.shape[1] * cell_size,
              z_min, z_min + density_img.shape[0] * cell_size]
    
    # Panel 1: Density + Hough walls
    ax = axes[0, 0]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal')
    for xw in x_walls:
        ax.axvline(xw, color='cyan', alpha=0.5, linewidth=1)
    for zw in z_walls:
        ax.axhline(zw, color='lime', alpha=0.5, linewidth=1)
    ax.set_title('1. Density + Hough Walls')
    
    # Panel 2: Grid cells classification
    ax = axes[0, 1]
    ax.set_facecolor('#333')
    for c in cells:
        color = '#4CAF50' if c['is_room'] else '#F44336'
        alpha = 0.6 if c['is_room'] else 0.3
        rect = patches.Rectangle((c['x0'], c['z0']), c['w'], c['h'],
                                  linewidth=0.5, edgecolor='white',
                                  facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        # Label occupancy
        cx = (c['x0'] + c['x1']) / 2
        cz = (c['z0'] + c['z1']) / 2
        if c['w'] > 0.5 and c['h'] > 0.5:
            ax.text(cx, cz, f"{c['occupancy']:.0%}", ha='center', va='center',
                    fontsize=6, color='white')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('2. Grid Cell Classification (green=room)')
    
    # Panel 3: Merged rooms
    ax = axes[1, 0]
    ax.set_facecolor('white')
    colors = ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#F3E5F5', '#FFFDE7', '#E0F7FA', '#FCE4EC']
    for idx, rc in enumerate(rooms_classified):
        color = '#F5F5F5' if rc['is_hallway'] else colors[idx % len(colors)]
        for c in rc['cells']:
            rect = patches.Rectangle((c['x0'], c['z0']), c['w'], c['h'],
                                      linewidth=0.5, edgecolor='#999',
                                      facecolor=color, alpha=0.8)
            ax.add_patch(rect)
        # Label
        x0, x1, z0, z1 = rc['bounds']
        ax.text((x0+x1)/2, (z0+z1)/2, 
                f"{'Hall' if rc['is_hallway'] else f'Room {idx+1}'}\n{rc['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('3. Merged Rooms')
    
    # Panel 4: Clean floor plan
    ax = axes[1, 1]
    ax.set_facecolor('white')
    color_idx = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3:
            continue
        color = HALLWAY_COLOR if rd['is_hallway'] else ROOM_COLORS[color_idx % len(ROOM_COLORS)]
        if not rd['is_hallway']:
            color_idx += 1
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.8)
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1) % n]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold', color='#333')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('4. Clean Floor Plan')
    
    fig.suptitle('v28 Wall Grid — Debug Panels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description='v28 Wall Grid floor plan extraction')
    parser.add_argument('mesh', help='Path to mesh file (.obj/.ply/.glb/.stl)')
    parser.add_argument('--cell', type=float, default=0.02, help='Density cell size (m)')
    parser.add_argument('--nms', type=float, default=0.15, help='Wall NMS distance (m)')
    parser.add_argument('--min-wall-length', type=float, default=1.0, help='Min wall evidence length (m)')
    parser.add_argument('--min-room-area', type=float, default=1.5, help='Min room area (m²)')
    parser.add_argument('-o', '--output', default='results/v28_wallgrid', help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_name = Path(args.mesh).stem
    
    # Load mesh
    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh')
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    # Detect coordinate system
    up_idx, up_name = detect_up_axis(mesh)
    print(f"  Up axis: {up_name} (index {up_idx})")
    
    # Project to 2D
    rx, rz = project_vertices(mesh, up_idx)
    print(f"  2D bounds: X=[{rx.min():.2f}, {rx.max():.2f}], Z=[{rz.min():.2f}, {rz.max():.2f}]")
    
    # Find dominant angle and rotate
    angle = find_dominant_angle(rx, rz, cell=args.cell)
    print(f"  Dominant angle: {angle:.1f}°")
    
    cos_a = math.cos(math.radians(-angle))
    sin_a = math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a
    print(f"  Rotated bounds: X=[{rx2.min():.2f}, {rx2.max():.2f}], Z=[{rz2.min():.2f}, {rz2.max():.2f}]")
    
    # Build density image
    print("Building density image...")
    density_img, img_x_min, img_z_min, cs = build_density_image(rx2, rz2, cell_size=args.cell)
    print(f"  Image size: {density_img.shape}")
    
    # Detect Hough wall positions
    print("Detecting walls...")
    x_walls, z_walls, x_str, z_str = hough_wall_positions(density_img, img_x_min, img_z_min, cs, nms_dist=args.nms)
    print(f"  Raw X walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Raw Z walls: {[f'{w:.2f}' for w in z_walls]}")
    
    # Filter walls by evidence (lower threshold for boundary walls)
    print("Filtering walls by evidence...")
    x_walls_f, z_walls_f = filter_walls_by_evidence(
        density_img, img_x_min, img_z_min, cs,
        x_walls, z_walls, x_str, z_str,
        min_length_m=args.min_wall_length
    )
    
    # Ensure we have boundary walls from the building envelope
    # Find the occupied region boundary
    occupied = (density_img > 0).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.3/cs), int(0.3/cs)))
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, k_close)
    rows_any = np.where(occupied.any(axis=1))[0]
    cols_any = np.where(occupied.any(axis=0))[0]
    if len(rows_any) > 0 and len(cols_any) > 0:
        env_x_min = img_x_min + cols_any[0] * cs
        env_x_max = img_x_min + cols_any[-1] * cs
        env_z_min = img_z_min + rows_any[0] * cs
        env_z_max = img_z_min + rows_any[-1] * cs
        
        # Add boundary walls if not already present
        snap_dist = 0.4
        if not any(abs(w - env_x_min) < snap_dist for w in x_walls_f):
            x_walls_f = np.sort(np.append(x_walls_f, env_x_min))
            print(f"    Added X boundary wall at {env_x_min:.2f}")
        if not any(abs(w - env_x_max) < snap_dist for w in x_walls_f):
            x_walls_f = np.sort(np.append(x_walls_f, env_x_max))
            print(f"    Added X boundary wall at {env_x_max:.2f}")
        if not any(abs(w - env_z_min) < snap_dist for w in z_walls_f):
            z_walls_f = np.sort(np.append(z_walls_f, env_z_min))
            print(f"    Added Z boundary wall at {env_z_min:.2f}")
        if not any(abs(w - env_z_max) < snap_dist for w in z_walls_f):
            z_walls_f = np.sort(np.append(z_walls_f, env_z_max))
            print(f"    Added Z boundary wall at {env_z_max:.2f}")
    
    # Also re-add walls that were filtered but have moderate evidence (>= 0.5m)
    # These might be short interior walls (closets, bathroom dividers)
    for xw in x_walls:
        if not any(abs(w - xw) < 0.15 for w in x_walls_f):
            # Check with lower threshold
            strip_w = max(1, int(0.08 / cs))
            px = int((xw - img_x_min) / cs)
            if 0 <= px < density_img.shape[1]:
                lo = max(0, px - strip_w)
                hi = min(density_img.shape[1], px + strip_w + 1)
                profile = density_img[:, lo:hi].max(axis=1)
                threshold = max(2, np.percentile(profile[profile > 0], 25) if np.any(profile > 0) else 2)
                is_wall = profile > threshold
                max_run = 0
                cur = 0
                for v in is_wall:
                    if v: cur += 1; max_run = max(max_run, cur)
                    else: cur = 0
                if max_run * cs >= 0.5:
                    x_walls_f = np.sort(np.append(x_walls_f, xw))
                    print(f"    Re-added X wall at {xw:.2f} (evidence={max_run*cs:.1f}m)")
    
    for zw in z_walls:
        if not any(abs(w - zw) < 0.15 for w in z_walls_f):
            strip_w = max(1, int(0.08 / cs))
            px = int((zw - img_z_min) / cs)
            if 0 <= px < density_img.shape[0]:
                lo = max(0, px - strip_w)
                hi = min(density_img.shape[0], px + strip_w + 1)
                profile = density_img[lo:hi, :].max(axis=0)
                threshold = max(2, np.percentile(profile[profile > 0], 25) if np.any(profile > 0) else 2)
                is_wall = profile > threshold
                max_run = 0
                cur = 0
                for v in is_wall:
                    if v: cur += 1; max_run = max(max_run, cur)
                    else: cur = 0
                if max_run * cs >= 0.5:
                    z_walls_f = np.sort(np.append(z_walls_f, zw))
                    print(f"    Re-added Z wall at {zw:.2f} (evidence={max_run*cs:.1f}m)")
    
    print(f"  Final X walls: {[f'{w:.2f}' for w in x_walls_f]}")
    print(f"  Final Z walls: {[f'{w:.2f}' for w in z_walls_f]}")
    
    # Classify grid cells
    print("Classifying grid cells...")
    cells = classify_grid_cells(density_img, img_x_min, img_z_min, cs, x_walls_f, z_walls_f)
    room_cells = [c for c in cells if c['is_room']]
    print(f"  Total cells: {len(cells)}, Room cells: {len(room_cells)}")
    
    # Merge cells into rooms
    print("Merging cells into rooms...")
    room_groups = merge_cells_into_rooms(cells, x_walls_f, z_walls_f, min_room_area=args.min_room_area)
    print(f"  Found {len(room_groups)} room groups")
    
    # Classify rooms vs hallways
    rooms_classified = classify_rooms_vs_hallways(room_groups)
    for i, rc in enumerate(rooms_classified):
        x0, x1, z0, z1 = rc['bounds']
        label = 'HALL' if rc['is_hallway'] else f'Room {i+1}'
        print(f"  {label}: {rc['area']:.1f}m² ({x1-x0:.1f}×{z1-z0:.1f}m, aspect={rc['aspect']:.1f})")
    
    # Extract polygons
    print("Extracting polygons...")
    rooms_data = []
    room_num = 1
    hall_num = 1
    for rc in rooms_classified:
        poly = room_to_polygon(rc['cells'])
        area = compute_polygon_area(poly) if len(poly) >= 3 else rc['area']
        
        if rc['is_hallway']:
            name = f"Hall {hall_num}" if hall_num > 1 else "Hall"
            hall_num += 1
        else:
            name = f"Room {room_num}"
            room_num += 1
        
        rooms_data.append({
            'name': name,
            'polygon': poly,
            'area': area,
            'is_hallway': rc['is_hallway'],
            'bounds': rc['bounds'],
        })
    
    # Detect doors and windows
    print("Detecting openings...")
    doors = detect_doors_between_rooms(density_img, img_x_min, img_z_min, cs, rooms_data)
    windows = detect_windows_on_exterior(density_img, img_x_min, img_z_min, cs, rooms_data)
    print(f"  Doors: {len(doors)}, Windows: {len(windows)}")
    
    # Render
    total_area = sum(rd['area'] for rd in rooms_data)
    n_rooms = sum(1 for rd in rooms_data if not rd['is_hallway'])
    n_halls = sum(1 for rd in rooms_data if rd['is_hallway'])
    
    print(f"\n=== RESULTS ===")
    print(f"  Rooms: {n_rooms}, Hallways: {n_halls}")
    print(f"  Total area: {total_area:.1f}m²")
    for rd in rooms_data:
        print(f"    {rd['name']}: {rd['area']:.1f}m² {'(hallway)' if rd['is_hallway'] else ''}")
    
    # Debug panels
    render_debug_panels(density_img, img_x_min, img_z_min, cs,
                        x_walls_f, z_walls_f, cells, rooms_classified,
                        rooms_data, out_dir / f"v28_{mesh_name}_debug.png")
    
    # Clean floor plan
    render_clean_floorplan(rooms_data, doors, windows,
                           out_dir / f"v28_{mesh_name}_clean.png",
                           title=f"v28 Wall Grid — {n_rooms} rooms, {n_halls} hallway(s), {total_area:.1f}m²")
    
    # Save results JSON
    results = {
        'summary': {
            'approach': 'v28_wall_grid',
            'num_rooms': n_rooms,
            'num_hallways': n_halls,
            'total_area_m2': round(total_area, 1),
        },
        'rooms': [{
            'name': rd['name'],
            'area_m2': round(rd['area'], 1),
            'is_hallway': rd['is_hallway'],
            'polygon': [[round(p[0], 3), round(p[1], 3)] for p in rd['polygon']],
        } for rd in rooms_data],
        'doors': doors,
        'windows': windows,
        'walls': {
            'x_positions': [round(w, 3) for w in x_walls_f],
            'z_positions': [round(w, 3) for w in z_walls_f],
        },
    }
    
    json_path = out_dir / f"v28_{mesh_name}_results.json"
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Saved: {json_path}")


if __name__ == '__main__':
    main()

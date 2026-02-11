#!/usr/bin/env python3
"""
mesh2plan v28b - Wall Grid with Edge-Based Room Separation

Key insight from v28: Classifying cells by interior occupancy doesn't work because
ALL scanned areas have points (floor/ceiling). Instead, detect walls ON THE BOUNDARIES 
between grid cells. Two cells are in different rooms if the boundary between them
has high density (a wall).

Steps:
1. Build density image, detect Hough walls → grid
2. For each pair of adjacent cells, check density along their shared boundary
3. If boundary density is high → wall between them → don't merge
4. If boundary density is low → same room → merge
5. Clean rectilinear polygons, doors, windows
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


# ─── Utilities (same as v28) ───

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


def add_envelope_walls(density_img, x_min, z_min, cell_size, x_walls, z_walls):
    """Add building envelope as outer walls."""
    occupied = (density_img > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.3/cell_size), int(0.3/cell_size)))
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, k)
    rows = np.where(occupied.any(axis=1))[0]
    cols = np.where(occupied.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return x_walls, z_walls
    
    env_x_min = x_min + cols[0] * cell_size
    env_x_max = x_min + cols[-1] * cell_size
    env_z_min = z_min + rows[0] * cell_size
    env_z_max = z_min + rows[-1] * cell_size
    
    snap = 0.4
    xw = list(x_walls)
    zw = list(z_walls)
    
    if not any(abs(w - env_x_min) < snap for w in xw): xw.append(env_x_min)
    if not any(abs(w - env_x_max) < snap for w in xw): xw.append(env_x_max)
    if not any(abs(w - env_z_min) < snap for w in zw): zw.append(env_z_min)
    if not any(abs(w - env_z_max) < snap for w in zw): zw.append(env_z_max)
    
    return np.sort(xw), np.sort(zw)


# ─── Edge-based wall detection between cells ───

def check_wall_between_cells(density_img, x_min, z_min, cell_size, 
                               c1, c2, wall_threshold_ratio=2.0):
    """
    Check if there's a wall between two adjacent cells.
    Sample density along the shared boundary. If it's significantly higher
    than the interior density of the cells, there's a wall.
    
    Returns (has_wall, wall_strength)
    """
    strip_half = max(2, int(0.05 / cell_size))  # 5cm strip on each side
    
    # Determine shared edge
    if c1['i'] == c2['i']:
        # Vertically adjacent (same column, different rows)
        if c1['j'] < c2['j']:
            # c1 is below c2, shared edge is at c1's top = c2's bottom
            z_boundary = c1['z1']  # = c2['z0']
        else:
            z_boundary = c2['z1']
        
        # Sample horizontal strip along boundary
        z_px = int((z_boundary - z_min) / cell_size)
        x_lo = int((min(c1['x0'], c2['x0']) - x_min) / cell_size)
        x_hi = int((max(c1['x1'], c2['x1']) - x_min) / cell_size)
        
        z_lo = max(0, z_px - strip_half)
        z_hi = min(density_img.shape[0], z_px + strip_half + 1)
        x_lo = max(0, x_lo)
        x_hi = min(density_img.shape[1], x_hi)
        
        if z_lo >= z_hi or x_lo >= x_hi:
            return False, 0
        
        boundary_density = density_img[z_lo:z_hi, x_lo:x_hi].mean()
        
    elif c1['j'] == c2['j']:
        # Horizontally adjacent (same row, different columns)
        if c1['i'] < c2['i']:
            x_boundary = c1['x1']
        else:
            x_boundary = c2['x1']
        
        x_px = int((x_boundary - x_min) / cell_size)
        z_lo = int((min(c1['z0'], c2['z0']) - z_min) / cell_size)
        z_hi = int((max(c1['z1'], c2['z1']) - z_min) / cell_size)
        
        x_lo_px = max(0, x_px - strip_half)
        x_hi_px = min(density_img.shape[1], x_px + strip_half + 1)
        z_lo = max(0, z_lo)
        z_hi = min(density_img.shape[0], z_hi)
        
        if x_lo_px >= x_hi_px or z_lo >= z_hi:
            return False, 0
        
        boundary_density = density_img[z_lo:z_hi, x_lo_px:x_hi_px].mean()
    else:
        return False, 0
    
    # Get interior density of both cells (center 60%)
    def cell_interior_density(c):
        inset = int(0.15 / cell_size)
        px0 = int((c['x0'] - x_min) / cell_size) + inset
        px1 = int((c['x1'] - x_min) / cell_size) - inset
        pz0 = int((c['z0'] - z_min) / cell_size) + inset
        pz1 = int((c['z1'] - z_min) / cell_size) - inset
        px0 = max(0, px0); px1 = min(density_img.shape[1], max(px0+1, px1))
        pz0 = max(0, pz0); pz1 = min(density_img.shape[0], max(pz0+1, pz1))
        region = density_img[pz0:pz1, px0:px1]
        return region.mean() if region.size > 0 else 0
    
    int1 = cell_interior_density(c1)
    int2 = cell_interior_density(c2)
    avg_interior = (int1 + int2) / 2
    
    # Wall if boundary density is significantly higher than interior
    if avg_interior < 0.1:
        # Both cells essentially empty
        has_wall = boundary_density > 1.0
    else:
        ratio = boundary_density / max(avg_interior, 0.01)
        has_wall = ratio > wall_threshold_ratio and boundary_density > 0.5
    
    return has_wall, boundary_density


def build_cell_graph(density_img, x_min, z_min, cell_size, x_walls, z_walls,
                      wall_threshold_ratio=2.0, min_cell_size=0.2):
    """
    Build grid cells and determine connectivity (wall/no-wall between neighbors).
    """
    # Create cells
    cells = []
    cell_idx = {}  # (i, j) -> index
    
    for i in range(len(x_walls) - 1):
        for j in range(len(z_walls) - 1):
            x0, x1 = x_walls[i], x_walls[i+1]
            z0, z1 = z_walls[j], z_walls[j+1]
            w, h = x1 - x0, z1 - z0
            
            if w < min_cell_size or h < min_cell_size:
                continue
            
            # Check if cell has any scan data
            inset = int(0.05 / cell_size)
            px0 = int((x0 - x_min) / cell_size) + inset
            px1 = int((x1 - x_min) / cell_size) - inset
            pz0 = int((z0 - z_min) / cell_size) + inset
            pz1 = int((z1 - z_min) / cell_size) - inset
            px0 = max(0, px0); px1 = min(density_img.shape[1], max(px0+1, px1))
            pz0 = max(0, pz0); pz1 = min(density_img.shape[0], max(pz0+1, pz1))
            
            region = density_img[pz0:pz1, px0:px1]
            occupancy = np.count_nonzero(region) / max(1, region.size)
            
            if occupancy < 0.05:  # Nearly empty — not scanned
                continue
            
            idx = len(cells)
            cell_idx[(i, j)] = idx
            cells.append({
                'i': i, 'j': j,
                'x0': x0, 'x1': x1, 'z0': z0, 'z1': z1,
                'w': w, 'h': h, 'area': w * h,
                'occupancy': occupancy,
                'idx': idx,
            })
    
    # Check walls between adjacent cells
    edges = []  # (idx1, idx2, has_wall, strength)
    
    for c in cells:
        i, j = c['i'], c['j']
        
        for di, dj in [(1, 0), (0, 1)]:  # right and up neighbors
            ni, nj = i + di, j + dj
            if (ni, nj) in cell_idx:
                n_idx = cell_idx[(ni, nj)]
                n_cell = cells[n_idx]
                
                has_wall, strength = check_wall_between_cells(
                    density_img, x_min, z_min, cell_size,
                    c, n_cell, wall_threshold_ratio
                )
                
                edges.append((c['idx'], n_idx, has_wall, strength))
    
    return cells, edges


def merge_cells_by_edges(cells, edges, min_room_area=1.5):
    """Merge cells connected by non-wall edges using union-find."""
    n = len(cells)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b
    
    for idx1, idx2, has_wall, _ in edges:
        if not has_wall:
            union(idx1, idx2)
    
    # Group by root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(cells[i])
    
    # Filter by min area
    rooms = []
    for g in groups.values():
        area = sum(c['area'] for c in g)
        if area >= min_room_area:
            rooms.append(g)
    
    rooms.sort(key=lambda g: -sum(c['area'] for c in g))
    return rooms


def room_to_polygon(cells):
    """Convert merged cells to rectilinear polygon."""
    if not cells:
        return []
    
    i_vals = [c['i'] for c in cells]
    j_vals = [c['j'] for c in cells]
    i_min, i_max = min(i_vals), max(i_vals)
    j_min, j_max = min(j_vals), max(j_vals)
    
    all_keys = set((c['i'], c['j']) for c in cells)
    
    # Check if rectangle
    expected = set((i, j) for i in range(i_min, i_max+1) for j in range(j_min, j_max+1))
    # Only count keys that are actually in our cell set
    if all_keys >= expected & all_keys:
        pass  # may be rectangle
    
    # Build binary grid
    ni = i_max - i_min + 1
    nj = j_max - j_min + 1
    grid = np.zeros((nj, ni), dtype=np.uint8)
    
    x_bounds = {}
    z_bounds = {}
    for c in cells:
        gi = c['i'] - i_min
        gj = c['j'] - j_min
        grid[gj, gi] = 1
        x_bounds[c['i']] = (c['x0'], c['x1'])
        z_bounds[c['j']] = (c['z0'], c['z1'])
    
    # Simple case: rectangle
    if grid.sum() == ni * nj:
        x0 = min(c['x0'] for c in cells)
        x1 = max(c['x1'] for c in cells)
        z0 = min(c['z0'] for c in cells)
        z1 = max(c['z1'] for c in cells)
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    # Non-rectangular: trace the outline
    # Use marching squares on the binary grid
    poly = trace_rectilinear_outline(grid, x_bounds, z_bounds, i_min, j_min)
    return poly


def trace_rectilinear_outline(grid, x_bounds, z_bounds, i_off, j_off):
    """Trace rectilinear outline of a binary grid, returning world-coordinate polygon."""
    nj, ni = grid.shape
    
    # Pad grid to ensure we can trace the boundary
    padded = np.zeros((nj + 2, ni + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = grid
    
    # Find contour using cv2
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback
        x0 = min(v[0] for v in x_bounds.values())
        x1 = max(v[1] for v in x_bounds.values())
        z0 = min(v[0] for v in z_bounds.values())
        z1 = max(v[1] for v in z_bounds.values())
        return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    
    contour = max(contours, key=cv2.contourArea)
    
    # Build sorted lists of cell boundaries
    all_i = sorted(x_bounds.keys())
    all_j = sorted(z_bounds.keys())
    
    # X boundaries: i_off+0 -> x_bounds[i_off][0], i_off+0 end -> x_bounds[i_off][1], etc.
    x_edges = []
    for i in all_i:
        x_edges.append(x_bounds[i][0])
    if all_i:
        x_edges.append(x_bounds[all_i[-1]][1])
    
    z_edges = []
    for j in all_j:
        z_edges.append(z_bounds[j][0])
    if all_j:
        z_edges.append(z_bounds[all_j[-1]][1])
    
    # Map contour pixels to world coords
    # Contour is in padded grid coords: (col, row) with padding offset of 1
    poly = []
    for pt in contour.reshape(-1, 2):
        col, row = pt[0] - 1, pt[1] - 1  # Remove padding offset
        
        # col maps to x_edges index, row maps to z_edges index
        xi = min(col, len(x_edges) - 1)
        zi = min(row, len(z_edges) - 1)
        xi = max(0, xi)
        zi = max(0, zi)
        
        wx = x_edges[xi] if xi < len(x_edges) else x_edges[-1]
        wz = z_edges[zi] if zi < len(z_edges) else z_edges[-1]
        
        poly.append([wx, wz])
    
    # Remove collinear and duplicate points
    cleaned = remove_collinear(poly)
    return cleaned if len(cleaned) >= 3 else poly[:4]


def remove_collinear(poly):
    """Remove collinear and duplicate points from polygon."""
    if len(poly) < 3:
        return poly
    
    # Remove near-duplicates
    cleaned = [poly[0]]
    for p in poly[1:]:
        if abs(p[0] - cleaned[-1][0]) > 0.01 or abs(p[1] - cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    
    if len(cleaned) < 3:
        return cleaned
    
    # Remove collinear
    result = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i-1) % n]
        curr = cleaned[i]
        nxt = cleaned[(i+1) % n]
        cross = (curr[0]-prev[0])*(nxt[1]-curr[1]) - (curr[1]-prev[1])*(nxt[0]-curr[0])
        if abs(cross) > 0.001:
            result.append(curr)
    
    return result if len(result) >= 3 else cleaned


def classify_rooms_hallways(room_groups, max_narrow_dim=1.2):
    """Classify based on narrowness."""
    results = []
    for group in room_groups:
        x0 = min(c['x0'] for c in group)
        x1 = max(c['x1'] for c in group)
        z0 = min(c['z0'] for c in group)
        z1 = max(c['z1'] for c in group)
        w, h = x1 - x0, z1 - z0
        area = sum(c['area'] for c in group)
        min_dim = min(w, h)
        aspect = max(w, h) / min_dim if min_dim > 0 else 999
        
        is_hallway = min_dim < max_narrow_dim and aspect > 2.5
        
        results.append({
            'cells': group,
            'is_hallway': is_hallway,
            'area': area,
            'bounds': (x0, x1, z0, z1),
            'w': w, 'h': h,
        })
    return results


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    return abs(sum(poly[i][0]*poly[(i+1)%n][1] - poly[(i+1)%n][0]*poly[i][1] for i in range(n))) / 2

def polygon_centroid(poly):
    n = len(poly)
    if n == 0: return 0, 0
    return sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n


# ─── Door/Window detection (simplified) ───

def find_shared_edges(poly1, poly2, tol=0.2):
    segs = []
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i+1)%n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j+1)%n2]
            if abs(a1[0]-a2[0]) < 0.05 and abs(b1[0]-b2[0]) < 0.05 and abs(a1[0]-b1[0]) < tol:
                alo, ahi = min(a1[1],a2[1]), max(a1[1],a2[1])
                blo, bhi = min(b1[1],b2[1]), max(b1[1],b2[1])
                lo, hi = max(alo,blo), min(ahi,bhi)
                if hi - lo > 0.3:
                    segs.append({'type':'v','x':(a1[0]+b1[0])/2,'lo':lo,'hi':hi})
            if abs(a1[1]-a2[1]) < 0.05 and abs(b1[1]-b2[1]) < 0.05 and abs(a1[1]-b1[1]) < tol:
                alo, ahi = min(a1[0],a2[0]), max(a1[0],a2[0])
                blo, bhi = min(b1[0],b2[0]), max(b1[0],b2[0])
                lo, hi = max(alo,blo), min(ahi,bhi)
                if hi - lo > 0.3:
                    segs.append({'type':'h','z':(a1[1]+b1[1])/2,'lo':lo,'hi':hi})
    return segs

def find_gap(density_img, x_min, z_min, cs, seg):
    sw = max(1, int(0.08/cs))
    if seg['type'] == 'v':
        xp = int((seg['x']-x_min)/cs)
        lp = int((seg['lo']-z_min)/cs)
        hp = int((seg['hi']-z_min)/cs)
        xl = max(0,xp-sw); xh = min(density_img.shape[1],xp+sw+1)
        lp = max(0,lp); hp = min(density_img.shape[0],hp)
        if xl >= xh or lp >= hp: return None
        prof = density_img[lp:hp, xl:xh].sum(axis=1)
    else:
        zp = int((seg['z']-z_min)/cs)
        lp = int((seg['lo']-x_min)/cs)
        hp = int((seg['hi']-x_min)/cs)
        zl = max(0,zp-sw); zh = min(density_img.shape[0],zp+sw+1)
        lp = max(0,lp); hp = min(density_img.shape[1],hp)
        if zl >= zh or lp >= hp: return None
        prof = density_img[zl:zh, lp:hp].sum(axis=0)
    
    if len(prof) == 0 or not np.any(prof > 0): return None
    thr = np.percentile(prof[prof>0], 30) if np.any(prof>0) else 1
    is_gap = prof < max(thr, 1)
    lbl, ng = ndlabel(is_gap)
    for g in range(1, ng+1):
        gi = np.where(lbl==g)[0]
        gl = len(gi) * cs
        if 0.6 < gl < 1.5:
            center = gi.mean()
            if seg['type'] == 'v':
                return {'x':seg['x'], 'z':z_min+(lp+center)*cs, 'width':gl, 'orientation':'vertical'}
            else:
                return {'x':x_min+(lp+center)*cs, 'z':seg['z'], 'width':gl, 'orientation':'horizontal'}
    return None


# ─── Rendering ───

ROOM_COLORS = ['#E8F5E9','#E3F2FD','#FFF3E0','#F3E5F5','#FFFDE7','#E0F7FA','#FCE4EC','#F1F8E9']
HALLWAY_COLOR = '#F5F5F5'

def render_debug(density_img, x_min, z_min, cs, x_walls, z_walls,
                  cells, edges, rooms_data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    extent = [x_min, x_min + density_img.shape[1]*cs, z_min, z_min + density_img.shape[0]*cs]
    
    # P1: Density + walls
    ax = axes[0,0]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal')
    for xw in x_walls: ax.axvline(xw, color='cyan', alpha=0.5, lw=1)
    for zw in z_walls: ax.axhline(zw, color='lime', alpha=0.5, lw=1)
    ax.set_title('1. Density + Walls')
    
    # P2: Cell graph with wall edges
    ax = axes[0,1]
    ax.set_facecolor('#222')
    for c in cells:
        rect = patches.Rectangle((c['x0'],c['z0']), c['w'], c['h'],
                                  lw=0.5, ec='white', fc='#4CAF50', alpha=0.4)
        ax.add_patch(rect)
    # Draw wall edges in red, open edges in green
    for idx1, idx2, has_wall, strength in edges:
        c1, c2 = cells[idx1], cells[idx2]
        cx1 = (c1['x0']+c1['x1'])/2
        cz1 = (c1['z0']+c1['z1'])/2
        cx2 = (c2['x0']+c2['x1'])/2
        cz2 = (c2['z0']+c2['z1'])/2
        color = 'red' if has_wall else 'lime'
        lw = 2 if has_wall else 1
        ax.plot([cx1,cx2], [cz1,cz2], color=color, lw=lw, alpha=0.8)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('2. Cell Graph (red=wall, green=open)')
    
    # P3: Merged rooms (cells colored by room)
    ax = axes[1,0]
    ax.set_facecolor('white')
    ci = 0
    for rd in rooms_data:
        color = HALLWAY_COLOR if rd['is_hallway'] else ROOM_COLORS[ci % len(ROOM_COLORS)]
        if not rd['is_hallway']: ci += 1
        for c in rd.get('room_cells', []):
            rect = patches.Rectangle((c['x0'],c['z0']), c['w'], c['h'],
                                      lw=0.5, ec='#999', fc=color, alpha=0.8)
            ax.add_patch(rect)
        bx = rd['bounds']
        ax.text((bx[0]+bx[1])/2, (bx[2]+bx[3])/2, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('3. Merged Rooms')
    
    # P4: Clean polygons
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
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=2.5)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold', color='#333')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('4. Clean Polygons')
    
    fig.suptitle('v28b Wall Grid (Edge-Based) — Debug', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def render_clean(rooms_data, doors, windows, output_path, title="v28b"):
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
    
    for win in windows:
        x, z, l = win['x'], win['z'], win.get('length', 1.0)
        if win['orientation'] == 'vertical':
            ax.plot([x,x], [z-l/2,z+l/2], color='white', lw=5)
            ax.plot([x-0.03,x-0.03], [z-l/2,z+l/2], 'k-', lw=1.5)
            ax.plot([x+0.03,x+0.03], [z-l/2,z+l/2], 'k-', lw=1.5)
        else:
            ax.plot([x-l/2,x+l/2], [z,z], color='white', lw=5)
            ax.plot([x-l/2,x+l/2], [z-0.03,z-0.03], 'k-', lw=1.5)
            ax.plot([x-l/2,x+l/2], [z+0.03,z+0.03], 'k-', lw=1.5)
    
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=9, fontweight='bold', color='#333')
    
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_z)-margin, max(all_z)+margin)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--nms', type=float, default=0.15)
    parser.add_argument('--wall-ratio', type=float, default=2.0, help='Boundary/interior density ratio for wall detection')
    parser.add_argument('--min-room-area', type=float, default=1.5)
    parser.add_argument('-o', '--output', default='results/v28b_wallgrid')
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
    
    x_walls, z_walls = hough_wall_positions(density_img, img_x_min, img_z_min, cs, nms_dist=args.nms)
    print(f"  Hough X walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Hough Z walls: {[f'{w:.2f}' for w in z_walls]}")
    
    x_walls, z_walls = add_envelope_walls(density_img, img_x_min, img_z_min, cs, x_walls, z_walls)
    print(f"  With envelope X: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  With envelope Z: {[f'{w:.2f}' for w in z_walls]}")
    
    print("Building cell graph...")
    cells, edges = build_cell_graph(density_img, img_x_min, img_z_min, cs,
                                     x_walls, z_walls, wall_threshold_ratio=args.wall_ratio)
    n_walls = sum(1 for _,_,hw,_ in edges if hw)
    n_open = sum(1 for _,_,hw,_ in edges if not hw)
    print(f"  Cells: {len(cells)}, Edges: {len(edges)} ({n_walls} walls, {n_open} open)")
    
    # Debug edge info
    for idx1, idx2, hw, strength in edges:
        c1, c2 = cells[idx1], cells[idx2]
        label = "WALL" if hw else "open"
        print(f"    ({c1['i']},{c1['j']})-({c2['i']},{c2['j']}): {label} str={strength:.2f}")
    
    print("Merging rooms...")
    room_groups = merge_cells_by_edges(cells, edges, min_room_area=args.min_room_area)
    print(f"  Room groups: {len(room_groups)}")
    
    classified = classify_rooms_hallways(room_groups)
    
    # Build rooms_data
    rooms_data = []
    rn, hn = 1, 1
    for rc in classified:
        poly = room_to_polygon(rc['cells'])
        area = compute_polygon_area(poly) if len(poly) >= 3 else rc['area']
        
        if rc['is_hallway']:
            name = f"Hall" if hn == 1 else f"Hall {hn}"
            hn += 1
        else:
            name = f"Room {rn}"
            rn += 1
        
        rooms_data.append({
            'name': name,
            'polygon': poly,
            'area': area,
            'is_hallway': bool(rc['is_hallway']),
            'bounds': rc['bounds'],
            'room_cells': rc['cells'],
        })
    
    # Doors
    doors = []
    for i in range(len(rooms_data)):
        for j in range(i+1, len(rooms_data)):
            shared = find_shared_edges(rooms_data[i]['polygon'], rooms_data[j]['polygon'])
            for seg in shared:
                d = find_gap(density_img, img_x_min, img_z_min, cs, seg)
                if d: doors.append(d)
    
    # Windows (on exterior edges not shared)
    windows = []
    
    total_area = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if not rd['is_hallway'])
    nh = sum(1 for rd in rooms_data if rd['is_hallway'])
    
    print(f"\n=== RESULTS ===")
    print(f"  Rooms: {nr}, Hallways: {nh}, Total: {total_area:.1f}m²")
    for rd in rooms_data:
        print(f"    {rd['name']}: {rd['area']:.1f}m² {'(hall)' if rd['is_hallway'] else ''} poly={len(rd['polygon'])}v")
    
    render_debug(density_img, img_x_min, img_z_min, cs, x_walls, z_walls,
                  cells, edges, rooms_data, out_dir / f"v28b_{mesh_name}_debug.png")
    
    title = f"v28b Wall Grid — {nr} rooms, {nh} hall(s), {total_area:.1f}m²"
    render_clean(rooms_data, doors, windows, out_dir / f"v28b_{mesh_name}_clean.png", title)
    
    # JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    
    results = {
        'approach': 'v28b_wall_grid_edge',
        'rooms': [{
            'name': rd['name'], 'area_m2': round(rd['area'],1),
            'is_hallway': rd['is_hallway'],
            'polygon': [[round(p[0],3), round(p[1],3)] for p in rd['polygon']],
        } for rd in rooms_data],
        'doors': doors, 'windows': windows,
        'walls': {'x': [round(w,3) for w in x_walls], 'z': [round(w,3) for w in z_walls]},
    }
    with open(out_dir / f"v28b_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    
    print("Done!")

if __name__ == '__main__':
    main()

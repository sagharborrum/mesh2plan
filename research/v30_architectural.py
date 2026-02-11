#!/usr/bin/env python3
"""
mesh2plan v30 - Architectural Floor Plan

Combines v29's mask-cut room detection with v27h's clean architectural rendering.
Goal: Match the AI-generated reference floor plan quality.

Improvements over v29b:
1. Better wall thickness rendering (draw walls as filled rectangles, not lines)
2. Door detection with proper arc rendering
3. Window detection with parallel line rendering
4. Room labels with area and estimated function
5. Scale bar and compass rose
6. Cleaner polygon extraction with density-aware bounds
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndlabel, uniform_filter1d


# ─── Utilities (from v29) ───

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


# ─── Room detection (from v29) ───

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


def extract_and_merge_rooms(cut_mask, density_img, x_min, z_min, cell_size, 
                             x_walls, z_walls, target_rooms=5):
    """Extract rooms from cut mask, merge fragments to target count."""
    lbl, n = ndlabel(cut_mask)
    raw_rooms = []
    for i in range(1, n + 1):
        mask = (lbl == i)
        area = np.sum(mask) * cell_size * cell_size
        if area < 0.5: continue
        raw_rooms.append({'mask': mask, 'area_m2': area})
    
    raw_rooms.sort(key=lambda r: -r['area_m2'])
    
    # Merge smallest into nearest neighbor until target count
    def is_narrow(r, cs):
        rows, cols = np.where(r['mask'])
        if len(rows) == 0: return False
        return min((rows.max()-rows.min())*cs, (cols.max()-cols.min())*cs) < 1.5
    
    while len(raw_rooms) > target_rooms:
        raw_rooms.sort(key=lambda r: r['area_m2'])
        merged = False
        for i, r in enumerate(raw_rooms):
            dilated = cv2.dilate(r['mask'].astype(np.uint8), np.ones((15,15), np.uint8))
            candidates = []
            for j, r2 in enumerate(raw_rooms):
                if i == j: continue
                overlap = np.sum(dilated & r2['mask'].astype(np.uint8))
                if overlap > 0:
                    both_narrow = is_narrow(r, cell_size) and is_narrow(r2, cell_size)
                    score = (10000 if both_narrow else 0) + overlap - r2['area_m2']
                    candidates.append((j, score))
            if candidates:
                candidates.sort(key=lambda c: -c[1])
                best_j = candidates[0][0]
                raw_rooms[best_j]['mask'] = raw_rooms[best_j]['mask'] | r['mask']
                raw_rooms[best_j]['area_m2'] += r['area_m2']
                raw_rooms.pop(i)
                merged = True
                break
        if not merged: break
    
    # Build room data with polygons
    rooms_data = []
    raw_rooms.sort(key=lambda r: -r['area_m2'])
    
    for r in raw_rooms:
        mask = r['mask']
        rows, cols = np.where(mask)
        if len(rows) == 0: continue
        
        # Density-aware bounds
        masked_density = density_img * mask
        dr, dc = np.where(masked_density > 0)
        if len(dr) > 0:
            wx_min = x_min + np.percentile(dc, 1) * cell_size
            wx_max = x_min + np.percentile(dc, 99) * cell_size
            wz_min = z_min + np.percentile(dr, 1) * cell_size
            wz_max = z_min + np.percentile(dr, 99) * cell_size
        else:
            wx_min = x_min + cols.min() * cell_size
            wx_max = x_min + cols.max() * cell_size
            wz_min = z_min + rows.min() * cell_size
            wz_max = z_min + rows.max() * cell_size
        
        def snap(val, positions, max_snap=0.25):
            if len(positions) == 0: return val
            dists = np.abs(np.array(positions) - val)
            idx = np.argmin(dists)
            return float(positions[idx]) if dists[idx] < max_snap else val
        
        x0 = snap(wx_min, x_walls)
        x1 = snap(wx_max, x_walls)
        z0 = snap(wz_min, z_walls)
        z1 = snap(wz_max, z_walls)
        
        # Check fill ratio for L-shape
        bbox_area = (cols.max()-cols.min()+1) * (rows.max()-rows.min()+1)
        fill_ratio = len(rows) / max(1, bbox_area)
        
        if fill_ratio > 0.80:
            poly = [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
        else:
            # Contour-based
            mask_u8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                epsilon = max(3, int(0.2 / cell_size))
                approx = cv2.approxPolyDP(contour, epsilon, True)
                poly = []
                for pt in approx.reshape(-1, 2):
                    wx = snap(x_min + pt[0] * cell_size, x_walls)
                    wz = snap(z_min + pt[1] * cell_size, z_walls)
                    poly.append([wx, wz])
                # Make rectilinear
                poly = make_rectilinear(poly)
                poly = remove_collinear(poly)
            else:
                poly = [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
        
        area = compute_polygon_area(poly) if len(poly) >= 3 else r['area_m2']
        
        # Classify
        w = abs(x1 - x0)
        h = abs(z1 - z0)
        min_dim = min(w, h)
        max_dim = max(w, h)
        
        # Use distance transform for better hallway detection
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_half_width = dist.max() * cell_size
        
        aspect_ratio = max_dim / max(min_dim, 0.01)
        print(f"    classify: half_w={max_half_width:.2f}m, dim={min_dim:.2f}×{max_dim:.2f}m, aspect={aspect_ratio:.1f}")
        
        if max_half_width < 1.0 and aspect_ratio > 2.5:
            room_type = 'hallway'
        elif area < 3.0:
            room_type = 'closet'
        else:
            room_type = 'room'
        
        rooms_data.append({
            'polygon': poly,
            'area': area,
            'type': room_type,
            'mask': mask,
            'bounds': (x0, x1, z0, z1),
        })
    
    # Name rooms
    rn, hn, cn = 1, 1, 1
    for rd in rooms_data:
        if rd['type'] == 'hallway':
            rd['name'] = "Hallway" if hn == 1 else f"Hallway {hn}"
            hn += 1
        elif rd['type'] == 'closet':
            rd['name'] = "Closet" if cn == 1 else f"Closet {cn}"
            cn += 1
        else:
            rd['name'] = f"Room {rn}"
            rn += 1
    
    return rooms_data


def make_rectilinear(poly):
    if len(poly) < 3: return poly
    result = [poly[0]]
    for i in range(1, len(poly)):
        prev = result[-1]; curr = poly[i]
        dx, dz = abs(curr[0]-prev[0]), abs(curr[1]-prev[1])
        if dx > 0.05 and dz > 0.05:
            if dx < dz: result.append([curr[0], prev[1]])
            else: result.append([prev[0], curr[1]])
        result.append(curr)
    return result

def remove_collinear(poly):
    if len(poly) < 3: return poly
    cleaned = [poly[0]]
    for p in poly[1:]:
        if abs(p[0]-cleaned[-1][0]) > 0.01 or abs(p[1]-cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    if len(cleaned) < 3: return cleaned
    result = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i-1)%n]; curr = cleaned[i]; nxt = cleaned[(i+1)%n]
        cross = (curr[0]-prev[0])*(nxt[1]-curr[1]) - (curr[1]-prev[1])*(nxt[0]-curr[0])
        if abs(cross) > 0.001: result.append(curr)
    return result if len(result) >= 3 else cleaned

def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    return abs(sum(poly[i][0]*poly[(i+1)%n][1] - poly[(i+1)%n][0]*poly[i][1] for i in range(n))) / 2

def polygon_centroid(poly):
    n = len(poly)
    if n == 0: return 0, 0
    return sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n


# ─── Door/Window Detection ───

def detect_doors(density_img, x_min, z_min, cs, rooms_data):
    """Detect doors as gaps between adjacent rooms."""
    doors = []
    for i in range(len(rooms_data)):
        pi = rooms_data[i]['polygon']
        for j in range(i+1, len(rooms_data)):
            pj = rooms_data[j]['polygon']
            # Check all edge pairs for shared walls
            for ei in range(len(pi)):
                a1, a2 = pi[ei], pi[(ei+1)%len(pi)]
                for ej in range(len(pj)):
                    b1, b2 = pj[ej], pj[(ej+1)%len(pj)]
                    door = check_shared_edge_for_door(density_img, x_min, z_min, cs, a1, a2, b1, b2)
                    if door:
                        doors.append(door)
    return doors

def check_shared_edge_for_door(density_img, x_min, z_min, cs, a1, a2, b1, b2, tol=0.2):
    """Check if two edges share a wall with a door gap."""
    sw = max(1, int(0.08/cs))
    
    # Vertical edges
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
    
    # Horizontal edges
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

def render_architectural(rooms_data, doors, output_path, rotation_angle=0):
    """Render clean architectural floor plan matching reference quality."""
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
    wall_w = 0.08  # wall thickness in meters
    
    # 1. Draw room fills
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        if rd['type'] == 'hallway':
            color = ROOM_COLORS['hallway']
        elif rd['type'] == 'closet':
            color = ROOM_COLORS['closet']
        else:
            color = ROOM_COLORS['room'][ci % len(ROOM_COLORS['room'])]
            ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.9, zorder=1)
    
    # 2. Draw walls as thick filled rectangles
    drawn_edges = set()
    for rd in rooms_data:
        poly = rd['polygon']
        n = len(poly)
        for i in range(n):
            p1 = poly[i]; p2 = poly[(i+1)%n]
            edge_key = (round(min(p1[0],p2[0]),2), round(min(p1[1],p2[1]),2),
                       round(max(p1[0],p2[0]),2), round(max(p1[1],p2[1]),2))
            
            if abs(p1[0]-p2[0]) < 0.05:  # Vertical wall
                x = p1[0]
                z_lo, z_hi = min(p1[1],p2[1]), max(p1[1],p2[1])
                rect = patches.Rectangle((x - wall_w/2, z_lo), wall_w, z_hi-z_lo,
                                          facecolor='black', edgecolor='none', zorder=2)
                ax.add_patch(rect)
            elif abs(p1[1]-p2[1]) < 0.05:  # Horizontal wall
                z = p1[1]
                x_lo, x_hi = min(p1[0],p2[0]), max(p1[0],p2[0])
                rect = patches.Rectangle((x_lo, z - wall_w/2), x_hi-x_lo, wall_w,
                                          facecolor='black', edgecolor='none', zorder=2)
                ax.add_patch(rect)
            else:
                ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=3, zorder=2)
    
    # 3. Draw doors (clear wall gap + arc)
    for door in doors:
        x, z = door['x'], door['z']
        w = door.get('width', 0.8)
        
        if door['orientation'] == 'vertical':
            # Clear wall at door
            rect = patches.Rectangle((x - wall_w, z - w/2), wall_w*2, w,
                                      facecolor='white', edgecolor='none', zorder=3)
            ax.add_patch(rect)
            # Door swing arc
            arc = Arc((x, z - w/2), w, w, angle=0, theta1=0, theta2=90,
                      color='#666', linewidth=1.5, zorder=4)
            ax.add_patch(arc)
            # Door line
            ax.plot([x, x + w/2], [z - w/2, z - w/2], color='#666', lw=1.5, zorder=4)
        else:
            rect = patches.Rectangle((x - w/2, z - wall_w), w, wall_w*2,
                                      facecolor='white', edgecolor='none', zorder=3)
            ax.add_patch(rect)
            arc = Arc((x - w/2, z), w, w, angle=0, theta1=0, theta2=90,
                      color='#666', linewidth=1.5, zorder=4)
            ax.add_patch(arc)
            ax.plot([x - w/2, x - w/2], [z, z + w/2], color='#666', lw=1.5, zorder=4)
    
    # 4. Room labels
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        cx, cz = polygon_centroid(poly)
        area = rd['area']
        name = rd['name']
        
        # Font size based on room area
        fs = 9 if area > 5 else 7
        
        ax.text(cx, cz + 0.15, name, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#333', zorder=5)
        ax.text(cx, cz - 0.15, f"{area:.1f} m²", ha='center', va='center',
                fontsize=fs-1, color='#666', zorder=5)
    
    # 5. Scale bar
    x_range = max(all_x) - min(all_x)
    scale_len = 1.0  # 1 meter
    sx = min(all_x) + 0.3
    sy = min(all_z) - 0.4
    ax.plot([sx, sx + scale_len], [sy, sy], 'k-', lw=2)
    ax.plot([sx, sx], [sy - 0.05, sy + 0.05], 'k-', lw=2)
    ax.plot([sx + scale_len, sx + scale_len], [sy - 0.05, sy + 0.05], 'k-', lw=2)
    ax.text(sx + scale_len/2, sy - 0.15, '1 m', ha='center', va='top', fontsize=8)
    
    # 6. Title
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
    ax.set_title(f"Floor Plan\n{subtitle}", fontsize=13, fontweight='bold', pad=15, color='#333')
    ax.axis('off')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--nms', type=float, default=0.3)
    parser.add_argument('-o', '--output', default='results/v30_architectural')
    args = parser.parse_args()
    
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = Path(args.mesh).stem
    
    print(f"Loading: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh')
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    up_idx, _ = detect_up_axis(mesh)
    rx, rz = project_vertices(mesh, up_idx)
    angle = find_dominant_angle(rx, rz, cell=args.cell)
    print(f"  Rotation: {angle:.1f}°")
    
    cos_a, sin_a = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a
    
    density_img, x_min, z_min, cs = build_density_image(rx2, rz2, cell_size=args.cell)
    room_mask = build_room_mask(density_img, cs)
    
    x_walls, z_walls, x_str, z_str = hough_wall_positions(density_img, x_min, z_min, cs, nms_dist=args.nms)
    print(f"  Hough walls: {len(x_walls)}X, {len(z_walls)}Z")
    
    # Score and select walls
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
    print(f"  Selected cuts: X={[f'{w:.2f}' for w in sel_x]}, Z={[f'{w:.2f}' for w in sel_z]}")
    
    sel_x_arr = np.array(sorted(sel_x))
    sel_z_arr = np.array(sorted(sel_z))
    sel_x_str = np.array([next(s for w,s,_ in x_scored if w==xw) for xw in sel_x_arr])
    sel_z_str = np.array([next(s for w,s,_ in z_scored if w==zw) for zw in sel_z_arr])
    
    cut_mask, valid_x, valid_z = cut_mask_with_walls(
        room_mask, density_img, x_min, z_min, cs,
        sel_x_arr, sel_z_arr, sel_x_str, sel_z_str, min_wall_run=0.5)
    
    print("Extracting rooms...")
    rooms_data = extract_and_merge_rooms(cut_mask, density_img, x_min, z_min, cs,
                                          x_walls, z_walls, target_rooms=5)
    
    for rd in rooms_data:
        print(f"  {rd['name']}: {rd['area']:.1f}m² ({rd['type']}) poly={len(rd['polygon'])}v")
    
    # Detect doors
    doors = detect_doors(density_img, x_min, z_min, cs, rooms_data)
    print(f"  Doors: {len(doors)}")
    
    # Render
    render_architectural(rooms_data, doors, out_dir / f"v30_{mesh_name}_plan.png", angle)
    
    # Also render debug overlay
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
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
    ax.set_title('Overlay')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"v30_{mesh_name}_debug.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    
    results = {
        'approach': 'v30_architectural',
        'rooms': [{
            'name': rd['name'], 'area_m2': round(rd['area'],1),
            'type': rd['type'],
            'polygon': [[round(p[0],3), round(p[1],3)] for p in rd['polygon']],
        } for rd in rooms_data],
        'doors': doors,
        'walls': {'x': [round(w,3) for w in x_walls], 'z': [round(w,3) for w in z_walls]},
    }
    with open(out_dir / f"v30_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    
    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if rd['type'] == 'room')
    nh = sum(1 for rd in rooms_data if rd['type'] == 'hallway')
    nc = sum(1 for rd in rooms_data if rd['type'] == 'closet')
    print(f"\n=== v30 RESULTS ===")
    print(f"  {nr} rooms, {nh} hallways, {nc} closets — {total:.1f}m²")
    print("Done!")

if __name__ == '__main__':
    main()

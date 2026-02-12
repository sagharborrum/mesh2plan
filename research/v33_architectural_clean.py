#!/usr/bin/env python3
"""
mesh2plan v33 - Architectural Clean Render

Based on v32_strip_merge pipeline. Changes:
- Professional architectural floor plan rendering (no colored fills, thick black walls,
  proper door arcs, window symbols on exterior walls)
- SVG output in addition to PNG
- Window detection on exterior walls
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
import svgwrite


# ─── Utilities (from v32) ───

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


# ─── Room detection (from v32) ───

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
                             x_walls, z_walls, x_cuts, z_cuts, target_rooms=5):
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
    
    if not raw_rooms: return []
    
    x_cuts_sorted = sorted(x_cuts)
    def get_strip(cx):
        for i, cut in enumerate(x_cuts_sorted):
            if cx < cut: return i
        return len(x_cuts_sorted)
    
    for r in raw_rooms:
        r['strip'] = get_strip(r['cx'])
    
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
            for r in big_rooms: merged_rooms.append(r)
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
    
    raw_rooms = merged_rooms
    rooms_data = []
    raw_rooms.sort(key=lambda r: -r['area_m2'])
    
    for r in raw_rooms:
        mask = r['mask']
        rows, cols = np.where(mask)
        if len(rows) == 0: continue
        
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
        
        x0 = snap(wx_min, x_walls); x1 = snap(wx_max, x_walls)
        z0 = snap(wz_min, z_walls); z1 = snap(wz_max, z_walls)
        
        bbox_area = (cols.max()-cols.min()+1) * (rows.max()-rows.min()+1)
        fill_ratio = len(rows) / max(1, bbox_area)
        
        if fill_ratio > 0.80:
            poly = [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
        else:
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
                poly = make_rectilinear(poly)
                poly = remove_collinear(poly)
            else:
                poly = [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
        
        area = compute_polygon_area(poly) if len(poly) >= 3 else r['area_m2']
        
        w = abs(x1 - x0); h = abs(z1 - z0)
        min_dim = min(w, h); max_dim = max(w, h)
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_half_width = dist.max() * cell_size
        aspect_ratio = max_dim / max(min_dim, 0.01)
        
        if len(x_cuts) >= 2:
            x_lo = min(x_cuts); x_hi = max(x_cuts)
            cx = (x0 + x1) / 2
            is_center = x_lo < cx < x_hi
        else:
            is_center = False
        
        if is_center and min_dim < 2.0: room_type = 'hallway'
        elif max_half_width < 0.8 and aspect_ratio > 2.5: room_type = 'hallway'
        elif area < 4.5: room_type = 'closet'
        else: room_type = 'room'
        
        rooms_data.append({
            'polygon': poly, 'area': area, 'type': room_type,
            'mask': mask, 'bounds': (x0, x1, z0, z1),
        })
    
    rn, hn, cn = 1, 1, 1
    for rd in rooms_data:
        if rd['type'] == 'hallway':
            rd['name'] = "Hallway" if hn == 1 else f"Hallway {hn}"; hn += 1
        elif rd['type'] == 'closet':
            rd['name'] = "Closet" if cn == 1 else f"Closet {cn}"; cn += 1
        else:
            rd['name'] = f"Room {rn}"; rn += 1
    
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

def simplify_polygon(poly, min_edge=0.4):
    if len(poly) < 4: return poly
    changed = True
    while changed and len(poly) > 3:
        changed = False
        new_poly = []; skip = set()
        for i in range(len(poly)):
            if i in skip: continue
            j = (i + 1) % len(poly)
            dx = abs(poly[j][0] - poly[i][0])
            dz = abs(poly[j][1] - poly[i][1])
            edge_len = max(dx, dz)
            if edge_len < min_edge and len(poly) - len(skip) > 3:
                prev = poly[(i - 1) % len(poly)]
                nxt = poly[(j + 1) % len(poly)]
                new_poly.append([poly[j][0] if abs(poly[j][0] - nxt[0]) < 0.01 else poly[i][0],
                                 poly[j][1] if abs(poly[j][1] - nxt[1]) < 0.01 else poly[i][1]])
                skip.add(j); changed = True
            else:
                new_poly.append(poly[i])
        poly = new_poly
    return poly

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


# ─── Door Detection (from v32) ───

def detect_doors(density_img, x_min, z_min, cs, rooms_data):
    doors = []
    for i in range(len(rooms_data)):
        pi = rooms_data[i]['polygon']; ni = rooms_data[i]['name']
        for j in range(i+1, len(rooms_data)):
            pj = rooms_data[j]['polygon']; nj = rooms_data[j]['name']
            for ei in range(len(pi)):
                a1, a2 = pi[ei], pi[(ei+1)%len(pi)]
                for ej in range(len(pj)):
                    b1, b2 = pj[ej], pj[(ej+1)%len(pj)]
                    door = check_shared_edge_for_door(density_img, x_min, z_min, cs, a1, a2, b1, b2)
                    if door:
                        print(f"    Door found: {ni} ↔ {nj} at ({door['x']:.2f}, {door['z']:.2f})")
                        doors.append(door)
    
    if not doors:
        print("    No doors found from polygon edges — trying mask adjacency")
        for i in range(len(rooms_data)):
            if 'mask' not in rooms_data[i]: continue
            mi = rooms_data[i]['mask']; ni = rooms_data[i]['name']
            di = cv2.dilate(mi.astype(np.uint8), np.ones((15,15), np.uint8))
            for j in range(i+1, len(rooms_data)):
                if 'mask' not in rooms_data[j]: continue
                mj = rooms_data[j]['mask']; nj = rooms_data[j]['name']
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


# ─── Window Detection (NEW in v33) ───

def detect_windows(density_img, x_min, z_min, cs, rooms_data):
    """Detect windows as medium-length low-density gaps on exterior walls."""
    windows = []
    
    # Collect all edges and determine which are exterior
    # An exterior edge is one that only belongs to one room
    edge_count = {}
    edge_room = {}
    for ri, rd in enumerate(rooms_data):
        poly = rd['polygon']
        n = len(poly)
        for i in range(n):
            p1 = poly[i]; p2 = poly[(i+1)%n]
            # Normalize edge key
            key = (round(min(p1[0],p2[0]),2), round(min(p1[1],p2[1]),2),
                   round(max(p1[0],p2[0]),2), round(max(p1[1],p2[1]),2))
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_room:
                edge_room[key] = (p1, p2)
    
    exterior_edges = [(edge_room[k], k) for k, c in edge_count.items() if c == 1]
    
    sw = max(1, int(0.08/cs))
    
    for (p1, p2), key in exterior_edges:
        if abs(p1[0]-p2[0]) < 0.05:  # Vertical exterior wall
            x = (p1[0] + p2[0]) / 2
            z_lo, z_hi = min(p1[1],p2[1]), max(p1[1],p2[1])
            if z_hi - z_lo < 1.0: continue  # Wall too short for window
            x_px = int((x - x_min)/cs)
            xl = max(0, x_px-sw); xh = min(density_img.shape[1], x_px+sw+1)
            lp = max(0, int((z_lo-z_min)/cs)); hp = min(density_img.shape[0], int((z_hi-z_min)/cs))
            if xl >= xh or lp >= hp: continue
            prof = density_img[lp:hp, xl:xh].sum(axis=1)
            win = find_window_in_profile(prof, cs, x, z_min + lp*cs, 'vertical')
            if win:
                windows.append(win)
                print(f"    Window found at ({win['x']:.2f}, {win['z']:.2f}) w={win['width']:.2f}m {win['orientation']}")
                
        elif abs(p1[1]-p2[1]) < 0.05:  # Horizontal exterior wall
            z = (p1[1] + p2[1]) / 2
            x_lo, x_hi = min(p1[0],p2[0]), max(p1[0],p2[0])
            if x_hi - x_lo < 1.0: continue
            z_px = int((z - z_min)/cs)
            zl = max(0, z_px-sw); zh = min(density_img.shape[0], z_px+sw+1)
            lp = max(0, int((x_lo-x_min)/cs)); hp = min(density_img.shape[1], int((x_hi-x_min)/cs))
            if zl >= zh or lp >= hp: continue
            prof = density_img[zl:zh, lp:hp].sum(axis=0)
            win = find_window_in_profile(prof, cs, x_min + lp*cs, z, 'horizontal')
            if win:
                windows.append(win)
                print(f"    Window found at ({win['x']:.2f}, {win['z']:.2f}) w={win['width']:.2f}m {win['orientation']}")
    
    return windows

def find_window_in_profile(prof, cs, origin_x, origin_z, orientation):
    """Find window gaps: 0.8-1.5m gaps with some density on both sides."""
    if len(prof) == 0 or not np.any(prof > 0): return None
    thr = np.percentile(prof[prof>0], 40) if np.any(prof>0) else 1
    is_gap = prof < max(thr, 1)
    lbl, ng = ndlabel(is_gap)
    for g in range(1, ng+1):
        gi = np.where(lbl==g)[0]
        gl = len(gi) * cs
        if 0.7 < gl < 1.8:
            # Check there's wall evidence on both sides
            left_ok = gi[0] > 3 and np.mean(prof[max(0,gi[0]-5):gi[0]]) > thr * 0.5
            right_ok = gi[-1] < len(prof)-4 and np.mean(prof[gi[-1]+1:min(len(prof),gi[-1]+6)]) > thr * 0.5
            if left_ok and right_ok:
                center = gi.mean() * cs
                if orientation == 'vertical':
                    return {'x': origin_x, 'z': origin_z + center, 'width': gl, 'orientation': orientation}
                else:
                    return {'x': origin_x + center, 'z': origin_z, 'width': gl, 'orientation': orientation}
    return None


# ─── Architectural Rendering (v33 - clean style) ───

def collect_all_wall_segments(rooms_data):
    """Collect all wall edge segments from room polygons, tracking which are shared (interior)."""
    edge_count = {}
    edge_data = {}
    
    for rd in rooms_data:
        poly = rd['polygon']
        n = len(poly)
        for i in range(n):
            p1 = poly[i]; p2 = poly[(i+1)%n]
            key = (round(min(p1[0],p2[0]),2), round(min(p1[1],p2[1]),2),
                   round(max(p1[0],p2[0]),2), round(max(p1[1],p2[1]),2))
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_data[key] = (p1, p2)
    
    segments = []
    for key, (p1, p2) in edge_data.items():
        is_exterior = edge_count[key] == 1
        segments.append({
            'p1': p1, 'p2': p2,
            'exterior': is_exterior,
        })
    return segments


def render_architectural(rooms_data, doors, windows, output_path, show_labels=False):
    """Render clean architectural floor plan — white rooms, thick black walls, door arcs, window symbols."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    all_x, all_z = [], []
    for rd in rooms_data:
        for p in rd['polygon']:
            all_x.append(p[0]); all_z.append(p[1])
    if not all_x:
        plt.savefig(output_path, dpi=200); plt.close(); return
    
    margin = 0.5
    WALL_T = 0.13  # wall thickness in meters (chunky architectural style)
    WALL_EXTEND = 0.07  # corner extension for clean joints
    
    wall_segments = collect_all_wall_segments(rooms_data)
    
    # 1. Draw thick black walls with corner overlap
    for seg in wall_segments:
        p1, p2 = seg['p1'], seg['p2']
        
        if abs(p1[0]-p2[0]) < 0.05:  # Vertical wall
            x = (p1[0] + p2[0]) / 2
            z_lo, z_hi = min(p1[1],p2[1]), max(p1[1],p2[1])
            # Extend for corner joints
            rect = patches.Rectangle(
                (x - WALL_T/2, z_lo - WALL_EXTEND),
                WALL_T, (z_hi - z_lo) + 2*WALL_EXTEND,
                facecolor='black', edgecolor='none', zorder=2, linewidth=0)
            ax.add_patch(rect)
        elif abs(p1[1]-p2[1]) < 0.05:  # Horizontal wall
            z = (p1[1] + p2[1]) / 2
            x_lo, x_hi = min(p1[0],p2[0]), max(p1[0],p2[0])
            rect = patches.Rectangle(
                (x_lo - WALL_EXTEND, z - WALL_T/2),
                (x_hi - x_lo) + 2*WALL_EXTEND, WALL_T,
                facecolor='black', edgecolor='none', zorder=2, linewidth=0)
            ax.add_patch(rect)
        else:
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=5, zorder=2, solid_capstyle='projecting')
    
    # 2. Draw doors (white gap in wall + quarter-circle arc + door leaf)
    for door in doors:
        x, z = door['x'], door['z']
        w = door.get('width', 0.8)
        door_len = w * 0.9  # door leaf slightly shorter than opening
        
        if door['orientation'] == 'vertical':
            # Clear wall at door position
            clear = patches.Rectangle(
                (x - WALL_T*0.8, z - w/2), WALL_T*1.6, w,
                facecolor='white', edgecolor='none', zorder=3, linewidth=0)
            ax.add_patch(clear)
            # Draw opening lines (thin lines at door jambs)
            ax.plot([x - WALL_T/2, x + WALL_T/2], [z - w/2, z - w/2], 'k-', lw=0.8, zorder=4)
            ax.plot([x - WALL_T/2, x + WALL_T/2], [z + w/2, z + w/2], 'k-', lw=0.8, zorder=4)
            # Door arc (quarter circle)
            arc = Arc((x, z - w/2), door_len*2, door_len*2,
                      angle=0, theta1=0, theta2=90,
                      color='black', linewidth=0.8, linestyle='-', zorder=4)
            ax.add_patch(arc)
            # Door leaf line
            ax.plot([x, x + door_len], [z - w/2, z - w/2], 'k-', lw=1.0, zorder=4)
        else:
            clear = patches.Rectangle(
                (x - w/2, z - WALL_T*0.8), w, WALL_T*1.6,
                facecolor='white', edgecolor='none', zorder=3, linewidth=0)
            ax.add_patch(clear)
            ax.plot([x - w/2, x - w/2], [z - WALL_T/2, z + WALL_T/2], 'k-', lw=0.8, zorder=4)
            ax.plot([x + w/2, x + w/2], [z - WALL_T/2, z + WALL_T/2], 'k-', lw=0.8, zorder=4)
            arc = Arc((x - w/2, z), door_len*2, door_len*2,
                      angle=0, theta1=0, theta2=90,
                      color='black', linewidth=0.8, linestyle='-', zorder=4)
            ax.add_patch(arc)
            ax.plot([x - w/2, x - w/2], [z, z + door_len], 'k-', lw=1.0, zorder=4)
    
    # 3. Draw windows (white gap + two parallel lines)
    for win in windows:
        x, z = win['x'], win['z']
        w = win.get('width', 1.0)
        line_offset = WALL_T * 0.25  # offset from center for parallel lines
        
        if win['orientation'] == 'vertical':
            # Clear wall
            clear = patches.Rectangle(
                (x - WALL_T*0.7, z - w/2), WALL_T*1.4, w,
                facecolor='white', edgecolor='none', zorder=3, linewidth=0)
            ax.add_patch(clear)
            # Two parallel lines
            ax.plot([x - line_offset, x - line_offset], [z - w/2, z + w/2], 'k-', lw=1.0, zorder=4)
            ax.plot([x + line_offset, x + line_offset], [z - w/2, z + w/2], 'k-', lw=1.0, zorder=4)
            # End caps
            ax.plot([x - WALL_T/2, x + WALL_T/2], [z - w/2, z - w/2], 'k-', lw=1.0, zorder=4)
            ax.plot([x - WALL_T/2, x + WALL_T/2], [z + w/2, z + w/2], 'k-', lw=1.0, zorder=4)
        else:
            clear = patches.Rectangle(
                (x - w/2, z - WALL_T*0.7), w, WALL_T*1.4,
                facecolor='white', edgecolor='none', zorder=3, linewidth=0)
            ax.add_patch(clear)
            ax.plot([x - w/2, x + w/2], [z - line_offset, z - line_offset], 'k-', lw=1.0, zorder=4)
            ax.plot([x - w/2, x + w/2], [z + line_offset, z + line_offset], 'k-', lw=1.0, zorder=4)
            ax.plot([x - w/2, x - w/2], [z - WALL_T/2, z + WALL_T/2], 'k-', lw=1.0, zorder=4)
            ax.plot([x + w/2, x + w/2], [z - WALL_T/2, z + WALL_T/2], 'k-', lw=1.0, zorder=4)
    
    # 4. Optional room labels (very subtle)
    if show_labels:
        for rd in rooms_data:
            poly = rd['polygon']
            if len(poly) < 3: continue
            cx, cz = polygon_centroid(poly)
            ax.text(cx, cz, rd['name'], ha='center', va='center',
                    fontsize=7, color='#999', zorder=5)
    
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"  Saved: {output_path}")


def render_svg(rooms_data, doors, windows, output_path, show_labels=False):
    """Render clean architectural floor plan as SVG."""
    all_x, all_z = [], []
    for rd in rooms_data:
        for p in rd['polygon']:
            all_x.append(p[0]); all_z.append(p[1])
    if not all_x: return
    
    margin = 0.5
    WALL_T = 0.13
    WALL_EXTEND = 0.07
    SCALE = 100  # pixels per meter
    
    x_min_w = min(all_x) - margin
    z_min_w = min(all_z) - margin
    x_max_w = max(all_x) + margin
    z_max_w = max(all_z) + margin
    
    svg_w = (x_max_w - x_min_w) * SCALE
    svg_h = (z_max_w - z_min_w) * SCALE
    
    def tx(x): return (x - x_min_w) * SCALE
    def tz(z): return (z_max_w - z) * SCALE  # flip Y for SVG
    def ts(s): return s * SCALE
    
    dwg = svgwrite.Drawing(str(output_path), size=(f'{svg_w:.0f}px', f'{svg_h:.0f}px'),
                           viewBox=f'0 0 {svg_w:.0f} {svg_h:.0f}')
    dwg.add(dwg.rect(insert=(0,0), size=(svg_w, svg_h), fill='white'))
    
    wall_segments = collect_all_wall_segments(rooms_data)
    
    # Walls group
    walls_g = dwg.g(id='walls')
    for seg in wall_segments:
        p1, p2 = seg['p1'], seg['p2']
        if abs(p1[0]-p2[0]) < 0.05:  # Vertical
            x = (p1[0]+p2[0])/2
            z_lo, z_hi = min(p1[1],p2[1]), max(p1[1],p2[1])
            walls_g.add(dwg.rect(
                insert=(tx(x - WALL_T/2), tz(z_hi + WALL_EXTEND)),
                size=(ts(WALL_T), ts(z_hi - z_lo + 2*WALL_EXTEND)),
                fill='black'))
        elif abs(p1[1]-p2[1]) < 0.05:  # Horizontal
            z = (p1[1]+p2[1])/2
            x_lo, x_hi = min(p1[0],p2[0]), max(p1[0],p2[0])
            walls_g.add(dwg.rect(
                insert=(tx(x_lo - WALL_EXTEND), tz(z + WALL_T/2)),
                size=(ts(x_hi - x_lo + 2*WALL_EXTEND), ts(WALL_T)),
                fill='black'))
    dwg.add(walls_g)
    
    # Doors group
    doors_g = dwg.g(id='doors')
    for door in doors:
        x, z = door['x'], door['z']
        w = door.get('width', 0.8)
        door_len = w * 0.9
        
        if door['orientation'] == 'vertical':
            # White gap
            doors_g.add(dwg.rect(
                insert=(tx(x - WALL_T*0.8), tz(z + w/2)),
                size=(ts(WALL_T*1.6), ts(w)), fill='white'))
            # Arc path
            cx_s, cz_s = tx(x), tz(z - w/2)
            r = ts(door_len)
            arc_end_x = tx(x + door_len)
            arc_end_z = tz(z - w/2 + door_len)
            # SVG arc: from door hinge to end of swing
            d = f'M {tx(x + door_len):.1f},{tz(z - w/2):.1f} A {r:.1f},{r:.1f} 0 0,1 {tx(x):.1f},{tz(z - w/2 + door_len):.1f}'
            doors_g.add(dwg.path(d=d, stroke='black', stroke_width='0.8', fill='none'))
            # Door leaf
            doors_g.add(dwg.line(
                start=(tx(x), tz(z - w/2)),
                end=(tx(x + door_len), tz(z - w/2)),
                stroke='black', stroke_width='1'))
        else:
            doors_g.add(dwg.rect(
                insert=(tx(x - w/2), tz(z + WALL_T*0.8)),
                size=(ts(w), ts(WALL_T*1.6)), fill='white'))
            r = ts(door_len)
            d = f'M {tx(x - w/2):.1f},{tz(z + door_len):.1f} A {r:.1f},{r:.1f} 0 0,0 {tx(x - w/2 + door_len):.1f},{tz(z):.1f}'
            doors_g.add(dwg.path(d=d, stroke='black', stroke_width='0.8', fill='none'))
            doors_g.add(dwg.line(
                start=(tx(x - w/2), tz(z)),
                end=(tx(x - w/2), tz(z + door_len)),
                stroke='black', stroke_width='1'))
    dwg.add(doors_g)
    
    # Windows group
    windows_g = dwg.g(id='windows')
    line_offset = WALL_T * 0.25
    for win in windows:
        x, z = win['x'], win['z']
        w = win.get('width', 1.0)
        if win['orientation'] == 'vertical':
            windows_g.add(dwg.rect(
                insert=(tx(x - WALL_T*0.7), tz(z + w/2)),
                size=(ts(WALL_T*1.4), ts(w)), fill='white'))
            windows_g.add(dwg.line(start=(tx(x-line_offset), tz(z-w/2)), end=(tx(x-line_offset), tz(z+w/2)), stroke='black', stroke_width='1'))
            windows_g.add(dwg.line(start=(tx(x+line_offset), tz(z-w/2)), end=(tx(x+line_offset), tz(z+w/2)), stroke='black', stroke_width='1'))
            windows_g.add(dwg.line(start=(tx(x-WALL_T/2), tz(z-w/2)), end=(tx(x+WALL_T/2), tz(z-w/2)), stroke='black', stroke_width='1'))
            windows_g.add(dwg.line(start=(tx(x-WALL_T/2), tz(z+w/2)), end=(tx(x+WALL_T/2), tz(z+w/2)), stroke='black', stroke_width='1'))
        else:
            windows_g.add(dwg.rect(
                insert=(tx(x - w/2), tz(z + WALL_T*0.7)),
                size=(ts(w), ts(WALL_T*1.4)), fill='white'))
            windows_g.add(dwg.line(start=(tx(x-w/2), tz(z-line_offset)), end=(tx(x+w/2), tz(z-line_offset)), stroke='black', stroke_width='1'))
            windows_g.add(dwg.line(start=(tx(x-w/2), tz(z+line_offset)), end=(tx(x+w/2), tz(z+line_offset)), stroke='black', stroke_width='1'))
            windows_g.add(dwg.line(start=(tx(x-w/2), tz(z-WALL_T/2)), end=(tx(x-w/2), tz(z+WALL_T/2)), stroke='black', stroke_width='1'))
            windows_g.add(dwg.line(start=(tx(x+w/2), tz(z-WALL_T/2)), end=(tx(x+w/2), tz(z+WALL_T/2)), stroke='black', stroke_width='1'))
    dwg.add(windows_g)
    
    dwg.save()
    print(f"  Saved SVG: {output_path}")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--nms', type=float, default=0.3)
    parser.add_argument('--labels', action='store_true', help='Show subtle room labels')
    parser.add_argument('-o', '--output', default='results/v33')
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
        print(f"  Single room mode — skipping cuts")
        sel_x = []; sel_z = []
    
    print(f"  Selected cuts: X={[f'{w:.2f}' for w in sel_x]}, Z={[f'{w:.2f}' for w in sel_z]}")
    
    if sel_x or sel_z:
        sel_x_arr = np.array(sorted(sel_x))
        sel_z_arr = np.array(sorted(sel_z))
        sel_x_str = np.array([next(s for w,s,_ in x_scored if w==xw) for xw in sel_x_arr]) if len(sel_x_arr) else np.array([])
        sel_z_str = np.array([next(s for w,s,_ in z_scored if w==zw) for zw in sel_z_arr]) if len(sel_z_arr) else np.array([])
        
        cut_mask, valid_x, valid_z = cut_mask_with_walls(
            room_mask, density_img, x_min, z_min, cs,
            sel_x_arr, sel_z_arr, sel_x_str, sel_z_str, min_wall_run=0.5)
        
        print("Extracting rooms...")
        rooms_data = extract_and_merge_rooms(cut_mask, density_img, x_min, z_min, cs,
                                              x_walls, z_walls, valid_x, valid_z, target_rooms=5)
    else:
        print("Single room mode — no cuts")
        valid_x, valid_z = [], []
        mask_u8 = room_mask.astype(np.uint8)
        
        def snap(val, positions, max_snap=0.25):
            if len(positions) == 0: return val
            dists = np.abs(np.array(positions) - val)
            idx = np.argmin(dists)
            return float(positions[idx]) if dists[idx] < max_snap else val
        
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            epsilon = max(5, int(0.3 / cs))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            poly = []
            for pt in approx.reshape(-1, 2):
                wx = snap(x_min + pt[0] * cs, x_walls)
                wz = snap(z_min + pt[1] * cs, z_walls)
                poly.append([wx, wz])
            poly = make_rectilinear(poly)
            poly = remove_collinear(poly)
            poly = simplify_polygon(poly, min_edge=0.8)
            poly = remove_collinear(poly)
        else:
            rows, cols = np.where(room_mask)
            x0 = snap(x_min + cols.min() * cs, x_walls)
            x1 = snap(x_min + cols.max() * cs, x_walls)
            z0 = snap(z_min + rows.min() * cs, z_walls)
            z1 = snap(z_min + rows.max() * cs, z_walls)
            poly = [[x0,z0],[x1,z0],[x1,z1],[x0,z1]]
        
        area = compute_polygon_area(poly) if len(poly) >= 3 else np.sum(room_mask) * cs * cs
        print(f"  Single room: {len(poly)}v, {area:.1f}m²")
        rooms_data = [{'name': 'Room', 'polygon': poly, 'area': area, 'type': 'room', 'mask': room_mask}]
    
    for rd in rooms_data:
        print(f"  {rd['name']}: {rd['area']:.1f}m² ({rd['type']}) poly={len(rd['polygon'])}v")
    
    # Detect doors
    doors = detect_doors(density_img, x_min, z_min, cs, rooms_data)
    print(f"  Doors: {len(doors)}")
    
    # Detect windows
    windows = detect_windows(density_img, x_min, z_min, cs, rooms_data)
    print(f"  Windows: {len(windows)}")
    
    # Render PNG
    png_path = out_dir / f"v33_{mesh_name}_plan.png"
    render_architectural(rooms_data, doors, windows, png_path, show_labels=args.labels)
    
    # Render SVG
    svg_path = out_dir / f"v33_{mesh_name}_plan.svg"
    render_svg(rooms_data, doors, windows, svg_path, show_labels=args.labels)
    
    # JSON results
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    
    results = {
        'approach': 'v33_architectural_clean',
        'rooms': [{
            'name': rd['name'], 'area_m2': round(rd['area'],1),
            'type': rd['type'],
            'polygon': [[round(p[0],3), round(p[1],3)] for p in rd['polygon']],
        } for rd in rooms_data],
        'doors': doors,
        'windows': [{'x': w['x'], 'z': w['z'], 'width': w['width'], 'orientation': w['orientation']} for w in windows],
        'walls': {'x': [round(w,3) for w in x_walls], 'z': [round(w,3) for w in z_walls]},
    }
    with open(out_dir / f"v33_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    
    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if rd['type'] == 'room')
    nh = sum(1 for rd in rooms_data if rd['type'] == 'hallway')
    nc = sum(1 for rd in rooms_data if rd['type'] == 'closet')
    print(f"\n=== v33 RESULTS ===")
    print(f"  {nr} rooms, {nh} hallways, {nc} closets — {total:.1f}m²")
    print(f"  {len(doors)} doors, {len(windows)} windows")
    print("Done!")

if __name__ == '__main__':
    main()

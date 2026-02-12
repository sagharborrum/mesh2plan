#!/usr/bin/env python3
"""
mesh2plan v52 - Line Graph v5: Structural Wall Recovery

Key insight: v50 had the right approach (density-scored walls, ~6 selected) but 
missed ONE critical interior wall, making Room 1 too large (13.4m²).
v51 tried selecting MORE walls but over-fragmented everything.

v52 approach:
1. Start with v50's mean-threshold wall selection (~6 walls)
2. Build initial room cells
3. For any room > MAX_ROOM_M2 (11m²), search UNSELECTED walls for the best 
   one that crosses it → add it and re-partition
4. This surgically adds only the walls needed to fix oversized rooms
5. Tighter rho clustering (min_gap=18px) to avoid merging close parallel walls
6. Gap-free room expansion (from v51)
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
import shutil


MAX_ROOM_M2 = 10.0  # Rooms larger than this trigger wall recovery


def detect_up_axis(mesh):
    v = np.array(mesh.vertices)
    spans = [v[:, i].max() - v[:, i].min() for i in range(3)]
    n = np.abs(mesh.face_normals)
    mean_abs = n.mean(axis=0)
    scores = mean_abs / (np.array(spans) + 0.01)
    up = int(np.argmax(scores))
    print(f"  Up axis: {'XYZ'[up]}")
    return up


def create_wall_density(mesh, resolution=0.02, normal_thresh=0.5, up_axis=1):
    normals = mesh.face_normals
    centroids = mesh.triangles_center
    up_comp = np.abs(normals[:, up_axis])
    wall_mask = up_comp < normal_thresh
    wall_strength = 1.0 - up_comp
    wall_centroids = centroids[wall_mask]
    wall_weights = wall_strength[wall_mask]
    print(f"  Wall faces: {wall_mask.sum()} / {len(normals)} ({wall_mask.mean()*100:.1f}%)")

    floor_axes = [i for i in range(3) if i != up_axis]
    ax0, ax1 = floor_axes
    x, z = wall_centroids[:, ax0], wall_centroids[:, ax1]
    pad = 0.3
    x_min, x_max = x.min() - pad, x.max() + pad
    z_min, z_max = z.min() - pad, z.max() + pad
    w = int((x_max - x_min) / resolution) + 1
    h = int((z_max - z_min) / resolution) + 1

    wall_density = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(wall_density, (zi, xi), wall_weights)

    all_x = mesh.vertices[:, ax0]
    all_z = mesh.vertices[:, ax1]
    all_density = np.zeros((h, w), dtype=np.float32)
    axi = np.clip(((all_x - x_min) / resolution).astype(int), 0, w - 1)
    azi = np.clip(((all_z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(all_density, (azi, axi), 1)

    print(f"  Image size: {w}x{h}")
    return wall_density, all_density, (x_min, z_min, resolution, w, h)


def get_apartment_mask(all_density, wall_density, threshold=1):
    mask = (all_density >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 1, -1)
    return mask


def find_dominant_angles(wall_density, mask):
    d = wall_density.copy()
    d[mask == 0] = 0
    pos_vals = d[d > 0]
    if len(pos_vals) == 0:
        return None
    p80 = np.percentile(pos_vals, 80)
    wall_img = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255

    lines = cv2.HoughLines(wall_img, 1, np.pi / 180, threshold=30)
    if lines is None:
        return None
    lines = lines.squeeze(axis=1)
    print(f"  Raw Hough lines: {len(lines)}")

    thetas_deg = np.degrees(lines[:, 1]) % 180
    hist = np.zeros(180)
    for t in thetas_deg:
        hist[int(t) % 180] += 1
    kernel_h = np.ones(9) / 9
    ext = np.concatenate([hist[-4:], hist, hist[:4]])
    smooth = np.convolve(ext, kernel_h, mode='same')[4:-4]

    peaks = []
    for i in range(180):
        prev, nxt = (i - 1) % 180, (i + 1) % 180
        if smooth[i] > smooth[prev] and smooth[i] > smooth[nxt] and smooth[i] > 3:
            peaks.append((smooth[i], i))
    peaks.sort(reverse=True)

    angle1 = peaks[0][1] if peaks else 0
    angle2 = None
    for _, a in peaks[1:]:
        diff = abs(a - angle1)
        diff = min(diff, 180 - diff)
        if abs(diff - 90) < 20:
            angle2 = a
            break
    if angle2 is None:
        angle2 = (angle1 + 90) % 180

    print(f"  Dominant angles: {angle1}° and {angle2}°")
    return lines, wall_img, angle1, angle2


def score_line_by_density(rho, theta, wall_density, mask, width_px=5):
    h, w = wall_density.shape
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x0, y0 = rho * cos_t, rho * sin_t
    dx, dy = -sin_t, cos_t

    t_range = max(w, h) * 2
    n_samples = int(t_range * 2)
    ts = np.linspace(-t_range, t_range, n_samples)
    xs = (x0 + ts * dx).astype(int)
    ys = (y0 + ts * dy).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 10:
        return 0, 0, 0

    in_mask = mask[ys, xs] > 0
    if not in_mask.any():
        return 0, 0, 0

    density_vals = np.zeros(len(xs), dtype=np.float32)
    for offset in range(-width_px//2, width_px//2 + 1):
        ox = np.clip(xs + int(offset * cos_t), 0, w - 1)
        oy = np.clip(ys + int(offset * sin_t), 0, h - 1)
        density_vals += wall_density[oy, ox]

    density_vals[~in_mask] = 0
    total = density_vals.sum()
    mask_count = in_mask.sum()
    
    high = density_vals > np.percentile(density_vals[in_mask], 30) if mask_count > 0 else density_vals > 0
    high = high & in_mask
    diff = np.diff(np.concatenate([[0], high.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    max_run = max((e - s for s, e in zip(starts, ends)), default=0)
    
    coverage = high.sum() / max(mask_count, 1)
    
    return float(total), int(max_run), float(coverage)


def cluster_and_score_walls(lines, wall_density, mask, angle1, angle2, angle_tol=12, min_gap_px=18):
    thetas_deg = np.degrees(lines[:, 1]) % 180
    
    families = {0: [], 1: []}
    for i, t in enumerate(thetas_deg):
        for fi, center in enumerate([angle1, angle2]):
            diff = abs(t - center)
            diff = min(diff, 180 - diff)
            if diff < angle_tol:
                families[fi].append(i)
                break

    all_walls = []
    for fi in [0, 1]:
        if not families[fi]:
            continue
        rhos = lines[families[fi], 0]
        thetas = lines[families[fi], 1]
        order = np.argsort(rhos)
        sorted_rhos = rhos[order]
        sorted_thetas = thetas[order]

        clusters = []
        cur_rhos = [sorted_rhos[0]]
        cur_thetas = [sorted_thetas[0]]
        cur_votes = 1

        for i in range(1, len(order)):
            if sorted_rhos[i] - np.mean(cur_rhos) < min_gap_px:
                cur_rhos.append(sorted_rhos[i])
                cur_thetas.append(sorted_thetas[i])
                cur_votes += 1
            else:
                clusters.append((np.mean(cur_rhos), np.mean(cur_thetas), cur_votes))
                cur_rhos = [sorted_rhos[i]]
                cur_thetas = [sorted_thetas[i]]
                cur_votes = 1
        clusters.append((np.mean(cur_rhos), np.mean(cur_thetas), cur_votes))

        for rho, theta, votes in clusters:
            total_d, max_run, coverage = score_line_by_density(rho, theta, wall_density, mask)
            all_walls.append({
                'rho': rho, 'theta': theta, 'votes': votes,
                'family': fi, 'density_score': total_d, 
                'max_run': max_run, 'coverage': coverage
            })

    print(f"  Clustered walls: {len(all_walls)} (fam0: {sum(1 for w in all_walls if w['family']==0)}, fam1: {sum(1 for w in all_walls if w['family']==1)})")
    return all_walls


def select_walls_initial(walls, max_walls=10):
    """Initial selection using mean composite threshold (like v50)."""
    if not walls:
        return [], []
    
    for w in walls:
        w['composite'] = w['density_score'] * math.sqrt(max(w['max_run'], 1))
    
    walls.sort(key=lambda w: w['composite'], reverse=True)
    
    for i, w in enumerate(walls):
        print(f"    [{i}] fam={w['family']} θ={math.degrees(w['theta']):.1f}° "
              f"ρ={w['rho']:.0f} votes={w['votes']} density={w['density_score']:.0f} "
              f"run={w['max_run']}px composite={w['composite']:.0f}")
    
    scores = [w['composite'] for w in walls]
    if len(scores) > 3:
        mean_s = np.mean(scores)
        above = [w for w in walls if w['composite'] >= mean_s]
        if len(above) < 4:
            selected = walls[:4]
        elif len(above) > max_walls:
            selected = above[:max_walls]
        else:
            selected = above
    else:
        selected = walls[:max_walls]
    
    # Ensure at least 2 per family
    for fi in [0, 1]:
        fam_in = [w for w in selected if w['family'] == fi]
        if len(fam_in) < 2:
            extras = [w for w in walls if w['family'] == fi and w not in selected]
            for e in extras[:2 - len(fam_in)]:
                selected.append(e)
    
    rejected = [w for w in walls if w not in selected]
    print(f"  Initial selection: {len(selected)} walls")
    return selected, rejected


def get_wall_endpoints(wall, mask):
    h, w = mask.shape
    rho, theta = wall['rho'], wall['theta']
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x0, y0 = rho * cos_t, rho * sin_t
    dx, dy = -sin_t, cos_t

    t_range = max(w, h) * 2
    ts = np.linspace(-t_range, t_range, 500)
    xs = (x0 + ts * dx).astype(int)
    ys = (y0 + ts * dy).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 2:
        return None
    in_mask = mask[ys, xs] > 0
    if not in_mask.any():
        return None
    mi = np.where(in_mask)[0]
    return (int(xs[mi[0]]), int(ys[mi[0]])), (int(xs[mi[-1]]), int(ys[mi[-1]]))


def wall_crosses_room(wall, room_mask, mask):
    """Check if a wall line crosses through a room mask. Returns the fraction 
    of the wall's in-mask length that overlaps with the room."""
    h, w = mask.shape
    rho, theta = wall['rho'], wall['theta']
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x0, y0 = rho * cos_t, rho * sin_t
    dx, dy = -sin_t, cos_t

    t_range = max(w, h) * 2
    ts = np.linspace(-t_range, t_range, 500)
    xs = (x0 + ts * dx).astype(int)
    ys = (y0 + ts * dy).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 2:
        return 0.0

    in_mask = mask[ys, xs] > 0
    in_room = room_mask[ys, xs] > 0
    
    mask_count = in_mask.sum()
    room_count = (in_mask & in_room).sum()
    
    if mask_count < 10:
        return 0.0
    return room_count / mask_count


def build_room_cells(wall_segments, mask, line_thickness=3):
    h, w = mask.shape
    line_img = np.zeros((h, w), dtype=np.uint8)
    for seg in wall_segments:
        if seg.get('pt1') and seg.get('pt2'):
            cv2.line(line_img, seg['pt1'], seg['pt2'], 255, line_thickness)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(line_img, contours, -1, 255, line_thickness)
    interior = ((line_img == 0) & (mask > 0)).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  Connected components: {n_labels - 1}")
    return labels, n_labels, line_img, interior


def extract_rooms(labels, n_labels, res, min_area_m2=0.5):
    min_area_px = int(min_area_m2 / (res * res))
    rooms = []
    for lbl in range(1, n_labels):
        room_mask = (labels == lbl).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px < min_area_px:
            continue
        rooms.append({
            'mask': room_mask,
            'area_m2': round(area_px * res * res, 1),
            'area_px': area_px,
        })
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    return rooms


def recover_structural_walls(rooms, rejected_walls, mask, res, max_room_m2=MAX_ROOM_M2):
    """For each oversized room, find the best rejected wall that crosses it and add it."""
    recovered = []
    for room in rooms:
        if room['area_m2'] <= max_room_m2:
            continue
        print(f"  Room {room['area_m2']}m² is oversized — searching for splitting wall...")
        best_wall = None
        best_score = 0
        for wall in rejected_walls:
            overlap = wall_crosses_room(wall, room['mask'], mask)
            if overlap > 0.1:  # Wall must cross at least 10% within this room
                # Score: density × overlap (prefer walls that strongly cross this room)
                score = wall['composite'] * overlap
                print(f"    Candidate: fam={wall['family']} ρ={wall['rho']:.0f} "
                      f"overlap={overlap:.2f} score={score:.0f}")
                if score > best_score:
                    best_score = score
                    best_wall = wall
        if best_wall:
            print(f"    → Recovered wall: fam={best_wall['family']} ρ={best_wall['rho']:.0f}")
            recovered.append(best_wall)
            rejected_walls = [w for w in rejected_walls if w is not best_wall]
    return recovered


def get_aspect(room):
    mask = room['mask']
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 1.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    if min(w, h) < 1:
        return 10.0
    return max(w, h) / min(w, h)


def smart_merge(rooms, resolution, target_count=5):
    res = resolution
    
    def is_hallway(room):
        return get_aspect(room) > 2.5 and room['area_m2'] > 1.5
    
    # Merge tiny rooms (<1.5m²)
    changed = True
    while changed:
        changed = False
        tiny = [r for r in rooms if r['area_m2'] < 1.5]
        if not tiny:
            break
        for sr in tiny:
            dilated = cv2.dilate(sr['mask'], np.ones((9, 9), np.uint8))
            best, best_ov = None, 0
            for r in rooms:
                if r is sr:
                    continue
                ov = (dilated & r['mask']).sum()
                if ov > best_ov:
                    best_ov = ov
                    best = r
            if best and best_ov > 0:
                best['mask'] = best['mask'] | sr['mask']
                best['area_px'] = best['mask'].sum()
                best['area_m2'] = round(best['area_px'] * res * res, 1)
                rooms = [r for r in rooms if r is not sr]
                changed = True
                break
    
    # Merge to target count
    while len(rooms) > target_count:
        candidates = [r for r in rooms if not is_hallway(r)]
        if not candidates:
            candidates = rooms
        smallest = min(candidates, key=lambda r: r['area_m2'])
        
        dilated = cv2.dilate(smallest['mask'], np.ones((9, 9), np.uint8))
        best, best_ov = None, 0
        for r in rooms:
            if r is smallest:
                continue
            ov = (dilated & r['mask']).sum()
            if ov > best_ov:
                best_ov = ov
                best = r
        if best and best_ov > 0:
            best['mask'] = best['mask'] | smallest['mask']
            best['area_px'] = best['mask'].sum()
            best['area_m2'] = round(best['area_px'] * res * res, 1)
            rooms = [r for r in rooms if r is not smallest]
        else:
            break
    
    return rooms


def expand_rooms_to_fill(rooms, mask):
    """Expand room masks to fill wall pixels between them."""
    h, w = mask.shape
    claimed = np.zeros((h, w), dtype=np.int32)
    for i, room in enumerate(rooms):
        claimed[room['mask'] > 0] = i + 1
    
    unclaimed = (mask > 0) & (claimed == 0)
    kernel = np.ones((3, 3), np.uint8)
    
    for _ in range(25):
        if not unclaimed.any():
            break
        new_claimed = claimed.copy()
        for i in range(len(rooms)):
            current = (claimed == i + 1).astype(np.uint8)
            dilated = cv2.dilate(current, kernel)
            can_claim = (dilated > 0) & unclaimed
            new_claimed[can_claim] = i + 1
        claimed = new_claimed
        unclaimed = (mask > 0) & (claimed == 0)
    
    for i, room in enumerate(rooms):
        room['mask'] = (claimed == i + 1).astype(np.uint8)
        room['area_px'] = room['mask'].sum()


def reextract_polygons(rooms, transform):
    x_min, z_min, res, w, h = transform
    for room in rooms:
        contours, _ = cv2.findContours(room['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, 0.015 * peri, True)
            pts_px = simplified.reshape(-1, 2).astype(float)
            pts_world = np.zeros((len(pts_px), 2))
            pts_world[:, 0] = pts_px[:, 0] * res + x_min
            pts_world[:, 1] = pts_px[:, 1] * res + z_min
            room['polygon'] = pts_world
            room['area_m2'] = round(room['mask'].sum() * res * res, 1)
            room['vertices'] = len(pts_world)


def snap_polygon_to_angles(pts, angles, thresh_deg=15):
    if len(pts) < 3:
        return pts
    thresh = math.radians(thresh_deg)
    n = len(pts)
    snapped = pts.copy().astype(float)
    for _ in range(8):
        new = snapped.copy()
        for i in range(n):
            j = (i + 1) % n
            dx = snapped[j, 0] - snapped[i, 0]
            dy = snapped[j, 1] - snapped[i, 1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.05:
                continue
            edge_angle = math.atan2(dy, dx) % np.pi
            best_a, best_d = None, float('inf')
            for a in angles:
                d = abs(edge_angle - a) % np.pi
                if d > np.pi/2:
                    d = np.pi - d
                if d < best_d:
                    best_d = d
                    best_a = a
            if best_d > thresh:
                continue
            mid = (snapped[i] + snapped[j]) / 2
            target = best_a
            orig = math.atan2(dy, dx)
            if abs(orig - target) > np.pi/2 and abs(orig - target) < 3*np.pi/2:
                target += np.pi
            half = length / 2
            new[i] = [mid[0] - half*math.cos(target), mid[1] - half*math.sin(target)]
            new[j] = [mid[0] + half*math.cos(target), mid[1] + half*math.sin(target)]
        snapped = new
    cleaned = [snapped[0]]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - cleaned[-1]) > 0.03:
            cleaned.append(snapped[i])
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) < 0.03:
        cleaned = cleaned[:-1]
    return np.array(cleaned) if len(cleaned) >= 3 else snapped


def intersect_consecutive_edges(pts):
    if len(pts) < 3:
        return pts
    n = len(pts)
    edges = []
    for i in range(n):
        j = (i+1) % n
        edges.append((pts[i].copy(), pts[j][0]-pts[i][0], pts[j][1]-pts[i][1]))
    new_pts = []
    for i in range(n):
        j = (i+1) % n
        p1, dx1, dy1 = edges[i]
        p2, dx2, dy2 = edges[j]
        det = dx1*dy2 - dy1*dx2
        if abs(det) < 1e-10:
            new_pts.append(pts[j].copy())
            continue
        t = ((p2[0]-p1[0])*dy2 - (p2[1]-p1[1])*dx2) / det
        ix, iy = p1[0]+t*dx1, p1[1]+t*dy1
        if np.linalg.norm([ix-pts[j][0], iy-pts[j][1]]) > 1.0:
            new_pts.append(pts[j].copy())
        else:
            new_pts.append(np.array([ix, iy]))
    return np.array(new_pts)


def remove_short_edges(pts, min_length=0.15):
    if len(pts) < 4:
        return pts
    changed = True
    while changed and len(pts) >= 4:
        changed = False
        n = len(pts)
        lengths = [np.linalg.norm(pts[(i+1)%n] - pts[i]) for i in range(n)]
        si = np.argmin(lengths)
        if lengths[si] < min_length:
            pts = np.delete(pts, (si+1)%n, axis=0)
            changed = True
    return pts


def remove_collinear(pts, thresh=0.05):
    if len(pts) < 4:
        return pts
    cleaned = []
    n = len(pts)
    for i in range(n):
        v1 = pts[i] - pts[(i-1)%n]
        v2 = pts[(i+1)%n] - pts[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 > 0 and l2 > 0:
            cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
            if cross / (l1*l2) > thresh:
                cleaned.append(pts[i])
        else:
            cleaned.append(pts[i])
    return np.array(cleaned) if len(cleaned) >= 3 else pts


def classify_room(polygon, area, aspect=None):
    if polygon is None or len(polygon) < 3:
        return "unknown"
    if aspect is None:
        xs, zs = polygon[:, 0], polygon[:, 1]
        w, h = xs.max() - xs.min(), zs.max() - zs.min()
        aspect = max(w, h) / (min(w, h) + 0.01)
    if area >= 8:
        return "room"
    if area >= 5:
        return "room" if aspect < 2.5 else "hallway"
    if area >= 3:
        if aspect > 2.5:
            return "hallway"
        return "bathroom"
    return "closet"


def detect_doors(rooms):
    doors = []
    for i in range(len(rooms)):
        for j in range(i+1, len(rooms)):
            d1 = cv2.dilate(rooms[i]['mask'], np.ones((9, 9), np.uint8))
            d2 = cv2.dilate(rooms[j]['mask'], np.ones((9, 9), np.uint8))
            overlap = d1 & d2
            if overlap.sum() > 15:
                ys, xs = np.where(overlap > 0)
                doors.append({'rooms': (i, j), 'pos_px': (xs.mean(), ys.mean())})
    return doors


def render_floorplan(rooms, doors, transform, angles, output_path, title,
                     wall_thickness_m=0.12):
    x_min, z_min, res, w, h = transform
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    fills = ['#FAFAFA', '#F5F5F5', '#F0F0F0', '#EFEFEF', '#EAEAEA', '#F8F8F8']
    half_w = wall_thickness_m / 2.0

    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:, 0], pc[:, 1], color=fills[i % len(fills)], zorder=1)
        n = len(poly)
        for vi in range(n):
            vj = (vi + 1) % n
            dx = poly[vj, 0] - poly[vi, 0]
            dy = poly[vj, 1] - poly[vi, 1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.01:
                continue
            nx, ny = -dy / length, dx / length
            for sign in [-1, 1]:
                p1 = [poly[vi, 0] + sign*half_w*nx, poly[vi, 1] + sign*half_w*ny]
                p2 = [poly[vj, 0] + sign*half_w*nx, poly[vj, 1] + sign*half_w*ny]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.2, zorder=3)
        for vi in range(n):
            vj = (vi + 1) % n
            dx = poly[vj, 0] - poly[vi, 0]
            dy = poly[vj, 1] - poly[vi, 1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.01:
                continue
            nx, ny = -dy / length, dx / length
            for sign in [-1, 1]:
                p = [poly[vi, 0] + sign*half_w*nx, poly[vi, 1] + sign*half_w*ny]
                q = [poly[vi, 0] - sign*half_w*nx, poly[vi, 1] - sign*half_w*ny]
                ax.plot([p[0], q[0]], [p[1], q[1]], 'k-', linewidth=0.8, zorder=3)
                
        cx, cz = poly.mean(axis=0)
        ax.text(cx, cz, f"{room.get('name','?')}\n{room.get('area_m2',0):.1f}m²",
                ha='center', va='center', fontsize=10, fontweight='bold', zorder=5)

    for door in doors:
        cx, cy = door['pos_px']
        wx, wz = cx * res + x_min, cy * res + z_min
        ax.add_patch(plt.Circle((wx, wz), 0.35, fill=False, color='#666',
                                linewidth=1, linestyle='--', zorder=4))

    angle_strs = [f"{math.degrees(a):.0f}°" for a in angles]
    total = sum(r.get('area_m2', 0) for r in rooms)
    ax.text(0.02, 0.98, f"Wall angles: {', '.join(angle_strs)}\n{len(rooms)} rooms, {total:.1f}m²",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(False)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.2]*2, 'k-', linewidth=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.05, '1m', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug(wall_density, mask, wall_img, all_walls, selected_walls, recovered_walls,
                 line_img, interior, labels, rooms, transform, angles, output_path):
    x_min, z_min, res, w, h = transform
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    axes[0, 0].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Wall-only density (log)')
    
    axes[0, 1].imshow(wall_img, cmap='gray', origin='lower')
    axes[0, 1].set_title('Wall mask (p80)')

    vis = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    for wall in all_walls:
        ep = get_wall_endpoints(wall, mask)
        if ep:
            cv2.line(vis, ep[0], ep[1], (100, 100, 100), 1)
    for wall in selected_walls:
        if wall.get('pt1') and wall.get('pt2'):
            c = (255, 0, 0) if wall['family'] == 0 else (0, 0, 255)
            cv2.line(vis, wall['pt1'], wall['pt2'], c, 2)
    for wall in recovered_walls:
        if wall.get('pt1') and wall.get('pt2'):
            cv2.line(vis, wall['pt1'], wall['pt2'], (0, 255, 0), 2)  # green = recovered
    axes[0, 2].imshow(vis[:, :, ::-1], origin='lower')
    axes[0, 2].set_title(f'Walls: {len(selected_walls)} initial + {len(recovered_walls)} recovered (green)')

    axes[0, 3].imshow(line_img, cmap='gray', origin='lower')
    axes[0, 3].set_title('Line arrangement + boundary')
    
    axes[1, 0].imshow(interior * 255, cmap='gray', origin='lower')
    axes[1, 0].set_title('Interior cells')

    room_colors = [(255,100,100),(100,255,100),(100,100,255),
                   (255,255,100),(255,100,255),(100,255,255),(200,150,100)]
    vis2 = np.zeros((h, w, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        vis2[room['mask'] > 0] = room_colors[i % len(room_colors)]
    axes[1, 1].imshow(vis2, origin='lower')
    axes[1, 1].set_title(f'Room cells ({len(rooms)})')

    if all_walls:
        sorted_walls = sorted(all_walls, key=lambda w: w['composite'], reverse=True)
        scores = [w['composite'] for w in sorted_walls]
        colors_list = []
        for w in sorted_walls:
            if w in recovered_walls:
                colors_list.append('lime')
            elif w in selected_walls:
                colors_list.append('green')
            else:
                colors_list.append('gray')
        axes[1, 2].bar(range(len(scores)), scores, color=colors_list)
        axes[1, 2].set_title('Scores: green=initial, lime=recovered, gray=rejected')

    axes[1, 3].set_facecolor('white')
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None: continue
        pc = np.vstack([poly, poly[0]])
        axes[1, 3].fill(pc[:, 0], pc[:, 1], color='#F0F0F0')
        axes[1, 3].plot(pc[:, 0], pc[:, 1], 'k-', linewidth=2)
        cx, cz = poly.mean(axis=0)
        axes[1, 3].text(cx, cz, f"{room.get('name','?')}\n{room.get('area_m2',0):.1f}m²",
                        ha='center', va='center', fontsize=7)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].set_title('Final floorplan')

    plt.suptitle('v52 Line Graph v5 — Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def process_mesh(mesh_path, out_dir, resolution=0.02):
    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    res = resolution

    up_axis = detect_up_axis(mesh)
    wall_density, all_density, transform = create_wall_density(mesh, res, up_axis=up_axis)
    x_min, z_min, _, w, h = transform
    mask = get_apartment_mask(all_density, wall_density)
    print(f"  Apartment area: {mask.sum() * res * res:.1f} m²")

    print("\nFind dominant angles...")
    result = find_dominant_angles(wall_density, mask)
    if result is None:
        print("  No lines!")
        return
    lines, wall_img, angle1_deg, angle2_deg = result

    print("\nCluster and score walls...")
    all_walls = cluster_and_score_walls(lines, wall_density, mask, angle1_deg, angle2_deg,
                                         angle_tol=12, min_gap_px=18)

    print("\nInitial wall selection (mean threshold)...")
    selected, rejected = select_walls_initial(all_walls)

    # Get endpoints for all walls (needed for crossing check)
    for wall in all_walls:
        ep = get_wall_endpoints(wall, mask)
        if ep:
            wall['pt1'], wall['pt2'] = ep

    selected = [w for w in selected if w.get('pt1')]
    rejected = [w for w in rejected if w.get('pt1')]
    print(f"  Initial walls with endpoints: {len(selected)}")

    # Build initial room cells
    print(f"\nBuild initial room cells...")
    labels, n_labels, line_img, interior = build_room_cells(selected, mask, line_thickness=3)
    initial_rooms = extract_rooms(labels, n_labels, res, min_area_m2=0.5)
    print(f"  Initial rooms: {len(initial_rooms)} (areas: {[r['area_m2'] for r in initial_rooms]})")

    # STRUCTURAL WALL RECOVERY: find walls to split oversized rooms
    print(f"\nStructural wall recovery (max room = {MAX_ROOM_M2}m²)...")
    recovered = recover_structural_walls(initial_rooms, rejected, mask, res)
    
    if recovered:
        # Re-build with recovered walls added
        all_selected = selected + recovered
        print(f"  Total walls after recovery: {len(all_selected)}")
        labels, n_labels, line_img, interior = build_room_cells(all_selected, mask, line_thickness=3)
    else:
        all_selected = selected
        print(f"  No walls recovered")

    rooms = extract_rooms(labels, n_labels, res, min_area_m2=0.5)
    print(f"  Rooms after recovery: {len(rooms)} (areas: {[r['area_m2'] for r in rooms]})")

    # Smart merge
    rooms = smart_merge(rooms, res, target_count=6)
    print(f"  After smart merge: {len(rooms)}")

    # Don't expand — use line-graph cell areas directly; double-line rendering handles visual gaps
    
    # Compute angles
    angle_rads = []
    for fi in [0, 1]:
        fam_walls = [w for w in all_selected if w['family'] == fi]
        if fam_walls:
            angle_rads.append(np.mean([w['theta'] for w in fam_walls]) % np.pi)
        else:
            angle_rads.append(math.radians(angle1_deg if fi == 0 else angle2_deg) % np.pi)

    reextract_polygons(rooms, transform)

    # Snap polygons
    for room in rooms:
        poly = room.get('polygon')
        if poly is not None and len(poly) >= 3:
            poly = snap_polygon_to_angles(poly, angle_rads)
            poly = intersect_consecutive_edges(poly)
            poly = remove_short_edges(poly)
            poly = remove_collinear(poly)
            room['polygon'] = poly
            room['vertices'] = len(poly)

    # Classify
    rc, hc, bc, cc = 1, 1, 1, 1
    for room in rooms:
        poly = room.get('polygon')
        asp = get_aspect(room)
        rtype = classify_room(poly, room['area_m2'], asp) if poly is not None else 'unknown'
        room['type'] = rtype
        if rtype == 'hallway':
            room['name'] = f"Hallway" if hc == 1 else f"Hallway {hc}"; hc += 1
        elif rtype == 'bathroom':
            room['name'] = f"Bathroom" if bc == 1 else f"Bathroom {bc}"; bc += 1
        elif rtype == 'closet':
            room['name'] = f"Closet" if cc == 1 else f"Closet {cc}"; cc += 1
        else:
            room['name'] = f"Room {rc}"; rc += 1

    doors = detect_doors(rooms)
    print(f"  {len(doors)} doors")

    title = f'v52 Line Graph v5 — {mesh_path.stem}'
    render_floorplan(rooms, doors, transform, angle_rads,
                     out_dir / 'floorplan.png', title)
    render_debug(wall_density, mask, wall_img, all_walls, selected, recovered,
                 line_img, interior, labels, rooms, transform, angle_rads,
                 out_dir / 'debug.png')

    total = sum(r.get('area_m2', 0) for r in rooms)
    results = {
        'approach': 'v52_line_graph_v5',
        'angles_deg': [round(math.degrees(a), 1) for a in angle_rads],
        'wall_segments': len(all_selected),
        'recovered_walls': len(recovered),
        'rooms': [{'name': r.get('name'), 'area_m2': r['area_m2'],
                    'type': r['type'], 'vertices': r['vertices']} for r in rooms],
        'doors': len(doors), 'total_m2': round(total, 1)
    }
    with open(out_dir / 'floorplan.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== v52 Summary ===")
    print(f"  Angles: {[f'{math.degrees(a):.0f}°' for a in angle_rads]}")
    print(f"  Walls: {len(all_selected)} ({len(selected)} initial + {len(recovered)} recovered)")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {total:.1f}m², {len(doors)} doors")
    return out_dir / 'floorplan.png'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', '-r', type=float, default=0.02)
    args = parser.parse_args()
    out_dir = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / 'results' / 'v52_line_graph_v5'
    result = process_mesh(args.mesh_path, out_dir, args.resolution)
    if result:
        shutil.copy2(result, Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')


if __name__ == '__main__':
    main()

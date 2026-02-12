#!/usr/bin/env python3
"""
mesh2plan v47 - Combined Multimodal Pipeline (Stage 4)

Fuses ALL data sources into a refined floor plan with doors, windows, materials:
- Mesh wall density (v41b: normal-filtered, cleanest signal)
- Room segmentation (v40: density-ridge watershed + Hough angle detection)
- Photo feature points (v46: 73K projected 3D points)
- Texture classification (v45: material profiles per room)
- Door/window candidates (v45: texture-based, v46: photo-based)

Outputs:
- floorplan_final.png — clean architectural floor plan
- floorplan_debug.png — debug overlay with all data sources
- floorplan.svg — vector output
- floorplan.json — structured data
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import json
import cv2
from pathlib import Path
from scipy import ndimage
from collections import Counter, defaultdict
import math
import shutil
import xml.etree.ElementTree as ET

# ============================================================
# Paths
# ============================================================
BASE = Path(__file__).resolve().parent.parent
SCAN_DIR = BASE / "data" / "multiroom" / "2026_02_10_18_31_36"
OUT_DIR = BASE / "results" / "v47_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_DIR = BASE / "results" / "v44_stage1"
STAGE2_DIR = BASE / "results" / "v45_stage2"
STAGE3_DIR = BASE / "results" / "v46_stage3"

RESOLUTION = 0.02  # m/px
PHOTO_WEIGHT = 0.3  # photo feature weight relative to mesh density

# ============================================================
# 1. Wall Density from Mesh (v41b approach)
# ============================================================
def create_wall_density(mesh, resolution=RESOLUTION, normal_thresh=0.5):
    """Wall-only density using face normals (v41b)."""
    normals = mesh.face_normals
    centroids = mesh.triangles_center

    y_comp = np.abs(normals[:, 1])
    wall_mask = y_comp < normal_thresh
    wall_strength = 1.0 - y_comp

    wall_centroids = centroids[wall_mask]
    wall_weights = wall_strength[wall_mask]

    x, z = wall_centroids[:, 0], wall_centroids[:, 2]
    pad = 0.3
    x_min, x_max = x.min() - pad, x.max() + pad
    z_min, z_max = z.min() - pad, z.max() + pad
    w = int((x_max - x_min) / resolution) + 1
    h = int((z_max - z_min) / resolution) + 1

    wall_density = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(wall_density, (zi, xi), wall_weights)

    # All-vertex density for apartment mask
    all_x, all_z = mesh.vertices[:, 0], mesh.vertices[:, 2]
    all_density = np.zeros((h, w), dtype=np.float32)
    axi = np.clip(((all_x - x_min) / resolution).astype(int), 0, w - 1)
    azi = np.clip(((all_z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(all_density, (azi, axi), 1)

    transform = (x_min, z_min, resolution)
    print(f"  Wall faces: {wall_mask.sum()}/{len(normals)} ({wall_mask.mean()*100:.1f}%)")
    print(f"  Grid: {w}x{h}, wall density range: {wall_density.min():.1f}-{wall_density.max():.1f}")
    return wall_density, all_density, transform


# ============================================================
# 2. Photo Feature Density
# ============================================================
def create_photo_feature_density(feature_points, shape, transform):
    """Create density image from photo-projected 3D feature points."""
    x_min, z_min, res = transform
    h, w = shape
    feat_density = np.zeros((h, w), dtype=np.float32)

    if not feature_points:
        return feat_density

    xs = np.array([p['x'] for p in feature_points])
    zs = np.array([p['z'] for p in feature_points])
    xi = np.clip(((xs - x_min) / res).astype(int), 0, w - 1)
    zi = np.clip(((zs - z_min) / res).astype(int), 0, h - 1)
    np.add.at(feat_density, (zi, xi), 1)

    print(f"  Photo feature density: {len(feature_points)} points, max={feat_density.max():.0f}")
    return feat_density


def fuse_densities(wall_density, photo_density, photo_weight=PHOTO_WEIGHT):
    """Combine mesh wall density + photo feature density."""
    # Normalize each to [0,1] range before combining
    wd = wall_density.copy()
    pd = photo_density.copy()

    wd_max = wd.max()
    pd_max = pd.max()
    if wd_max > 0:
        wd /= wd_max
    if pd_max > 0:
        pd /= pd_max

    # Gaussian blur photo density slightly (points are sparser)
    pd = cv2.GaussianBlur(pd, (5, 5), 1.5)

    fused = wd + photo_weight * pd
    print(f"  Fused density range: {fused.min():.3f}-{fused.max():.3f}")
    return fused


# ============================================================
# 3. Apartment Mask + Room Segmentation (v39/v40 approach)
# ============================================================
def get_apartment_mask(density, threshold=1):
    mask = (density >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 1, -1)
    return mask


def find_seeds(wall_density, mask, min_dist=25, n_target=6):
    """Find room seeds in low wall-density areas."""
    d = wall_density.copy()
    d[mask == 0] = 999

    d_smooth = cv2.GaussianBlur(d, (11, 11), 3)
    masked_vals = d_smooth[mask > 0]
    d_max = masked_vals.max() if len(masked_vals) > 0 else 1
    inv = d_max - d_smooth
    inv[mask == 0] = 0

    boundary_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    combined = inv * 0.5 + boundary_dist * 0.5
    combined[mask == 0] = 0

    for md in [min_dist, min_dist - 5, min_dist - 10, 10, 5]:
        if md < 5:
            md = 5
        ks = md * 2 + 1
        local_max = ndimage.maximum_filter(combined, size=ks)
        peaks = (combined == local_max) & (combined > md * 0.3)
        n_labels, peak_labels = cv2.connectedComponents(peaks.astype(np.uint8))
        if n_labels - 1 >= n_target:
            break

    seeds = []
    for lbl in range(1, n_labels):
        ys, xs = np.where(peak_labels == lbl)
        best = np.argmax(combined[ys, xs])
        seeds.append((xs[best], ys[best], combined[ys[best], xs[best]]))
    seeds.sort(key=lambda s: s[2], reverse=True)
    if len(seeds) > n_target * 2:
        seeds = seeds[:n_target * 2]

    print(f"  Seeds found: {len(seeds)}")
    return seeds, combined


def watershed_on_walls(seeds, wall_density, mask):
    d = wall_density.copy()
    d[mask == 0] = 0
    d_smooth = cv2.GaussianBlur(d, (7, 7), 2)

    d_max = d_smooth[mask > 0].max() if mask.any() else 1
    d_norm = np.zeros_like(d_smooth)
    d_norm[mask > 0] = d_smooth[mask > 0] / max(d_max, 1e-6) * 255
    d_uint8 = d_norm.astype(np.uint8)

    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1
    for i, (sx, sy, _) in enumerate(seeds):
        cv2.circle(markers, (sx, sy), 3, i + 2, -1)

    grad_color = cv2.cvtColor(d_uint8, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(grad_color, markers.copy())
    return ws


def extract_rooms(ws, mask, seeds, res=RESOLUTION, min_room_m2=2.5):
    min_px = int(min_room_m2 / (res * res))
    rooms = []
    for i in range(len(seeds)):
        lbl = i + 2
        room_mask = ((ws == lbl) & (mask > 0)).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px >= min_px:
            rooms.append({'id': len(rooms), 'mask': room_mask, 'area_px': area_px})
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    for i, r in enumerate(rooms):
        r['id'] = i
    return rooms


def merge_small(rooms, min_area_px=None, res=RESOLUTION):
    if min_area_px is None:
        min_area_px = int(3.0 / (res * res))
    changed = True
    while changed:
        changed = False
        small = [r for r in rooms if r['area_px'] < min_area_px]
        if not small:
            break
        for sr in small:
            dilated = cv2.dilate(sr['mask'], np.ones((7, 7), np.uint8))
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
                rooms.remove(sr)
                changed = True
                break
    for i, r in enumerate(rooms):
        r['id'] = i
    return rooms


# ============================================================
# 4. Dominant Angle Detection (v40 Hough)
# ============================================================
def detect_dominant_angles(wall_density, mask, n_angles=4):
    d = wall_density.copy()
    d[mask == 0] = 0
    masked_vals = d[mask > 0]
    if len(masked_vals[masked_vals > 0]) == 0:
        return [0, np.pi / 2]
    p80 = np.percentile(masked_vals[masked_vals > 0], 80)
    wall_mask_img = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255

    lines = cv2.HoughLinesP(wall_mask_img, 1, np.pi / 180, threshold=30,
                            minLineLength=20, maxLineGap=10)
    if lines is None:
        return [0, np.pi / 2]

    n_bins = 36
    bins = np.zeros(n_bins)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = math.atan2(y2-y1, x2-x1) % np.pi
        bin_idx = int(angle / np.pi * n_bins) % n_bins
        bins[bin_idx] += length

    ext = np.concatenate([bins[-2:], bins, bins[:2]])
    smooth = np.convolve(ext, [0.15, 0.25, 0.2, 0.25, 0.15], mode='same')[2:-2]

    peaks = []
    for i in range(n_bins):
        prev, nxt = (i-1) % n_bins, (i+1) % n_bins
        if smooth[i] > smooth[prev] and smooth[i] > smooth[nxt]:
            peaks.append((smooth[i], i))
    peaks.sort(reverse=True)

    dominant = []
    for w, bi in peaks[:n_angles*2]:
        a = (bi + 0.5) / n_bins * np.pi
        too_close = any(abs(a - e) % np.pi < np.pi/18 for e in dominant)
        if not too_close:
            dominant.append(a)
        if len(dominant) >= n_angles:
            break

    dominant.sort()
    print(f"  Dominant angles: {[f'{math.degrees(a):.1f}°' for a in dominant]}")
    return dominant if dominant else [0, np.pi / 2]


# ============================================================
# 5. Polygon Extraction with Angle Snap (v40/v41b)
# ============================================================
def snap_to_angles(pts, angles, angle_thresh_deg=15):
    if len(pts) < 3:
        return pts
    angle_thresh = math.radians(angle_thresh_deg)
    n = len(pts)
    snapped = pts.copy().astype(float)

    for iteration in range(8):
        new = snapped.copy()
        for i in range(n):
            j = (i + 1) % n
            dx = snapped[j, 0] - snapped[i, 0]
            dy = snapped[j, 1] - snapped[i, 1]
            length = math.sqrt(dx**2 + dy**2)
            if length < 0.05:
                continue
            edge_angle = math.atan2(dy, dx) % np.pi

            best_angle, best_diff = None, float('inf')
            for a in angles:
                diff = abs(edge_angle - a) % np.pi
                if diff > np.pi/2:
                    diff = np.pi - diff
                if diff < best_diff:
                    best_diff = diff
                    best_angle = a

            if best_diff > angle_thresh:
                continue

            mid = (snapped[i] + snapped[j]) / 2
            target = best_angle
            orig_dir = math.atan2(dy, dx)
            if abs(orig_dir - target) > np.pi/2 and abs(orig_dir - target) < 3*np.pi/2:
                target += np.pi

            half = length / 2
            new[i, 0] = mid[0] - half * math.cos(target)
            new[i, 1] = mid[1] - half * math.sin(target)
            new[j, 0] = mid[0] + half * math.cos(target)
            new[j, 1] = mid[1] + half * math.sin(target)
        snapped = new

    cleaned = [snapped[0]]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - cleaned[-1]) > 0.03:
            cleaned.append(snapped[i])
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) < 0.03:
        cleaned = cleaned[:-1]
    return np.array(cleaned) if len(cleaned) >= 3 else snapped


def intersect_consecutive_edges(pts, angles):
    if len(pts) < 3:
        return pts
    n = len(pts)
    edges = []
    for i in range(n):
        j = (i+1) % n
        dx, dy = pts[j][0]-pts[i][0], pts[j][1]-pts[i][1]
        edges.append((pts[i], pts[j], dx, dy))

    new_pts = []
    for i in range(n):
        j = (i+1) % n
        p1, p2, dx1, dy1 = edges[i]
        p3, p4, dx2, dy2 = edges[j]
        det = dx1*dy2 - dy1*dx2
        if abs(det) < 1e-10:
            new_pts.append(pts[j].copy())
            continue
        t = ((p3[0]-p1[0])*dy2 - (p3[1]-p1[1])*dx2) / det
        ix = p1[0] + t*dx1
        iy = p1[1] + t*dy1
        if np.linalg.norm([ix-pts[j][0], iy-pts[j][1]]) > 1.0:
            new_pts.append(pts[j].copy())
        else:
            new_pts.append(np.array([ix, iy]))
    return np.array(new_pts)


def remove_short_edges(pts, min_length=0.20):
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


def extract_polygon(room_mask, transform, angles, epsilon_factor=0.008):
    x_min, z_min, res = transform
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

    pts_px = simplified.reshape(-1, 2).astype(float)
    pts = np.zeros((len(pts_px), 2))
    pts[:, 0] = pts_px[:, 0] * res + x_min
    pts[:, 1] = pts_px[:, 1] * res + z_min

    pts = snap_to_angles(pts, angles)
    pts = intersect_consecutive_edges(pts, angles)
    pts = remove_short_edges(pts, min_length=0.20)
    pts = remove_collinear(pts)
    return pts


def polygon_area(pts):
    n = len(pts)
    if n < 3:
        return 0
    a = sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))
    return abs(a) / 2


# ============================================================
# 6. Room Classification (material-aware)
# ============================================================
def classify_room_material(polygon, area, material_profile=None):
    """Classify room using geometry + material profile from Stage 2."""
    xs, zs = polygon[:, 0], polygon[:, 1]
    w, h = xs.max() - xs.min(), zs.max() - zs.min()
    aspect = max(w, h) / (min(w, h) + 0.01)

    # Material-based classification (primary)
    if material_profile:
        tile_pct = material_profile.get('tile', 0)
        wood_pct = material_profile.get('wood_door', 0)
        window_pct = material_profile.get('window_glass', 0)
        painted_pct = material_profile.get('painted_wall', 0)

        if tile_pct > 15:
            return "Bathroom"
        if wood_pct > 20 and area < 6:
            return "Entry"
        if painted_pct > 60 and area > 8:
            return "Living Room"
        if window_pct > 30 and area > 10:
            return "Living Room"
        if window_pct > 30 and area > 5:
            return "Bedroom"

    # Geometry-based fallback
    if area < 3:
        return "Closet"
    if area < 5 and aspect > 2.0:
        return "Hallway"
    if area < 5:
        return "Bathroom"
    if aspect > 2.5:
        return "Hallway"
    if area > 15:
        return "Living Room"
    if area > 8:
        return "Bedroom"
    return "Room"


# ============================================================
# 7. Door Detection (multi-source fusion)
# ============================================================
def detect_doors_density_gap(rooms, transform):
    """Detect doors via density gap analysis along shared walls."""
    doors = []
    x_min, z_min, res = transform
    for i in range(len(rooms)):
        for j in range(i+1, len(rooms)):
            d1 = cv2.dilate(rooms[i]['mask'], np.ones((9, 9), np.uint8))
            d2 = cv2.dilate(rooms[j]['mask'], np.ones((9, 9), np.uint8))
            overlap = d1 & d2
            if overlap.sum() > 15:
                ys, xs = np.where(overlap > 0)
                cx_px, cy_px = xs.mean(), ys.mean()
                cx = cx_px * res + x_min
                cz = cy_px * res + z_min
                doors.append({
                    'source': 'density_gap',
                    'rooms': (rooms[i]['id'], rooms[j]['id']),
                    'x': float(cx), 'z': float(cz),
                    'px': float(cx_px), 'py': float(cy_px),
                    'confidence': 0.5,
                })
    return doors


def fuse_doors(density_doors, texture_doors, photo_doors, merge_dist=0.8):
    """Fuse door candidates from multiple sources. 2+ sources = high confidence."""
    all_candidates = []
    for d in density_doors:
        all_candidates.append({**d, 'sources': {'density_gap'}})
    for d in texture_doors:
        all_candidates.append({
            'source': 'texture', 'x': d['x'], 'z': d['z'],
            'confidence': 0.4, 'sources': {'texture'},
            'n_faces': d.get('n_faces', 0),
        })
    for d in photo_doors:
        all_candidates.append({
            'source': 'photo', 'x': d['x'], 'z': d['z'],
            'confidence': 0.3, 'sources': {'photo'},
        })

    # Merge nearby candidates
    fused = []
    used = [False] * len(all_candidates)
    for i in range(len(all_candidates)):
        if used[i]:
            continue
        cluster = [all_candidates[i]]
        used[i] = True
        for j in range(i+1, len(all_candidates)):
            if used[j]:
                continue
            dx = all_candidates[i]['x'] - all_candidates[j]['x']
            dz = all_candidates[i]['z'] - all_candidates[j]['z']
            if math.sqrt(dx**2 + dz**2) < merge_dist:
                cluster.append(all_candidates[j])
                used[j] = True

        # Merge cluster
        sources = set()
        for c in cluster:
            sources |= c.get('sources', set())
        avg_x = np.mean([c['x'] for c in cluster])
        avg_z = np.mean([c['z'] for c in cluster])
        confidence = min(1.0, 0.3 * len(sources) + 0.1 * (len(cluster) - 1))
        rooms_info = None
        for c in cluster:
            if 'rooms' in c:
                rooms_info = c['rooms']
                break

        fused.append({
            'x': float(avg_x), 'z': float(avg_z),
            'sources': list(sources),
            'n_sources': len(sources),
            'confidence': float(confidence),
            'rooms': rooms_info,
        })

    fused.sort(key=lambda d: d['confidence'], reverse=True)
    print(f"  Fused doors: {len(fused)} (from {len(all_candidates)} candidates)")
    for d in fused:
        print(f"    ({d['x']:.2f}, {d['z']:.2f}) conf={d['confidence']:.2f} sources={d['sources']}")
    return fused


# ============================================================
# 8. Window Detection (multi-source)
# ============================================================
def find_exterior_edges(rooms):
    """Find polygon edges not shared with another room (= exterior walls)."""
    exterior_edges = []
    all_edges = []

    for room in rooms:
        poly = room.get('polygon')
        if poly is None:
            continue
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = (poly[i].copy(), poly[j].copy())
            all_edges.append((room['id'], i, edge))

    # An edge is exterior if no other room has a nearby parallel edge
    for rid, eidx, (p1, p2) in all_edges:
        mid = (p1 + p2) / 2
        is_shared = False
        for rid2, _, (q1, q2) in all_edges:
            if rid2 == rid:
                continue
            mid2 = (q1 + q2) / 2
            dist = np.linalg.norm(mid - mid2)
            if dist < 0.5:  # close midpoints = shared wall
                is_shared = True
                break
        if not is_shared:
            exterior_edges.append({
                'room_id': rid, 'edge_idx': eidx,
                'p1': p1.tolist(), 'p2': p2.tolist(),
                'mid': mid.tolist(),
                'length': float(np.linalg.norm(p2 - p1)),
            })
    return exterior_edges


def fuse_windows(texture_windows, exterior_edges, merge_dist=1.0):
    """Place windows on exterior walls near texture-based window candidates."""
    windows = []
    for tw in texture_windows:
        # Find nearest exterior edge
        best_edge = None
        best_dist = float('inf')
        for edge in exterior_edges:
            mid = np.array(edge['mid'])
            dist = math.sqrt((tw['x'] - mid[0])**2 + (tw['z'] - mid[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_edge = edge

        if best_edge and best_dist < merge_dist:
            windows.append({
                'x': float(tw['x']), 'z': float(tw['z']),
                'edge': best_edge,
                'confidence': min(1.0, tw.get('n_faces', 100) / 500),
                'sources': ['texture'],
            })

    # Deduplicate
    unique = []
    for w in windows:
        too_close = False
        for u in unique:
            if math.sqrt((w['x']-u['x'])**2 + (w['z']-u['z'])**2) < 0.5:
                too_close = True
                break
        if not too_close:
            unique.append(w)

    print(f"  Windows: {len(unique)} on exterior walls")
    return unique


# ============================================================
# 9. Final Rendering — Clean Architectural Style
# ============================================================
WALL_THICKNESS = 0.12  # meters

def render_final_floorplan(rooms, doors, windows, transform, angles, output_path):
    """Render clean architectural floor plan."""
    x_min, z_min, res = transform
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    room_fills = [
        '#FFFFFF', '#F8F8F8', '#FFFFFF', '#F5F5F5', '#FAFAFA',
        '#F8F8F8', '#FFFFFF', '#F5F5F5',
    ]

    # Draw room fills
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:, 0], pc[:, 1], color=room_fills[i % len(room_fills)],
                edgecolor='none', zorder=1)

    # Draw walls (thick black lines)
    for room in rooms:
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        pc = np.vstack([poly, poly[0]])
        ax.plot(pc[:, 0], pc[:, 1], 'k-', linewidth=3.5, solid_capstyle='round',
                solid_joinstyle='round', zorder=3)

    # Draw doors
    for door in doors:
        if door['confidence'] < 0.3:
            continue
        dx, dz = door['x'], door['z']
        door_width = 0.8
        # Draw door as gap + arc
        circle = plt.Circle((dx, dz), door_width/2, fill=False,
                            edgecolor='#444444', linewidth=1.5,
                            linestyle='-', zorder=4)
        ax.add_patch(circle)
        # Small square marker
        ax.plot(dx, dz, 's', color='white', markersize=6,
                markeredgecolor='#444444', markeredgewidth=1.5, zorder=5)

    # Draw windows
    for win in windows:
        wx, wz = win['x'], win['z']
        edge = win.get('edge')
        if edge:
            p1 = np.array(edge['p1'])
            p2 = np.array(edge['p2'])
            d = p2 - p1
            d_len = np.linalg.norm(d)
            if d_len > 0:
                d_norm = d / d_len
                perp = np.array([-d_norm[1], d_norm[0]])
                # Window symbol: three parallel lines
                w_half = min(0.5, d_len * 0.3)
                mid = np.array([wx, wz])
                for offset in [-0.04, 0, 0.04]:
                    p_start = mid - d_norm * w_half + perp * offset
                    p_end = mid + d_norm * w_half + perp * offset
                    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
                            color='#2196F3', linewidth=2, zorder=4)

    # Room labels
    for room in rooms:
        poly = room.get('polygon')
        if poly is None:
            continue
        cx, cz = poly.mean(axis=0)
        name = room.get('name', '?')
        area = room.get('area_m2', 0)
        ax.text(cx, cz, f"{name}", ha='center', va='center',
                fontsize=11, fontweight='bold', color='#333333', zorder=6)
        ax.text(cx, cz - 0.35, f"{area:.1f} m²", ha='center', va='center',
                fontsize=9, color='#666666', zorder=6)

    # Scale bar
    ax.set_aspect('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_y = ylim[0] + (ylim[1] - ylim[0]) * 0.03
    bar_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    ax.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], 'k-', linewidth=3, zorder=10)
    ax.plot([bar_x, bar_x], [bar_y - 0.05, bar_y + 0.05], 'k-', linewidth=2, zorder=10)
    ax.plot([bar_x + 1, bar_x + 1], [bar_y - 0.05, bar_y + 0.05], 'k-', linewidth=2, zorder=10)
    ax.text(bar_x + 0.5, bar_y - 0.15, '1 m', ha='center', fontsize=9, zorder=10)

    # North arrow
    arrow_x = xlim[1] - (xlim[1] - xlim[0]) * 0.08
    arrow_y = ylim[1] - (ylim[1] - ylim[0]) * 0.08
    ax.annotate('', xy=(arrow_x, arrow_y + 0.4), xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    ax.text(arrow_x, arrow_y + 0.5, 'N', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#333333')

    # Title
    total_area = sum(r.get('area_m2', 0) for r in rooms)
    ax.set_title(f"Floor Plan — {len(rooms)} rooms, {total_area:.1f} m²",
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.grid(False)

    # Remove tick marks for cleaner look
    ax.tick_params(axis='both', which='both', length=0, labelsize=8, colors='#999999')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# 10. Debug Rendering
# ============================================================
def render_debug(wall_density, photo_density, fused_density, all_density, mask,
                 rooms, doors, windows, wall_segments, transform, angles, output_path):
    """Debug overlay showing all data sources."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    x_min, z_min, res = transform

    room_colors = [(255,100,100), (100,255,100), (100,100,255),
                   (255,255,100), (255,100,255), (100,255,255),
                   (200,150,100), (150,100,200)]

    # 1. Wall density (mesh)
    ax = axes[0, 0]
    ax.imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    ax.set_title('Mesh Wall Density (v41b)', fontsize=10)

    # 2. Photo feature density
    ax = axes[0, 1]
    ax.imshow(np.log1p(photo_density), cmap='hot', origin='lower')
    ax.set_title(f'Photo Feature Density (Stage 3)', fontsize=10)

    # 3. Fused density
    ax = axes[0, 2]
    h, w = fused_density.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    wd_n = np.log1p(wall_density)
    wd_n = wd_n / wd_n.max() if wd_n.max() > 0 else wd_n
    pd_n = np.log1p(photo_density)
    pd_n = pd_n / pd_n.max() if pd_n.max() > 0 else pd_n
    overlay[:, :, 2] = wd_n  # mesh = blue
    overlay[:, :, 0] = pd_n  # photo = red
    overlay[:, :, 1] = np.minimum(wd_n, pd_n) * 0.5
    ax.imshow(overlay, origin='lower')
    ax.set_title('Fused: Blue=mesh, Red=photo', fontsize=10)

    # 4. Room segmentation
    ax = axes[1, 0]
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for room in rooms:
        vis[room['mask'] > 0] = room_colors[room['id'] % len(room_colors)]
    ax.imshow(vis, origin='lower')
    ax.set_title(f'Room Segmentation ({len(rooms)} rooms)', fontsize=10)

    # 5. Polygons + wall segments
    ax = axes[1, 1]
    ax.imshow(np.log1p(wall_density), cmap='gray', origin='lower', alpha=0.5)
    for room in rooms:
        poly = room.get('polygon')
        if poly is None:
            continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        c = np.array(room_colors[room['id'] % len(room_colors)]) / 255.0
        pp = np.vstack([np.column_stack([px, pz]), [px[0], pz[0]]])
        ax.plot(pp[:, 0], pp[:, 1], '-', color=c, linewidth=2)
        ax.fill(pp[:, 0], pp[:, 1], color=c, alpha=0.15)
    # Wall segments from photos
    for seg in wall_segments:
        ax.plot([seg['px1'], seg['px2']], [seg['py1'], seg['py2']],
                'g-', linewidth=1, alpha=0.6)
    ax.set_title('Polygons + Photo Wall Segments (green)', fontsize=10)

    # 6. Doors + Windows
    ax = axes[1, 2]
    ax.imshow(np.log1p(wall_density), cmap='gray', origin='lower', alpha=0.3)
    for room in rooms:
        poly = room.get('polygon')
        if poly is None:
            continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        pp = np.vstack([np.column_stack([px, pz]), [px[0], pz[0]]])
        ax.plot(pp[:, 0], pp[:, 1], 'k-', linewidth=1.5)
    for door in doors:
        dpx = (door['x'] - x_min) / res
        dpy = (door['z'] - z_min) / res
        color = 'red' if door['n_sources'] >= 2 else 'orange'
        ax.plot(dpx, dpy, 's', color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(f"D({door['n_sources']}s)", (dpx, dpy), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    for win in windows:
        wpx = (win['x'] - x_min) / res
        wpy = (win['z'] - z_min) / res
        ax.plot(wpx, wpy, 'D', color='deepskyblue', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)
    ax.set_title(f'Doors ({len(doors)}) + Windows ({len(windows)})', fontsize=10)

    angle_strs = [f"{math.degrees(a):.0f}°" for a in angles]
    fig.suptitle(f'v47 Combined Pipeline Debug — Angles: {", ".join(angle_strs)}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# 11. SVG Output
# ============================================================
def render_svg(rooms, doors, windows, transform, output_path):
    """Generate SVG floor plan."""
    x_min, z_min, res = transform

    # Compute bounds
    all_pts = []
    for room in rooms:
        poly = room.get('polygon')
        if poly is not None:
            all_pts.extend(poly.tolist())
    if not all_pts:
        return
    all_pts = np.array(all_pts)
    bx_min, bz_min = all_pts.min(axis=0) - 0.5
    bx_max, bz_max = all_pts.max(axis=0) + 0.5

    scale = 100  # px per meter
    svg_w = (bx_max - bx_min) * scale
    svg_h = (bz_max - bz_min) * scale

    def tx(x):
        return (x - bx_min) * scale
    def tz(z):
        return (z - bz_min) * scale

    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': f'{svg_w:.0f}', 'height': f'{svg_h:.0f}',
        'viewBox': f'0 0 {svg_w:.0f} {svg_h:.0f}',
    })

    # Background
    ET.SubElement(svg, 'rect', {
        'width': '100%', 'height': '100%', 'fill': 'white'
    })

    # Room fills
    for room in rooms:
        poly = room.get('polygon')
        if poly is None:
            continue
        points = ' '.join(f'{tx(p[0]):.1f},{tz(p[1]):.1f}' for p in poly)
        ET.SubElement(svg, 'polygon', {
            'points': points,
            'fill': '#F8F8F8', 'stroke': 'black', 'stroke-width': '3',
            'stroke-linejoin': 'round',
        })

    # Room labels
    for room in rooms:
        poly = room.get('polygon')
        if poly is None:
            continue
        cx, cz = poly.mean(axis=0)
        name = room.get('name', '?')
        area = room.get('area_m2', 0)

        text = ET.SubElement(svg, 'text', {
            'x': f'{tx(cx):.1f}', 'y': f'{tz(cz):.1f}',
            'text-anchor': 'middle', 'dominant-baseline': 'middle',
            'font-family': 'Arial, sans-serif', 'font-size': '12',
            'font-weight': 'bold', 'fill': '#333333',
        })
        text.text = name

        text2 = ET.SubElement(svg, 'text', {
            'x': f'{tx(cx):.1f}', 'y': f'{tz(cz) + 16:.1f}',
            'text-anchor': 'middle', 'dominant-baseline': 'middle',
            'font-family': 'Arial, sans-serif', 'font-size': '10',
            'fill': '#666666',
        })
        text2.text = f'{area:.1f} m²'

    # Door symbols
    for door in doors:
        if door['confidence'] < 0.3:
            continue
        cx, cz = tx(door['x']), tz(door['z'])
        ET.SubElement(svg, 'circle', {
            'cx': f'{cx:.1f}', 'cy': f'{cz:.1f}', 'r': '8',
            'fill': 'white', 'stroke': '#444444', 'stroke-width': '2',
        })

    # Window symbols
    for win in windows:
        cx, cz = tx(win['x']), tz(win['z'])
        ET.SubElement(svg, 'rect', {
            'x': f'{cx - 6:.1f}', 'y': f'{cz - 2:.1f}',
            'width': '12', 'height': '4',
            'fill': '#2196F3', 'stroke': 'none',
        })

    # Scale bar
    sb_x, sb_y = 20, svg_h - 20
    ET.SubElement(svg, 'line', {
        'x1': f'{sb_x}', 'y1': f'{sb_y}',
        'x2': f'{sb_x + scale}', 'y2': f'{sb_y}',
        'stroke': 'black', 'stroke-width': '3',
    })
    sbt = ET.SubElement(svg, 'text', {
        'x': f'{sb_x + scale/2}', 'y': f'{sb_y - 8}',
        'text-anchor': 'middle', 'font-family': 'Arial', 'font-size': '10',
    })
    sbt.text = '1 m'

    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ')
    tree.write(output_path, xml_declaration=True, encoding='unicode')
    print(f"  Saved: {output_path}")


# ============================================================
# 12. JSON Output
# ============================================================
def save_json(rooms, doors, windows, angles, transform, output_path):
    """Save structured floor plan data."""
    data = {
        'version': 'v47_multimodal_combined',
        'transform': {
            'x_min': transform[0], 'z_min': transform[1], 'resolution': transform[2],
        },
        'dominant_angles_deg': [round(math.degrees(a), 1) for a in angles],
        'rooms': [],
        'doors': [],
        'windows': [],
        'total_area_m2': round(sum(r.get('area_m2', 0) for r in rooms), 1),
    }

    for room in rooms:
        r = {
            'id': room['id'],
            'name': room.get('name', '?'),
            'type': room.get('type', '?'),
            'area_m2': room.get('area_m2', 0),
            'vertices': room.get('vertices', 0),
            'polygon': room['polygon'].tolist() if room.get('polygon') is not None else None,
            'materials': room.get('materials', {}),
        }
        data['rooms'].append(r)

    for door in doors:
        d = {
            'x': door['x'], 'z': door['z'],
            'confidence': door['confidence'],
            'sources': door['sources'],
            'rooms': door.get('rooms'),
        }
        data['doors'].append(d)

    for win in windows:
        w = {
            'x': win['x'], 'z': win['z'],
            'confidence': win.get('confidence', 0.5),
            'sources': win.get('sources', []),
        }
        data['windows'].append(w)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("mesh2plan v47 — Combined Multimodal Pipeline")
    print("=" * 60)

    # --- Load mesh ---
    print("\n[1/10] Loading mesh...")
    mesh = trimesh.load_mesh(str(SCAN_DIR / "export_refined.obj"))
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # --- Wall density from mesh ---
    print("\n[2/10] Creating wall-only density (v41b)...")
    wall_density, all_density, transform = create_wall_density(mesh)

    # --- Load photo feature points ---
    print("\n[3/10] Loading photo feature points (Stage 3)...")
    with open(STAGE3_DIR / "feature_points_3d.json") as f:
        fp_data = json.load(f)
    feature_points = fp_data['points']
    print(f"  Loaded {len(feature_points)} points (total: {fp_data['n_total']})")

    # --- Photo feature density ---
    print("\n[4/10] Creating photo feature density + fusing...")
    photo_density = create_photo_feature_density(feature_points, wall_density.shape, transform)
    fused_density = fuse_densities(wall_density, photo_density)

    # --- Apartment mask ---
    print("\n[5/10] Apartment mask + room segmentation...")
    mask = get_apartment_mask(all_density)
    print(f"  Apartment area: {mask.sum() * RESOLUTION * RESOLUTION:.1f} m²")

    # --- Dominant angles (use raw wall density — proven best) ---
    angles = detect_dominant_angles(wall_density, mask)

    # --- Room seeds + watershed (use raw wall density for segmentation) ---
    seeds, seed_score = find_seeds(wall_density, mask, min_dist=25, n_target=6)
    ws = watershed_on_walls(seeds, wall_density, mask)

    # --- Extract rooms ---
    rooms = extract_rooms(ws, mask, seeds)
    print(f"  Raw rooms: {len(rooms)}")
    rooms = merge_small(rooms, min_area_px=int(2.0 / (RESOLUTION * RESOLUTION)))
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    for i, r in enumerate(rooms):
        r['id'] = i
    print(f"  After merge: {len(rooms)}")

    # --- Polygons ---
    print("\n[6/10] Polygon extraction with angle snap...")
    for room in rooms:
        poly = extract_polygon(room['mask'], transform, angles)
        if poly is not None:
            room['polygon'] = poly
            room['area_m2'] = round(polygon_area(poly), 1)
            room['vertices'] = len(poly)
        else:
            room['polygon'] = None
            room['area_m2'] = 0
            room['vertices'] = 0

    rooms = [r for r in rooms if r.get('polygon') is not None]
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    for i, r in enumerate(rooms):
        r['id'] = i

    # --- Load material profiles ---
    print("\n[7/10] Room classification (material-aware)...")
    with open(STAGE2_DIR / "room_materials.json") as f:
        material_profiles = json.load(f)

    # Assign materials to rooms by matching room IDs (best effort)
    # Room order may differ; match by spatial overlap with stage1 rooms
    for room in rooms:
        # Use room index as proxy (same watershed approach)
        rid_str = str(room['id'])
        if rid_str in material_profiles:
            room['materials'] = material_profiles[rid_str]
        else:
            room['materials'] = {}

        mat_prof = room.get('materials')
        room['type'] = classify_room_material(room['polygon'], room['area_m2'], mat_prof)

    # Name rooms (deduplicate names)
    name_counts = defaultdict(int)
    for room in rooms:
        base = room['type']
        name_counts[base] += 1
    name_idx = defaultdict(int)
    for room in rooms:
        base = room['type']
        name_idx[base] += 1
        if name_counts[base] > 1:
            room['name'] = f"{base} {name_idx[base]}"
        else:
            room['name'] = base

    for room in rooms:
        print(f"  Room {room['id']}: {room['name']} — {room['area_m2']}m², {room['vertices']}v")

    # --- Doors ---
    print("\n[8/10] Door detection (multi-source fusion)...")
    density_doors = detect_doors_density_gap(rooms, transform)

    with open(STAGE2_DIR / "door_window_candidates.json") as f:
        tex_candidates = json.load(f)
    texture_doors = tex_candidates.get('doors', [])
    texture_windows_raw = tex_candidates.get('windows', [])

    with open(STAGE3_DIR / "wall_segments.json") as f:
        ws_data = json.load(f)
    wall_segments = ws_data.get('segments', [])
    photo_dw = ws_data.get('door_window_candidates', [])
    photo_doors = [d for d in photo_dw if d.get('type') == 'door']

    doors = fuse_doors(density_doors, texture_doors, photo_doors)

    # --- Windows ---
    print("\n[9/10] Window detection...")
    exterior_edges = find_exterior_edges(rooms)
    print(f"  Exterior edges: {len(exterior_edges)}")
    windows = fuse_windows(texture_windows_raw, exterior_edges)

    # --- Render ---
    print("\n[10/10] Rendering outputs...")
    render_final_floorplan(rooms, doors, windows, transform, angles,
                           OUT_DIR / "floorplan_final.png")
    render_debug(wall_density, photo_density, fused_density, all_density, mask,
                 rooms, doors, windows, wall_segments, transform, angles,
                 OUT_DIR / "floorplan_debug.png")
    render_svg(rooms, doors, windows, transform, OUT_DIR / "floorplan.svg")
    save_json(rooms, doors, windows, angles, transform, OUT_DIR / "floorplan.json")

    # Copy to workspace
    workspace = Path.home() / '.openclaw' / 'workspace'
    shutil.copy2(OUT_DIR / "floorplan_final.png", workspace / "v47_floorplan.png")
    print(f"\n  Copied floorplan_final.png → workspace/v47_floorplan.png")

    # Summary
    total_area = sum(r.get('area_m2', 0) for r in rooms)
    print(f"\n{'='*60}")
    print(f"v47 Combined Pipeline Complete!")
    print(f"  Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    print(f"  Doors: {len(doors)} ({sum(1 for d in doors if d['n_sources']>=2)} multi-source)")
    print(f"  Windows: {len(windows)}")
    print(f"  Angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

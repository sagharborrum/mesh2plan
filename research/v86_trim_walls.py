#!/usr/bin/env python3
"""
mesh2plan v86 - Trim partition walls to room boundaries

Key fixes over v85:
1. Trim partition walls: only render the segment that actually separates two different rooms
   (Wall 11 was 6.35m but should only be ~2m between bathroom/hallway)
2. Better boundary: remove diagonal artifacts with stricter angle snapping
3. Room fills: use Shapely union properly for merged rooms
4. Hallway boundary: constrain to actual room extent
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Arc
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from shapely.geometry import Polygon as ShapelyPoly, MultiPolygon, LineString, box
from shapely.ops import unary_union
from pathlib import Path
import shutil

RESOLUTION = 0.012
WALL_THICK_M = 0.15

def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh

def rotate_points(pts, angle_deg, center=None):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if center is not None: pts = pts - center
    r = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: r += center
    return r

def detect_wall_angles_mirrored(mesh):
    normals = mesh.face_normals
    areas = mesh.area_faces
    wm = np.abs(normals[:, 1]) < 0.3
    wn = normals[wm][:, [0, 2]].copy()
    wn[:, 0] = -wn[:, 0]
    wa = areas[wm]
    angles = np.degrees(np.arctan2(wn[:, 1], wn[:, 0])) % 180
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=wa)
    hist = ndimage.gaussian_filter1d(hist, sigma=2)
    peaks, props = find_peaks(hist, height=hist.max() * 0.2, distance=20)
    top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
    wall_angles = sorted([(a + 90) % 180 for a in top2])
    print(f"Wall angles: {wall_angles}°")
    return wall_angles

def make_density(mesh, face_mask, xmin, zmin, w, h, res):
    areas = mesh.area_faces[face_mask]
    centers = mesh.triangles_center[face_mask][:, [0, 2]].copy()
    centers[:, 0] = -centers[:, 0]
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((centers[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((centers[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    return density

def hough_walls(density, wall_angles, res, angle_tol=15):
    thresh = np.percentile(density[density > 0], 55) if (density > 0).any() else 0
    binary = (density > thresh).astype(np.uint8) * 255
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10,
                            minLineLength=int(0.2 / res), maxLineGap=int(0.08 / res))
    if lines is None: return {i: [] for i in range(len(wall_angles))}
    lines = [l[0] for l in lines]
    wall_lines = {i: [] for i in range(len(wall_angles))}
    for x1, y1, x2, y2 in lines:
        la = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        for i, wa in enumerate(wall_angles):
            diff = min(abs(la - wa), 180 - abs(la - wa))
            if diff < angle_tol:
                wall_lines[i].append((x1, y1, x2, y2))
                break
    return wall_lines

def cluster_walls(wall_lines, wall_angles, density, res, min_span=1.0, min_coverage=0.25):
    cluster_dist = int(0.20 / res)
    all_walls = []
    for angle_idx, wa in enumerate(wall_angles):
        lines = wall_lines[angle_idx]
        if not lines: continue
        items = []
        for x1, y1, x2, y2 in lines:
            rad = np.radians(wa)
            nx, ny = -np.sin(rad), np.cos(rad)
            perp = ((x1+x2)/2)*nx + ((y1+y2)/2)*ny
            dx, dy = np.cos(rad), np.sin(rad)
            p1 = x1*dx + y1*dy
            p2 = x2*dx + y2*dy
            pmin, pmax = min(p1, p2), max(p1, p2)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            items.append((perp, pmin, pmax, length))
        items.sort(key=lambda x: x[0])
        clusters = [[items[0]]]
        for i in range(1, len(items)):
            if items[i][0] - items[i-1][0] < cluster_dist:
                clusters[-1].append(items[i])
            else:
                clusters.append([items[i]])
        for cluster in clusters:
            perp_avg = np.mean([c[0] for c in cluster])
            para_min = min(c[1] for c in cluster)
            para_max = max(c[2] for c in cluster)
            total_length = sum(c[3] for c in cluster)
            span = para_max - para_min
            if span * res < min_span: continue
            if total_length < int(0.5 / res): continue
            rad = np.radians(wa)
            dx, dy = np.cos(rad), np.sin(rad)
            nx, ny = -np.sin(rad), np.cos(rad)
            sx = perp_avg * nx + para_min * dx
            sy = perp_avg * ny + para_min * dy
            ex = perp_avg * nx + para_max * dx
            ey = perp_avg * ny + para_max * dy
            n_samples = 50
            xs = np.linspace(sx, ex, n_samples).astype(int)
            ys = np.linspace(sy, ey, n_samples).astype(int)
            h_d, w_d = density.shape
            valid = (xs >= 0) & (xs < w_d) & (ys >= 0) & (ys < h_d)
            if valid.sum() == 0: continue
            vals = density[ys[valid], xs[valid]]
            d_thresh = np.percentile(density[density > 0], 30) if (density > 0).any() else 0
            coverage = (vals > d_thresh).mean()
            if coverage < min_coverage: continue
            all_walls.append({
                'angle': wa, 'perp': perp_avg,
                'start_px': (int(sx), int(sy)),
                'end_px': (int(ex), int(ey)),
                'span_m': span * res,
                'coverage': coverage,
            })
    return all_walls

def walls_to_m(walls, xmin, zmin, res):
    for w in walls:
        sx, sy = w['start_px']
        ex, ey = w['end_px']
        w['start_m'] = (sx*res+xmin, sy*res+zmin)
        w['end_m'] = (ex*res+xmin, ey*res+zmin)
    return walls

def check_dual_band(wall, density_upper, density_lower):
    sx, sy = wall['start_px']
    ex, ey = wall['end_px']
    xs = np.linspace(sx, ex, 50).astype(int)
    ys = np.linspace(sy, ey, 50).astype(int)
    h, w = density_upper.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() == 0: return False
    ut = np.percentile(density_upper[density_upper > 0], 20) if (density_upper > 0).any() else 0
    lt = np.percentile(density_lower[density_lower > 0], 20) if (density_lower > 0).any() else 0
    uc = (density_upper[ys[valid], xs[valid]] > ut).mean()
    lc = (density_lower[ys[valid], xs[valid]] > lt).mean()
    return uc > 0.15 and lc > 0.15

def snap_contour_to_angles(contour_pts, wall_angles, min_seg_len=0.3):
    n = len(contour_pts)
    snapped = []
    for i in range(n):
        p1 = contour_pts[i]
        p2 = contour_pts[(i+1) % n]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        seg_len = np.sqrt(dx*dx + dy*dy)
        if seg_len < min_seg_len * 0.3: continue
        seg_angle = np.degrees(np.arctan2(dy, dx)) % 180
        best_wa, best_diff = wall_angles[0], 999
        for wa in wall_angles:
            diff = min(abs(seg_angle - wa), 180 - abs(seg_angle - wa))
            if diff < best_diff:
                best_diff = diff
                best_wa = wa
        if best_diff < 25:
            rad = np.radians(best_wa)
            d = np.array([np.cos(rad), np.sin(rad)])
            n_vec = np.array([-np.sin(rad), np.cos(rad)])
            proj1, proj2 = np.dot(p1, d), np.dot(p2, d)
            perp = (np.dot(p1, n_vec) + np.dot(p2, n_vec)) / 2
            snapped.append((proj1*d + perp*n_vec, proj2*d + perp*n_vec))
        else:
            snapped.append((p1, p2))
    if not snapped: return contour_pts
    result = []
    for i in range(len(snapped)):
        s1, e1 = snapped[i]
        s2, e2 = snapped[(i+1) % len(snapped)]
        d1, d2 = e1 - s1, e2 - s2
        det = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(det) < 1e-10:
            result.append(e1)
        else:
            t = ((s2[0]-s1[0])*d2[1] - (s2[1]-s1[1])*d2[0]) / det
            result.append(s1 + t * d1)
    return np.array(result)

def draw_wall_rect(ax, start, end, thickness, color='#2A2A2A', zorder=10):
    s, e = np.array(start), np.array(end)
    d = e - s
    length = np.linalg.norm(d)
    if length < 0.01: return
    d_norm = d / length
    n = np.array([-d_norm[1], d_norm[0]])
    half_t = thickness / 2
    corners = np.array([s - n*half_t, s + n*half_t, e + n*half_t, e - n*half_t])
    poly = MplPoly(corners, closed=True, fc=color, ec=color, lw=0.5, zorder=zorder)
    ax.add_patch(poly)

def trim_wall_to_rooms(wall, room_label_map, xmin, zmin, res, w_img, h_img):
    """Trim a wall to only the segment separating different rooms.
    Walk along the wall, check room labels on both sides. Keep only the portion
    where the two sides have different room labels (both > 0)."""
    sx, sy = wall['start_px']
    ex, ey = wall['end_px']
    rad = np.radians(wall['angle'])
    nx_w, ny_w = -np.sin(rad), np.cos(rad)
    off_px = int(0.20 / res)
    
    n_samples = 100
    ts = np.linspace(0, 1, n_samples)
    partition_mask = np.zeros(n_samples, dtype=bool)
    
    for si, t in enumerate(ts):
        px_s = int(sx + (ex-sx)*t)
        py_s = int(sy + (ey-sy)*t)
        sa_x = np.clip(int(px_s + nx_w * off_px), 0, w_img-1)
        sa_y = np.clip(int(py_s + ny_w * off_px), 0, h_img-1)
        sb_x = np.clip(int(px_s - nx_w * off_px), 0, w_img-1)
        sb_y = np.clip(int(py_s - ny_w * off_px), 0, h_img-1)
        la = room_label_map[sa_y, sa_x]
        lb = room_label_map[sb_y, sb_x]
        if la > 0 and lb > 0 and la != lb:
            partition_mask[si] = True
    
    if not partition_mask.any():
        return None
    
    # Find contiguous runs of True
    runs = []
    start_idx = None
    for i in range(n_samples):
        if partition_mask[i] and start_idx is None:
            start_idx = i
        elif not partition_mask[i] and start_idx is not None:
            runs.append((start_idx, i-1))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, n_samples-1))
    
    # Take the longest run
    if not runs: return None
    best_run = max(runs, key=lambda r: r[1]-r[0])
    t_start = ts[best_run[0]]
    t_end = ts[best_run[1]]
    
    # Add small margin
    margin = 0.02
    t_start = max(0, t_start - margin)
    t_end = min(1, t_end + margin)
    
    new_sx = sx + (ex-sx)*t_start
    new_sy = sy + (ey-sy)*t_start
    new_ex = sx + (ex-sx)*t_end
    new_ey = sy + (ey-sy)*t_end
    
    span = np.sqrt((new_ex-new_sx)**2 + (new_ey-new_sy)**2) * res
    if span < 0.3:
        return None
    
    return {
        'start_m': (new_sx*res+xmin, new_sy*res+zmin),
        'end_m': (new_ex*res+xmin, new_ey*res+zmin),
        'span_m': span,
    }


def main():
    mesh_path = Path('/Users/thelodge/projects/mesh2plan/data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    wall_angles = detect_wall_angles_mirrored(mesh)
    render_rotation = -wall_angles[0]
    
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center_m = pts_xz.mean(axis=0)
    
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    heights = mesh.triangles_center[:, 1]
    
    res = RESOLUTION
    xmin, zmin = pts_xz.min(axis=0) - 0.5
    xmax, zmax = pts_xz.max(axis=0) + 0.5
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    
    # ============ TWO-PASS WALL DETECTION ============
    upper_mask = wall_mask & (heights > -0.8) & (heights < -0.3)
    density_upper = make_density(mesh, upper_mask, xmin, zmin, w, h, res)
    hough_upper = hough_walls(density_upper, wall_angles, res)
    walls_upper = cluster_walls(hough_upper, wall_angles, density_upper, res, min_span=0.8, min_coverage=0.20)
    walls_to_m(walls_upper, xmin, zmin, res)
    print(f"Pass 1 walls: {len(walls_upper)}")
    
    lower_mask = wall_mask & (heights > -1.5) & (heights < -0.8)
    density_lower = make_density(mesh, lower_mask, xmin, zmin, w, h, res)
    density_all = make_density(mesh, wall_mask, xmin, zmin, w, h, res)
    hough_all = hough_walls(density_all, wall_angles, res)
    walls_all_raw = cluster_walls(hough_all, wall_angles, density_all, res, min_span=0.4, min_coverage=0.15)
    
    walls_pass2 = []
    for wall in walls_all_raw:
        if check_dual_band(wall, density_upper, density_lower):
            walls_to_m([wall], xmin, zmin, res)
            walls_pass2.append(wall)
    print(f"Pass 2 validated walls: {len(walls_pass2)}")
    
    all_walls = list(walls_upper)
    for w2 in walls_pass2:
        dup = any(w1['angle'] == w2['angle'] and abs(w1['perp'] - w2['perp']) < int(0.25 / res) for w1 in walls_upper)
        if not dup:
            all_walls.append(w2)
    
    # Connectivity filter
    filtered = []
    for i, wl in enumerate(all_walls):
        if wl['span_m'] > 2.0:
            filtered.append(wl)
        else:
            connected = any(
                np.linalg.norm(np.array(p1) - np.array(p2)) < 0.5
                for j, w2 in enumerate(all_walls) if i != j
                for p1 in [wl['start_m'], wl['end_m']]
                for p2 in [w2['start_m'], w2['end_m']]
            )
            if connected: filtered.append(wl)
    all_walls = filtered
    print(f"Total walls: {len(all_walls)}")
    for i, ww in enumerate(all_walls):
        print(f"  Wall {i}: angle={ww['angle']}° span={ww['span_m']:.2f}m cov={ww['coverage']:.2f}")
    
    # ============ PARTITION & ROOMS ============
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((pts_xz[:, 0] - xmin) / res).astype(int), 0, w - 1)
    apy = np.clip(((pts_xz[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(all_density, (apy, apx), 1)
    all_density = cv2.GaussianBlur(all_density, (11, 11), 3.0)
    apt_thresh = np.percentile(all_density[all_density > 0], 10)
    apt_mask = (all_density > apt_thresh).astype(np.uint8) * 255
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    apt_mask = cv2.erode(apt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours_apt, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apt_contour = max(contours_apt, key=cv2.contourArea)
    
    wall_px = max(4, int(WALL_THICK_M / res))
    partition = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(partition, [apt_contour], 0, 255, wall_px + 2)
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition, outside, (0, 0), 128)
    for wall in all_walls:
        sx, sy = wall['start_px']
        ex, ey = wall['end_px']
        mx, my = (sx+ex)//2, (sy+ey)//2
        if 0 <= my < h and 0 <= mx < w and partition[my, mx] != 128:
            cv2.line(partition, (sx, sy), (ex, ey), 255, wall_px)
    partition[outside[1:-1, 1:-1] == 1] = 128
    
    interior = (partition == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    
    rooms_raw = []
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_m2 = region.sum() * res * res
        if area_m2 < 1.0: continue
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        contour_m = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx])
        room_snapped = snap_contour_to_angles(contour_m, wall_angles, min_seg_len=0.3)
        rooms_raw.append({
            'area': area_m2, 'contour_m': room_snapped,
            'centroid_m': (room_snapped[:,0].mean(), room_snapped[:,1].mean()),
            'label_id': label_id,
        })
    rooms_raw.sort(key=lambda r: r['area'], reverse=True)
    
    print(f"\nDetected {len(rooms_raw)} rooms (before merge):")
    for i, r in enumerate(rooms_raw):
        c = r['contour_m']
        print(f"  Room {i+1}: {c[:,0].max()-c[:,0].min():.2f}×{c[:,1].max()-c[:,1].min():.2f} = {r['area']:.1f}m²")
    
    pre_merge_contours = [r['contour_m'].copy() for r in rooms_raw]
    
    # Merge rooms
    rooms = list(rooms_raw)
    merge_groups = {i: [i] for i in range(len(rooms))}
    
    def try_merge(rooms, merge_groups):
        for i in range(len(rooms)):
            if rooms[i] is None: continue
            for j in range(i+1, len(rooms)):
                if rooms[j] is None: continue
                a_i, a_j = rooms[i]['area'], rooms[j]['area']
                combined = a_i + a_j
                rule1 = (3.0 < min(a_i, a_j) < 6.0 and max(a_i, a_j) < 12.0 and 10.0 < combined < 20.0)
                rule2 = (min(a_i, a_j) < 2.0 and max(a_i, a_j) < 4.0 and combined < 6.0)
                if not (rule1 or rule2): continue
                ci = np.array(rooms[i]['centroid_m'])
                cj = np.array(rooms[j]['centroid_m'])
                if np.linalg.norm(ci - cj) > 4.0: continue
                try:
                    pi = ShapelyPoly(rooms[i]['contour_m']).buffer(0)
                    pj = ShapelyPoly(rooms[j]['contour_m']).buffer(0)
                    if not pi.buffer(0.15).intersects(pj.buffer(0.15)): continue
                except: continue
                print(f"  Merging: {a_i:.1f}m² + {a_j:.1f}m² → {combined:.1f}m²")
                all_pts = np.vstack([rooms[i]['contour_m'], rooms[j]['contour_m']])
                centroid = all_pts.mean(axis=0)
                rooms[i] = {
                    'area': combined, 'contour_m': all_pts,
                    'centroid_m': (centroid[0], centroid[1]),
                    'label_id': rooms[i]['label_id'],
                }
                merge_groups[i] = merge_groups[i] + merge_groups[j]
                del merge_groups[j]
                rooms[j] = None
                return True
        return False
    
    while try_merge(rooms, merge_groups):
        pass
    
    # Rebuild rooms list
    final_rooms = []
    final_render_contours = []
    for i, r in enumerate(rooms):
        if r is None: continue
        group_indices = merge_groups[i]
        group_contours = [pre_merge_contours[gi] for gi in group_indices]
        
        # Use Shapely union for merged room contour
        try:
            polys = [ShapelyPoly(c).buffer(0) for c in group_contours if len(c) >= 3]
            union = unary_union(polys)
            if isinstance(union, MultiPolygon):
                union = max(union.geoms, key=lambda g: g.area)
            hull_pts = np.array(union.exterior.coords)[:-1]
        except:
            hull_pts = np.vstack(group_contours)
        
        final_rooms.append({
            'area': r['area'],
            'contour_m': hull_pts,
            'centroid_m': r['centroid_m'],
        })
        final_render_contours.append(group_contours)
    
    rooms = final_rooms
    rooms_contours = final_render_contours
    
    # Name rooms
    names = []
    bed_n = 0
    for r in rooms:
        a = r['area']
        if a > 10: bed_n += 1; names.append(f'Bedroom {bed_n}')
        elif a > 3.5: names.append('Hallway')
        elif a > 2: names.append('Bathroom')
        else: names.append('WC')
    
    print(f"\nFinal {len(rooms)} rooms:")
    for i, r in enumerate(rooms):
        c = r['contour_m']
        print(f"  {names[i]}: {c[:,0].max()-c[:,0].min():.2f}×{c[:,1].max()-c[:,1].min():.2f} = {r['area']:.1f}m²")
    
    # ============ BUILD ROOM LABEL MAP (for wall trimming) ============
    room_label_map = np.zeros((h, w), dtype=np.int32)
    for ri, contours_list in enumerate(rooms_contours):
        for contour in contours_list:
            pts_px = ((contour - [xmin, zmin]) / res).astype(np.int32)
            cv2.fillPoly(room_label_map, [pts_px], ri + 1)
    
    # ============ CLASSIFY AND TRIM PARTITION WALLS ============
    trimmed_partition_walls = []
    for wi, wall in enumerate(all_walls):
        trimmed = trim_wall_to_rooms(wall, room_label_map, xmin, zmin, res, w, h)
        if trimmed is not None:
            print(f"  Partition wall {wi}: trimmed to {trimmed['span_m']:.2f}m")
            trimmed_partition_walls.append(trimmed)
    
    print(f"Trimmed partition walls: {len(trimmed_partition_walls)}")
    
    # ============ APARTMENT BOUNDARY ============
    wa0 = np.radians(wall_angles[0])
    d0 = np.array([np.cos(wa0), np.sin(wa0)])
    d1 = np.array([-np.sin(wa0), np.cos(wa0)])
    
    room_obbs = []
    for contours_list in rooms_contours:
        all_pts = np.vstack(contours_list)
        p0 = all_pts @ d0
        p1 = all_pts @ d1
        min0, max0 = p0.min(), p0.max()
        min1, max1 = p1.min(), p1.max()
        corners = np.array([
            min0 * d0 + min1 * d1,
            max0 * d0 + min1 * d1,
            max0 * d0 + max1 * d1,
            min0 * d0 + max1 * d1,
        ])
        try:
            p = ShapelyPoly(corners).buffer(WALL_THICK_M * 0.9)
            if p.is_valid: room_obbs.append(p)
        except: pass
    
    apt_union = unary_union(room_obbs).simplify(0.10)  # more aggressive simplification
    if isinstance(apt_union, MultiPolygon):
        apt_union = max(apt_union.geoms, key=lambda g: g.area)
    
    # Buffer out then back in to smooth corners
    apt_union = apt_union.buffer(0.05).buffer(-0.05)
    apt_union = apt_union.simplify(0.15)
    
    apt_pts = np.array(apt_union.exterior.coords)[:-1]
    apt_snapped = snap_contour_to_angles(apt_pts, wall_angles, min_seg_len=0.3)
    
    # Remove short edges iteratively
    for _ in range(5):
        if len(apt_snapped) < 4: break
        edges = np.array([np.linalg.norm(apt_snapped[(i+1)%len(apt_snapped)] - apt_snapped[i])
                          for i in range(len(apt_snapped))])
        shortest = np.argmin(edges)
        if edges[shortest] > 0.40: break
        # Remove the vertex creating the shortest edge by merging with neighbor
        apt_snapped = np.delete(apt_snapped, shortest, axis=0)
    
    # Re-snap after cleanup
    apt_snapped = snap_contour_to_angles(apt_snapped, wall_angles, min_seg_len=0.3)
    
    print(f"Apartment boundary: {len(apt_pts)} → {len(apt_snapped)} vertices")
    
    # ============ RENDER ============
    rot = render_rotation
    
    rooms_rot_fills = []
    for contours_list in rooms_contours:
        rotated = [rotate_points(c, rot, center_m) for c in contours_list]
        rooms_rot_fills.append(rotated)
    
    rooms_rot = []
    for room in rooms:
        rc = rotate_points(room['contour_m'], rot, center_m)
        rooms_rot.append({'contour': rc, 'centroid': rc.mean(axis=0), 'area': room['area']})
    
    # Rotate trimmed partition walls
    part_walls_rot = []
    for pw in trimmed_partition_walls:
        s = rotate_points(np.array([pw['start_m']]), rot, center_m)[0]
        e = rotate_points(np.array([pw['end_m']]), rot, center_m)[0]
        part_walls_rot.append((s, e))
    
    bnd_rot = rotate_points(apt_snapped, rot, center_m)
    
    # Compute plot bounds
    all_pts_list = []
    for fills in rooms_rot_fills:
        for c in fills:
            all_pts_list.append(c)
    all_rc = np.vstack(all_pts_list)
    rxmin, rymin = all_rc.min(axis=0) - 1.0
    rxmax, rymax = all_rc.max(axis=0) + 1.0
    
    fig_w = 14
    fig_h = fig_w * (rymax-rymin) / (rxmax-rxmin)
    fig_h = max(fig_h, 8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_facecolor('#F8F8F8')
    
    # Grid
    for x in np.arange(int(rxmin)-1, int(rxmax)+2, 0.5):
        ax.axvline(x, color='#EEEEEE', lw=0.3, zorder=0)
    for y in np.arange(int(rymin)-1, int(rymax)+2, 0.5):
        ax.axhline(y, color='#EEEEEE', lw=0.3, zorder=0)
    
    # Room fills — axis-aligned bboxes in rotated space (all rooms are rectangular)
    # Inset slightly to avoid bleeding past walls
    inset = 0.06
    for i, fills in enumerate(rooms_rot_fills):
        fc = '#EDE0CC' if rooms[i]['area'] > 8 else '#FFFFFF'
        all_pts = np.vstack(fills)
        x0, y0 = all_pts.min(axis=0) + inset
        x1, y1 = all_pts.max(axis=0) - inset
        if x1 <= x0 or y1 <= y0: continue
        bbox = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        poly = MplPoly(bbox, closed=True, fc=fc, ec='none', zorder=1, alpha=0.9)
        ax.add_patch(poly)
    
    # Outer boundary walls
    outer_thick = WALL_THICK_M * 1.8
    n_bnd = len(bnd_rot)
    for i in range(n_bnd):
        p1, p2 = bnd_rot[i], bnd_rot[(i+1) % n_bnd]
        if np.linalg.norm(p2 - p1) < 0.05: continue
        draw_wall_rect(ax, p1, p2, outer_thick, color='#2A2A2A', zorder=10)
    
    # Interior partition walls (TRIMMED)
    for s, e in part_walls_rot:
        draw_wall_rect(ax, s, e, WALL_THICK_M, color='#2A2A2A', zorder=9)
    
    # Room labels
    for i, rr in enumerate(rooms_rot):
        cx, cy = rr['centroid']
        name = names[i]
        ax.text(cx, cy + 0.15, name, ha='center', va='center', fontsize=12,
                color='#333', fontweight='bold', zorder=20)
        ax.text(cx, cy - 0.2, f'({rooms[i]["area"]:.1f} m²)', ha='center', va='center',
                fontsize=9, color='#888', zorder=20)
    
    # Dimensions for bedrooms
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 8: continue
        c = rr['contour']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        w_m, h_m = x1 - x0, y1 - y0
        off = 0.35
        ax.annotate('', xy=(x1, y1+off), xytext=(x0, y1+off),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{w_m:.2f} m', ha='center', fontsize=8, color='#666', zorder=15)
        ax.annotate('', xy=(x1+off, y1), xytext=(x1+off, y0),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+off+0.1, (y0+y1)/2, f'{h_m:.2f} m', ha='left', va='center', fontsize=8,
                color='#666', rotation=90, zorder=15)
    
    # Door arcs
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 2: continue
        c = rr['contour']
        cx, cy = rr['centroid']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        door_w = 0.80
        if rooms[i]['area'] > 8:
            arc = Arc((cx, y0), door_w, door_w, angle=0, theta1=0, theta2=90,
                      color='#555', lw=1.0, zorder=15)
            ax.add_patch(arc)
        elif rooms[i]['area'] > 2:
            arc = Arc((x1, cy), door_w, door_w, angle=0, theta1=90, theta2=180,
                      color='#555', lw=1.0, zorder=15)
            ax.add_patch(arc)
    
    # Scale bar
    sb_x, sb_y = rxmin + 0.5, rymin + 0.3
    ax.plot([sb_x, sb_x+1], [sb_y, sb_y], 'k-', lw=2.5, zorder=20)
    ax.plot([sb_x, sb_x], [sb_y-0.06, sb_y+0.06], 'k-', lw=2, zorder=20)
    ax.plot([sb_x+1, sb_x+1], [sb_y-0.06, sb_y+0.06], 'k-', lw=2, zorder=20)
    ax.text(sb_x+0.5, sb_y+0.15, '1 m', ha='center', fontsize=9, color='#333', fontweight='bold', zorder=20)
    
    ax.set_xlim(rxmin, rxmax)
    ax.set_ylim(rymin, rymax)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    out = '/tmp/v86_trim_walls.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))

if __name__ == '__main__':
    main()

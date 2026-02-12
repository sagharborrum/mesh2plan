#!/usr/bin/env python3
"""
mesh2plan v89 - Build boundary in rotated (axis-aligned) space

Key insight: work in rotated space where walls are H/V, then all boundary
segments are naturally H/V. No angle snapping needed.

Approach:
1. Rotate all data by -60° so walls are axis-aligned
2. Run Hough in rotated space (walls at 0° and 90°)
3. Build partition, flood fill, merge rooms - all in rotated space
4. Build rectilinear boundary from room bboxes (already H/V)
5. Render directly (already axis-aligned)
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
from shapely.geometry import Polygon as ShapelyPoly, MultiPolygon, box as shapely_box
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

def rotate_2d(pts, angle_deg, center=None):
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
    return wall_angles

def make_density_from_pts(centers, areas, xmin, ymin, w, h, res):
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((centers[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((centers[:, 1] - ymin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    return density

def hough_hv(density, res):
    """Hough for strictly H and V lines."""
    thresh = np.percentile(density[density > 0], 55) if (density > 0).any() else 0
    binary = (density > thresh).astype(np.uint8) * 255
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10,
                            minLineLength=int(0.2 / res), maxLineGap=int(0.08 / res))
    if lines is None: return {'H': [], 'V': []}
    lines = [l[0] for l in lines]
    hv = {'H': [], 'V': []}
    for x1, y1, x2, y2 in lines:
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        if angle < 20:
            hv['H'].append((x1, y1, x2, y2))
        elif angle > 70:
            hv['V'].append((x1, y1, x2, y2))
    return hv

def cluster_hv_walls(hv_lines, density, res, min_span=1.0, min_coverage=0.25):
    cluster_dist = int(0.20 / res)
    all_walls = []
    
    for direction in ['H', 'V']:
        lines = hv_lines[direction]
        if not lines: continue
        items = []
        for x1, y1, x2, y2 in lines:
            if direction == 'H':
                perp = (y1 + y2) / 2  # y position
                para_min = min(x1, x2)
                para_max = max(x1, x2)
            else:
                perp = (x1 + x2) / 2  # x position
                para_min = min(y1, y2)
                para_max = max(y1, y2)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            items.append((perp, para_min, para_max, length))
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
            
            if direction == 'H':
                sx, sy = int(para_min), int(perp_avg)
                ex, ey = int(para_max), int(perp_avg)
            else:
                sx, sy = int(perp_avg), int(para_min)
                ex, ey = int(perp_avg), int(para_max)
            
            # Coverage check
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
                'direction': direction, 'perp': perp_avg,
                'start_px': (sx, sy), 'end_px': (ex, ey),
                'span_m': span * res, 'coverage': coverage,
            })
    return all_walls

def check_dual_band_pts(wall, density_upper, density_lower):
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

def trim_wall_to_rooms(wall, room_label_map, res, w_img, h_img):
    sx, sy = wall['start_px']
    ex, ey = wall['end_px']
    direction = wall['direction']
    off_px = int(0.20 / res)
    n_samples = 100
    ts = np.linspace(0, 1, n_samples)
    partition_mask = np.zeros(n_samples, dtype=bool)
    
    for si, t in enumerate(ts):
        px_s = int(sx + (ex-sx)*t)
        py_s = int(sy + (ey-sy)*t)
        if direction == 'H':
            sa_x, sa_y = px_s, min(py_s + off_px, h_img-1)
            sb_x, sb_y = px_s, max(py_s - off_px, 0)
        else:
            sa_x, sa_y = min(px_s + off_px, w_img-1), py_s
            sb_x, sb_y = max(px_s - off_px, 0), py_s
        sa_x = np.clip(sa_x, 0, w_img-1)
        sa_y = np.clip(sa_y, 0, h_img-1)
        sb_x = np.clip(sb_x, 0, w_img-1)
        sb_y = np.clip(sb_y, 0, h_img-1)
        la = room_label_map[sa_y, sa_x]
        lb = room_label_map[sb_y, sb_x]
        if la > 0 and lb > 0 and la != lb:
            partition_mask[si] = True
    
    if not partition_mask.any(): return None
    runs = []
    start_idx = None
    for i in range(n_samples):
        if partition_mask[i] and start_idx is None: start_idx = i
        elif not partition_mask[i] and start_idx is not None:
            runs.append((start_idx, i-1)); start_idx = None
    if start_idx is not None: runs.append((start_idx, n_samples-1))
    if not runs: return None
    best_run = max(runs, key=lambda r: r[1]-r[0])
    t_start = max(0, ts[best_run[0]] - 0.02)
    t_end = min(1, ts[best_run[1]] + 0.02)
    new_sx = sx + (ex-sx)*t_start
    new_sy = sy + (ey-sy)*t_start
    new_ex = sx + (ex-sx)*t_end
    new_ey = sy + (ey-sy)*t_end
    span = np.sqrt((new_ex-new_sx)**2 + (new_ey-new_sy)**2) * res
    if span < 0.3: return None
    return {
        'start_px': (new_sx, new_sy), 'end_px': (new_ex, new_ey),
        'span_m': span, 'direction': direction,
    }


def main():
    mesh_path = Path('/Users/thelodge/projects/mesh2plan/data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    wall_angles = detect_wall_angles_mirrored(mesh)
    rot_angle = -wall_angles[0]  # -60°
    print(f"Wall angles: {wall_angles}°, rotation: {rot_angle}°")
    
    # Mirror X and get XZ projection
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center_m = pts_xz.mean(axis=0)
    
    # Rotate ALL points to axis-aligned space
    pts_rot = rotate_2d(pts_xz, rot_angle, center_m)
    
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    heights = mesh.triangles_center[:, 1]
    
    # Get rotated face centers
    face_centers_xz = mesh.triangles_center[:, [0, 2]].copy()
    face_centers_xz[:, 0] = -face_centers_xz[:, 0]
    face_centers_rot = rotate_2d(face_centers_xz, rot_angle, center_m)
    
    res = RESOLUTION
    xmin, ymin = pts_rot.min(axis=0) - 0.5
    xmax, ymax = pts_rot.max(axis=0) + 0.5
    w = int((xmax - xmin) / res)
    h = int((ymax - ymin) / res)
    
    # ============ TWO-PASS WALL DETECTION (in rotated space) ============
    upper_mask = wall_mask & (heights > -0.8) & (heights < -0.3)
    upper_centers = face_centers_rot[upper_mask]
    upper_areas = mesh.area_faces[upper_mask]
    density_upper = make_density_from_pts(upper_centers, upper_areas, xmin, ymin, w, h, res)
    
    hough_upper = hough_hv(density_upper, res)
    walls_upper = cluster_hv_walls(hough_upper, density_upper, res, min_span=0.8, min_coverage=0.20)
    print(f"Pass 1 walls: {len(walls_upper)}")
    
    lower_mask = wall_mask & (heights > -1.5) & (heights < -0.8)
    lower_centers = face_centers_rot[lower_mask]
    lower_areas = mesh.area_faces[lower_mask]
    density_lower = make_density_from_pts(lower_centers, lower_areas, xmin, ymin, w, h, res)
    
    all_centers = face_centers_rot[wall_mask]
    all_areas_w = mesh.area_faces[wall_mask]
    density_all = make_density_from_pts(all_centers, all_areas_w, xmin, ymin, w, h, res)
    
    hough_all = hough_hv(density_all, res)
    walls_all_raw = cluster_hv_walls(hough_all, density_all, res, min_span=0.4, min_coverage=0.15)
    
    walls_pass2 = []
    for wall in walls_all_raw:
        if check_dual_band_pts(wall, density_upper, density_lower):
            walls_pass2.append(wall)
    print(f"Pass 2 validated walls: {len(walls_pass2)}")
    
    # Merge pass1 + pass2
    all_walls = list(walls_upper)
    for w2 in walls_pass2:
        dup = any(w1['direction'] == w2['direction'] and abs(w1['perp'] - w2['perp']) < int(0.25 / res) 
                  for w1 in walls_upper)
        if not dup:
            all_walls.append(w2)
    
    # Connectivity filter
    filtered = []
    for i, wl in enumerate(all_walls):
        if wl['span_m'] > 2.0:
            filtered.append(wl)
        else:
            sp = np.array(wl['start_px'])
            ep = np.array(wl['end_px'])
            connected = False
            for j, w2 in enumerate(all_walls):
                if i == j: continue
                for p1 in [sp, ep]:
                    for p2 in [np.array(w2['start_px']), np.array(w2['end_px'])]:
                        if np.linalg.norm(p1 - p2) * res < 0.5:
                            connected = True
                            break
                    if connected: break
                if connected: break
            if connected: filtered.append(wl)
    all_walls = filtered
    
    # Convert to meters
    for ww in all_walls:
        sx, sy = ww['start_px']
        ex, ey = ww['end_px']
        ww['start_m'] = (sx*res+xmin, sy*res+ymin)
        ww['end_m'] = (ex*res+xmin, ey*res+ymin)
    
    print(f"Total walls: {len(all_walls)}")
    for i, ww in enumerate(all_walls):
        print(f"  Wall {i}: {ww['direction']} span={ww['span_m']:.2f}m cov={ww['coverage']:.2f}")
    
    # ============ PARTITION & ROOMS ============
    # Apartment mask from all points
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((pts_rot[:, 0] - xmin) / res).astype(int), 0, w - 1)
    apy = np.clip(((pts_rot[:, 1] - ymin) / res).astype(int), 0, h - 1)
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
        # Get bbox in rotated space (already H/V aligned)
        ys_r, xs_r = np.where(region)
        x0_px, x1_px = xs_r.min(), xs_r.max()
        y0_px, y1_px = ys_r.min(), ys_r.max()
        x0_m = x0_px * res + xmin
        y0_m = y0_px * res + ymin
        x1_m = x1_px * res + xmin
        y1_m = y1_px * res + ymin
        contour_m = np.array([(x0_m, y0_m), (x1_m, y0_m), (x1_m, y1_m), (x0_m, y1_m)])
        rooms_raw.append({
            'area': area_m2,
            'contour_m': contour_m,
            'centroid_m': ((x0_m+x1_m)/2, (y0_m+y1_m)/2),
            'label_id': label_id,
            'bbox_m': (x0_m, y0_m, x1_m, y1_m),
        })
    rooms_raw.sort(key=lambda r: r['area'], reverse=True)
    
    print(f"\nDetected {len(rooms_raw)} rooms (before merge):")
    for i, r in enumerate(rooms_raw):
        x0,y0,x1,y1 = r['bbox_m']
        print(f"  Room {i+1}: {x1-x0:.2f}×{y1-y0:.2f} = {r['area']:.1f}m²")
    
    pre_merge_bboxes = [r['bbox_m'] for r in rooms_raw]
    
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
                # Check adjacency
                bi = rooms[i]['bbox_m']
                bj = rooms[j]['bbox_m']
                gap = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]),
                          max(bi[1], bj[1]) - min(bi[3], bj[3]))
                if gap > 0.3: continue
                print(f"  Merging: {a_i:.1f}m² + {a_j:.1f}m² → {combined:.1f}m²")
                # Merged bbox
                x0 = min(bi[0], bj[0])
                y0 = min(bi[1], bj[1])
                x1 = max(bi[2], bj[2])
                y1 = max(bi[3], bj[3])
                rooms[i] = {
                    'area': combined,
                    'contour_m': np.array([(x0,y0),(x1,y0),(x1,y1),(x0,y1)]),
                    'centroid_m': ((x0+x1)/2, (y0+y1)/2),
                    'label_id': rooms[i]['label_id'],
                    'bbox_m': (x0, y0, x1, y1),
                }
                merge_groups[i] = merge_groups[i] + merge_groups[j]
                del merge_groups[j]
                rooms[j] = None
                return True
        return False
    
    while try_merge(rooms, merge_groups): pass
    
    final_rooms = []
    final_groups = []
    for i, r in enumerate(rooms):
        if r is None: continue
        final_rooms.append(r)
        final_groups.append(merge_groups[i])
    rooms = final_rooms
    
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
        x0,y0,x1,y1 = r['bbox_m']
        print(f"  {names[i]}: {x1-x0:.2f}×{y1-y0:.2f} = {r['area']:.1f}m²")
    
    # ============ BUILD ROOM LABEL MAP ============
    room_label_map = np.zeros((h, w), dtype=np.int32)
    for ri, groups in enumerate(final_groups):
        for gi in groups:
            bb = pre_merge_bboxes[gi]
            x0p = int((bb[0] - xmin) / res)
            y0p = int((bb[1] - ymin) / res)
            x1p = int((bb[2] - xmin) / res)
            y1p = int((bb[3] - ymin) / res)
            room_label_map[y0p:y1p, x0p:x1p] = ri + 1
    
    # ============ TRIM PARTITION WALLS ============
    trimmed_walls = []
    for wi, wall in enumerate(all_walls):
        trimmed = trim_wall_to_rooms(wall, room_label_map, res, w, h)
        if trimmed is not None:
            # Convert trimmed px to meters
            sx, sy = trimmed['start_px']
            ex, ey = trimmed['end_px']
            trimmed['start_m'] = (sx*res+xmin, sy*res+ymin)
            trimmed['end_m'] = (ex*res+xmin, ey*res+ymin)
            print(f"  Partition wall {wi}: trimmed to {trimmed['span_m']:.2f}m")
            trimmed_walls.append(trimmed)
    print(f"Trimmed partition walls: {len(trimmed_walls)}")
    
    # ============ APARTMENT BOUNDARY (rectilinear from room boxes) ============
    room_shapely = []
    for r in rooms:
        x0,y0,x1,y1 = r['bbox_m']
        room_shapely.append(shapely_box(x0, y0, x1, y1).buffer(WALL_THICK_M * 0.8))
    
    apt_union = unary_union(room_shapely)
    if isinstance(apt_union, MultiPolygon):
        apt_union = max(apt_union.geoms, key=lambda g: g.area)
    
    # Simplify to get clean boundary
    apt_union = apt_union.buffer(0.05).buffer(-0.05).simplify(0.08)
    apt_pts = np.array(apt_union.exterior.coords)[:-1]
    
    # Force rectilinear: snap each vertex to nearest H/V grid
    grid_snap = 0.05
    for i in range(len(apt_pts)):
        apt_pts[i, 0] = round(apt_pts[i, 0] / grid_snap) * grid_snap
        apt_pts[i, 1] = round(apt_pts[i, 1] / grid_snap) * grid_snap
    
    # Rebuild as strictly rectilinear
    rect_pts = [apt_pts[0].tolist()]
    for i in range(1, len(apt_pts)):
        prev = rect_pts[-1]
        curr = apt_pts[i].tolist()
        if abs(prev[0] - curr[0]) > grid_snap and abs(prev[1] - curr[1]) > grid_snap:
            # Diagonal - insert corner
            rect_pts.append([curr[0], prev[1]])
        rect_pts.append(curr)
    
    # Check closing segment
    if abs(rect_pts[-1][0] - rect_pts[0][0]) > grid_snap and abs(rect_pts[-1][1] - rect_pts[0][1]) > grid_snap:
        rect_pts.append([rect_pts[0][0], rect_pts[-1][1]])
    
    # Remove very short edges
    for _ in range(10):
        if len(rect_pts) < 4: break
        changed = False
        new_pts = []
        i = 0
        while i < len(rect_pts):
            j = (i + 1) % len(rect_pts)
            dx = abs(rect_pts[j][0] - rect_pts[i][0])
            dy = abs(rect_pts[j][1] - rect_pts[i][1])
            if dx + dy < 0.20 and len(rect_pts) > 4:
                # Skip this vertex (merge with next)
                mid = [(rect_pts[i][0]+rect_pts[j][0])/2, (rect_pts[i][1]+rect_pts[j][1])/2]
                new_pts.append(mid)
                i += 2
                changed = True
            else:
                new_pts.append(rect_pts[i])
                i += 1
        if not changed: break
        rect_pts = new_pts
    
    bnd_pts = np.array(rect_pts)
    print(f"Boundary: {len(bnd_pts)} vertices")
    
    # ============ RENDER (already in rotated space) ============
    all_room_pts = np.vstack([r['contour_m'] for r in rooms])
    rxmin, rymin = all_room_pts.min(axis=0) - 1.2
    rxmax, rymax = all_room_pts.max(axis=0) + 1.2
    
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
    
    # Build partition wall strips for clipping
    wall_strips = []
    for tw in trimmed_walls:
        s, e = np.array(tw['start_m']), np.array(tw['end_m'])
        d = e - s
        length = np.linalg.norm(d)
        if length < 0.01: continue
        d_n = d / length
        n_vec = np.array([-d_n[1], d_n[0]])
        half_t = WALL_THICK_M * 0.5
        corners = [s - n_vec*half_t, s + n_vec*half_t, e + n_vec*half_t, e - n_vec*half_t]
        wall_strips.append(ShapelyPoly(corners))
    
    try:
        apt_poly = ShapelyPoly(bnd_pts).buffer(0)
    except:
        apt_poly = None
    
    # Room fills
    inset = 0.06
    for i, r in enumerate(rooms):
        fc = '#EDE0CC' if r['area'] > 8 else '#F5F0EA'
        x0,y0,x1,y1 = r['bbox_m']
        room_box = ShapelyPoly([(x0+inset,y0+inset),(x1-inset,y0+inset),(x1-inset,y1-inset),(x0+inset,y1-inset)])
        if apt_poly and apt_poly.is_valid:
            room_box = room_box.intersection(apt_poly.buffer(-inset))
        for ws in wall_strips:
            room_box = room_box.difference(ws.buffer(0.02))
        if room_box.is_empty: continue
        geoms = list(room_box.geoms) if isinstance(room_box, MultiPolygon) else [room_box]
        for geom in geoms:
            if geom.area < 0.3: continue
            coords = np.array(geom.exterior.coords)
            poly = MplPoly(coords, closed=True, fc=fc, ec='none', zorder=1, alpha=0.9)
            ax.add_patch(poly)
    
    # Outer boundary walls
    outer_thick = WALL_THICK_M * 1.8
    n_bnd = len(bnd_pts)
    for i in range(n_bnd):
        p1, p2 = bnd_pts[i], bnd_pts[(i+1) % n_bnd]
        if np.linalg.norm(p2 - p1) < 0.05: continue
        draw_wall_rect(ax, p1, p2, outer_thick, color='#2A2A2A', zorder=10)
    
    # Interior partition walls
    for tw in trimmed_walls:
        draw_wall_rect(ax, tw['start_m'], tw['end_m'], WALL_THICK_M, color='#2A2A2A', zorder=9)
    
    # Room labels
    for i, r in enumerate(rooms):
        cx, cy = r['centroid_m']
        name = names[i]
        ax.text(cx, cy + 0.15, name, ha='center', va='center', fontsize=12,
                color='#333', fontweight='bold', zorder=20)
        ax.text(cx, cy - 0.2, f'({r["area"]:.1f} m²)', ha='center', va='center',
                fontsize=9, color='#888', zorder=20)
    
    # Dimensions for bedrooms
    for i, r in enumerate(rooms):
        if r['area'] < 8: continue
        x0,y0,x1,y1 = r['bbox_m']
        w_m, h_m = x1-x0, y1-y0
        off = 0.35
        ax.annotate('', xy=(x1, y1+off), xytext=(x0, y1+off),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{w_m:.2f} m', ha='center', fontsize=8, color='#666', zorder=15)
        ax.annotate('', xy=(x1+off, y1), xytext=(x1+off, y0),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+off+0.1, (y0+y1)/2, f'{h_m:.2f} m', ha='left', va='center', fontsize=8,
                color='#666', rotation=90, zorder=15)
    
    # Bathroom dimension
    for i, r in enumerate(rooms):
        if 'Bathroom' not in names[i]: continue
        x0,y0,x1,y1 = r['bbox_m']
        w_m = x1 - x0
        off = 0.25
        ax.annotate('', xy=(x1, y1+off), xytext=(x0, y1+off),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{w_m:.2f} m', ha='center', fontsize=7, color='#666', zorder=15)
    
    # Door arcs
    for i, r in enumerate(rooms):
        if r['area'] < 2: continue
        x0,y0,x1,y1 = r['bbox_m']
        cx, cy = r['centroid_m']
        door_w = 0.80
        if r['area'] > 8:
            arc = Arc((cx, y0), door_w, door_w, angle=0, theta1=0, theta2=90,
                      color='#555', lw=1.0, zorder=15)
            ax.add_patch(arc)
        elif r['area'] > 2:
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
    out = '/tmp/v89_rotspace.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))

if __name__ == '__main__':
    main()

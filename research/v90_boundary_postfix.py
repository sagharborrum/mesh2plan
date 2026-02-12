#!/usr/bin/env python3
"""
mesh2plan v90 - v87 pipeline + post-process boundary to strict H/V in rotated space

Strategy: Keep v87's wall detection + room detection (which works well), 
but after computing the OBB-based boundary, rotate to axis-aligned space,
force all vertices to lie on H/V grid, then rotate back.
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
            items.append((perp, min(p1,p2), max(p1,p2), np.sqrt((x2-x1)**2+(y2-y1)**2)))
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
            if span * res < min_span or total_length < int(0.5 / res): continue
            rad = np.radians(wa)
            dx, dy = np.cos(rad), np.sin(rad)
            nx, ny = -np.sin(rad), np.cos(rad)
            sx = perp_avg*nx + para_min*dx
            sy = perp_avg*ny + para_min*dy
            ex = perp_avg*nx + para_max*dx
            ey = perp_avg*ny + para_max*dy
            xs = np.linspace(sx, ex, 50).astype(int)
            ys = np.linspace(sy, ey, 50).astype(int)
            h_d, w_d = density.shape
            valid = (xs >= 0) & (xs < w_d) & (ys >= 0) & (ys < h_d)
            if valid.sum() == 0: continue
            d_thresh = np.percentile(density[density > 0], 30) if (density > 0).any() else 0
            coverage = (density[ys[valid], xs[valid]] > d_thresh).mean()
            if coverage < min_coverage: continue
            all_walls.append({
                'angle': wa, 'perp': perp_avg,
                'start_px': (int(sx), int(sy)), 'end_px': (int(ex), int(ey)),
                'span_m': span * res, 'coverage': coverage,
            })
    return all_walls

def walls_to_m(walls, xmin, zmin, res):
    for w in walls:
        sx, sy = w['start_px']; ex, ey = w['end_px']
        w['start_m'] = (sx*res+xmin, sy*res+zmin)
        w['end_m'] = (ex*res+xmin, ey*res+zmin)

def check_dual_band(wall, density_upper, density_lower):
    sx, sy = wall['start_px']; ex, ey = wall['end_px']
    xs = np.linspace(sx, ex, 50).astype(int)
    ys = np.linspace(sy, ey, 50).astype(int)
    h, w = density_upper.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() == 0: return False
    ut = np.percentile(density_upper[density_upper > 0], 20) if (density_upper > 0).any() else 0
    lt = np.percentile(density_lower[density_lower > 0], 20) if (density_lower > 0).any() else 0
    return (density_upper[ys[valid], xs[valid]] > ut).mean() > 0.15 and \
           (density_lower[ys[valid], xs[valid]] > lt).mean() > 0.15

def snap_contour_to_angles(contour_pts, wall_angles, min_seg_len=0.3):
    n = len(contour_pts)
    snapped = []
    for i in range(n):
        p1, p2 = contour_pts[i], contour_pts[(i+1) % n]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if np.sqrt(dx*dx + dy*dy) < min_seg_len * 0.3: continue
        seg_angle = np.degrees(np.arctan2(dy, dx)) % 180
        best_wa, best_diff = wall_angles[0], 999
        for wa in wall_angles:
            diff = min(abs(seg_angle - wa), 180 - abs(seg_angle - wa))
            if diff < best_diff: best_diff, best_wa = diff, wa
        if best_diff < 25:
            rad = np.radians(best_wa)
            d = np.array([np.cos(rad), np.sin(rad)])
            nv = np.array([-np.sin(rad), np.cos(rad)])
            perp = (np.dot(p1, nv) + np.dot(p2, nv)) / 2
            snapped.append((np.dot(p1, d)*d + perp*nv, np.dot(p2, d)*d + perp*nv))
        else:
            snapped.append((p1, p2))
    if not snapped: return contour_pts
    result = []
    for i in range(len(snapped)):
        s1, e1 = snapped[i]; s2, e2 = snapped[(i+1) % len(snapped)]
        d1, d2 = e1 - s1, e2 - s2
        det = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(det) < 1e-10: result.append(e1)
        else:
            t = ((s2[0]-s1[0])*d2[1] - (s2[1]-s1[1])*d2[0]) / det
            result.append(s1 + t * d1)
    return np.array(result)

def draw_wall_rect(ax, start, end, thickness, color='#2A2A2A', zorder=10):
    s, e = np.array(start), np.array(end)
    d = e - s; length = np.linalg.norm(d)
    if length < 0.01: return
    d_n = d / length; n = np.array([-d_n[1], d_n[0]]); ht = thickness / 2
    corners = np.array([s-n*ht, s+n*ht, e+n*ht, e-n*ht])
    ax.add_patch(MplPoly(corners, closed=True, fc=color, ec=color, lw=0.5, zorder=zorder))

def trim_wall_to_rooms(wall, room_label_map, xmin, zmin, res, w_img, h_img):
    sx, sy = wall['start_px']; ex, ey = wall['end_px']
    rad = np.radians(wall['angle'])
    nx_w, ny_w = -np.sin(rad), np.cos(rad)
    off_px = int(0.20 / res)
    n_samples = 100; ts = np.linspace(0, 1, n_samples)
    pmask = np.zeros(n_samples, dtype=bool)
    for si, t in enumerate(ts):
        px_s, py_s = int(sx+(ex-sx)*t), int(sy+(ey-sy)*t)
        ax_, ay_ = np.clip(int(px_s+nx_w*off_px),0,w_img-1), np.clip(int(py_s+ny_w*off_px),0,h_img-1)
        bx_, by_ = np.clip(int(px_s-nx_w*off_px),0,w_img-1), np.clip(int(py_s-ny_w*off_px),0,h_img-1)
        la, lb = room_label_map[ay_, ax_], room_label_map[by_, bx_]
        if la > 0 and lb > 0 and la != lb: pmask[si] = True
    if not pmask.any(): return None
    runs = []; start_idx = None
    for i in range(n_samples):
        if pmask[i] and start_idx is None: start_idx = i
        elif not pmask[i] and start_idx is not None: runs.append((start_idx, i-1)); start_idx = None
    if start_idx is not None: runs.append((start_idx, n_samples-1))
    if not runs: return None
    best = max(runs, key=lambda r: r[1]-r[0])
    t0, t1 = max(0, ts[best[0]]-0.02), min(1, ts[best[1]]+0.02)
    nsx, nsy = sx+(ex-sx)*t0, sy+(ey-sy)*t0
    nex, ney = sx+(ex-sx)*t1, sy+(ey-sy)*t1
    span = np.sqrt((nex-nsx)**2+(ney-nsy)**2) * res
    if span < 0.3: return None
    return {'start_m': (nsx*res+xmin, nsy*res+zmin), 'end_m': (nex*res+xmin, ney*res+zmin), 'span_m': span}


def make_rectilinear_boundary(rooms_contours, wall_angles, center_m):
    """Build boundary: compute OBBs along wall angles, union, then force H/V in rotated space."""
    rot = -wall_angles[0]
    
    # Compute room bboxes in rotated space
    room_boxes = []
    for contours_list in rooms_contours:
        all_pts = np.vstack(contours_list)
        rotated = rotate_points(all_pts, rot, center_m)
        x0, y0 = rotated.min(axis=0)
        x1, y1 = rotated.max(axis=0)
        room_boxes.append(shapely_box(x0, y0, x1, y1))
    
    # Buffer and union
    buffered = [b.buffer(WALL_THICK_M * 0.9) for b in room_boxes]
    union = unary_union(buffered)
    if isinstance(union, MultiPolygon):
        union = max(union.geoms, key=lambda g: g.area)
    
    union = union.buffer(0.03).buffer(-0.03)
    
    # Get coords in rotated space
    coords = np.array(union.exterior.coords)[:-1]
    
    # Snap to grid in rotated space (where walls are H/V)
    snap = 0.05
    coords[:, 0] = np.round(coords[:, 0] / snap) * snap
    coords[:, 1] = np.round(coords[:, 1] / snap) * snap
    
    # Force rectilinear: for each pair, if diagonal, insert corner
    rect = [coords[0].tolist()]
    for i in range(1, len(coords)):
        prev = rect[-1]
        curr = coords[i].tolist()
        dx, dy = abs(curr[0]-prev[0]), abs(curr[1]-prev[1])
        if dx > snap and dy > snap:
            # Insert corner: choose direction that creates shorter stub
            rect.append([curr[0], prev[1]])
        rect.append(curr)
    
    # Close
    prev = rect[-1]; first = rect[0]
    dx, dy = abs(first[0]-prev[0]), abs(first[1]-prev[1])
    if dx > snap and dy > snap:
        rect.append([first[0], prev[1]])
    
    # Remove short edges (< 0.2m)
    for _ in range(15):
        if len(rect) < 6: break
        changed = False
        new_rect = []
        i = 0
        while i < len(rect):
            j = (i + 1) % len(rect)
            seg_len = abs(rect[j][0]-rect[i][0]) + abs(rect[j][1]-rect[i][1])
            if seg_len < 0.20 and len(rect) > 6:
                # Remove short edge by averaging the two endpoints
                # But we need to maintain rectilinearity
                # Skip this point, let next one connect to previous
                k = (j + 1) % len(rect)
                if k < len(rect):
                    # Project: keep the coordinate that aligns with previous
                    prev_idx = (i - 1) % len(rect)
                    if abs(rect[prev_idx][0] - rect[i][0]) < snap:
                        # Previous edge is vertical, so keep x from rect[i]
                        new_rect.append([rect[i][0], rect[k][1]])
                    else:
                        new_rect.append([rect[k][0], rect[i][1]])
                i += 2
                changed = True
            else:
                new_rect.append(rect[i])
                i += 1
        if not changed: break
        rect = new_rect
    
    # Convert back to original space
    rect_arr = np.array(rect)
    result = rotate_points(rect_arr, -rot, center_m)
    return result, rect_arr


def main():
    mesh_path = Path('/Users/thelodge/projects/mesh2plan/data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    wall_angles = detect_wall_angles_mirrored(mesh)
    render_rotation = -wall_angles[0]
    print(f"Wall angles: {wall_angles}°")
    
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
    
    # TWO-PASS WALL DETECTION
    upper_mask = wall_mask & (heights > -0.8) & (heights < -0.3)
    density_upper = make_density(mesh, upper_mask, xmin, zmin, w, h, res)
    walls_upper = cluster_walls(hough_walls(density_upper, wall_angles, res), wall_angles, density_upper, res, 0.8, 0.20)
    walls_to_m(walls_upper, xmin, zmin, res)
    print(f"Pass 1: {len(walls_upper)} walls")
    
    lower_mask = wall_mask & (heights > -1.5) & (heights < -0.8)
    density_lower = make_density(mesh, lower_mask, xmin, zmin, w, h, res)
    density_all = make_density(mesh, wall_mask, xmin, zmin, w, h, res)
    walls_all_raw = cluster_walls(hough_walls(density_all, wall_angles, res), wall_angles, density_all, res, 0.4, 0.15)
    
    walls_pass2 = [w2 for w2 in walls_all_raw if check_dual_band(w2, density_upper, density_lower)]
    for w2 in walls_pass2: walls_to_m([w2], xmin, zmin, res)
    print(f"Pass 2: {len(walls_pass2)} validated")
    
    all_walls = list(walls_upper)
    for w2 in walls_pass2:
        if not any(w1['angle']==w2['angle'] and abs(w1['perp']-w2['perp'])<int(0.25/res) for w1 in walls_upper):
            all_walls.append(w2)
    
    # Connectivity filter
    filtered = []
    for i, wl in enumerate(all_walls):
        if wl['span_m'] > 2.0: filtered.append(wl); continue
        for j, w2 in enumerate(all_walls):
            if i == j: continue
            for p1 in [wl['start_m'], wl['end_m']]:
                for p2 in [w2['start_m'], w2['end_m']]:
                    if np.linalg.norm(np.array(p1)-np.array(p2)) < 0.5:
                        filtered.append(wl); break
                else: continue
                break
            else: continue
            break
    all_walls = filtered
    print(f"Total: {len(all_walls)} walls")
    
    # PARTITION & ROOMS
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((pts_xz[:, 0]-xmin)/res).astype(int), 0, w-1)
    apy = np.clip(((pts_xz[:, 1]-zmin)/res).astype(int), 0, h-1)
    np.add.at(all_density, (apy, apx), 1)
    all_density = cv2.GaussianBlur(all_density, (11, 11), 3.0)
    apt_thresh = np.percentile(all_density[all_density > 0], 10)
    apt_mask = (all_density > apt_thresh).astype(np.uint8) * 255
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    apt_mask = cv2.erode(apt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apt_contour = max(cnts, key=cv2.contourArea)
    
    wall_px = max(4, int(WALL_THICK_M / res))
    partition = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(partition, [apt_contour], 0, 255, wall_px + 2)
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition, outside, (0, 0), 128)
    for wall in all_walls:
        sx, sy = wall['start_px']; ex, ey = wall['end_px']
        mx, my = (sx+ex)//2, (sy+ey)//2
        if 0 <= my < h and 0 <= mx < w and partition[my, mx] != 128:
            cv2.line(partition, (sx, sy), (ex, ey), 255, wall_px)
    partition[outside[1:-1, 1:-1] == 1] = 128
    
    interior = (partition == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    
    rooms_raw = []
    for lid in range(1, n_labels + 1):
        region = (labeled == lid)
        area = region.sum() * res * res
        if area < 1.0: continue
        ru8 = region.astype(np.uint8) * 255
        cnts2, _ = cv2.findContours(ru8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts2: continue
        cnt = max(cnts2, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.012 * cv2.arcLength(cnt, True), True)
        cm = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx])
        cm = snap_contour_to_angles(cm, wall_angles, 0.3)
        rooms_raw.append({'area': area, 'contour_m': cm, 'centroid_m': (cm[:,0].mean(), cm[:,1].mean()), 'label_id': lid})
    rooms_raw.sort(key=lambda r: r['area'], reverse=True)
    
    print(f"\n{len(rooms_raw)} rooms before merge:")
    for i, r in enumerate(rooms_raw):
        c = r['contour_m']
        print(f"  {i+1}: {c[:,0].max()-c[:,0].min():.2f}×{c[:,1].max()-c[:,1].min():.2f} = {r['area']:.1f}m²")
    
    pre_merge = [r['contour_m'].copy() for r in rooms_raw]
    
    # Merge
    rooms = list(rooms_raw)
    mg = {i: [i] for i in range(len(rooms))}
    def try_merge():
        for i in range(len(rooms)):
            if rooms[i] is None: continue
            for j in range(i+1, len(rooms)):
                if rooms[j] is None: continue
                ai, aj = rooms[i]['area'], rooms[j]['area']
                combined = ai + aj
                r1 = 3.0 < min(ai,aj) < 6.0 and max(ai,aj) < 12.0 and 10.0 < combined < 20.0
                r2 = min(ai,aj) < 2.0 and max(ai,aj) < 4.0 and combined < 6.0
                if not (r1 or r2): continue
                if np.linalg.norm(np.array(rooms[i]['centroid_m'])-np.array(rooms[j]['centroid_m'])) > 4: continue
                try:
                    pi = ShapelyPoly(rooms[i]['contour_m']).buffer(0)
                    pj = ShapelyPoly(rooms[j]['contour_m']).buffer(0)
                    if not pi.buffer(0.15).intersects(pj.buffer(0.15)): continue
                except: continue
                print(f"  Merging: {ai:.1f}m² + {aj:.1f}m² → {combined:.1f}m²")
                pts = np.vstack([rooms[i]['contour_m'], rooms[j]['contour_m']])
                rooms[i] = {'area': combined, 'contour_m': pts, 'centroid_m': (pts[:,0].mean(), pts[:,1].mean()), 'label_id': rooms[i]['label_id']}
                mg[i] = mg[i] + mg[j]; del mg[j]; rooms[j] = None
                return True
        return False
    while try_merge(): pass
    
    final_rooms, rooms_contours = [], []
    for i, r in enumerate(rooms):
        if r is None: continue
        gcs = [pre_merge[gi] for gi in mg[i]]
        try:
            polys = [ShapelyPoly(c).buffer(0) for c in gcs if len(c)>=3]
            u = unary_union(polys)
            if isinstance(u, MultiPolygon): u = max(u.geoms, key=lambda g: g.area)
            hp = np.array(u.exterior.coords)[:-1]
        except: hp = np.vstack(gcs)
        final_rooms.append({'area': r['area'], 'contour_m': hp, 'centroid_m': r['centroid_m']})
        rooms_contours.append(gcs)
    rooms = final_rooms
    
    names = []; bed_n = 0
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
    
    # ROOM LABEL MAP
    rlm = np.zeros((h, w), dtype=np.int32)
    for ri, clist in enumerate(rooms_contours):
        for c in clist:
            pts_px = ((c - [xmin, zmin]) / res).astype(np.int32)
            cv2.fillPoly(rlm, [pts_px], ri + 1)
    
    # TRIM PARTITION WALLS
    trimmed = []
    for wi, wall in enumerate(all_walls):
        t = trim_wall_to_rooms(wall, rlm, xmin, zmin, res, w, h)
        if t: print(f"  Part wall {wi}: {t['span_m']:.2f}m"); trimmed.append(t)
    print(f"Partition walls: {len(trimmed)}")
    
    # RECTILINEAR BOUNDARY
    bnd_orig, bnd_rotated = make_rectilinear_boundary(rooms_contours, wall_angles, center_m)
    print(f"Boundary: {len(bnd_orig)} vertices")
    
    # RENDER
    rot = render_rotation
    rooms_rot = [{'contour': rotate_points(r['contour_m'], rot, center_m),
                  'centroid': rotate_points(np.array([r['centroid_m']]), rot, center_m)[0],
                  'area': r['area']} for r in rooms]
    
    rooms_rot_fills = [[rotate_points(c, rot, center_m) for c in clist] for clist in rooms_contours]
    
    pw_rot = [(rotate_points(np.array([t['start_m']]), rot, center_m)[0],
               rotate_points(np.array([t['end_m']]), rot, center_m)[0]) for t in trimmed]
    
    bnd_rot = rotate_points(bnd_orig, rot, center_m)
    
    all_pts = np.vstack([np.vstack(f) for f in rooms_rot_fills])
    rxmin, rymin = all_pts.min(axis=0) - 1.0
    rxmax, rymax = all_pts.max(axis=0) + 1.0
    
    fw = 14; fh = max(8, fw * (rymax-rymin)/(rxmax-rxmin))
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
    ax.set_facecolor('#F8F8F8')
    
    for x in np.arange(int(rxmin)-1, int(rxmax)+2, 0.5): ax.axvline(x, color='#EEE', lw=0.3, zorder=0)
    for y in np.arange(int(rymin)-1, int(rymax)+2, 0.5): ax.axhline(y, color='#EEE', lw=0.3, zorder=0)
    
    # Wall strips for clipping
    wstrips = []
    for s, e in pw_rot:
        d = e - s; l = np.linalg.norm(d)
        if l < 0.01: continue
        dn = d/l; nv = np.array([-dn[1], dn[0]]); ht = WALL_THICK_M * 0.5
        wstrips.append(ShapelyPoly([s-nv*ht, s+nv*ht, e+nv*ht, e-nv*ht]))
    
    try: apt_poly = ShapelyPoly(bnd_rot).buffer(0)
    except: apt_poly = None
    
    # Room fills
    inset = 0.06
    for i, fills in enumerate(rooms_rot_fills):
        fc = '#EDE0CC' if rooms[i]['area'] > 8 else '#F5F0EA'
        ap = np.vstack(fills)
        x0, y0 = ap.min(axis=0) + inset; x1, y1 = ap.max(axis=0) - inset
        if x1 <= x0 or y1 <= y0: continue
        rb = ShapelyPoly([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
        if apt_poly and apt_poly.is_valid: rb = rb.intersection(apt_poly.buffer(-inset))
        for ws in wstrips: rb = rb.difference(ws.buffer(0.02))
        if rb.is_empty: continue
        geoms = list(rb.geoms) if isinstance(rb, MultiPolygon) else [rb]
        for g in geoms:
            if g.area < 0.3: continue
            ax.add_patch(MplPoly(np.array(g.exterior.coords), closed=True, fc=fc, ec='none', zorder=1, alpha=0.9))
    
    # Outer boundary
    ot = WALL_THICK_M * 1.8
    for i in range(len(bnd_rot)):
        p1, p2 = bnd_rot[i], bnd_rot[(i+1) % len(bnd_rot)]
        if np.linalg.norm(p2-p1) < 0.05: continue
        draw_wall_rect(ax, p1, p2, ot, '#2A2A2A', 10)
    
    # Interior walls
    for s, e in pw_rot: draw_wall_rect(ax, s, e, WALL_THICK_M, '#2A2A2A', 9)
    
    # Labels
    for i, rr in enumerate(rooms_rot):
        cx, cy = rr['centroid']
        ax.text(cx, cy+0.15, names[i], ha='center', va='center', fontsize=12, color='#333', fontweight='bold', zorder=20)
        ax.text(cx, cy-0.2, f'({rooms[i]["area"]:.1f} m²)', ha='center', va='center', fontsize=9, color='#888', zorder=20)
    
    # Dimensions
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 8: continue
        c = rr['contour']; x0, y0 = c.min(axis=0); x1, y1 = c.max(axis=0)
        wm, hm = x1-x0, y1-y0; off = 0.35
        ax.annotate('', xy=(x1,y1+off), xytext=(x0,y1+off), arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{wm:.2f} m', ha='center', fontsize=8, color='#666', zorder=15)
        ax.annotate('', xy=(x1+off,y1), xytext=(x1+off,y0), arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+off+0.1, (y0+y1)/2, f'{hm:.2f} m', ha='left', va='center', fontsize=8, color='#666', rotation=90, zorder=15)
    
    for i, rr in enumerate(rooms_rot):
        if 'Bathroom' not in names[i]: continue
        c = rr['contour']; x0 = c[:,0].min(); x1 = c[:,0].max(); y1 = c[:,1].max()
        off = 0.25
        ax.annotate('', xy=(x1,y1+off), xytext=(x0,y1+off), arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{x1-x0:.2f} m', ha='center', fontsize=7, color='#666', zorder=15)
    
    # Door arcs
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 2: continue
        cx, cy = rr['centroid']; c = rr['contour']
        x0, y0 = c.min(axis=0); x1, y1 = c.max(axis=0)
        dw = 0.80
        if rooms[i]['area'] > 8:
            ax.add_patch(Arc((cx, y0), dw, dw, angle=0, theta1=0, theta2=90, color='#555', lw=1.0, zorder=15))
        elif rooms[i]['area'] > 2:
            ax.add_patch(Arc((x1, cy), dw, dw, angle=0, theta1=90, theta2=180, color='#555', lw=1.0, zorder=15))
    
    # Scale bar
    sbx, sby = rxmin + 0.5, rymin + 0.3
    ax.plot([sbx, sbx+1], [sby, sby], 'k-', lw=2.5, zorder=20)
    ax.plot([sbx, sbx], [sby-0.06, sby+0.06], 'k-', lw=1.5, zorder=20)
    ax.plot([sbx+1, sbx+1], [sby-0.06, sby+0.06], 'k-', lw=1.5, zorder=20)
    ax.text(sbx+0.5, sby+0.15, '1 m', ha='center', fontsize=9, color='#333', fontweight='bold', zorder=20)
    
    ax.set_xlim(rxmin, rxmax); ax.set_ylim(rymin, rymax)
    ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout()
    out = '/tmp/v90_boundary_postfix.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))

if __name__ == '__main__':
    main()

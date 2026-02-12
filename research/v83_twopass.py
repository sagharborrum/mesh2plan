#!/usr/bin/env python3
"""
mesh2plan v83 - Two-pass wall detection

Pass 1: Height-filtered (upper band -0.8 to -0.3m) for clean main walls (no furniture)
Pass 2: Full-height wall faces, but only KEEP segments that have density in BOTH
        upper (-0.8 to -0.3m) AND lower (-1.5 to -0.8m) bands.
        Walls span floor-to-ceiling → present in both bands.
        Furniture only in lower band → rejected.

Merge pass 1 + pass 2 walls, then normal pipeline.
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
from shapely.geometry import Polygon as ShapelyPoly, MultiPolygon
from shapely.ops import unary_union
from pathlib import Path
import shutil

RESOLUTION = 0.012
WALL_THICK_M = 0.12

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
    """Build 2D density map from selected faces."""
    areas = mesh.area_faces[face_mask]
    centers = mesh.triangles_center[face_mask][:, [0, 2]].copy()
    centers[:, 0] = -centers[:, 0]  # mirror X
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((centers[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((centers[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    return density

def hough_walls(density, wall_angles, res, xmin, zmin, angle_tol=15):
    """Detect wall segments via Hough on density map."""
    thresh = np.percentile(density[density > 0], 55) if (density > 0).any() else 0
    binary = (density > thresh).astype(np.uint8) * 255
    edges = cv2.Canny(binary, 50, 150)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10,
                            minLineLength=int(0.2 / res), maxLineGap=int(0.08 / res))
    if lines is None:
        return []
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
    """Cluster Hough segments into wall lines."""
    cluster_dist = int(0.20 / res)
    all_walls = []
    
    for angle_idx, wa in enumerate(wall_angles):
        lines = wall_lines[angle_idx]
        if not lines:
            continue
        
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
            
            if span * res < min_span:
                continue
            if total_length < int(0.5 / res):
                continue
            
            rad = np.radians(wa)
            dx, dy = np.cos(rad), np.sin(rad)
            nx, ny = -np.sin(rad), np.cos(rad)
            
            sx = perp_avg * nx + para_min * dx
            sy = perp_avg * ny + para_min * dy
            ex = perp_avg * nx + para_max * dx
            ey = perp_avg * ny + para_max * dy
            
            # Sample density coverage
            n_samples = 50
            xs = np.linspace(sx, ex, n_samples).astype(int)
            ys = np.linspace(sy, ey, n_samples).astype(int)
            h_d, w_d = density.shape
            valid = (xs >= 0) & (xs < w_d) & (ys >= 0) & (ys < h_d)
            if valid.sum() == 0:
                continue
            vals = density[ys[valid], xs[valid]]
            d_thresh = np.percentile(density[density > 0], 30) if (density > 0).any() else 0
            coverage = (vals > d_thresh).mean()
            if coverage < min_coverage:
                continue
            
            all_walls.append({
                'angle': wa, 'perp': perp_avg,
                'start_px': (int(sx), int(sy)),
                'end_px': (int(ex), int(ey)),
                'span_m': span * res,
                'coverage': coverage,
                'start_m': (sx*res, sy*res),  # relative to xmin, zmin
            })
    
    return all_walls

def walls_to_m(walls, xmin, zmin, res):
    """Convert pixel walls to meter coordinates."""
    result = []
    for w in walls:
        sx, sy = w['start_px']
        ex, ey = w['end_px']
        result.append({
            **w,
            'start_m': (sx*res+xmin, sy*res+zmin),
            'end_m': (ex*res+xmin, ey*res+zmin),
        })
    return result

def snap_contour_to_angles(contour_pts, wall_angles, min_seg_len=0.3):
    n = len(contour_pts)
    snapped = []
    for i in range(n):
        p1 = contour_pts[i]
        p2 = contour_pts[(i+1) % n]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        seg_len = np.sqrt(dx*dx + dy*dy)
        if seg_len < min_seg_len * 0.3:
            continue
        seg_angle = np.degrees(np.arctan2(dy, dx)) % 180
        best_wa = wall_angles[0]
        best_diff = 999
        for wa in wall_angles:
            diff = min(abs(seg_angle - wa), 180 - abs(seg_angle - wa))
            if diff < best_diff:
                best_diff = diff
                best_wa = wa
        if best_diff < 25:
            rad = np.radians(best_wa)
            d = np.array([np.cos(rad), np.sin(rad)])
            n_vec = np.array([-np.sin(rad), np.cos(rad)])
            proj1 = np.dot(p1, d)
            proj2 = np.dot(p2, d)
            perp = (np.dot(p1, n_vec) + np.dot(p2, n_vec)) / 2
            new_p1 = proj1 * d + perp * n_vec
            new_p2 = proj2 * d + perp * n_vec
            snapped.append((new_p1, new_p2))
        else:
            snapped.append((p1, p2))
    if not snapped:
        return contour_pts
    result = []
    for i in range(len(snapped)):
        s1, e1 = snapped[i]
        s2, e2 = snapped[(i+1) % len(snapped)]
        d1 = e1 - s1
        d2 = e2 - s2
        det = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(det) < 1e-10:
            result.append(e1)
        else:
            t = ((s2[0]-s1[0])*d2[1] - (s2[1]-s1[1])*d2[0]) / det
            pt = s1 + t * d1
            result.append(pt)
    return np.array(result)

def check_dual_band_coverage(wall, density_upper, density_lower, res):
    """Check if a wall segment has density in BOTH upper and lower bands.
    True walls span full height; furniture only in lower band."""
    sx, sy = wall['start_px']
    ex, ey = wall['end_px']
    n_samples = 50
    xs = np.linspace(sx, ex, n_samples).astype(int)
    ys = np.linspace(sy, ey, n_samples).astype(int)
    h, w = density_upper.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() == 0:
        return False, 0, 0
    
    upper_thresh = np.percentile(density_upper[density_upper > 0], 20) if (density_upper > 0).any() else 0
    lower_thresh = np.percentile(density_lower[density_lower > 0], 20) if (density_lower > 0).any() else 0
    
    upper_vals = density_upper[ys[valid], xs[valid]]
    lower_vals = density_lower[ys[valid], xs[valid]]
    
    upper_cov = (upper_vals > upper_thresh).mean()
    lower_cov = (lower_vals > lower_thresh).mean()
    
    return (upper_cov > 0.15 and lower_cov > 0.15), upper_cov, lower_cov


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
    
    # ============================================================
    # PASS 1: Upper band only (-0.8 to -0.3m) — clean, no furniture
    # ============================================================
    upper_mask = wall_mask & (heights > -0.8) & (heights < -0.3)
    density_upper = make_density(mesh, upper_mask, xmin, zmin, w, h, res)
    print(f"Pass 1 (upper band): {upper_mask.sum()} faces")
    
    hough_upper = hough_walls(density_upper, wall_angles, res, xmin, zmin)
    walls_upper = cluster_walls(hough_upper, wall_angles, density_upper, res,
                                min_span=0.8, min_coverage=0.20)
    walls_upper = walls_to_m(walls_upper, xmin, zmin, res)
    print(f"  Pass 1 walls: {len(walls_upper)}")
    
    # ============================================================
    # PASS 2: Lower band (-1.5 to -0.8m) — has furniture AND walls
    # Only keep segments that ALSO have upper-band density (= real walls)
    # ============================================================
    lower_mask = wall_mask & (heights > -1.5) & (heights < -0.8)
    density_lower = make_density(mesh, lower_mask, xmin, zmin, w, h, res)
    print(f"Pass 2 (lower band): {lower_mask.sum()} faces")
    
    # Also try full-height density for short interior walls
    all_wall_mask = wall_mask  # no height filter
    density_all = make_density(mesh, all_wall_mask, xmin, zmin, w, h, res)
    
    hough_lower = hough_walls(density_all, wall_angles, res, xmin, zmin)
    walls_lower_raw = cluster_walls(hough_lower, wall_angles, density_all, res,
                                     min_span=0.4, min_coverage=0.15)
    
    # Filter: only keep lower-band walls that also appear in upper band
    walls_pass2 = []
    for wall in walls_lower_raw:
        is_dual, ucov, lcov = check_dual_band_coverage(wall, density_upper, density_lower, res)
        if is_dual:
            wall_m = walls_to_m([wall], xmin, zmin, res)[0]
            walls_pass2.append(wall_m)
            print(f"    ✓ dual-band wall: span={wall['span_m']:.2f}m, upper={ucov:.2f}, lower={lcov:.2f}")
        else:
            print(f"    ✗ furniture: span={wall['span_m']:.2f}m, upper={ucov:.2f}, lower={lcov:.2f}")
    
    print(f"  Pass 2 validated walls: {len(walls_pass2)}")
    
    # ============================================================
    # Merge: deduplicate walls from both passes
    # ============================================================
    all_walls = list(walls_upper)
    
    for w2 in walls_pass2:
        # Check if this wall is already in pass 1 (similar perp distance and angle)
        duplicate = False
        for w1 in walls_upper:
            if w1['angle'] == w2['angle']:
                if abs(w1['perp'] - w2['perp']) < int(0.25 / res):
                    duplicate = True
                    break
        if not duplicate:
            all_walls.append(w2)
            print(f"  + added pass2 wall: angle={w2['angle']:.0f}°, span={w2['span_m']:.2f}m")
    
    print(f"\nTotal merged walls (pre-filter): {len(all_walls)}")
    
    # ============================================================
    # Post-filter: remove short interior walls that don't connect
    # to other walls (likely window/door frames or artifacts)
    # ============================================================
    def wall_connects(w1, w2, tolerance=0.3):
        """Check if two walls' endpoints are close (connected)."""
        pts1 = [np.array(w1['start_m']), np.array(w1['end_m'])]
        pts2 = [np.array(w2['start_m']), np.array(w2['end_m'])]
        for p1 in pts1:
            for p2 in pts2:
                if np.linalg.norm(p1 - p2) * res < tolerance:
                    return True
        return False
    
    # Keep walls that are either long (>2m span) or connect to at least one other wall
    filtered_walls = []
    for i, wl in enumerate(all_walls):
        if wl['span_m'] > 2.0:
            filtered_walls.append(wl)
        else:
            connected = False
            for j, w2 in enumerate(all_walls):
                if i == j:
                    continue
                if wall_connects(wl, w2, tolerance=0.5):
                    connected = True
                    break
            if connected:
                filtered_walls.append(wl)
                print(f"  kept short wall: span={wl['span_m']:.2f}m (connected)")
            else:
                print(f"  removed orphan wall: span={wl['span_m']:.2f}m")
    
    all_walls = filtered_walls
    print(f"Total merged walls (post-filter): {len(all_walls)}")
    for i, wl in enumerate(all_walls):
        sm, em = wl['start_m'], wl['end_m']
        print(f"  Wall {i}: angle={wl['angle']}° span={wl['span_m']:.2f}m  ({sm[0]:.2f},{sm[1]:.2f}) → ({em[0]:.2f},{em[1]:.2f})")
    
    # ============================================================
    # Also build a combined density for visualization
    # ============================================================
    # Use upper-band density as primary (clean)
    # Add validated lower-band contribution
    density_combined = density_upper.copy()
    
    # ============================================================
    # Apartment boundary from all points
    # ============================================================
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((pts_xz[:, 0] - xmin) / res).astype(int), 0, w - 1)
    apy = np.clip(((pts_xz[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(all_density, (apy, apx), 1)
    all_density = cv2.GaussianBlur(all_density, (11, 11), 3.0)
    apt_thresh = np.percentile(all_density[all_density > 0], 10)
    apt_mask = (all_density > apt_thresh).astype(np.uint8) * 255
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    apt_mask = cv2.erode(apt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apt_contour = max(contours, key=cv2.contourArea)
    
    # ============================================================
    # Partition with walls → flood fill rooms
    # ============================================================
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
    
    rooms = []
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_m2 = region.sum() * res * res
        if area_m2 < 1.0:
            continue
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        contour_m = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx])
        room_snapped = snap_contour_to_angles(contour_m, wall_angles, min_seg_len=0.3)
        rooms.append({
            'area': area_m2,
            'contour_m': room_snapped,
            'n_verts': len(room_snapped),
            'centroid_m': (room_snapped[:,0].mean(), room_snapped[:,1].mean()),
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nDetected {len(rooms)} rooms (before merge):")
    total = 0
    for i, r in enumerate(rooms):
        c = r['contour_m']
        w_m = c[:,0].max()-c[:,0].min()
        h_m = c[:,1].max()-c[:,1].min()
        print(f"  Room {i+1}: {w_m:.2f}×{h_m:.2f} = {r['area']:.1f}m² ({r['n_verts']}v)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # ============================================================
    # Room merging: if two adjacent rooms combine to bedroom size
    # and one is suspiciously small (3-6m²), merge them.
    # Adjacency = centroids within ~3m and combined area is 10-20m².
    # ============================================================
    merged = True
    while merged:
        merged = False
        for i in range(len(rooms)):
            if rooms[i] is None:
                continue
            for j in range(i+1, len(rooms)):
                if rooms[j] is None:
                    continue
                a_i, a_j = rooms[i]['area'], rooms[j]['area']
                combined = a_i + a_j
                # Only merge if one is small (3-6m²) and combined is bedroom-ish
                if not (3.0 < min(a_i, a_j) < 6.0 and 10.0 < combined < 20.0):
                    continue
                # Check adjacency via centroid distance
                ci = np.array(rooms[i]['centroid_m'])
                cj = np.array(rooms[j]['centroid_m'])
                dist = np.linalg.norm(ci - cj)
                if dist > 4.0:
                    continue
                # Check Shapely adjacency
                try:
                    pi = ShapelyPoly(rooms[i]['contour_m'])
                    pj = ShapelyPoly(rooms[j]['contour_m'])
                    if not pi.is_valid:
                        pi = pi.buffer(0)
                    if not pj.is_valid:
                        pj = pj.buffer(0)
                    # Buffer slightly to catch near-adjacent rooms
                    if not pi.buffer(0.15).intersects(pj.buffer(0.15)):
                        continue
                except:
                    continue
                
                # Merge!
                print(f"  Merging rooms {i+1} ({a_i:.1f}m²) + {j+1} ({a_j:.1f}m²) → {combined:.1f}m²")
                merged_poly = unary_union([pi.buffer(0.01), pj.buffer(0.01)])
                if isinstance(merged_poly, MultiPolygon):
                    merged_poly = max(merged_poly.geoms, key=lambda g: g.area)
                merged_pts = np.array(merged_poly.exterior.coords)[:-1]
                merged_snapped = snap_contour_to_angles(merged_pts, wall_angles, min_seg_len=0.3)
                rooms[i] = {
                    'area': combined,
                    'contour_m': merged_snapped,
                    'n_verts': len(merged_snapped),
                    'centroid_m': (merged_snapped[:,0].mean(), merged_snapped[:,1].mean()),
                }
                rooms[j] = None
                merged = True
                break
            if merged:
                break
    
    rooms = [r for r in rooms if r is not None]
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nAfter merge: {len(rooms)} rooms:")
    total = 0
    for i, r in enumerate(rooms):
        c = r['contour_m']
        w_m = c[:,0].max()-c[:,0].min()
        h_m = c[:,1].max()-c[:,1].min()
        print(f"  Room {i+1}: {w_m:.2f}×{h_m:.2f} = {r['area']:.1f}m² ({r['n_verts']}v)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # Name rooms
    names = []
    bed_n = 0
    hall_n = 0
    for r in rooms:
        a = r['area']
        if a > 10:
            bed_n += 1
            names.append(f'Bedroom {bed_n}')
        elif a > 3.5:
            hall_n += 1
            names.append('Hallway' if hall_n == 1 else 'Entry')
        elif a > 2:
            names.append('Bathroom')
        else:
            names.append('WC')
    
    # ============================================================
    # Apartment boundary from room union
    # ============================================================
    room_polys = []
    for r in rooms:
        if len(r['contour_m']) >= 3:
            try:
                p = ShapelyPoly(r['contour_m']).buffer(WALL_THICK_M / 2)
                if p.is_valid:
                    room_polys.append(p)
            except:
                pass
    
    apt_union = unary_union(room_polys).simplify(0.05)
    if isinstance(apt_union, MultiPolygon):
        apt_boundary = max(apt_union.geoms, key=lambda g: g.area)
    else:
        apt_boundary = apt_union
    
    apt_boundary_pts = np.array(apt_boundary.exterior.coords)
    apt_boundary_snapped = snap_contour_to_angles(apt_boundary_pts[:-1], wall_angles, min_seg_len=0.3)
    
    # ============================================================
    # Render
    # ============================================================
    rot = render_rotation
    
    rooms_rot = []
    for room in rooms:
        rc = rotate_points(room['contour_m'], rot, center_m)
        rooms_rot.append({'contour': rc, 'centroid': rc.mean(axis=0), 'area': room['area']})
    
    walls_rot = []
    for wall in all_walls:
        s = rotate_points(np.array([wall['start_m']]), rot, center_m)[0]
        e = rotate_points(np.array([wall['end_m']]), rot, center_m)[0]
        walls_rot.append((s, e))
    
    bnd_rot = rotate_points(apt_boundary_snapped, rot, center_m)
    
    all_rc = np.vstack([r['contour'] for r in rooms_rot])
    rxmin, rymin = all_rc.min(axis=0) - 1.2
    rxmax, rymax = all_rc.max(axis=0) + 1.2
    
    fig_w = 14
    fig_h = fig_w * (rymax-rymin) / (rxmax-rxmin)
    fig_h = max(fig_h, 8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_facecolor('#F5F5F5')
    
    # Grid
    for x in np.arange(int(rxmin)-1, int(rxmax)+2, 0.5):
        ax.axvline(x, color='#EEEEEE', lw=0.3, zorder=0)
    for y in np.arange(int(rymin)-1, int(rymax)+2, 0.5):
        ax.axhline(y, color='#EEEEEE', lw=0.3, zorder=0)
    
    data_range = rxmax - rxmin
    wt_pts = WALL_THICK_M / data_range * fig_w * 72
    wt_pts = max(6, min(wt_pts, 14))
    
    # Room fills
    bedroom_color = '#E8D5B7'
    for i, rr in enumerate(rooms_rot):
        if len(rr['contour']) >= 3:
            fc = bedroom_color if rooms[i]['area'] > 8 else '#FFFFFF'
            poly = MplPoly(rr['contour'], closed=True, fc=fc, ec='none', zorder=1, alpha=0.8)
            ax.add_patch(poly)
    
    # Outer walls (boundary)
    n_bnd = len(bnd_rot)
    for i in range(n_bnd):
        p1 = bnd_rot[i]
        p2 = bnd_rot[(i+1) % n_bnd]
        seg_len = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if seg_len < 0.05:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#2A2A2A', lw=wt_pts * 1.2,
                solid_capstyle='projecting', zorder=10)
    
    # Interior walls — only draw if they partition two different rooms
    for wi, wall in enumerate(all_walls):
        sx, sy = wall['start_px']
        ex, ey = wall['end_px']
        mx, my = (sx+ex)//2, (sy+ey)//2
        # Check room labels on both sides of wall midpoint (perpendicular offset)
        rad = np.radians(wall['angle'])
        nx_w, ny_w = -np.sin(rad), np.cos(rad)
        off_px = int(0.15 / res)
        side_a = (int(mx + nx_w * off_px), int(my + ny_w * off_px))
        side_b = (int(mx - nx_w * off_px), int(my - ny_w * off_px))
        label_a = labeled[np.clip(side_a[1], 0, h-1), np.clip(side_a[0], 0, w-1)]
        label_b = labeled[np.clip(side_b[1], 0, h-1), np.clip(side_b[0], 0, w-1)]
        
        if label_a == label_b and label_a > 0:
            print(f"  Skipping non-partition wall {wi}: same room ({label_a}) on both sides")
            continue
        
        s, e = walls_rot[wi]
        ax.plot([s[0], e[0]], [s[1], e[1]], color='#2A2A2A', lw=wt_pts * 0.7,
                solid_capstyle='butt', zorder=9)
    
    # Room labels
    for i, rr in enumerate(rooms_rot):
        cx, cy = rr['centroid']
        name = names[i] if i < len(names) else f'Room {i+1}'
        ax.text(cx, cy + 0.2, name, ha='center', va='center', fontsize=11,
                color='#444', fontweight='bold', zorder=20)
        ax.text(cx, cy - 0.15, f'({rooms[i]["area"]:.1f} m²)', ha='center', va='center',
                fontsize=9, color='#888', zorder=20)
    
    # Dimensions for big rooms
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 6:
            continue
        c = rr['contour']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        w_m = x1 - x0
        h_m = y1 - y0
        off = 0.3
        ax.annotate('', xy=(x1, y1+off), xytext=(x0, y1+off),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.12, f'{w_m:.2f} m', ha='center', fontsize=7, color='#666', zorder=15)
        ax.annotate('', xy=(x1+off, y1), xytext=(x1+off, y0),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+off+0.12, (y0+y1)/2, f'{h_m:.2f} m', ha='left', va='center', fontsize=7,
                color='#666', rotation=90, zorder=15)
    
    # Door arcs
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 2:
            continue
        c = rr['contour']
        cx, cy = rr['centroid']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        door_w = 0.75
        if rooms[i]['area'] > 8:
            arc = Arc((cx, y0), door_w, door_w, angle=0, theta1=0, theta2=90,
                      color='#666', lw=0.8, zorder=15)
            ax.add_patch(arc)
        elif rooms[i]['area'] > 2:
            arc = Arc((x0, cy), door_w, door_w, angle=0, theta1=0, theta2=90,
                      color='#666', lw=0.8, zorder=15)
            ax.add_patch(arc)
    
    # Scale bar
    sb_x = rxmin + 0.5
    sb_y = rymin + 0.3
    ax.plot([sb_x, sb_x+1], [sb_y, sb_y], 'k-', lw=2, zorder=20)
    ax.plot([sb_x, sb_x], [sb_y-0.05, sb_y+0.05], 'k-', lw=1.5, zorder=20)
    ax.plot([sb_x+1, sb_x+1], [sb_y-0.05, sb_y+0.05], 'k-', lw=1.5, zorder=20)
    ax.text(sb_x+0.5, sb_y+0.12, '1 m', ha='center', fontsize=8, color='#333', zorder=20)
    
    ax.set_xlim(rxmin, rxmax)
    ax.set_ylim(rymin, rymax)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    out = '/tmp/v83_twopass.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))
    
    # Also save a diagnostic with density maps
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    axes2[0].imshow(density_upper, cmap='hot', origin='lower')
    axes2[0].set_title('Upper band (-0.8 to -0.3m)')
    axes2[1].imshow(density_lower, cmap='hot', origin='lower')
    axes2[1].set_title('Lower band (-1.5 to -0.8m)')
    axes2[2].imshow(density_combined, cmap='hot', origin='lower')
    axes2[2].set_title('Combined (upper + validated)')
    for ax2 in axes2:
        ax2.axis('off')
    diag = '/tmp/v83_diagnostic.png'
    plt.savefig(diag, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic: {diag}")

if __name__ == '__main__':
    main()

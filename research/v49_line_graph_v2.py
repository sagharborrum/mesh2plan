#!/usr/bin/env python3
"""
mesh2plan v49 - Line Graph v2: Clean Two-Angle Partition

Based on v48 (which works!) with targeted fixes:
1. STRICT 2-angle filtering — only accept lines within ±12° of the two dominant angles
   (v48 had spurious 135° lines creating messy upper-left)
2. Merge threshold raised to 3m² (v48's 2m² left 3 tiny closets)
3. Better wall validation: require density evidence AND minimum length
4. Double-line wall rendering
5. Line extension only within mask (same as v48, which works)
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


def hough_and_filter(wall_density, mask):
    """Standard Hough, then STRICT angle filtering to exactly 2 perpendicular families."""
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

    # Find 2 dominant perpendicular angles via histogram
    thetas_deg = np.degrees(lines[:, 1]) % 180
    hist = np.zeros(180)
    for t in thetas_deg:
        hist[int(t) % 180] += 1
    kernel = np.ones(9) / 9
    ext = np.concatenate([hist[-4:], hist, hist[:4]])
    smooth = np.convolve(ext, kernel, mode='same')[4:-4]

    peaks = []
    for i in range(180):
        prev, nxt = (i - 1) % 180, (i + 1) % 180
        if smooth[i] > smooth[prev] and smooth[i] > smooth[nxt] and smooth[i] > 3:
            peaks.append((smooth[i], i))
    peaks.sort(reverse=True)

    # Take strongest, then best perpendicular partner
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

    # STRICT filter: only keep lines within ±12° of these two angles
    angle_tol = 12
    families = [[], []]
    for i, t in enumerate(thetas_deg):
        for fi, center in enumerate([angle1, angle2]):
            diff = abs(t - center)
            diff = min(diff, 180 - diff)
            if diff < angle_tol:
                families[fi].append(i)
                break
    
    kept = sum(len(f) for f in families)
    rejected = len(lines) - kept
    print(f"  Kept {kept} lines, rejected {rejected} off-angle lines")

    return lines, wall_img, families, [angle1, angle2]


def cluster_by_rho(lines, family_indices, min_gap_px=7):
    """Cluster lines within a family by rho distance."""
    if not family_indices:
        return []
    rhos = lines[family_indices, 0]
    thetas = lines[family_indices, 1]
    order = np.argsort(rhos)
    sorted_rhos = rhos[order]
    sorted_thetas = thetas[order]

    clusters = []
    cur_rhos = [sorted_rhos[0]]
    cur_thetas = [sorted_thetas[0]]
    cur_count = 1

    for i in range(1, len(order)):
        if sorted_rhos[i] - np.mean(cur_rhos) < min_gap_px:
            cur_rhos.append(sorted_rhos[i])
            cur_thetas.append(sorted_thetas[i])
            cur_count += 1
        else:
            clusters.append((np.mean(cur_rhos), np.mean(cur_thetas), cur_count))
            cur_rhos = [sorted_rhos[i]]
            cur_thetas = [sorted_thetas[i]]
            cur_count = 1
    clusters.append((np.mean(cur_rhos), np.mean(cur_thetas), cur_count))
    return clusters


def validate_wall_segment(rho, theta, wall_img, mask, min_run_px=25):
    """Check wall line against density image. Returns endpoints or None."""
    h, w = wall_img.shape
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x0, y0 = rho * cos_t, rho * sin_t
    dx, dy = -sin_t, cos_t

    t_range = max(w, h) * 2
    n_samples = int(t_range * 2)
    ts = np.linspace(-t_range, t_range, n_samples)
    xs = (x0 + ts * dx).astype(int)
    ys = (y0 + ts * dy).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    ts, xs, ys = ts[valid], xs[valid], ys[valid]
    if len(ts) < 10:
        return None

    in_mask = mask[ys, xs] > 0
    has_wall = wall_img[ys, xs] > 0
    for offset in range(1, 4):
        ox, oy = int(offset * cos_t), int(offset * sin_t)
        for sign in [-1, 1]:
            nx = np.clip(xs + sign * ox, 0, w - 1)
            ny = np.clip(ys + sign * oy, 0, h - 1)
            has_wall = has_wall | (wall_img[ny, nx] > 0)

    evidence = in_mask & has_wall
    in_mask_int = in_mask.astype(int)
    diff = np.diff(np.concatenate([[0], in_mask_int, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    best_start, best_end, best_evidence = None, None, 0
    for s, e in zip(starts, ends):
        seg_len = e - s
        wall_count = evidence[s:e].sum()
        if wall_count >= min_run_px * 0.3 and seg_len >= min_run_px:
            if wall_count > best_evidence:
                best_evidence = wall_count
                best_start, best_end = s, e

    if best_start is None:
        return None
    return (int(xs[best_start]), int(ys[best_start])), (int(xs[best_end-1]), int(ys[best_end-1]))


def build_room_cells(wall_segments, mask, line_thickness=3):
    h, w = mask.shape
    line_img = np.zeros((h, w), dtype=np.uint8)
    for seg in wall_segments:
        cv2.line(line_img, seg['pt1'], seg['pt2'], 255, line_thickness)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(line_img, contours, -1, 255, line_thickness)
    interior = ((line_img == 0) & (mask > 0)).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  Connected components: {n_labels - 1}")
    return labels, n_labels, line_img, interior


def extract_rooms(labels, n_labels, transform, min_area_m2=1.0):
    x_min, z_min, res, w, h = transform
    min_area_px = int(min_area_m2 / (res * res))
    rooms = []
    for lbl in range(1, n_labels):
        room_mask = (labels == lbl).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px < min_area_px:
            continue
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, 0.015 * peri, True)
        pts_px = simplified.reshape(-1, 2).astype(float)
        pts_world = np.zeros((len(pts_px), 2))
        pts_world[:, 0] = pts_px[:, 0] * res + x_min
        pts_world[:, 1] = pts_px[:, 1] * res + z_min
        rooms.append({
            'mask': room_mask,
            'polygon': pts_world,
            'area_m2': round(area_px * res * res, 1),
            'vertices': len(pts_world),
        })
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    return rooms


def merge_small_rooms(rooms, min_area_m2=3.0, resolution=0.02):
    """Merge rooms < threshold into adjacent larger rooms."""
    changed = True
    while changed:
        changed = False
        small = [r for r in rooms if r['area_m2'] < min_area_m2]
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
                best['area_m2'] = round(best['area_px'] * resolution * resolution, 1)
                rooms = [r for r in rooms if r is not sr]
                changed = True
                break
    return rooms


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


def classify_room(polygon, area):
    xs, zs = polygon[:, 0], polygon[:, 1]
    w, h = xs.max() - xs.min(), zs.max() - zs.min()
    aspect = max(w, h) / (min(w, h) + 0.01)
    if area < 3:
        return "closet"
    if area < 5:
        return "hallway" if aspect > 1.8 else "bathroom"
    if aspect > 2.5:
        return "hallway"
    return "room"


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


def render_debug(wall_density, mask, wall_img, wall_segments, line_img, interior,
                 labels, rooms, transform, angles, output_path):
    x_min, z_min, res, w, h = transform
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    axes[0, 0].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Wall-only density (log)')
    axes[0, 1].imshow(wall_img, cmap='gray', origin='lower')
    axes[0, 1].set_title('Wall mask (p80)')

    vis = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    for seg in wall_segments:
        c = (255, 0, 0) if seg['family'] == 0 else (0, 0, 255)
        cv2.line(vis, seg['pt1'], seg['pt2'], c, 2)
    axes[0, 2].imshow(vis[:, :, ::-1], origin='lower')
    axes[0, 2].set_title(f'Wall segments ({len(wall_segments)})')

    axes[0, 3].imshow(line_img, cmap='gray', origin='lower')
    axes[0, 3].set_title('Line arrangement')
    axes[1, 0].imshow(interior * 255, cmap='gray', origin='lower')
    axes[1, 0].set_title('Interior cells')

    room_colors = [(255,100,100),(100,255,100),(100,100,255),
                   (255,255,100),(255,100,255),(100,255,255),(200,150,100)]
    vis2 = np.zeros((h, w, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        vis2[room['mask'] > 0] = room_colors[i % len(room_colors)]
    axes[1, 1].imshow(vis2, origin='lower')
    axes[1, 1].set_title(f'Room cells ({len(rooms)})')

    axes[1, 2].imshow(np.log1p(wall_density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None: continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        c = np.array(room_colors[i % len(room_colors)]) / 255.0
        pp = np.vstack([np.column_stack([px, pz]), [px[0], pz[0]]])
        axes[1, 2].plot(pp[:, 0], pp[:, 1], '-', color=c, linewidth=2)
        axes[1, 2].fill(pp[:, 0], pp[:, 1], color=c, alpha=0.2)
    axes[1, 2].set_title('Polygons on density')

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

    plt.suptitle('v49 Line Graph v2 — Debug', fontsize=14)
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
    mask = get_apartment_mask(all_density)
    print(f"  Apartment area: {mask.sum() * res * res:.1f} m²")

    print("\nHough + angle filtering...")
    result = hough_and_filter(wall_density, mask)
    if result is None:
        print("  No lines!")
        return
    lines, wall_img, families, dominant_angles_deg = result

    print("\nCluster by rho...")
    all_clustered = []
    angle_rads = []
    for fi, fam in enumerate(families):
        if not fam:
            continue
        clustered = cluster_by_rho(lines, fam, min_gap_px=25)
        mean_theta = np.mean([c[1] for c in clustered])
        angle_rads.append(mean_theta % np.pi)
        print(f"  Family {fi} ({dominant_angles_deg[fi]}°): {len(clustered)} wall positions")
        for rho, theta, count in clustered:
            all_clustered.append({'rho': rho, 'theta': theta, 'count': count, 'family': fi})

    print(f"\nValidate walls ({len(all_clustered)} candidates)...")
    min_run_px = max(15, int(0.5 / res))
    wall_segments = []
    for cl in all_clustered:
        result = validate_wall_segment(cl['rho'], cl['theta'], wall_img, mask, min_run_px)
        if result:
            pt1, pt2 = result
            seg_len = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) * res
            wall_segments.append({
                'rho': cl['rho'], 'theta': cl['theta'],
                'pt1': pt1, 'pt2': pt2, 'length_m': round(seg_len, 2),
                'family': cl['family'], 'votes': cl['count']
            })
            print(f"    θ={math.degrees(cl['theta']):.1f}° len={seg_len:.1f}m votes={cl['count']}")

    print(f"  Validated: {len(wall_segments)}")

    # Keep only the strongest walls — target ~7-8 total
    # Sort by votes, keep top walls but ensure at least 2 per family
    wall_segments.sort(key=lambda s: s['votes'], reverse=True)
    max_walls = 8
    if len(wall_segments) > max_walls:
        # Ensure minimum per family
        kept = []
        fam_counts = {0: 0, 1: 0}
        for seg in wall_segments:
            if len(kept) >= max_walls:
                # Still add if family is underrepresented
                if fam_counts[seg['family']] < 2:
                    kept.append(seg)
                    fam_counts[seg['family']] += 1
                continue
            kept.append(seg)
            fam_counts[seg['family']] += 1
        wall_segments = kept
        print(f"  After top-{max_walls} filter: {len(wall_segments)}")

    # Extend to mask boundary (same as v48)
    extended = []
    for seg in wall_segments:
        rho, theta = seg['rho'], seg['theta']
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        dx, dy = -sin_t, cos_t
        ts = np.linspace(-max(w,h)*2, max(w,h)*2, 500)
        xs = (x0 + ts * dx).astype(int)
        ys = (y0 + ts * dy).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[valid], ys[valid]
        if len(xs) < 2:
            extended.append(seg)
            continue
        in_mask = mask[ys, xs] > 0
        if not in_mask.any():
            extended.append(seg)
            continue
        mi = np.where(in_mask)[0]
        ext = dict(seg)
        ext['pt1'] = (int(xs[mi[0]]), int(ys[mi[0]]))
        ext['pt2'] = (int(xs[mi[-1]]), int(ys[mi[-1]]))
        extended.append(ext)

    print(f"\nBuild room cells...")
    labels, n_labels, line_img, interior = build_room_cells(extended, mask, line_thickness=3)

    rooms = extract_rooms(labels, n_labels, transform, min_area_m2=1.0)
    print(f"  Raw rooms: {len(rooms)}")

    # First pass: merge tiny rooms
    rooms = merge_small_rooms(rooms, min_area_m2=2.5, resolution=res)
    print(f"  After merge (2.5m²): {len(rooms)}")
    
    # Second pass: if still too many rooms, keep merging smallest
    target_rooms = 6
    while len(rooms) > target_rooms:
        # Find smallest room
        smallest = min(rooms, key=lambda r: r['area_m2'])
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
    print(f"  After merge: {len(rooms)}")

    reextract_polygons(rooms, transform)

    # Snap to angles
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
        rtype = classify_room(poly, room['area_m2']) if poly is not None else 'unknown'
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

    render_floorplan(rooms, doors, transform, angle_rads,
                     out_dir / 'floorplan.png',
                     f'v49 Line Graph v2 — {mesh_path.stem}')
    render_debug(wall_density, mask, wall_img, wall_segments, line_img, interior,
                 labels, rooms, transform, angle_rads, out_dir / 'debug.png')

    total = sum(r.get('area_m2', 0) for r in rooms)
    results = {
        'approach': 'v49_line_graph_v2',
        'angles_deg': [round(math.degrees(a), 1) for a in angle_rads],
        'wall_segments': len(wall_segments),
        'rooms': [{'name': r.get('name'), 'area_m2': r['area_m2'],
                    'type': r['type'], 'vertices': r['vertices']} for r in rooms],
        'doors': len(doors), 'total_m2': round(total, 1)
    }
    with open(out_dir / 'floorplan.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== v49 Summary ===")
    print(f"  Angles: {[f'{math.degrees(a):.0f}°' for a in angle_rads]}")
    print(f"  Walls: {len(wall_segments)}")
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
    out_dir = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / 'results' / 'v49_line_graph_v2'
    result = process_mesh(args.mesh_path, out_dir, args.resolution)
    if result:
        shutil.copy2(result, Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')


if __name__ == '__main__':
    main()

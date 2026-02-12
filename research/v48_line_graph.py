#!/usr/bin/env python3
"""
mesh2plan v48 - Line Graph Room Extraction

KEY INSIGHT: Extract room polygons directly from Hough wall lines instead of
watershed segmentation. The wall-only density (from face normals) shows clear
wall lines. Standard Hough finds lines at two perpendicular angle families.
Cluster by rho → unique wall positions. Draw wall lines → connected components = rooms.

Pipeline:
1. Wall-only density from face normals (v41b)
2. Standard Hough transform → infinite lines (rho, theta)
3. Cluster by angle family, then by rho → unique wall positions
4. Validate lines against density → get actual wall segments
5. Draw wall lines on image → black connected components = room cells
6. Extract room polygons from cell contours
7. Classify rooms, render architectural floorplan
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
    """Auto-detect which axis is 'up' based on vertex spread and face normals."""
    v = np.array(mesh.vertices)
    spans = [v[:, i].max() - v[:, i].min() for i in range(3)]
    # The floor plane has the two largest spans; up axis has smallest or
    # is the axis where most face normals point (floor/ceiling faces)
    # Use normal distribution: up axis has strongest normal concentration
    n = np.abs(mesh.face_normals)
    mean_abs = n.mean(axis=0)
    # Up axis: many faces have normals pointing along it (floor/ceiling)
    # But also check span — up axis typically has smaller span for rooms
    # Combine: score = normal_concentration / span
    scores = mean_abs / (np.array(spans) + 0.01)
    up = int(np.argmax(scores))
    axes = 'XYZ'
    print(f"  Up axis: {axes[up]} (spans: X={spans[0]:.2f} Y={spans[1]:.2f} Z={spans[2]:.2f})")
    return up


def create_wall_density(mesh, resolution=0.02, normal_thresh=0.5, up_axis=1):
    """Create density image using ONLY wall-face centroids."""
    normals = mesh.face_normals
    centroids = mesh.triangles_center
    up_comp = np.abs(normals[:, up_axis])
    wall_mask = up_comp < normal_thresh
    wall_strength = 1.0 - up_comp
    wall_centroids = centroids[wall_mask]
    wall_weights = wall_strength[wall_mask]
    print(f"  Wall faces: {wall_mask.sum()} / {len(normals)} ({wall_mask.mean()*100:.1f}%)")

    # Floor plane axes (the two that aren't up)
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

    # All-vertex density for apartment mask
    all_x, all_z = mesh.vertices[:, ax0], mesh.vertices[:, ax1]
    all_density = np.zeros((h, w), dtype=np.float32)
    axi = np.clip(((all_x - x_min) / resolution).astype(int), 0, w - 1)
    azi = np.clip(((all_z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(all_density, (azi, axi), 1)

    print(f"  Image size: {w}x{h}, wall density range: {wall_density.min():.1f}-{wall_density.max():.1f}")
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


def hough_lines(wall_density, mask):
    """Standard Hough transform on thresholded wall density."""
    d = wall_density.copy()
    d[mask == 0] = 0
    masked_vals = d[mask > 0]
    pos_vals = masked_vals[masked_vals > 0]
    if len(pos_vals) == 0:
        return None
    p80 = np.percentile(pos_vals, 80)
    wall_img = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255

    lines = cv2.HoughLines(wall_img, 1, np.pi / 180, threshold=30)
    if lines is None:
        return None
    lines = lines.squeeze(axis=1)  # shape (N, 2) → rho, theta
    print(f"  Hough lines: {len(lines)}")
    return lines, wall_img


def group_angle_families(lines, angle_gap_deg=20):
    """Group lines into perpendicular angle families using histogram peaks."""
    thetas_deg = np.degrees(lines[:, 1]) % 180  # [0, 180)

    # Histogram with 1-degree bins
    n_bins = 180
    hist = np.zeros(n_bins)
    for t in thetas_deg:
        b = int(t) % n_bins
        hist[b] += 1

    # Smooth histogram (circular)
    kernel = np.ones(11) / 11
    ext = np.concatenate([hist[-5:], hist, hist[:5]])
    smooth = np.convolve(ext, kernel, mode='same')[5:-5]

    # Find peaks
    peaks = []
    for i in range(n_bins):
        prev, nxt = (i - 1) % n_bins, (i + 1) % n_bins
        if smooth[i] > smooth[prev] and smooth[i] > smooth[nxt] and smooth[i] > 2:
            peaks.append((smooth[i], i))
    peaks.sort(reverse=True)

    if len(peaks) == 0:
        # Fallback: one family
        return [list(range(len(lines)))]

    # Take top 2 peaks (should be ~90° apart)
    gap_rad = angle_gap_deg
    family_centers = []
    for weight, center in peaks:
        too_close = False
        for fc in family_centers:
            diff = abs(center - fc)
            diff = min(diff, 180 - diff)
            if diff < gap_rad:
                too_close = True
                break
        if not too_close:
            family_centers.append(center)
        if len(family_centers) >= 2:
            break

    # Assign each line to nearest family center
    families = [[] for _ in family_centers]
    for i, t in enumerate(thetas_deg):
        best_f, best_d = 0, 999
        for fi, fc in enumerate(family_centers):
            diff = abs(t - fc)
            diff = min(diff, 180 - diff)
            if diff < best_d:
                best_d = diff
                best_f = fi
        if best_d < gap_rad:
            families[best_f].append(i)

    # Sort by size
    families.sort(key=len, reverse=True)
    families = [f for f in families if len(f) > 0]

    for i, fam in enumerate(families):
        angles = thetas_deg[fam] if isinstance(fam, np.ndarray) else np.array([thetas_deg[j] for j in fam])
        mean_angle = np.mean(angles)
        print(f"  Family {i+1}: {len(fam)} lines, mean angle {mean_angle:.1f}°")

    return families


def cluster_by_rho(lines, family_indices, min_gap_m, resolution):
    """Cluster lines within an angle family by perpendicular distance (rho)."""
    min_gap_px = min_gap_m / resolution
    rhos = lines[family_indices, 0]
    thetas = lines[family_indices, 1]

    order = np.argsort(rhos)
    sorted_rhos = rhos[order]
    sorted_thetas = thetas[order]
    sorted_indices = np.array(family_indices)[order]

    clusters = []
    current_rhos = [sorted_rhos[0]]
    current_thetas = [sorted_thetas[0]]
    current_indices = [sorted_indices[0]]

    for i in range(1, len(order)):
        if sorted_rhos[i] - sorted_rhos[i-1] < min_gap_px:
            current_rhos.append(sorted_rhos[i])
            current_thetas.append(sorted_thetas[i])
            current_indices.append(sorted_indices[i])
        else:
            clusters.append((current_rhos, current_thetas, current_indices))
            current_rhos = [sorted_rhos[i]]
            current_thetas = [sorted_thetas[i]]
            current_indices = [sorted_indices[i]]
    clusters.append((current_rhos, current_thetas, current_indices))

    # Weighted average per cluster (weight = number of votes, approximated by count)
    result = []
    for c_rhos, c_thetas, c_idx in clusters:
        avg_rho = np.mean(c_rhos)
        avg_theta = np.mean(c_thetas)
        result.append((avg_rho, avg_theta, len(c_rhos)))

    return result


def validate_wall_segment(rho, theta, wall_img, mask, min_run_px=25):
    """
    Check wall line against density image. Find continuous segment with evidence.
    Returns (pt1, pt2) of the validated wall segment, or None.
    """
    h, w = wall_img.shape
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Sample points along the line
    # Line equation: x*cos(theta) + y*sin(theta) = rho
    # Parameterize along the line direction (-sin(theta), cos(theta))
    # Point on line closest to origin: (rho*cos_t, rho*sin_t)
    x0, y0 = rho * cos_t, rho * sin_t
    dx, dy = -sin_t, cos_t  # direction along line

    # Find extent of line within image
    # We need t range such that (x0+t*dx, y0+t*dy) is within [0,w)x[0,h)
    t_min, t_max = -max(w, h) * 2, max(w, h) * 2

    n_samples = int(t_max - t_min)
    if n_samples < 10:
        return None

    ts = np.linspace(t_min, t_max, n_samples)
    xs = (x0 + ts * dx).astype(int)
    ys = (y0 + ts * dy).astype(int)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    ts, xs, ys = ts[valid], xs[valid], ys[valid]
    if len(ts) < 10:
        return None

    # Check mask and wall density along line
    in_mask = mask[ys, xs] > 0
    has_wall = wall_img[ys, xs] > 0

    # Dilate the wall check slightly (check ±2 pixels perpendicular)
    for offset in range(1, 4):
        ox, oy = int(offset * cos_t), int(offset * sin_t)
        for sign in [-1, 1]:
            nx = np.clip(xs + sign * ox, 0, w - 1)
            ny = np.clip(ys + sign * oy, 0, h - 1)
            has_wall = has_wall | (wall_img[ny, nx] > 0)

    # Find runs of (in_mask AND has_wall)
    evidence = in_mask & has_wall
    # Also accept: in_mask alone (for gaps in walls like doors)
    # But require at least some wall evidence

    # Find the longest run where in_mask is true and has some wall evidence
    # Strategy: find segments within mask, keep those with enough wall pixels
    in_mask_int = in_mask.astype(int)
    diff = np.diff(np.concatenate([[0], in_mask_int, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    best_start, best_end = None, None
    best_evidence = 0

    for s, e in zip(starts, ends):
        seg_len = e - s
        wall_count = evidence[s:e].sum()
        wall_frac = wall_count / max(seg_len, 1)
        if wall_count >= min_run_px * 0.3 and seg_len >= min_run_px:
            if wall_count > best_evidence:
                best_evidence = wall_count
                best_start, best_end = s, e

    if best_start is None:
        return None

    # Return endpoints
    pt1 = (int(xs[best_start]), int(ys[best_start]))
    pt2 = (int(xs[best_end - 1]), int(ys[best_end - 1]))
    return pt1, pt2


def find_intersections(segments, mask, margin=10):
    """Find intersections of perpendicular wall segments."""
    if len(segments) < 2:
        return []

    h, w = mask.shape
    intersections = []

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg_i = segments[i]
            seg_j = segments[j]
            # Check if they're from different families (roughly perpendicular)
            angle_diff = abs(seg_i['theta'] - seg_j['theta'])
            if angle_diff < math.radians(30) or angle_diff > math.radians(150):
                continue  # Same family, skip

            # Intersection of two lines: rho1 = x*cos(t1)+y*sin(t1), rho2 = x*cos(t2)+y*sin(t2)
            ct1, st1 = math.cos(seg_i['theta']), math.sin(seg_i['theta'])
            ct2, st2 = math.cos(seg_j['theta']), math.sin(seg_j['theta'])
            det = ct1 * st2 - ct2 * st1
            if abs(det) < 1e-10:
                continue
            ix = (seg_i['rho'] * st2 - seg_j['rho'] * st1) / det
            iy = (ct1 * seg_j['rho'] - ct2 * seg_i['rho']) / det
            ix, iy = int(round(ix)), int(round(iy))

            # Check within image and near mask
            if 0 <= ix < w and 0 <= iy < h:
                # Check if near mask (within margin)
                x_lo = max(0, ix - margin)
                x_hi = min(w, ix + margin)
                y_lo = max(0, iy - margin)
                y_hi = min(h, iy + margin)
                if mask[y_lo:y_hi, x_lo:x_hi].any():
                    intersections.append((ix, iy))

    print(f"  Intersections: {len(intersections)}")
    return intersections


def build_room_cells(wall_segments, mask, transform, line_thickness=3):
    """
    Draw wall lines on image, find connected components = room cells.
    """
    h, w = mask.shape
    line_img = np.zeros((h, w), dtype=np.uint8)

    for seg in wall_segments:
        pt1, pt2 = seg['pt1'], seg['pt2']
        cv2.line(line_img, pt1, pt2, 255, line_thickness)

    # Also draw apartment boundary as wall
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(line_img, contours, -1, 255, line_thickness)

    # Room interiors = black connected components inside mask
    # Invert: rooms are where there are NO wall lines AND inside mask
    interior = ((line_img == 0) & (mask > 0)).astype(np.uint8)

    # Connected components
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  Connected components (raw): {n_labels - 1}")

    return labels, n_labels, line_img, interior


def extract_room_polygons(labels, n_labels, transform, min_area_m2=2.0):
    """Extract room polygons from connected component labels."""
    x_min, z_min, res, w, h = transform
    min_area_px = int(min_area_m2 / (res * res))

    rooms = []
    for lbl in range(1, n_labels):
        room_mask = (labels == lbl).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px < min_area_px:
            continue

        # Find contour
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        # Use approxPolyDP with tight epsilon to keep rectilinear shape
        simplified = cv2.approxPolyDP(contour, 0.015 * peri, True)
        pts_px = simplified.reshape(-1, 2).astype(float)

        # Convert to world coordinates
        pts_world = np.zeros((len(pts_px), 2))
        pts_world[:, 0] = pts_px[:, 0] * res + x_min
        pts_world[:, 1] = pts_px[:, 1] * res + z_min

        area_m2 = area_px * res * res
        rooms.append({
            'mask': room_mask,
            'polygon': pts_world,
            'polygon_px': pts_px,
            'area_m2': round(area_m2, 1),
            'area_px': area_px,
            'vertices': len(pts_world),
        })

    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    return rooms


def merge_small_rooms(rooms, min_area_m2=2.0):
    """Merge small cells into adjacent larger rooms."""
    changed = True
    while changed:
        changed = False
        small = [r for r in rooms if r['area_m2'] < min_area_m2]
        if not small:
            break
        for sr in small:
            dilated = cv2.dilate(sr['mask'], np.ones((5, 5), np.uint8))
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
                best['area_m2'] = round(best['area_px'] * 0.02 * 0.02, 1)
                rooms.remove(sr)
                changed = True
                break
    return rooms


def snap_polygon_to_angles(pts_world, angles, angle_thresh_deg=15):
    """Snap polygon edges to dominant wall angles."""
    if len(pts_world) < 3:
        return pts_world
    angle_thresh = math.radians(angle_thresh_deg)
    n = len(pts_world)
    snapped = pts_world.copy().astype(float)

    for iteration in range(6):
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

    # Remove near-duplicate vertices
    cleaned = [snapped[0]]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - cleaned[-1]) > 0.03:
            cleaned.append(snapped[i])
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) < 0.03:
        cleaned = cleaned[:-1]
    return np.array(cleaned) if len(cleaned) >= 3 else snapped


def intersect_consecutive_edges(pts):
    """Recompute vertices as intersections of consecutive edges."""
    if len(pts) < 3:
        return pts
    n = len(pts)
    edges = []
    for i in range(n):
        j = (i+1) % n
        dx, dy = pts[j][0]-pts[i][0], pts[j][1]-pts[i][1]
        edges.append((pts[i], dx, dy))
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
        ix = p1[0] + t*dx1
        iy = p1[1] + t*dy1
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
        return "hallway" if aspect > 2.0 else "bathroom"
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


def render_floorplan(rooms, doors, transform, angles, output_path, title):
    x_min, z_min, res, w, h = transform
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    colors = ['#F5F5F5', '#EFEFEF', '#E8E8E8', '#F0F0F0', '#EBEBEB',
              '#E3E3E3', '#F2F2F2', '#EDEDED']

    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:, 0], pc[:, 1], color=colors[i % len(colors)])
        ax.plot(pc[:, 0], pc[:, 1], 'k-', linewidth=3.0)
        cx, cz = poly.mean(axis=0)
        name = room.get('name', '?')
        area = room.get('area_m2', 0)
        ax.text(cx, cz, f"{name}\n{area:.1f}m²",
                ha='center', va='center', fontsize=10, fontweight='bold')

    # Door arcs
    for door in doors:
        cx, cy = door['pos_px']
        wx = cx * res + x_min
        wz = cy * res + z_min
        arc = plt.Circle((wx, wz), 0.4, fill=False, color='brown',
                         linewidth=1.5, linestyle='--')
        ax.add_patch(arc)

    angle_strs = [f"{math.degrees(a):.0f}°" for a in angles]
    ax.text(0.02, 0.98, f"Wall angles: {', '.join(angle_strs)}",
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(False)
    # Scale bar
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.2]*2, 'k-', linewidth=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.05, '1m', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug(wall_density, all_density, mask, wall_img, wall_segments,
                 line_img, interior, labels, rooms, transform, angles, output_path):
    x_min, z_min, res, w, h = transform
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # 1. Wall-only density
    axes[0, 0].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Wall-only density (log)')

    # 2. Thresholded wall mask
    axes[0, 1].imshow(wall_img, cmap='gray', origin='lower')
    axes[0, 1].set_title('Wall mask (p80)')

    # 3. Hough lines (all)
    hough_vis = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    axes[0, 2].imshow(hough_vis, origin='lower')
    axes[0, 2].set_title(f'Clustered wall lines ({len(wall_segments)})')
    colors_list = [(255, 0, 0), (0, 0, 255), (0, 200, 0), (255, 165, 0)]
    for i, seg in enumerate(wall_segments):
        c = colors_list[i % len(colors_list)]
        cv2.line(hough_vis, seg['pt1'], seg['pt2'], c, 2)
    axes[0, 2].imshow(hough_vis[:, :, ::-1], origin='lower')

    # 4. Line arrangement
    axes[0, 3].imshow(line_img, cmap='gray', origin='lower')
    axes[0, 3].set_title('Line arrangement (walls)')

    # 5. Interior mask
    axes[1, 0].imshow(interior * 255, cmap='gray', origin='lower')
    axes[1, 0].set_title('Interior (no walls)')

    # 6. Connected components
    room_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                   (255, 255, 100), (255, 100, 255), (100, 255, 255),
                   (200, 150, 100), (150, 100, 200)]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        vis[room['mask'] > 0] = room_colors[i % len(room_colors)]
    axes[1, 1].imshow(vis, origin='lower')
    axes[1, 1].set_title(f'Room cells ({len(rooms)})')

    # 7. Polygons on wall density
    axes[1, 2].imshow(np.log1p(wall_density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None:
            continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        c = np.array(room_colors[i % len(room_colors)]) / 255.0
        pp = np.vstack([np.column_stack([px, pz]), [px[0], pz[0]]])
        axes[1, 2].plot(pp[:, 0], pp[:, 1], '-', color=c, linewidth=2)
        axes[1, 2].fill(pp[:, 0], pp[:, 1], color=c, alpha=0.2)
    axes[1, 2].set_title('Room polygons')

    # 8. Final floorplan polygons
    axes[1, 3].set_facecolor('white')
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None:
            continue
        pc = np.vstack([poly, poly[0]])
        axes[1, 3].fill(pc[:, 0], pc[:, 1], color='#F0F0F0')
        axes[1, 3].plot(pc[:, 0], pc[:, 1], 'k-', linewidth=2)
        cx, cz = poly.mean(axis=0)
        axes[1, 3].text(cx, cz, room.get('name', '?'), ha='center', va='center', fontsize=7)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].set_title('Final floorplan')

    plt.suptitle('v48 Line Graph — Debug', fontsize=14)
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

    print("\nStep 0: Detect up axis...")
    up_axis = detect_up_axis(mesh)

    print("\nStep 1: Wall-only density...")
    wall_density, all_density, transform = create_wall_density(mesh, res, up_axis=up_axis)
    x_min, z_min, _, w, h = transform

    print("\nStep 2: Apartment mask...")
    mask = get_apartment_mask(all_density)
    apt_area = mask.sum() * res * res
    print(f"  Apartment area: {apt_area:.1f} m²")

    print("\nStep 3: Standard Hough transform...")
    result = hough_lines(wall_density, mask)
    if result is None:
        print("  No lines found!")
        return
    lines, wall_img = result

    print("\nStep 4: Group into angle families...")
    families = group_angle_families(lines)

    print("\nStep 5: Cluster by rho within each family...")
    all_clustered = []
    angles = []
    for fi, fam in enumerate(families):
        clustered = cluster_by_rho(lines, fam, min_gap_m=0.15, resolution=res)
        mean_theta = np.mean([c[1] for c in clustered])
        angles.append(mean_theta % np.pi)
        print(f"  Family {fi+1}: {len(clustered)} unique wall positions")
        for rho, theta, count in clustered:
            all_clustered.append({'rho': rho, 'theta': theta, 'count': count, 'family': fi})

    print(f"\nStep 6: Validate wall lines ({len(all_clustered)} candidates)...")
    wall_segments = []
    # Adapt min_run based on apartment size
    min_run_px = max(15, int(0.5 / res))
    for cl in all_clustered:
        result = validate_wall_segment(cl['rho'], cl['theta'], wall_img, mask,
                                       min_run_px=min_run_px)
        if result is not None:
            pt1, pt2 = result
            seg_len = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) * res
            wall_segments.append({
                'rho': cl['rho'], 'theta': cl['theta'],
                'pt1': pt1, 'pt2': pt2,
                'length_m': round(seg_len, 2),
                'family': cl['family'],
                'votes': cl['count']
            })
            print(f"    Wall: rho={cl['rho']:.0f} θ={math.degrees(cl['theta']):.1f}° len={seg_len:.1f}m ({cl['count']} votes)")

    print(f"  Validated wall segments: {len(wall_segments)}")

    if len(wall_segments) < 2:
        print("  Too few wall segments, trying with lower threshold...")
        min_run_px = max(8, int(0.3 / res))
        wall_segments = []
        for cl in all_clustered:
            result = validate_wall_segment(cl['rho'], cl['theta'], wall_img, mask,
                                           min_run_px=min_run_px)
            if result is not None:
                pt1, pt2 = result
                seg_len = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) * res
                wall_segments.append({
                    'rho': cl['rho'], 'theta': cl['theta'],
                    'pt1': pt1, 'pt2': pt2,
                    'length_m': round(seg_len, 2),
                    'family': cl['family'],
                    'votes': cl['count']
                })
        print(f"  Validated wall segments (relaxed): {len(wall_segments)}")

    print("\nStep 7: Build room cells from line arrangement...")
    # Extend wall segments to fill their full extent within mask
    # For better room separation, extend segments to apartment boundary
    extended_segments = []
    for seg in wall_segments:
        rho, theta = seg['rho'], seg['theta']
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        dx, dy = -sin_t, cos_t

        # Extend to image bounds
        t_vals = []
        if abs(dx) > 1e-10:
            t_vals.extend([(0 - x0) / dx, (w - 1 - x0) / dx])
        if abs(dy) > 1e-10:
            t_vals.extend([(0 - y0) / dy, (h - 1 - y0) / dy])
        if not t_vals:
            extended_segments.append(seg)
            continue

        t_min_val = min(t_vals)
        t_max_val = max(t_vals)

        # Find the range within mask
        n_samples = 500
        ts = np.linspace(t_min_val, t_max_val, n_samples)
        xs = (x0 + ts * dx).astype(int)
        ys = (y0 + ts * dy).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        ts, xs, ys = ts[valid], xs[valid], ys[valid]
        if len(ts) < 2:
            extended_segments.append(seg)
            continue

        in_mask = mask[ys, xs] > 0
        if not in_mask.any():
            extended_segments.append(seg)
            continue

        # Find first and last in-mask positions
        mask_indices = np.where(in_mask)[0]
        first, last = mask_indices[0], mask_indices[-1]
        ext_seg = dict(seg)
        ext_seg['pt1'] = (int(xs[first]), int(ys[first]))
        ext_seg['pt2'] = (int(xs[last]), int(ys[last]))
        extended_segments.append(ext_seg)

    labels, n_labels, line_img, interior = build_room_cells(
        extended_segments, mask, transform, line_thickness=3)

    print("\nStep 8: Extract room polygons...")
    rooms = extract_room_polygons(labels, n_labels, transform, min_area_m2=1.0)
    print(f"  Raw rooms: {len(rooms)}")

    rooms = merge_small_rooms(rooms, min_area_m2=2.0)
    print(f"  After merge: {len(rooms)}")

    # Re-extract polygons after merge (contours may have changed)
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

    # Snap polygons to dominant angles
    print("\nStep 9: Snap polygons to wall angles...")
    for room in rooms:
        poly = room.get('polygon')
        if poly is not None and len(poly) >= 3:
            poly = snap_polygon_to_angles(poly, angles)
            poly = intersect_consecutive_edges(poly)
            poly = remove_short_edges(poly, min_length=0.15)
            poly = remove_collinear(poly)
            room['polygon'] = poly
            room['vertices'] = len(poly)

    # Classify and name
    rc, hc, bc, cc = 1, 1, 1, 1
    for room in rooms:
        area = room.get('area_m2', 0)
        poly = room.get('polygon')
        if poly is not None:
            rtype = classify_room(poly, area)
        else:
            rtype = 'unknown'
        room['type'] = rtype
        if rtype == 'hallway':
            room['name'] = "Hallway" if hc == 1 else f"Hallway {hc}"
            hc += 1
        elif rtype == 'bathroom':
            room['name'] = "Bathroom" if bc == 1 else f"Bathroom {bc}"
            bc += 1
        elif rtype == 'closet':
            room['name'] = "Closet" if cc == 1 else f"Closet {cc}"
            cc += 1
        else:
            room['name'] = f"Room {rc}"
            rc += 1

    print("\nStep 10: Doors...")
    doors = detect_doors(rooms)
    print(f"  {len(doors)} doors")

    print("\nStep 11: Rendering...")
    render_floorplan(rooms, doors, transform, angles,
                     out_dir / 'floorplan.png',
                     f'v48 Line Graph — {mesh_path.stem}')
    render_debug(wall_density, all_density, mask, wall_img, wall_segments,
                 line_img, interior, labels, rooms, transform, angles,
                 out_dir / 'debug.png')

    # JSON output
    total_area = sum(r.get('area_m2', 0) for r in rooms)
    results = {
        'approach': 'v48_line_graph',
        'dominant_angles_deg': [round(math.degrees(a), 1) for a in angles],
        'wall_segments': len(wall_segments),
        'rooms': [{
            'name': r.get('name', '?'),
            'area_m2': r.get('area_m2', 0),
            'type': r.get('type', '?'),
            'vertices': r.get('vertices', 0),
            'polygon': r['polygon'].tolist() if r.get('polygon') is not None else None
        } for r in rooms],
        'doors': len(doors),
        'total_area_m2': round(total_area, 1)
    }
    with open(out_dir / 'floorplan.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== v48 Summary ===")
    print(f"  Wall angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    print(f"  Wall segments: {len(wall_segments)}")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")

    return out_dir / 'floorplan.png'


def main():
    parser = argparse.ArgumentParser(description='v48 Line Graph Room Extraction')
    parser.add_argument('mesh_path')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', '-r', type=float, default=0.02)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v48_line_graph'

    result = process_mesh(args.mesh_path, out_dir, args.resolution)

    if result:
        shutil.copy2(result, Path.home() / '.openclaw' / 'workspace' / 'v48_floorplan.png')
        print(f"  Copied to workspace")


if __name__ == '__main__':
    main()

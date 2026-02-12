#!/usr/bin/env python3
"""
mesh2plan v40 - Hough Angle Snap

Combines v39's density-ridge watershed (best room finding) with Hough-based
angle detection for polygon simplification.

Key insight from v39: apartment walls aren't axis-aligned (~30° rotated).
Rectilinear snap (forcing H/V) fights the real wall angles → diagonal artifacts.

New approach:
1. v39 watershed → room masks (proven to find 5 correct rooms)
2. Hough line detection on density image → find dominant wall angles
3. Per-room contour → snap edges to DETECTED angles (not H/V)
4. Merge near-parallel edges within angle clusters

This should produce clean architectural polygons that follow real wall angles.
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
from collections import Counter


def mesh_to_density(mesh, resolution=0.02):
    verts = np.array(mesh.vertices)
    x, z = verts[:, 0], verts[:, 2]
    pad = 0.3
    x_min, x_max = x.min() - pad, x.max() + pad
    z_min, z_max = z.min() - pad, z.max() + pad
    w = int((x_max - x_min) / resolution) + 1
    h = int((z_max - z_min) / resolution) + 1
    density = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(density, (zi, xi), 1)
    return density, (x_min, z_min, resolution)


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


def find_seeds_adaptive(density, mask, min_dist=25, n_target=6):
    d = density.copy()
    d[mask == 0] = 0
    d_smooth = cv2.GaussianBlur(d, (11, 11), 3)
    masked_vals = d_smooth[mask > 0]
    median = np.median(masked_vals[masked_vals > 0]) if np.any(masked_vals > 0) else 1
    floor_mask = ((d_smooth < median) & (mask > 0)).astype(np.uint8)
    dist = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
    boundary_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    combined = dist * 0.7 + boundary_dist * 0.3
    combined[mask == 0] = 0
    for md in [min_dist, min_dist - 5, min_dist - 10, 10, 5]:
        if md < 5:
            md = 5
        ks = md * 2 + 1
        local_max = ndimage.maximum_filter(combined, size=ks)
        peaks = (combined == local_max) & (combined > md)
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
    print(f"  Found {len(seeds)} seeds")
    return seeds, dist, combined


def watershed_density(seeds, density, mask):
    d = density.copy()
    d[mask == 0] = 0
    d_smooth = cv2.GaussianBlur(d, (7, 7), 2)
    d_norm = np.zeros_like(d_smooth)
    d_max = d_smooth[mask > 0].max() if np.any(mask > 0) else 1
    d_norm[mask > 0] = (d_smooth[mask > 0] / d_max * 255)
    d_uint8 = d_norm.astype(np.uint8)
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1
    for i, (sx, sy, _) in enumerate(seeds):
        cv2.circle(markers, (sx, sy), 3, i + 2, -1)
    grad_color = cv2.cvtColor(d_uint8, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(grad_color, markers.copy())
    return ws


def extract_rooms(ws, mask, seeds, res=0.02, min_room_m2=2.5):
    min_px = int(min_room_m2 / (res * res))
    rooms = []
    for i in range(len(seeds)):
        lbl = i + 2
        room_mask = ((ws == lbl) & (mask > 0)).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px >= min_px:
            rooms.append({'mask': room_mask, 'area_px': area_px})
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    return rooms


def merge_small(rooms, min_area_px=1500):
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
    return rooms


# ========== NEW: Hough angle detection ==========

def detect_dominant_angles(density, mask, n_angles=4):
    """
    Detect dominant wall angles from Hough lines on the density image.
    Returns sorted list of dominant angles in radians [0, pi).
    """
    d = density.copy()
    d[mask == 0] = 0
    
    # High density = walls
    masked_vals = d[mask > 0]
    p80 = np.percentile(masked_vals[masked_vals > 0], 80)
    wall_mask = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255
    
    # Hough lines on wall mask
    lines = cv2.HoughLinesP(wall_mask, 1, np.pi / 180, threshold=30,
                            minLineLength=20, maxLineGap=10)
    if lines is None:
        print("  No Hough lines found, defaulting to H/V")
        return [0, np.pi / 2]
    
    # Collect angles weighted by line length
    angle_weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = math.atan2(y2 - y1, x2 - x1) % np.pi  # [0, pi)
        angle_weights.append((angle, length))
    
    # Bin angles into 5-degree bins
    n_bins = 36
    bins = np.zeros(n_bins)
    for angle, weight in angle_weights:
        bin_idx = int(angle / np.pi * n_bins) % n_bins
        bins[bin_idx] += weight
    
    # Smooth bins (circular)
    bins_ext = np.concatenate([bins[-2:], bins, bins[:2]])
    bins_smooth = np.convolve(bins_ext, [0.15, 0.25, 0.2, 0.25, 0.15], mode='same')[2:-2]
    
    # Find peaks
    peaks = []
    for i in range(n_bins):
        prev = (i - 1) % n_bins
        nxt = (i + 1) % n_bins
        if bins_smooth[i] > bins_smooth[prev] and bins_smooth[i] > bins_smooth[nxt]:
            peaks.append((bins_smooth[i], i))
    
    peaks.sort(reverse=True)
    
    # Take top angles, merge near-perpendicular pairs
    dominant = []
    for weight, bin_idx in peaks[:n_angles * 2]:
        angle = (bin_idx + 0.5) / n_bins * np.pi
        # Check if too close to existing
        too_close = False
        for existing in dominant:
            diff = abs(angle - existing) % np.pi
            if diff < np.pi / 18:  # 10 degrees
                too_close = True
                break
        if not too_close:
            dominant.append(angle)
        if len(dominant) >= n_angles:
            break
    
    if not dominant:
        dominant = [0, np.pi / 2]
    
    dominant.sort()
    print(f"  Dominant wall angles: {[f'{math.degrees(a):.1f}°' for a in dominant]}")
    print(f"  Hough lines found: {len(lines)}")
    return dominant


def snap_to_angles(pts, angles, angle_thresh_deg=15):
    """
    Snap polygon edges to the nearest detected wall angle.
    For each edge, find closest dominant angle, then adjust endpoints
    to make the edge exactly that angle.
    """
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
            length = math.sqrt(dx ** 2 + dy ** 2)
            if length < 0.05:
                continue
            
            edge_angle = math.atan2(dy, dx) % np.pi
            
            # Find closest dominant angle
            best_angle = None
            best_diff = float('inf')
            for a in angles:
                # Check both a and a+pi (same line direction)
                for candidate in [a, (a + np.pi) % (2 * np.pi)]:
                    diff = abs(edge_angle - candidate % np.pi)
                    if diff > np.pi / 2:
                        diff = np.pi - diff
                    if diff < best_diff:
                        best_diff = diff
                        best_angle = a
            
            if best_diff > angle_thresh:
                continue
            
            # Snap: rotate edge to target angle, keeping midpoint fixed
            mid = (snapped[i] + snapped[j]) / 2
            target = best_angle
            # Direction: keep the sign of the original direction
            orig_dir = math.atan2(dy, dx)
            if abs(orig_dir - target) > np.pi / 2 and abs(orig_dir - target) < 3 * np.pi / 2:
                target = target + np.pi
            
            half_len = length / 2
            new[i, 0] = mid[0] - half_len * math.cos(target)
            new[i, 1] = mid[1] - half_len * math.sin(target)
            new[j, 0] = mid[0] + half_len * math.cos(target)
            new[j, 1] = mid[1] + half_len * math.sin(target)
        
        snapped = new
    
    # Clean up near-duplicate vertices
    cleaned = [snapped[0]]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - cleaned[-1]) > 0.03:
            cleaned.append(snapped[i])
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) < 0.03:
        cleaned = cleaned[:-1]
    
    return np.array(cleaned) if len(cleaned) >= 3 else snapped


def intersect_consecutive_edges(pts, angles, angle_thresh_deg=15):
    """
    After snapping edges to dominant angles, recompute vertices as
    intersection points of consecutive edges. This gives clean corners.
    """
    if len(pts) < 3:
        return pts
    
    n = len(pts)
    # For each edge, compute its line equation
    edges = []
    for i in range(n):
        j = (i + 1) % n
        p1, p2 = pts[i], pts[j]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        edges.append((p1, p2, dx, dy))
    
    # Intersect consecutive edges
    new_pts = []
    for i in range(n):
        j = (i + 1) % n
        p1, p2, dx1, dy1 = edges[i]
        p3, p4, dx2, dy2 = edges[j]
        
        # Line intersection
        det = dx1 * dy2 - dy1 * dx2
        if abs(det) < 1e-10:
            # Parallel edges, keep original vertex
            new_pts.append(pts[j].copy())
            continue
        
        t = ((p3[0] - p1[0]) * dy2 - (p3[1] - p1[1]) * dx2) / det
        ix = p1[0] + t * dx1
        iy = p1[1] + t * dy1
        
        # Sanity check: intersection shouldn't be too far from original vertex
        orig = pts[j]
        if np.linalg.norm([ix - orig[0], iy - orig[1]]) > 1.0:
            new_pts.append(orig.copy())
        else:
            new_pts.append(np.array([ix, iy]))
    
    return np.array(new_pts)


def remove_collinear(pts, thresh=0.03):
    if len(pts) < 4:
        return pts
    cleaned = []
    n = len(pts)
    for i in range(n):
        v1 = pts[i] - pts[(i - 1) % n]
        v2 = pts[(i + 1) % n] - pts[i]
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 > 0 and l2 > 0:
            sin_angle = cross / (l1 * l2)
            if sin_angle > 0.05:  # ~3 degrees
                cleaned.append(pts[i])
        else:
            cleaned.append(pts[i])
    return np.array(cleaned) if len(cleaned) >= 3 else pts


def remove_short_edges(pts, min_length=0.15):
    """Remove vertices that create very short edges (noise)."""
    if len(pts) < 4:
        return pts
    changed = True
    while changed and len(pts) >= 4:
        changed = False
        n = len(pts)
        lengths = []
        for i in range(n):
            j = (i + 1) % n
            lengths.append(np.linalg.norm(pts[j] - pts[i]))
        shortest_idx = np.argmin(lengths)
        if lengths[shortest_idx] < min_length:
            # Remove the vertex that changes the polygon least
            # (keep the one that's more "on the line" of its neighbors)
            i = shortest_idx
            j = (i + 1) % n
            # Remove vertex j (merge into i)
            pts = np.delete(pts, j, axis=0)
            changed = True
    return pts


def extract_polygon_hough(room_mask, transform, angles, epsilon_factor=0.01):
    """Extract polygon and snap to Hough-detected angles."""
    x_min, z_min, res = transform
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    
    # Use tighter epsilon for more vertices (we'll simplify via angle snapping)
    simplified = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
    
    pts_px = simplified.reshape(-1, 2).astype(float)
    pts = np.zeros((len(pts_px), 2))
    pts[:, 0] = pts_px[:, 0] * res + x_min
    pts[:, 1] = pts_px[:, 1] * res + z_min
    
    # Snap edges to dominant angles
    pts = snap_to_angles(pts, angles, angle_thresh_deg=15)
    
    # Recompute vertices as edge intersections for clean corners
    pts = intersect_consecutive_edges(pts, angles)
    
    # Clean up
    pts = remove_short_edges(pts, min_length=0.15)
    pts = remove_collinear(pts)
    
    return pts


def polygon_area(pts):
    n = len(pts)
    if n < 3:
        return 0
    a = 0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(a) / 2


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
        for j in range(i + 1, len(rooms)):
            d1 = cv2.dilate(rooms[i]['mask'], np.ones((9, 9), np.uint8))
            d2 = cv2.dilate(rooms[j]['mask'], np.ones((9, 9), np.uint8))
            overlap = d1 & d2
            if overlap.sum() > 15:
                ys, xs = np.where(overlap > 0)
                doors.append({'rooms': (i, j), 'pos_px': (xs.mean(), ys.mean())})
    return doors


def render_floorplan(rooms, doors, transform, angles, output_path, title="v40"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    colors = ['#E8E8E8', '#F0F0F0', '#E0E0E0', '#F5F5F5', '#EBEBEB',
              '#E3E3E3', '#F2F2F2', '#EDEDED']
    x_min, z_min, res = transform

    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:, 0], pc[:, 1], color=colors[i % len(colors)], alpha=0.5)
        ax.plot(pc[:, 0], pc[:, 1], 'k-', linewidth=2.5)
        cx, cz = poly[:, 0].mean(), poly[:, 1].mean()
        name = room.get('name', f'Room {i+1}')
        area = room.get('area_m2', 0)
        nv = room.get('vertices', 0)
        ax.text(cx, cz, f"{name}\n{area:.1f}m²\n({nv}v)", ha='center', va='center',
                fontsize=9, fontweight='bold')

    for door in doors:
        cx, cy = door['pos_px']
        ax.plot(cx * res + x_min, cy * res + z_min, 's', color='brown', markersize=8, zorder=5)

    # Show dominant angles in legend
    angle_strs = [f"{math.degrees(a):.0f}°" for a in angles]
    ax.text(0.02, 0.98, f"Wall angles: {', '.join(angle_strs)}", transform=ax.transAxes,
            fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot([xlim[0]+0.5, xlim[0]+1.5], [ylim[0]+0.3]*2, 'k-', linewidth=3)
    ax.text(xlim[0]+1.0, ylim[0]+0.15, '1m', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug(density, mask, seeds, seed_score, rooms, ws, angles, transform, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    x_min, z_min, res = transform

    axes[0, 0].imshow(np.log1p(density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Log Density')

    # Wall mask + Hough lines
    d = density.copy()
    d[mask == 0] = 0
    masked_vals = d[mask > 0]
    p80 = np.percentile(masked_vals[masked_vals > 0], 80)
    wall_mask = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255
    axes[0, 1].imshow(wall_mask, cmap='gray', origin='lower')
    lines = cv2.HoughLinesP(wall_mask, 1, np.pi / 180, threshold=30,
                            minLineLength=20, maxLineGap=10)
    if lines is not None:
        for line in lines[:100]:
            x1, y1, x2, y2 = line[0]
            axes[0, 1].plot([x1, x2], [y1, y2], 'r-', linewidth=0.5, alpha=0.5)
    axes[0, 1].set_title(f'Wall Mask + Hough ({len(lines) if lines is not None else 0} lines)')

    # Angle histogram
    if lines is not None:
        line_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = math.atan2(y2 - y1, x2 - x1) % np.pi
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            line_angles.extend([math.degrees(a)] * int(length))
        axes[0, 2].hist(line_angles, bins=72, range=(0, 180), color='steelblue', alpha=0.7)
        for a in angles:
            axes[0, 2].axvline(math.degrees(a), color='red', linewidth=2, linestyle='--')
        axes[0, 2].set_title(f'Angle Histogram (peaks: {[f"{math.degrees(a):.0f}°" for a in angles]})')
        axes[0, 2].set_xlabel('Angle (degrees)')

    # Watershed
    ws_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    ws_vis[ws == -1] = [255, 255, 255]
    axes[1, 0].imshow(np.log1p(density), cmap='gray', origin='lower', alpha=0.7)
    axes[1, 0].imshow(ws_vis, origin='lower', alpha=0.5)
    axes[1, 0].set_title('Watershed Boundaries')

    room_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                   (255, 255, 100), (255, 100, 255), (100, 255, 255),
                   (200, 150, 100), (150, 100, 200)]
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        vis[room['mask'] > 0] = room_colors[i % len(room_colors)]
    axes[1, 1].imshow(vis, origin='lower')
    axes[1, 1].set_title(f'Room Masks ({len(rooms)} rooms)')

    axes[1, 2].imshow(np.log1p(density), cmap='gray', origin='lower', alpha=0.5)
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
    axes[1, 2].set_title('Angle-Snapped Polygons')

    plt.suptitle('v40 Hough Angle Snap Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', '-r', type=float, default=0.02)
    parser.add_argument('--min-seed-dist', type=int, default=25)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v40_hough_angle_snap'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    res = args.resolution

    print("Step 1: Density image...")
    density, transform = mesh_to_density(mesh, res)

    print("Step 2: Apartment mask...")
    mask = get_apartment_mask(density)
    print(f"  Area: {mask.sum() * res * res:.1f} m²")

    print("Step 3: Detecting dominant wall angles...")
    angles = detect_dominant_angles(density, mask)

    print("Step 4: Finding room seeds...")
    seeds, floor_dist, seed_score = find_seeds_adaptive(density, mask, min_dist=args.min_seed_dist)

    print("Step 5: Watershed on density...")
    ws = watershed_density(seeds, density, mask)

    print("Step 6: Extracting rooms...")
    rooms = extract_rooms(ws, mask, seeds, res)
    print(f"  Raw rooms: {len(rooms)}")

    min_merge_px = int(3.0 / (res * res))
    rooms = merge_small(rooms, min_merge_px)
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  After merge: {len(rooms)}")

    print("Step 7: Polygon extraction with angle snapping...")
    for i, room in enumerate(rooms):
        poly = extract_polygon_hough(room['mask'], transform, angles)
        if poly is not None:
            area = polygon_area(poly)
            rtype = classify_room(poly, area)
            room['polygon'] = poly
            room['area_m2'] = round(area, 1)
            room['type'] = rtype
            room['vertices'] = len(poly)
            print(f"  Room {i+1}: {area:.1f}m², {len(poly)}v, type={rtype}")
        else:
            room['polygon'] = None
            room['area_m2'] = 0
            room['type'] = 'unknown'
            room['vertices'] = 0

    rooms_valid = sorted([r for r in rooms if r.get('polygon') is not None],
                         key=lambda r: r['area_m2'], reverse=True)
    rc, hc, bc, cc = 1, 1, 1, 1
    for room in rooms_valid:
        t = room['type']
        if t == 'hallway':
            room['name'] = "Hallway" if hc == 1 else f"Hallway {hc}"
            hc += 1
        elif t == 'bathroom':
            room['name'] = "Bathroom" if bc == 1 else f"Bathroom {bc}"
            bc += 1
        elif t == 'closet':
            room['name'] = "Closet" if cc == 1 else f"Closet {cc}"
            cc += 1
        else:
            room['name'] = f"Room {rc}"
            rc += 1

    print("Step 8: Doors...")
    doors = detect_doors(rooms_valid)
    print(f"  {len(doors)} doors")

    print("Step 9: Rendering...")
    mesh_name = mesh_path.stem
    render_floorplan(rooms_valid, doors, transform, angles,
                     out_dir / f"v40_{mesh_name}_plan.png",
                     f"v40 Hough Angle Snap — {mesh_name}")
    render_debug(density, mask, seeds, seed_score, rooms_valid, ws, angles, transform,
                 out_dir / f"v40_{mesh_name}_debug.png")

    # Copy main plan to workspace
    import shutil
    shutil.copy2(out_dir / f"v40_{mesh_name}_plan.png",
                 Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')

    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v40_hough_angle_snap',
        'dominant_angles_deg': [round(math.degrees(a), 1) for a in angles],
        'rooms': [{
            'name': r.get('name', '?'),
            'area_m2': r.get('area_m2', 0),
            'type': r.get('type', '?'),
            'vertices': r.get('vertices', 0),
            'polygon': r['polygon'].tolist() if r.get('polygon') is not None else None
        } for r in rooms_valid],
        'doors': len(doors),
        'total_area_m2': round(total_area, 1)
    }
    with open(out_dir / f"v40_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Dominant angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

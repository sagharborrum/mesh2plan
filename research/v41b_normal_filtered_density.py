#!/usr/bin/env python3
"""
mesh2plan v41b - Normal-Filtered Density

HYBRID approach combining v41's face-normal insight with v39/v40's proven pipeline.

Problem with v27-v40: density image includes ALL vertices (floor, ceiling, furniture).
Wall detection relies on "walls = high density" heuristic, which is noisy.

Problem with v41: pure vector approach too noisy for LiDAR mesh (261K segments → 9 walls).

New idea: Use face normals to CREATE a wall-only density image:
1. Classify faces by normal: wall (horizontal normal), floor/ceiling (vertical normal)
2. Project ONLY wall-face centroids to XZ → "wall density" image
3. This gives a much cleaner signal than vertex density
4. Apply v39 watershed for room finding on clean wall density
5. Apply v40 Hough angle snap for polygon extraction

Expected benefit: cleaner wall boundaries → better watershed → more accurate room polygons.
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
from collections import defaultdict
import shutil


def create_wall_density(mesh, resolution=0.02, normal_thresh=0.5):
    """
    Create density image using ONLY wall-face centroids.
    Wall faces = faces with near-horizontal normals (abs(normal.y) < thresh).
    """
    normals = mesh.face_normals
    centroids = mesh.triangles_center  # (N, 3)
    
    # Wall faces: horizontal normal
    y_comp = np.abs(normals[:, 1])
    wall_mask = y_comp < normal_thresh
    
    # Also compute "wall strength" = how horizontal the normal is
    # 0 = perfectly vertical face (floor), 1 = perfectly horizontal normal (wall)
    wall_strength = 1.0 - y_comp
    
    wall_centroids = centroids[wall_mask]
    wall_weights = wall_strength[wall_mask]
    
    print(f"  Wall faces: {wall_mask.sum()} / {len(normals)} ({wall_mask.mean()*100:.1f}%)")
    
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
    
    # Also create regular density for apartment mask
    all_x, all_z = mesh.vertices[:, 0], mesh.vertices[:, 2]
    all_density = np.zeros((h, w), dtype=np.float32)
    axi = np.clip(((all_x - x_min) / resolution).astype(int), 0, w - 1)
    azi = np.clip(((all_z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(all_density, (azi, axi), 1)
    
    print(f"  Wall density range: {wall_density.min():.1f} - {wall_density.max():.1f}")
    print(f"  Image size: {w}x{h}")
    
    return wall_density, all_density, (x_min, z_min, resolution)


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
    """
    Find room seeds in LOW wall-density areas (room interiors have few walls).
    """
    d = wall_density.copy()
    d[mask == 0] = 999  # high value outside apartment
    
    d_smooth = cv2.GaussianBlur(d, (11, 11), 3)
    
    # Invert: low wall density = high seed score
    masked_vals = d_smooth[mask > 0]
    d_max = masked_vals.max() if len(masked_vals) > 0 else 1
    inv = d_max - d_smooth
    inv[mask == 0] = 0
    
    # Also weight by distance from apartment boundary
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
    """
    Watershed using wall density as the gradient.
    Walls = ridges (high density) → watershed boundaries.
    """
    d = wall_density.copy()
    d[mask == 0] = 0
    d_smooth = cv2.GaussianBlur(d, (7, 7), 2)
    
    # Normalize to uint8
    d_max = d_smooth[mask > 0].max() if mask.any() else 1
    d_norm = np.zeros_like(d_smooth)
    d_norm[mask > 0] = d_smooth[mask > 0] / max(d_max, 1e-6) * 255
    d_uint8 = d_norm.astype(np.uint8)
    
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1  # background
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
            dx = snapped[j,0] - snapped[i,0]
            dy = snapped[j,1] - snapped[i,1]
            length = math.sqrt(dx**2 + dy**2)
            if length < 0.05:
                continue
            edge_angle = math.atan2(dy, dx) % np.pi
            
            best_angle, best_diff = None, float('inf')
            for a in angles:
                diff = abs(edge_angle - a) % np.pi
                if diff > np.pi/2: diff = np.pi - diff
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
            new[i,0] = mid[0] - half * math.cos(target)
            new[i,1] = mid[1] - half * math.sin(target)
            new[j,0] = mid[0] + half * math.cos(target)
            new[j,1] = mid[1] + half * math.sin(target)
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
    if n < 3: return 0
    a = sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))
    return abs(a) / 2


def classify_room(polygon, area):
    xs, zs = polygon[:, 0], polygon[:, 1]
    w, h = xs.max() - xs.min(), zs.max() - zs.min()
    aspect = max(w, h) / (min(w, h) + 0.01)
    if area < 3: return "closet"
    if area < 5: return "hallway" if aspect > 2.0 else "bathroom"
    if aspect > 2.5: return "hallway"
    return "room"


def detect_doors(rooms):
    doors = []
    for i in range(len(rooms)):
        for j in range(i+1, len(rooms)):
            d1 = cv2.dilate(rooms[i]['mask'], np.ones((9,9), np.uint8))
            d2 = cv2.dilate(rooms[j]['mask'], np.ones((9,9), np.uint8))
            overlap = d1 & d2
            if overlap.sum() > 15:
                ys, xs = np.where(overlap > 0)
                doors.append({'rooms': (i, j), 'pos_px': (xs.mean(), ys.mean())})
    return doors


def render_floorplan(rooms, doors, transform, angles, output_path, title):
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    colors = ['#E8E8E8', '#F0F0F0', '#E0E0E0', '#F5F5F5', '#EBEBEB',
              '#E3E3E3', '#F2F2F2', '#EDEDED']
    x_min, z_min, res = transform
    
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3: continue
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:,0], pc[:,1], color=colors[i%len(colors)], alpha=0.5)
        ax.plot(pc[:,0], pc[:,1], 'k-', linewidth=2.5)
        cx, cz = poly.mean(axis=0)
        ax.text(cx, cz, f"{room.get('name','?')}\n{room.get('area_m2',0):.1f}m²\n({room.get('vertices',0)}v)",
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    for door in doors:
        cx, cy = door['pos_px']
        ax.plot(cx*res+x_min, cy*res+z_min, 's', color='brown', markersize=8, zorder=5)
    
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


def render_debug(wall_density, all_density, mask, seeds, seed_score, rooms, ws, 
                 angles, transform, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].imshow(np.log1p(all_density), cmap='hot', origin='lower')
    axes[0,0].set_title('All-vertex density (log)')
    
    axes[0,1].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    axes[0,1].set_title('Wall-only density (log)')
    
    # Comparison: wall density is cleaner
    d = wall_density.copy()
    d[mask == 0] = 0
    masked = d[mask > 0]
    if len(masked[masked > 0]) > 0:
        p80 = np.percentile(masked[masked > 0], 80)
        wall_high = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255
    else:
        wall_high = np.zeros_like(mask, dtype=np.uint8)
    axes[0,2].imshow(wall_high, cmap='gray', origin='lower')
    axes[0,2].set_title('Wall mask (p80 of wall density)')
    
    # Seeds
    axes[1,0].imshow(seed_score, cmap='viridis', origin='lower')
    for sx, sy, _ in seeds:
        axes[1,0].plot(sx, sy, 'r*', markersize=10)
    axes[1,0].set_title(f'Seed score + {len(seeds)} seeds')
    
    # Watershed
    room_colors = [(255,100,100), (100,255,100), (100,100,255),
                   (255,255,100), (255,100,255), (100,255,255),
                   (200,150,100), (150,100,200)]
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        vis[room['mask'] > 0] = room_colors[i % len(room_colors)]
    axes[1,1].imshow(vis, origin='lower')
    axes[1,1].set_title(f'Room masks ({len(rooms)})')
    
    # Polygons on wall density
    x_min, z_min, res = transform
    axes[1,2].imshow(np.log1p(wall_density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None: continue
        px = (poly[:,0] - x_min) / res
        pz = (poly[:,1] - z_min) / res
        c = np.array(room_colors[i % len(room_colors)]) / 255.0
        pp = np.vstack([np.column_stack([px, pz]), [px[0], pz[0]]])
        axes[1,2].plot(pp[:,0], pp[:,1], '-', color=c, linewidth=2)
        axes[1,2].fill(pp[:,0], pp[:,1], color=c, alpha=0.2)
    axes[1,2].set_title('Polygons on wall density')
    
    plt.suptitle('v41b Normal-Filtered Density — Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', '-r', type=float, default=0.02)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v41b_normal_filtered_density'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    res = args.resolution

    print("\nStep 1: Create wall-only density image...")
    wall_density, all_density, transform = create_wall_density(mesh, res)

    print("\nStep 2: Apartment mask (from all-vertex density)...")
    mask = get_apartment_mask(all_density)
    print(f"  Apartment area: {mask.sum() * res * res:.1f} m²")

    print("\nStep 3: Dominant wall angles (from wall density)...")
    angles = detect_dominant_angles(wall_density, mask)

    print("\nStep 4: Room seeds (from inverse wall density)...")
    seeds, seed_score = find_seeds(wall_density, mask, min_dist=25, n_target=6)

    print("\nStep 5: Watershed on wall density...")
    ws = watershed_on_walls(seeds, wall_density, mask)

    print("\nStep 6: Extract rooms...")
    rooms = extract_rooms(ws, mask, seeds, res)
    print(f"  Raw rooms: {len(rooms)}")
    
    min_merge_px = int(3.0 / (res * res))
    rooms = merge_small(rooms, min_merge_px)
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  After merge: {len(rooms)}")

    print("\nStep 7: Polygon extraction with angle snap...")
    for i, room in enumerate(rooms):
        poly = extract_polygon(room['mask'], transform, angles)
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

    rooms_valid = sorted([r for r in rooms if r.get('polygon') is not None],
                         key=lambda r: r['area_m2'], reverse=True)
    
    rc, hc, bc, cc = 1, 1, 1, 1
    for room in rooms_valid:
        t = room.get('type', 'room')
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

    print("\nStep 8: Doors...")
    doors = detect_doors(rooms_valid)
    print(f"  {len(doors)} doors")

    print("\nStep 9: Rendering...")
    mesh_name = mesh_path.stem
    render_floorplan(rooms_valid, doors, transform, angles,
                     out_dir / f"v41b_{mesh_name}_plan.png",
                     f"v41b Normal-Filtered Density — {mesh_name}")
    render_debug(wall_density, all_density, mask, seeds, seed_score, rooms_valid, ws,
                 angles, transform, out_dir / f"v41b_{mesh_name}_debug.png")

    shutil.copy2(out_dir / f"v41b_{mesh_name}_plan.png",
                 Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')

    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v41b_normal_filtered_density',
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
    with open(out_dir / f"v41b_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== v41b Summary ===")
    print(f"  Dominant angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

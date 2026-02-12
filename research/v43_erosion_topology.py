#!/usr/bin/env python3
"""
mesh2plan v43 - Erosion Topology

FUNDAMENTALLY NEW approach: Use progressive erosion to detect narrow passages.

Key insight: When you erode a floor plan mask, narrow spaces (hallways, doorways)
disappear first. By tracking when connected components SPLIT during erosion,
we find the exact locations where hallways connect rooms.

Pipeline:
1. Normal-filtered wall density → apartment mask
2. Progressive erosion: for each erosion radius, count connected components
3. At each topology change (components increase), record the split location
4. Use split locations + erosion level to identify hallway/corridor regions
5. At the critical erosion level, connected components = individual rooms
6. Map rooms back to full-resolution mask using watershed from eroded seeds
7. Identify hallway as the region between room expansions
8. Per-room polygon extraction with Hough angle snap

Why this should find hallways:
- Hallways are NARROW — they erode away before rooms do
- The erosion level where a hallway disappears tells us its width
- Room seeds from heavy erosion are guaranteed to be "deep" room centers
- Expanding seeds back catches room geometry without absorbing hallways
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


def create_wall_density(mesh, resolution=0.02, normal_thresh=0.5):
    """Create wall-only density image using face normals."""
    normals = mesh.face_normals
    centroids = mesh.triangles_center
    y_comp = np.abs(normals[:, 1])
    wall_mask = y_comp < normal_thresh
    wall_weights = 1.0 - y_comp
    wall_centroids = centroids[wall_mask]
    ww = wall_weights[wall_mask]
    
    x, z = wall_centroids[:, 0], wall_centroids[:, 2]
    pad = 0.3
    x_min, x_max = x.min() - pad, x.max() + pad
    z_min, z_max = z.min() - pad, z.max() + pad
    w = int((x_max - x_min) / resolution) + 1
    h = int((z_max - z_min) / resolution) + 1
    
    wall_density = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(wall_density, (zi, xi), ww)
    
    all_x, all_z = mesh.vertices[:, 0], mesh.vertices[:, 2]
    all_density = np.zeros((h, w), dtype=np.float32)
    axi = np.clip(((all_x - x_min) / resolution).astype(int), 0, w - 1)
    azi = np.clip(((all_z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(all_density, (azi, axi), 1)
    
    print(f"  Wall faces: {wall_mask.sum()} / {len(normals)} ({wall_mask.mean()*100:.1f}%)")
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


def detect_dominant_angles(wall_density, mask, n_angles=4):
    d = wall_density.copy()
    d[mask == 0] = 0
    masked_vals = d[mask > 0]
    if len(masked_vals[masked_vals > 0]) == 0:
        return [0, np.pi / 2]
    p80 = np.percentile(masked_vals[masked_vals > 0], 80)
    wall_img = ((d >= p80) & (mask > 0)).astype(np.uint8) * 255
    
    lines = cv2.HoughLinesP(wall_img, 1, np.pi/180, threshold=30,
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


def progressive_erosion_analysis(mask, wall_mask, max_radius=60, step=2):
    """
    Progressively erode the interior (non-wall) mask and track topology changes.
    Returns list of (radius, n_components, component_labels) at each step.
    """
    # Interior = apartment mask minus wall pixels
    interior = ((mask > 0) & (wall_mask == 0)).astype(np.uint8)
    
    history = []
    prev_n = -1
    
    for r in range(0, max_radius + 1, step):
        if r == 0:
            eroded = interior.copy()
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            eroded = cv2.erode(interior, kernel)
        
        n_labels, labels = cv2.connectedComponents(eroded)
        n_comps = n_labels - 1  # exclude background
        
        # Filter tiny components (noise)
        valid_comps = 0
        for lbl in range(1, n_labels):
            if (labels == lbl).sum() > 50:  # at least 50px
                valid_comps += 1
        
        history.append({
            'radius': r,
            'n_components': valid_comps,
            'labels': labels,
            'eroded': eroded,
            'area': eroded.sum()
        })
        
        if valid_comps != prev_n:
            print(f"    r={r}: {valid_comps} components (area: {eroded.sum()}px)")
            prev_n = valid_comps
        
        # Stop when everything erodes away
        if eroded.sum() < 100:
            break
    
    return history


def find_optimal_erosion(history, target_rooms=5):
    """
    Find the erosion level that gives the best room segmentation.
    Strategy: find where component count is closest to target, preferring
    the LARGEST erosion radius (most separated rooms) at that count.
    """
    # Find all radii where component count matches or is close to target
    best = None
    best_diff = float('inf')
    
    for entry in history:
        n = entry['n_components']
        diff = abs(n - target_rooms)
        # Prefer: closest to target, then largest radius (more separation)
        if diff < best_diff or (diff == best_diff and entry['radius'] > best['radius']):
            best_diff = diff
            best = entry
    
    return best


def find_room_seeds_multiscale(history, mask, res=0.02, min_room_m2=2.5):
    """
    Extract room seeds at multiple erosion scales.
    Large rooms survive heavy erosion; small rooms (hallways) only survive light erosion.
    We want seeds from the heaviest erosion level where each room still exists.
    """
    min_px = int(min_room_m2 / (res * res))
    
    # Collect all unique room seeds across erosion levels
    # Start from heaviest erosion (most reliable seeds) and work backwards
    all_seeds = []
    claimed = np.zeros_like(mask, dtype=bool)
    
    # First pass: find large room seeds from heavy erosion
    for entry in reversed(history):
        labels = entry['labels']
        for lbl in range(1, labels.max() + 1):
            comp_mask = (labels == lbl).astype(np.uint8)
            area = comp_mask.sum()
            if area < 200:  # skip tiny fragments
                continue
            # Check if this seed overlaps with already-claimed seeds
            overlap = (comp_mask > 0) & claimed
            if overlap.sum() > area * 0.5:
                continue  # already covered by a deeper seed
            all_seeds.append({
                'mask': comp_mask,
                'area_px': area,
                'erosion_radius': entry['radius'],
                'centroid': ndimage.center_of_mass(comp_mask)
            })
            claimed |= (comp_mask > 0)
    
    # Deduplicate: merge seeds whose centroids are very close
    merged = []
    used = set()
    for i, s in enumerate(all_seeds):
        if i in used:
            continue
        group = [s]
        for j, s2 in enumerate(all_seeds[i+1:], i+1):
            if j in used:
                continue
            dist = math.sqrt((s['centroid'][0]-s2['centroid'][0])**2 + 
                           (s['centroid'][1]-s2['centroid'][1])**2)
            if dist < 30:  # ~0.6m
                group.append(s2)
                used.add(j)
        # Keep the one from highest erosion (most reliable)
        best = max(group, key=lambda g: g['erosion_radius'])
        merged.append(best)
        used.add(i)
    
    print(f"  Multi-scale seeds: {len(all_seeds)} raw → {len(merged)} merged")
    for s in merged:
        print(f"    seed: r={s['erosion_radius']}, area={s['area_px']}px, "
              f"centroid=({s['centroid'][1]:.0f},{s['centroid'][0]:.0f})")
    
    return merged


def watershed_from_seeds(seeds, mask, wall_density, res=0.02):
    """
    Expand room seeds into full rooms using watershed.
    Wall density is the gradient barrier.
    """
    # Create markers: 0=unknown, 1=background (outside mask), 2+=rooms
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[mask == 0] = 1  # background
    
    for i, seed in enumerate(seeds):
        markers[seed['mask'] > 0] = i + 2
    
    # Create gradient from wall density (walls = high gradient = barriers)
    d = wall_density.copy()
    d[mask == 0] = 0
    nz = d[d > 0]
    if len(nz) > 0:
        d = d / nz.max() * 255
    grad = d.astype(np.uint8)
    # Enhance wall signal
    grad = cv2.GaussianBlur(grad, (5, 5), 1)
    
    # Watershed needs 3-channel image
    grad_3c = cv2.merge([grad, grad, grad])
    cv2.watershed(grad_3c, markers)
    
    # Extract rooms
    rooms = []
    for i, seed in enumerate(seeds):
        room_mask = (markers == i + 2).astype(np.uint8)
        # Restrict to apartment mask
        room_mask &= mask
        area_px = room_mask.sum()
        rooms.append({
            'mask': room_mask,
            'area_px': area_px,
            'area_m2': round(area_px * res * res, 1),
            'erosion_radius': seed['erosion_radius']
        })
    
    return rooms


def find_hallway_regions(rooms, mask, res=0.02, min_hallway_m2=1.5):
    """
    After watershed expansion, any unclaimed apartment area between rooms
    is likely hallway/corridor space.
    """
    claimed = np.zeros_like(mask, dtype=bool)
    for room in rooms:
        claimed |= (room['mask'] > 0)
    
    unclaimed = ((mask > 0) & ~claimed).astype(np.uint8)
    if unclaimed.sum() == 0:
        return rooms
    
    min_px = int(min_hallway_m2 / (res * res))
    n_labels, labels = cv2.connectedComponents(unclaimed)
    
    hallway_rooms = []
    for lbl in range(1, n_labels):
        comp = (labels == lbl).astype(np.uint8)
        area = comp.sum()
        if area >= min_px:
            hallway_rooms.append({
                'mask': comp,
                'area_px': area,
                'area_m2': round(area * res * res, 1),
                'erosion_radius': 0,
                'is_hallway': True
            })
    
    if hallway_rooms:
        print(f"  Found {len(hallway_rooms)} hallway region(s): "
              f"{[h['area_m2'] for h in hallway_rooms]}m²")
        rooms.extend(hallway_rooms)
    
    return rooms


def snap_to_angles(pts, angles, angle_thresh_deg=15):
    if len(pts) < 3:
        return pts
    angle_thresh = math.radians(angle_thresh_deg)
    n = len(pts)
    snapped = pts.copy().astype(float)
    for _ in range(8):
        new = snapped.copy()
        for i in range(n):
            j = (i + 1) % n
            dx = snapped[j,0] - snapped[i,0]
            dy = snapped[j,1] - snapped[i,1]
            length = math.sqrt(dx**2 + dy**2)
            if length < 0.05: continue
            edge_angle = math.atan2(dy, dx) % np.pi
            best_angle, best_diff = None, float('inf')
            for a in angles:
                diff = abs(edge_angle - a) % np.pi
                if diff > np.pi/2: diff = np.pi - diff
                if diff < best_diff:
                    best_diff = diff
                    best_angle = a
            if best_diff > angle_thresh: continue
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
    if len(pts) < 3: return pts
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
    if len(pts) < 4: return pts
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
    if len(pts) < 4: return pts
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
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, 
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
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


def classify_room(polygon, area, is_hallway=False):
    if is_hallway:
        return "hallway"
    xs, zs = polygon[:, 0], polygon[:, 1]
    w, h = xs.max() - xs.min(), zs.max() - zs.min()
    aspect = max(w, h) / (min(w, h) + 0.01)
    if area < 3: return "closet"
    if area < 5: return "hallway" if aspect > 1.8 else "bathroom"
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
    colors = ['#D4E6F1', '#D5F5E3', '#FADBD8', '#F9E79F', '#E8DAEF',
              '#D6DBDF', '#FDEBD0', '#D1F2EB']
    x_min, z_min, res = transform
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3: continue
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:,0], pc[:,1], color=colors[i%len(colors)], alpha=0.6)
        ax.plot(pc[:,0], pc[:,1], 'k-', linewidth=2.5)
        cx, cz = poly.mean(axis=0)
        label = f"{room.get('name','?')}\n{room.get('area_m2',0):.1f}m²\n({room.get('vertices',0)}v)"
        if room.get('erosion_radius') is not None:
            label += f"\n[er={room['erosion_radius']}]"
        ax.text(cx, cz, label, ha='center', va='center', fontsize=9, fontweight='bold')
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


def render_debug(wall_density, all_density, mask, wall_mask, history, 
                 seeds, rooms, transform, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Wall density
    axes[0,0].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    axes[0,0].set_title('Wall-only density (log)')
    
    # Wall mask used for erosion
    axes[0,1].imshow(wall_mask * 255, cmap='gray', origin='lower')
    axes[0,1].set_title('Wall mask (erosion barrier)')
    
    # Erosion topology chart
    radii = [h['radius'] for h in history]
    n_comps = [h['n_components'] for h in history]
    axes[0,2].plot(radii, n_comps, 'b-o', markersize=3)
    axes[0,2].set_xlabel('Erosion radius (px)')
    axes[0,2].set_ylabel('Connected components')
    axes[0,2].set_title('Topology changes during erosion')
    axes[0,2].grid(True, alpha=0.3)
    # Mark topology changes
    for i in range(1, len(n_comps)):
        if n_comps[i] != n_comps[i-1]:
            axes[0,2].axvline(radii[i], color='r', alpha=0.3, linestyle='--')
    
    # Seeds visualization
    seed_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    seed_vis[mask > 0] = [40, 40, 40]
    colors = [(255,100,100), (100,255,100), (100,100,255),
              (255,255,100), (255,100,255), (100,255,255),
              (200,150,100), (150,100,200)]
    for i, seed in enumerate(seeds):
        seed_vis[seed['mask'] > 0] = colors[i % len(colors)]
    axes[1,0].imshow(seed_vis, origin='lower')
    axes[1,0].set_title(f'Room seeds ({len(seeds)})')
    
    # Room masks after watershed
    room_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        room_vis[room['mask'] > 0] = colors[i % len(colors)]
    axes[1,1].imshow(room_vis, origin='lower')
    axes[1,1].set_title(f'Rooms after watershed ({len(rooms)})')
    
    # Polygons overlay
    x_min, z_min, res = transform
    axes[1,2].imshow(np.log1p(wall_density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None: continue
        px = (poly[:,0] - x_min) / res
        pz = (poly[:,1] - z_min) / res
        c = np.array(colors[i % len(colors)]) / 255.0
        pp = np.vstack([np.column_stack([px, pz]), [px[0], pz[0]]])
        axes[1,2].plot(pp[:,0], pp[:,1], '-', color=c, linewidth=2)
        axes[1,2].fill(pp[:,0], pp[:,1], color=c, alpha=0.2)
    axes[1,2].set_title('Polygons on wall density')
    
    plt.suptitle('v43 Erosion Topology — Debug', fontsize=14)
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
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v43_erosion_topology'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    res = args.resolution

    print("\nStep 1: Wall-only density image...")
    wall_density, all_density, transform = create_wall_density(mesh, res)

    print("\nStep 2: Apartment mask...")
    mask = get_apartment_mask(all_density)
    apt_area = mask.sum() * res * res
    print(f"  Apartment area: {apt_area:.1f} m²")

    print("\nStep 3: Dominant wall angles...")
    angles = detect_dominant_angles(wall_density, mask)

    print("\nStep 4: Wall mask for erosion...")
    # Use wall density threshold to create wall mask
    d = wall_density.copy()
    d[mask == 0] = 0
    nz = d[d > 0]
    if len(nz) > 0:
        # Use a moderate threshold — walls are high density
        p75 = np.percentile(nz, 75)
        wall_mask = ((d >= p75) & (mask > 0)).astype(np.uint8)
    else:
        wall_mask = np.zeros_like(mask)
    
    # Light morphological cleanup — connect nearby wall pixels along dominant angles
    for angle in angles:
        klen = 15  # ~0.3m
        k = np.zeros((klen, klen), dtype=np.uint8)
        cx, cy = klen // 2, klen // 2
        rad = angle
        for t in range(-klen//2, klen//2 + 1):
            x = int(cx + t * math.cos(rad))
            y = int(cy + t * math.sin(rad))
            if 0 <= x < klen and 0 <= y < klen:
                k[y, x] = 1
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k) | wall_mask
    
    wall_frac = wall_mask.sum() / max(mask.sum(), 1)
    print(f"  Wall mask coverage: {wall_frac*100:.1f}%")

    print("\nStep 5: Progressive erosion analysis...")
    # Erode the interior (non-wall) space
    history = progressive_erosion_analysis(mask, wall_mask, max_radius=60, step=2)

    print("\nStep 6: Multi-scale room seeds...")
    seeds = find_room_seeds_multiscale(history, mask, res=res)
    
    if not seeds:
        print("  ERROR: No room seeds found!")
        return

    print(f"\nStep 7: Watershed expansion from {len(seeds)} seeds...")
    rooms = watershed_from_seeds(seeds, mask, wall_density, res=res)
    
    print("\nStep 8: Find hallway regions (unclaimed space)...")
    rooms = find_hallway_regions(rooms, mask, res=res)
    
    # Filter tiny rooms
    rooms = [r for r in rooms if r['area_m2'] >= 2.0]
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)

    print(f"\nStep 9: Polygon extraction ({len(rooms)} rooms)...")
    for i, room in enumerate(rooms):
        poly = extract_polygon(room['mask'], transform, angles)
        if poly is not None:
            area = polygon_area(poly)
            is_hw = room.get('is_hallway', False)
            rtype = classify_room(poly, area, is_hallway=is_hw)
            room['polygon'] = poly
            room['area_m2'] = round(area, 1)
            room['type'] = rtype
            room['vertices'] = len(poly)
            print(f"  Room {i+1}: {area:.1f}m², {len(poly)}v, type={rtype}, "
                  f"er={room.get('erosion_radius', '?')}")
        else:
            room['polygon'] = None
            room['area_m2'] = 0

    rooms_valid = [r for r in rooms if r.get('polygon') is not None]
    rooms_valid.sort(key=lambda r: r['area_m2'], reverse=True)
    
    # Name rooms
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

    print("\nStep 10: Doors...")
    doors = detect_doors(rooms_valid)
    print(f"  {len(doors)} doors")

    print("\nStep 11: Rendering...")
    mesh_name = mesh_path.stem
    render_floorplan(rooms_valid, doors, transform, angles,
                     out_dir / f"v43_{mesh_name}_plan.png",
                     f"v43 Erosion Topology — {mesh_name}")
    render_debug(wall_density, all_density, mask, wall_mask, history,
                 seeds, rooms_valid, transform, out_dir / f"v43_{mesh_name}_debug.png")

    shutil.copy2(out_dir / f"v43_{mesh_name}_plan.png",
                 Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')

    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v43_erosion_topology',
        'dominant_angles_deg': [round(math.degrees(a), 1) for a in angles],
        'rooms': [{
            'name': r.get('name', '?'),
            'area_m2': r.get('area_m2', 0),
            'type': r.get('type', '?'),
            'vertices': r.get('vertices', 0),
            'erosion_radius': r.get('erosion_radius', None),
            'polygon': r['polygon'].tolist() if r.get('polygon') is not None else None
        } for r in rooms_valid],
        'doors': len(doors),
        'total_area_m2': round(total_area, 1)
    }
    with open(out_dir / f"v43_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"=== v43 Erosion Topology Summary ===")
    print(f"{'='*50}")
    print(f"  Dominant angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']}) [er={r['erosion_radius']}]")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

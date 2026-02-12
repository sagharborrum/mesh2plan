#!/usr/bin/env python3
"""
mesh2plan v37 - Skeleton Partition

NEW APPROACH: Instead of watershed (which grows from seeds and creates arbitrary
boundaries), this uses the wall skeleton topology to partition space:

1. Project mesh → density image
2. Density threshold → wall mask (high density = vertical surfaces)
3. Morphological skeleton → 1px-wide wall lines
4. Close gaps in skeleton to form closed room boundaries
5. Flood fill non-skeleton pixels → each connected region = a room
6. Merge tiny regions (<3m²) with their largest neighbor
7. Per-room polygon extraction with strong rectilinear snapping

Key insight: The wall skeleton IS the room boundary graph. We don't need to
"grow" rooms — we just need to find the closed regions the skeleton creates.
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
    """Project mesh vertices to XZ plane density image."""
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
    """Get binary mask of apartment area."""
    mask = (density >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 1, -1)
    return mask


def find_wall_barrier(density, mask, wall_percentile=80):
    """
    Create a wall barrier that partitions the apartment into rooms.
    
    Instead of skeletonizing (which fails on fragmented walls), we:
    1. Threshold density to get wall mask
    2. Use directional (H/V) morphological closing to connect wall fragments
       along their dominant axis
    3. Dilate to form thick barriers
    """
    d = density.copy()
    d[mask == 0] = 0
    masked_vals = d[mask > 0]
    if len(masked_vals) == 0:
        return np.zeros_like(mask), np.zeros_like(mask)
    
    thresh = np.percentile(masked_vals[masked_vals > 0], wall_percentile)
    wall_mask = ((d >= thresh) & (mask > 0)).astype(np.uint8)
    
    # Close gaps first, then remove noise
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel3)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel3)
    
    print(f"  Wall mask after clean: {wall_mask.sum()} px")
    
    # For barrier: use higher threshold (strongest walls only) + directional closing
    thresh_strong = np.percentile(masked_vals[masked_vals > 0], 85)
    strong_walls = ((d >= thresh_strong) & (mask > 0)).astype(np.uint8)
    strong_walls = cv2.morphologyEx(strong_walls, cv2.MORPH_CLOSE, kernel3)
    print(f"  Strong wall pixels (p85): {strong_walls.sum()}")
    
    # Directional closing on strong walls — bridges gaps along wall axes
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 35))
    
    walls_h = cv2.morphologyEx(strong_walls, cv2.MORPH_CLOSE, kernel_h)
    walls_v = cv2.morphologyEx(strong_walls, cv2.MORPH_CLOSE, kernel_v)
    
    # Combine
    wall_barrier = ((walls_h > 0) | (walls_v > 0)).astype(np.uint8)
    
    # Thin the barrier: erode back to ~wall thickness
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wall_barrier = cv2.erode(wall_barrier, kernel_erode, iterations=1)
    wall_barrier = cv2.dilate(wall_barrier, kernel3, iterations=1)
    
    # Constrain to apartment
    wall_barrier = wall_barrier & mask
    
    return wall_mask, wall_barrier


def close_skeleton_gaps(skeleton, mask, max_gap=30):
    """
    Close gaps in the skeleton to form closed room boundaries.
    Strategy: find skeleton endpoints and try to connect them to nearby
    skeleton pixels or to the apartment boundary.
    """
    skel = (skeleton > 0).astype(np.uint8)
    
    # Find endpoints (pixels with exactly 1 neighbor in 8-connectivity)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    endpoints = (skel > 0) & (neighbor_count == 1)
    
    ey, ex = np.where(endpoints)
    print(f"  Skeleton endpoints: {len(ex)}")
    
    # For each endpoint, try to bridge to nearest other skeleton pixel
    # (that isn't in the same branch)
    result = skel.copy()
    
    for i in range(len(ex)):
        px, py = ex[i], ey[i]
        if result[py, px] == 0:
            continue
        
        # Search in a radius for another skeleton pixel
        for radius in range(5, max_gap, 3):
            y1 = max(0, py - radius)
            y2 = min(skel.shape[0], py + radius)
            x1 = max(0, px - radius)
            x2 = min(skel.shape[1], px + radius)
            
            patch = skel[y1:y2, x1:x2].copy()
            # Mask out our own branch (flood fill from endpoint)
            local_py, local_px = py - y1, px - x1
            if 0 <= local_py < patch.shape[0] and 0 <= local_px < patch.shape[1]:
                # Zero out a small region around the endpoint to avoid self-connection
                cv2.circle(patch, (local_px, local_py), 5, 0, -1)
            
            targets_y, targets_x = np.where(patch > 0)
            if len(targets_x) > 0:
                # Find closest
                dists = (targets_x - local_px)**2 + (targets_y - local_py)**2
                best = np.argmin(dists)
                if dists[best] <= max_gap**2:
                    tx, ty = targets_x[best] + x1, targets_y[best] + y1
                    cv2.line(result, (px, py), (tx, ty), 1, 1)
                break
        
        # Also connect to apartment boundary if close
        boundary = (mask == 0).astype(np.uint8)
        boundary_dilated = cv2.dilate(boundary, np.ones((3, 3), np.uint8))
        boundary_edge = boundary_dilated & mask
        
        y1 = max(0, py - max_gap)
        y2 = min(mask.shape[0], py + max_gap)
        x1 = max(0, px - max_gap)
        x2 = min(mask.shape[1], px + max_gap)
        patch = boundary_edge[y1:y2, x1:x2]
        targets_y, targets_x = np.where(patch > 0)
        if len(targets_x) > 0:
            local_py, local_px = py - y1, px - x1
            dists = (targets_x - local_px)**2 + (targets_y - local_py)**2
            best = np.argmin(dists)
            if dists[best] <= (max_gap // 2)**2:
                tx, ty = targets_x[best] + x1, targets_y[best] + y1
                cv2.line(result, (px, py), (tx, ty), 1, 1)
    
    return result * 255


def partition_rooms(wall_barrier, wall_mask_thin, mask, min_room_px=500):
    """
    Two-stage room finding:
    1. Use thick wall_barrier to find room seeds (connected components)
    2. Expand seeds outward using thin wall_mask as barriers (watershed)
    This gives accurate room boundaries at the thin wall locations.
    """
    # Stage 1: Find seeds using thick barrier
    interior = ((mask > 0) & (wall_barrier == 0)).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  Raw seed regions (thick barrier): {n_labels - 1}")
    
    # Keep seeds that represent at least ~1.5m² (smaller seeds merge later)
    seed_min = min(min_room_px // 4, 1000)
    seeds = []
    for lbl in range(1, n_labels):
        area = (labels == lbl).sum()
        if area >= seed_min:
            seeds.append(lbl)
    print(f"  Seeds above threshold: {len(seeds)}")
    
    if len(seeds) < 2:
        # Fallback: just use thick barrier partitioning
        rooms = []
        for lbl in range(1, n_labels):
            room_mask = (labels == lbl).astype(np.uint8)
            if room_mask.sum() >= min_room_px:
                rooms.append({'label': lbl, 'mask': room_mask, 'area_px': room_mask.sum()})
        rooms.sort(key=lambda r: r['area_px'], reverse=True)
        return rooms, labels
    
    # Stage 2: Watershed expansion using thin walls as barriers
    # Thin wall barrier (the original density-thresholded wall mask, dilated slightly)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thin_barrier = cv2.dilate(wall_mask_thin, kernel, iterations=1)
    
    # Create markers: 0=unknown, 1=background, 2+=seeds
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1  # background
    for i, seed_lbl in enumerate(seeds):
        markers[labels == seed_lbl] = i + 2
    
    # Create gradient image from thin walls (high gradient at walls)
    gradient = thin_barrier.astype(np.float32) * 255
    # Add density-based gradient
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    grad_color = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    
    ws_markers = cv2.watershed(grad_color, markers.copy())
    
    # Extract rooms
    rooms = []
    for i, seed_lbl in enumerate(seeds):
        lbl = i + 2
        room_mask = ((ws_markers == lbl) & (mask > 0)).astype(np.uint8)
        area = room_mask.sum()
        if area >= min_room_px:
            rooms.append({'label': lbl, 'mask': room_mask, 'area_px': area})
    
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  Rooms after watershed expansion: {len(rooms)}")
    return rooms, ws_markers


def merge_small_rooms(rooms, labels, mask, min_area_px=1500):
    """Merge rooms smaller than min_area with their largest adjacent neighbor."""
    merged = True
    while merged:
        merged = False
        small = [r for r in rooms if r['area_px'] < min_area_px]
        if not small:
            break
        
        for sr in small:
            # Find adjacent rooms
            dilated = cv2.dilate(sr['mask'], np.ones((7, 7), np.uint8))
            neighbors = []
            for r in rooms:
                if r is sr:
                    continue
                overlap = (dilated & r['mask']).sum()
                if overlap > 0:
                    neighbors.append((overlap, r))
            
            if neighbors:
                # Merge with largest neighbor
                neighbors.sort(key=lambda x: x[1]['area_px'], reverse=True)
                target = neighbors[0][1]
                target['mask'] = target['mask'] | sr['mask']
                target['area_px'] = target['mask'].sum()
                rooms.remove(sr)
                merged = True
                break
    
    return rooms


def extract_room_polygon(room_mask, transform, epsilon_factor=0.015):
    """Extract polygon from room mask with strong rectilinear snapping."""
    x_min, z_min, res = transform
    
    # Smooth mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    
    # Simplify
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert to world coordinates
    pts_px = simplified.reshape(-1, 2).astype(float)
    pts_world = np.zeros((len(pts_px), 2))
    pts_world[:, 0] = pts_px[:, 0] * res + x_min
    pts_world[:, 1] = pts_px[:, 1] * res + z_min
    
    # Strong rectilinear snap
    pts_world = rectilinear_snap(pts_world, angle_thresh=25)
    
    # Remove collinear points
    pts_world = remove_collinear(pts_world)
    
    return pts_world


def rectilinear_snap(pts, angle_thresh=25):
    """Snap polygon edges to axis-aligned directions."""
    if len(pts) < 3:
        return pts
    
    n = len(pts)
    snapped = pts.copy()
    
    for iteration in range(5):
        new_pts = snapped.copy()
        for i in range(n):
            j = (i + 1) % n
            dx = snapped[j, 0] - snapped[i, 0]
            dz = snapped[j, 1] - snapped[i, 1]
            length = math.sqrt(dx**2 + dz**2)
            if length < 0.05:
                continue
            angle = abs(math.degrees(math.atan2(dz, dx))) % 180
            
            if angle < angle_thresh or angle > (180 - angle_thresh):
                # Near horizontal
                avg_z = (snapped[i, 1] + snapped[j, 1]) / 2
                new_pts[i, 1] = avg_z
                new_pts[j, 1] = avg_z
            elif abs(angle - 90) < angle_thresh:
                # Near vertical
                avg_x = (snapped[i, 0] + snapped[j, 0]) / 2
                new_pts[i, 0] = avg_x
                new_pts[j, 0] = avg_x
        snapped = new_pts
    
    # Remove near-duplicate vertices
    cleaned = [snapped[0]]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - cleaned[-1]) > 0.05:
            cleaned.append(snapped[i])
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) < 0.05:
        cleaned = cleaned[:-1]
    
    return np.array(cleaned) if len(cleaned) >= 3 else snapped


def remove_collinear(pts, thresh=0.05):
    """Remove points that are collinear with their neighbors."""
    if len(pts) < 4:
        return pts
    
    cleaned = []
    n = len(pts)
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        
        # Cross product magnitude
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        
        if cross > thresh:
            cleaned.append(p_curr)
    
    return np.array(cleaned) if len(cleaned) >= 3 else pts


def polygon_area(pts):
    """Shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2


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
    """Find doors between adjacent rooms."""
    doors = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            d1 = cv2.dilate(rooms[i]['mask'], np.ones((9, 9), np.uint8))
            d2 = cv2.dilate(rooms[j]['mask'], np.ones((9, 9), np.uint8))
            overlap = d1 & d2
            if overlap.sum() > 15:
                ys, xs = np.where(overlap > 0)
                doors.append({
                    'rooms': (i, j),
                    'pos_px': (xs.mean(), ys.mean()),
                    'room_names': (rooms[i].get('name', ''), rooms[j].get('name', ''))
                })
    return doors


def render_floorplan(rooms, doors, transform, output_path, title="v37"):
    """Render floor plan."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    
    colors = ['#E8E8E8', '#F0F0F0', '#E0E0E0', '#F5F5F5', '#EBEBEB',
              '#E3E3E3', '#F2F2F2', '#EDEDED']
    x_min, z_min, res = transform
    
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        
        poly_closed = np.vstack([poly, poly[0]])
        ax.fill(poly_closed[:, 0], poly_closed[:, 1], color=colors[i % len(colors)], alpha=0.5)
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', linewidth=2.5)
        
        cx, cz = poly[:, 0].mean(), poly[:, 1].mean()
        name = room.get('name', f'Room {i+1}')
        area = room.get('area_m2', 0)
        nverts = room.get('vertices', 0)
        ax.text(cx, cz, f"{name}\n{area:.1f}m²\n({nverts}v)", ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    for door in doors:
        cx, cy = door['pos_px']
        ax.plot(cx * res + x_min, cy * res + z_min, 's', color='brown', markersize=8, zorder=5)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_x = xlim[0] + 0.5
    bar_y = ylim[0] + 0.3
    ax.plot([bar_x, bar_x + 1], [bar_y, bar_y], 'k-', linewidth=3)
    ax.text(bar_x + 0.5, bar_y - 0.15, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug(density, mask, wall_mask, wall_barrier, rooms, transform, output_path):
    """Debug visualization showing pipeline stages."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(np.log1p(density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Log Density')
    
    axes[0, 1].imshow(wall_mask, cmap='gray', origin='lower')
    axes[0, 1].set_title('Wall Mask (density thresh)')
    
    axes[0, 2].imshow(wall_barrier, cmap='gray', origin='lower')
    axes[0, 2].set_title('Wall Barrier (directional close)')
    
    # Distance transform of interior
    interior = ((mask > 0) & (wall_barrier == 0)).astype(np.uint8)
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    axes[1, 0].imshow(dist, cmap='jet', origin='lower')
    axes[1, 0].set_title('Interior Distance Transform')
    
    # Room partition colored
    room_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                   (255, 255, 100), (255, 100, 255), (100, 255, 255),
                   (200, 150, 100), (150, 100, 200)]
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        c = room_colors[i % len(room_colors)]
        vis[room['mask'] > 0] = c
    axes[1, 1].imshow(vis, origin='lower')
    axes[1, 1].set_title(f'Room Partitions ({len(rooms)} rooms)')
    
    # Polygons overlay
    x_min, z_min, res = transform
    axes[1, 2].imshow(np.log1p(density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        c = np.array(room_colors[i % len(room_colors)]) / 255.0
        px_c = np.append(px, px[0])
        pz_c = np.append(pz, pz[0])
        axes[1, 2].plot(px_c, pz_c, '-', color=c, linewidth=2)
        axes[1, 2].fill(px_c, pz_c, color=c, alpha=0.2)
    axes[1, 2].set_title('Final Polygons')
    
    plt.suptitle('v37 Skeleton Partition Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--resolution', '-r', type=float, default=0.02)
    parser.add_argument('--wall-percentile', type=float, default=80)
    parser.add_argument('--min-room-m2', type=float, default=3.0)
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v37_skeleton_partition'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    res = args.resolution
    min_room_px = int(args.min_room_m2 / (res ** 2))
    
    print("Step 1: Density image...")
    density, transform = mesh_to_density(mesh, res)
    print(f"  Size: {density.shape}")
    
    print("Step 2: Apartment mask...")
    mask = get_apartment_mask(density)
    print(f"  Area: {mask.sum()} px ({mask.sum() * res * res:.1f} m²)")
    
    print(f"Step 3: Wall barrier (percentile={args.wall_percentile})...")
    wall_mask, wall_barrier = find_wall_barrier(density, mask, args.wall_percentile)
    wall_pct = wall_mask.sum() / mask.sum() * 100
    barrier_pct = wall_barrier.sum() / mask.sum() * 100
    print(f"  Wall pixels: {wall_mask.sum()} ({wall_pct:.1f}%)")
    print(f"  Barrier pixels: {wall_barrier.sum()} ({barrier_pct:.1f}%)")
    
    print("Step 4: Partitioning rooms (thick barrier → seeds → watershed expand)...")
    rooms, labels = partition_rooms(wall_barrier, wall_mask, mask, min_room_px=min_room_px)
    
    print(f"Step 5: Merging small rooms (min {args.min_room_m2}m²)...")
    merge_thresh_px = int(args.min_room_m2 / (res * res))
    rooms = merge_small_rooms(rooms, labels, mask, min_area_px=merge_thresh_px)
    print(f"  Rooms after merging: {len(rooms)}")
    
    print("Step 6: Extracting polygons...")
    for i, room in enumerate(rooms):
        poly = extract_room_polygon(room['mask'], transform)
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
    
    # Name rooms
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
    
    print("Step 7: Doors...")
    doors = detect_doors(rooms)
    print(f"  {len(doors)} doors")
    
    print("Step 8: Rendering...")
    mesh_name = mesh_path.stem
    
    render_floorplan(rooms_valid, doors, transform,
                     out_dir / f"v37_{mesh_name}_plan.png",
                     f"v37 Skeleton Partition — {mesh_name}")
    
    render_debug(density, mask, wall_mask, wall_barrier, rooms_valid, transform,
                 out_dir / f"v37_{mesh_name}_debug.png")
    
    # Save results
    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v37_skeleton_partition',
        'rooms': [],
        'doors': len(doors),
        'total_area_m2': round(total_area, 1)
    }
    for room in rooms_valid:
        poly = room.get('polygon')
        results['rooms'].append({
            'name': room.get('name', 'unknown'),
            'area_m2': room.get('area_m2', 0),
            'type': room.get('type', 'unknown'),
            'vertices': room.get('vertices', 0),
            'polygon': poly.tolist() if poly is not None else None
        })
    
    json_path = out_dir / f"v37_{mesh_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")
    
    print("\n=== Summary ===")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

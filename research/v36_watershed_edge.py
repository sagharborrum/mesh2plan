#!/usr/bin/env python3
"""
mesh2plan v36 - Watershed + Density-Wall Segmentation

NEW APPROACH: Use density thresholding to find walls (high density = wall surfaces),
then watershed to segment rooms.

Key insight from debug: Canny edges on the density image are too noisy (picks up
furniture, scanning artifacts). Instead, walls are the HIGHEST density areas
(vertical surfaces accumulate many projected points).

Pipeline:
1. Project mesh to XZ density image (Y-up)
2. Identify walls via density thresholding (top ~15% density within mask)
3. Morphological thinning to get wall skeletons
4. Distance transform on non-wall areas → room centers
5. Watershed segmentation
6. Per-room contour extraction with rectilinear simplification
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


def find_wall_mask(density, mask, percentile=80):
    """Find walls as high-density regions within the apartment mask.
    
    Walls (vertical surfaces) project many vertices into the same XZ pixel,
    creating high-density ridges. Floor/ceiling areas have lower density.
    """
    d = density.copy()
    d[mask == 0] = 0
    
    # Get density values within mask
    masked_vals = d[mask > 0]
    if len(masked_vals) == 0:
        return np.zeros_like(mask)
    
    # Walls are in the top density percentile
    thresh = np.percentile(masked_vals[masked_vals > 0], percentile)
    wall_mask = ((d >= thresh) & (mask > 0)).astype(np.uint8)
    
    # Clean: close small gaps, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)
    
    return wall_mask


def thin_walls(wall_mask):
    """Thin wall regions to skeleton lines."""
    # Dilate slightly to connect nearby wall fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    
    # Skeletonize
    skeleton = cv2.ximgproc.thinning(connected * 255) if hasattr(cv2, 'ximgproc') else morphological_thin(connected)
    
    return skeleton


def morphological_thin(mask):
    """Simple morphological thinning fallback."""
    # Use iterative erosion approach
    thin = mask.copy() * 255
    # Just use the thick walls directly — dilate to ensure connectivity
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick = cv2.dilate(mask * 255, kernel, iterations=1)
    return thick


def watershed_segment(density, mask, wall_mask, min_room_px=800):
    """Watershed segmentation using walls as barriers.
    
    1. Create barrier from walls
    2. Distance transform of non-wall interior
    3. Find peaks as room seeds
    4. Watershed
    """
    # Wall barrier: use the wall mask, thickened slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls_thick = cv2.dilate(wall_mask, kernel, iterations=1)
    
    # Interior = mask minus walls
    interior = mask.copy()
    interior[walls_thick > 0] = 0
    
    # Distance transform
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    
    # Find room seeds via adaptive thresholding of distance
    # Try different thresholds to get 4-7 rooms
    best_labels = None
    best_n = 0
    for thresh_frac in [0.6, 0.5, 0.4, 0.3, 0.25, 0.2]:
        _, fg = cv2.threshold(dist, thresh_frac * dist.max(), 255, cv2.THRESH_BINARY)
        fg = fg.astype(np.uint8)
        # Remove tiny seeds
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n_labels, labels = cv2.connectedComponents(fg)
        n = n_labels - 1
        print(f"  dist threshold {thresh_frac:.2f}: {n} seeds")
        if 4 <= n <= 8:
            best_labels = labels
            best_n = n
            break
        if n > best_n:
            best_labels = labels
            best_n = n
    
    if best_labels is None or best_n < 2:
        print("  WARNING: Could not find enough room seeds")
        return [], np.zeros_like(mask, dtype=np.int32)
    
    print(f"  Using {best_n} seeds for watershed")
    
    # Prepare markers for watershed:
    # 0 = unknown (to be determined by watershed)
    # 1 = background (outside apartment)  
    # 2+ = room seeds
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1  # outside = background
    # Copy seed labels (1-based from connectedComponents) as 2+
    for lbl in range(1, best_n + 1):
        markers[best_labels == lbl] = lbl + 1
    # Everything else inside mask stays 0 = unknown
    
    # Watershed
    d_norm = density.copy()
    if d_norm.max() > 0:
        d_norm = (d_norm / d_norm.max() * 255).astype(np.uint8)
    img_color = cv2.cvtColor(d_norm, cv2.COLOR_GRAY2BGR)
    
    markers_ws = cv2.watershed(img_color, markers.astype(np.int32))
    
    # Extract rooms (label 1 = background, 2+ = rooms, -1 = boundaries)
    rooms = []
    for label in range(2, markers_ws.max() + 1):
        room_mask = ((markers_ws == label) & (mask > 0)).astype(np.uint8)
        area = room_mask.sum()
        if area >= min_room_px:
            rooms.append({
                'label': label,
                'mask': room_mask,
                'area_px': area
            })
    
    # Sort by area descending
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  Watershed: {len(rooms)} rooms (min {min_room_px}px)")
    
    return rooms, markers_ws


def extract_room_polygon(room_mask, transform, epsilon_factor=0.012):
    """Extract a clean rectilinear polygon from room mask contour."""
    x_min, z_min, res = transform
    
    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
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
    
    # Rectilinear snapping
    pts_world = rectilinear_snap(pts_world)
    
    return pts_world


def rectilinear_snap(pts, angle_thresh=20):
    """Snap polygon edges to axis-aligned (0° or 90°)."""
    if len(pts) < 3:
        return pts
    
    n = len(pts)
    snapped = pts.copy()
    
    for _ in range(3):
        new_pts = snapped.copy()
        for i in range(n):
            j = (i + 1) % n
            dx = snapped[j, 0] - snapped[i, 0]
            dz = snapped[j, 1] - snapped[i, 1]
            if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                continue
            angle = abs(math.degrees(math.atan2(dz, dx))) % 180
            
            if angle < angle_thresh or angle > (180 - angle_thresh):
                avg_z = (snapped[i, 1] + snapped[j, 1]) / 2
                new_pts[i, 1] = avg_z
                new_pts[j, 1] = avg_z
            elif abs(angle - 90) < angle_thresh:
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
    if area < 3:
        return "closet"
    xs, zs = polygon[:, 0], polygon[:, 1]
    w, h = xs.max() - xs.min(), zs.max() - zs.min()
    aspect = max(w, h) / (min(w, h) + 0.01)
    if area < 5:
        return "hallway" if aspect > 2.5 else "closet"
    if aspect > 2.5:
        return "hallway"
    return "room"


def detect_doors(rooms):
    """Find doors between adjacent rooms."""
    doors = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            d1 = cv2.dilate(rooms[i]['mask'], np.ones((7, 7), np.uint8))
            d2 = cv2.dilate(rooms[j]['mask'], np.ones((7, 7), np.uint8))
            overlap = d1 & d2
            if overlap.sum() > 10:
                ys, xs = np.where(overlap > 0)
                doors.append({
                    'rooms': (i, j),
                    'pos_px': (xs.mean(), ys.mean()),
                    'room_names': (rooms[i].get('name', ''), rooms[j].get('name', ''))
                })
    return doors


def render_floorplan(rooms, doors, transform, output_path, title="v36"):
    """Render architectural floor plan."""
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
        ax.text(cx, cz, f"{name}\n{area:.1f}m²", ha='center', va='center',
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


def render_debug(density, mask, wall_mask, rooms, markers, transform, output_path):
    """Debug visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    x_min, z_min, res = transform
    
    axes[0, 0].imshow(np.log1p(density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Log Density')
    
    axes[0, 1].imshow(mask, cmap='gray', origin='lower')
    axes[0, 1].set_title('Apartment Mask')
    
    axes[0, 2].imshow(wall_mask, cmap='gray', origin='lower')
    axes[0, 2].set_title('Wall Mask (density threshold)')
    
    # Distance transform
    walls_thick = cv2.dilate(wall_mask, np.ones((5, 5), np.uint8))
    interior = mask.copy()
    interior[walls_thick > 0] = 0
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    axes[1, 0].imshow(dist, cmap='jet', origin='lower')
    axes[1, 0].set_title('Distance Transform')
    
    # Watershed colored
    room_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                   (255, 255, 100), (255, 100, 255), (100, 255, 255),
                   (200, 150, 100), (150, 100, 200)]
    ws_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        c = room_colors[i % len(room_colors)]
        ws_vis[room['mask'] > 0] = c
    axes[1, 1].imshow(ws_vis, origin='lower')
    axes[1, 1].set_title(f'Watershed ({len(rooms)} rooms)')
    
    # Polygons on density
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
    axes[1, 2].set_title('Final Polygons on Density')
    
    plt.suptitle('v36 Watershed + Density-Wall Debug', fontsize=14)
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
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v36_watershed_edge'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading mesh: {args.mesh_path}")
    mesh = trimesh.load_mesh(args.mesh_path)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    print("Step 1: Density image...")
    density, transform = mesh_to_density(mesh, args.resolution)
    print(f"  Size: {density.shape}")
    
    print("Step 2: Apartment mask...")
    mask = get_apartment_mask(density)
    print(f"  Area: {mask.sum()} px")
    
    print(f"Step 3: Wall detection (percentile={args.wall_percentile})...")
    wall_mask = find_wall_mask(density, mask, args.wall_percentile)
    wall_pct = wall_mask.sum() / mask.sum() * 100
    print(f"  Wall pixels: {wall_mask.sum()} ({wall_pct:.1f}% of apartment)")
    
    # If walls cover too much or too little, adjust
    if wall_pct > 40:
        print("  Too many wall pixels, raising percentile to 90...")
        wall_mask = find_wall_mask(density, mask, 90)
        wall_pct = wall_mask.sum() / mask.sum() * 100
        print(f"  Wall pixels: {wall_mask.sum()} ({wall_pct:.1f}%)")
    elif wall_pct < 5:
        print("  Too few wall pixels, lowering percentile to 70...")
        wall_mask = find_wall_mask(density, mask, 70)
        wall_pct = wall_mask.sum() / mask.sum() * 100
        print(f"  Wall pixels: {wall_mask.sum()} ({wall_pct:.1f}%)")
    
    print("Step 4: Watershed segmentation...")
    rooms, markers = watershed_segment(density, mask, wall_mask)
    print(f"  {len(rooms)} rooms found")
    
    print("Step 5: Extracting room polygons...")
    room_counter = 1
    hallway_counter = 0
    closet_counter = 0
    
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
    rooms_sorted = sorted([r for r in rooms if r.get('polygon') is not None], 
                          key=lambda r: r['area_m2'], reverse=True)
    rc, hc, cc = 1, 1, 1
    for room in rooms_sorted:
        if room['type'] == 'hallway':
            room['name'] = "Hallway" if hc == 1 else f"Hallway {hc}"
            hc += 1
        elif room['type'] == 'closet':
            room['name'] = "Closet" if cc == 1 else f"Closet {cc}"
            cc += 1
        else:
            room['name'] = f"Room {rc}"
            rc += 1
    
    print("Step 6: Doors...")
    doors = detect_doors(rooms)
    print(f"  {len(doors)} doors")
    
    print("Step 7: Rendering...")
    mesh_name = Path(args.mesh_path).stem
    
    render_floorplan(rooms, doors, transform,
                     out_dir / f"v36_{mesh_name}_plan.png",
                     f"v36 Watershed + Density Walls — {mesh_name}")
    
    render_debug(density, mask, wall_mask, rooms, markers, transform,
                 out_dir / f"v36_{mesh_name}_debug.png")
    
    results = {
        'approach': 'v36_watershed_edge',
        'rooms': [],
        'doors': len(doors),
        'total_area_m2': sum(r.get('area_m2', 0) for r in rooms)
    }
    for room in rooms:
        poly = room.get('polygon')
        results['rooms'].append({
            'name': room.get('name', 'unknown'),
            'area_m2': room.get('area_m2', 0),
            'type': room.get('type', 'unknown'),
            'polygon': poly.tolist() if poly is not None else None,
            'vertices': room.get('vertices', 0)
        })
    
    json_path = out_dir / f"v36_{mesh_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")
    
    print("\n=== Summary ===")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']:.1f}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

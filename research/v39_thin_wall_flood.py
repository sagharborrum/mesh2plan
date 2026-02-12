#!/usr/bin/env python3
"""
mesh2plan v39 - Density Ridge Watershed

FUNDAMENTALLY DIFFERENT approach: treat the density image as a topographic map.
- Walls = ridges (high density — vertical surfaces produce more points per pixel)
- Room interiors = basins (lower density — floors)
- Use watershed on the density image directly: basins = rooms, ridges = walls

No manual wall thresholding. No barrier construction. Let watershed find the 
natural room boundaries defined by density ridges.

Steps:
1. Project mesh → density image
2. Gaussian blur to smooth noise
3. Find room centers via distance transform of low-density regions
4. Watershed on the density image with room center seeds
5. Extract and simplify room polygons
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
    """
    Find room seeds from low-density regions (floor areas).
    Uses adaptive thresholding: floor = below median density.
    """
    d = density.copy()
    d[mask == 0] = 0
    
    # Smooth to reduce furniture noise
    d_smooth = cv2.GaussianBlur(d, (11, 11), 3)
    
    # Floor = low density within apartment
    masked_vals = d_smooth[mask > 0]
    median = np.median(masked_vals[masked_vals > 0]) if np.any(masked_vals > 0) else 1
    floor_mask = ((d_smooth < median) & (mask > 0)).astype(np.uint8)
    
    # Distance transform on floor areas
    dist = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
    
    # Also compute distance from apartment boundary (favors interior points)
    boundary_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Combined: prefer points far from walls AND far from boundary
    combined = dist * 0.7 + boundary_dist * 0.3
    combined[mask == 0] = 0
    
    # Find local maxima
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
    # Keep top seeds (avoid over-segmentation)
    if len(seeds) > n_target * 2:
        seeds = seeds[:n_target * 2]
    
    print(f"  Found {len(seeds)} seeds")
    for i, (sx, sy, sd) in enumerate(seeds[:10]):
        print(f"    Seed {i+1}: ({sx}, {sy}), score={sd:.1f}")
    
    return seeds, dist, combined


def watershed_density(seeds, density, mask):
    """
    Watershed on the density image itself.
    High density = ridges (walls) = watershed boundaries.
    """
    # Normalize density to 0-255 for watershed
    d = density.copy()
    d[mask == 0] = 0
    d_smooth = cv2.GaussianBlur(d, (7, 7), 2)
    
    # Invert: watershed finds basins, so invert to make walls = barriers
    d_norm = np.zeros_like(d_smooth)
    d_max = d_smooth[mask > 0].max() if np.any(mask > 0) else 1
    d_norm[mask > 0] = (d_smooth[mask > 0] / d_max * 255)
    d_uint8 = d_norm.astype(np.uint8)
    
    # Create markers
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1  # background
    
    for i, (sx, sy, _) in enumerate(seeds):
        cv2.circle(markers, (sx, sy), 3, i + 2, -1)
    
    # Watershed needs 3-channel image
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


def extract_polygon(room_mask, transform, epsilon_factor=0.012):
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
    
    pts = rectilinear_snap(pts)
    pts = remove_collinear(pts)
    return pts


def rectilinear_snap(pts, angle_thresh=25):
    if len(pts) < 3:
        return pts
    n = len(pts)
    snapped = pts.copy()
    for _ in range(5):
        new = snapped.copy()
        for i in range(n):
            j = (i + 1) % n
            dx = snapped[j, 0] - snapped[i, 0]
            dz = snapped[j, 1] - snapped[i, 1]
            if math.sqrt(dx**2 + dz**2) < 0.05:
                continue
            angle = abs(math.degrees(math.atan2(dz, dx))) % 180
            if angle < angle_thresh or angle > (180 - angle_thresh):
                avg = (snapped[i, 1] + snapped[j, 1]) / 2
                new[i, 1] = avg
                new[j, 1] = avg
            elif abs(angle - 90) < angle_thresh:
                avg = (snapped[i, 0] + snapped[j, 0]) / 2
                new[i, 0] = avg
                new[j, 0] = avg
        snapped = new
    cleaned = [snapped[0]]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - cleaned[-1]) > 0.05:
            cleaned.append(snapped[i])
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) < 0.05:
        cleaned = cleaned[:-1]
    return np.array(cleaned) if len(cleaned) >= 3 else snapped


def remove_collinear(pts, thresh=0.05):
    if len(pts) < 4:
        return pts
    cleaned = []
    n = len(pts)
    for i in range(n):
        v1 = pts[i] - pts[(i - 1) % n]
        v2 = pts[(i + 1) % n] - pts[i]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) > thresh:
            cleaned.append(pts[i])
    return np.array(cleaned) if len(cleaned) >= 3 else pts


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


def render_floorplan(rooms, doors, transform, output_path, title="v39"):
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


def render_debug(density, mask, seeds, seed_score, rooms, ws, transform, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(np.log1p(density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Log Density')

    d_smooth = cv2.GaussianBlur(density, (7, 7), 2)
    d_smooth[mask == 0] = 0
    axes[0, 1].imshow(d_smooth, cmap='hot', origin='lower')
    axes[0, 1].set_title('Smoothed Density (watershed input)')

    axes[0, 2].imshow(seed_score, cmap='jet', origin='lower')
    for sx, sy, sd in seeds[:15]:
        axes[0, 2].plot(sx, sy, 'w*', markersize=12)
    axes[0, 2].set_title(f'Seed Score + {len(seeds)} Seeds')

    # Watershed boundaries
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

    x_min, z_min, res = transform
    axes[1, 2].imshow(np.log1p(density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None:
            continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        c = np.array(room_colors[i % len(room_colors)]) / 255.0
        axes[1, 2].plot(np.append(px, px[0]), np.append(pz, pz[0]), '-', color=c, linewidth=2)
        axes[1, 2].fill(np.append(px, px[0]), np.append(pz, pz[0]), color=c, alpha=0.2)
    axes[1, 2].set_title('Final Polygons')

    plt.suptitle('v39 Density Ridge Watershed Debug', fontsize=14)
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
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v39_thin_wall_flood'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    res = args.resolution

    print("Step 1: Density image...")
    density, transform = mesh_to_density(mesh, res)
    print(f"  Size: {density.shape}")

    print("Step 2: Apartment mask...")
    mask = get_apartment_mask(density)
    print(f"  Area: {mask.sum() * res * res:.1f} m²")

    print("Step 3: Finding room seeds...")
    seeds, floor_dist, seed_score = find_seeds_adaptive(density, mask, min_dist=args.min_seed_dist)

    print("Step 4: Watershed on density...")
    ws = watershed_density(seeds, density, mask)

    print("Step 5: Extracting rooms...")
    rooms = extract_rooms(ws, mask, seeds, res)
    print(f"  Raw rooms: {len(rooms)}")

    min_merge_px = int(3.0 / (res * res))
    rooms = merge_small(rooms, min_merge_px)
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  After merge: {len(rooms)}")

    print("Step 6: Polygon extraction...")
    for i, room in enumerate(rooms):
        poly = extract_polygon(room['mask'], transform)
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

    print("Step 7: Doors...")
    doors = detect_doors(rooms_valid)
    print(f"  {len(doors)} doors")

    print("Step 8: Rendering...")
    mesh_name = mesh_path.stem
    render_floorplan(rooms_valid, doors, transform,
                     out_dir / f"v39_{mesh_name}_plan.png",
                     f"v39 Density Ridge Watershed — {mesh_name}")
    render_debug(density, mask, seeds, seed_score, rooms_valid, ws, transform,
                 out_dir / f"v39_{mesh_name}_debug.png")

    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v39_density_ridge_watershed',
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
    with open(out_dir / f"v39_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

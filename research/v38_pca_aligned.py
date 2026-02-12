#!/usr/bin/env python3
"""
mesh2plan v38 - PCA-Aligned Flood Fill

KEY INSIGHT: The apartment mesh is rotated ~30° from axis-aligned. Previous
approaches (v32-v37) all work in the raw coordinate system, so rectilinear
snapping fights the actual wall orientations. Hough lines find diagonal walls
but can't snap them cleanly.

NEW APPROACH:
1. PCA on mesh XZ vertices → find principal axes of the apartment
2. Rotate all vertices to align walls with X/Y axes
3. Build density image in aligned space
4. Wall detection via gradient magnitude (walls = high density gradient)
5. Distance transform → room seeds → flood fill with wall barriers
6. Rectilinear simplification works perfectly in aligned space
7. Rotate final polygons back to original coordinates for display

This should give clean right-angle room polygons that match real wall geometry.
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
from sklearn.decomposition import PCA


def find_principal_angle(verts_xz):
    """Find rotation angle to axis-align the apartment using PCA."""
    pca = PCA(n_components=2)
    pca.fit(verts_xz)
    # Principal component direction
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    print(f"  PCA principal angle: {np.degrees(angle):.1f}°")
    return angle


def rotate_points(pts, angle, center=None):
    """Rotate 2D points by angle around center."""
    if center is None:
        center = pts.mean(axis=0)
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    return (pts - center) @ R.T + center


def mesh_to_density(xz_pts, resolution=0.02):
    """Project XZ points to density image."""
    x, z = xz_pts[:, 0], xz_pts[:, 1]
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


def detect_walls_gradient(density, mask, wall_percentile=80):
    """
    Detect walls using density thresholding + morphological cleanup.
    In axis-aligned space, walls should be thin H/V lines.
    """
    d = density.copy()
    d[mask == 0] = 0
    masked_vals = d[mask > 0]
    nonzero = masked_vals[masked_vals > 0]
    if len(nonzero) == 0:
        return np.zeros_like(mask), np.zeros_like(mask)

    # Thin walls (for expansion guidance)
    thresh_thin = np.percentile(nonzero, wall_percentile)
    wall_thin = ((d >= thresh_thin) & (mask > 0)).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_thin = cv2.morphologyEx(wall_thin, cv2.MORPH_CLOSE, k3)
    wall_thin = cv2.morphologyEx(wall_thin, cv2.MORPH_OPEN, k3)

    # Thick barrier for finding seeds - use directional closing in aligned space
    # Since we're axis-aligned, H/V kernels should work much better now
    thresh_strong = np.percentile(nonzero, 85)
    strong = ((d >= thresh_strong) & (mask > 0)).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, k3)

    # Directional closing - connect wall fragments along H and V
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    walls_h = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, kh)
    walls_v = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, kv)
    barrier = ((walls_h > 0) | (walls_v > 0)).astype(np.uint8)

    # Thin back
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    barrier = cv2.erode(barrier, ke, iterations=1)
    barrier = cv2.dilate(barrier, k3, iterations=1)
    barrier = barrier & mask

    print(f"  Wall thin: {wall_thin.sum()} px, barrier: {barrier.sum()} px")
    return wall_thin, barrier


def find_rooms_flood(barrier, wall_thin, mask, min_room_px=500, res=0.02):
    """Find rooms via thick barrier seeds + watershed expansion."""
    interior = ((mask > 0) & (barrier == 0)).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  Seed regions: {n_labels - 1}")

    # Keep meaningful seeds
    seeds = []
    for lbl in range(1, n_labels):
        area = (labels == lbl).sum()
        if area >= min_room_px // 4:
            seeds.append((lbl, area))
    seeds.sort(key=lambda x: x[1], reverse=True)
    print(f"  Valid seeds: {len(seeds)}")

    if len(seeds) < 2:
        # Fallback
        rooms = []
        for lbl in range(1, n_labels):
            m = (labels == lbl).astype(np.uint8)
            if m.sum() >= min_room_px:
                rooms.append({'mask': m, 'area_px': m.sum()})
        return rooms

    # Watershed
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask == 0] = 1
    for i, (lbl, _) in enumerate(seeds):
        markers[labels == lbl] = i + 2

    # Gradient from thin walls
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad = cv2.dilate(wall_thin, kd, iterations=1).astype(np.float32) * 255
    grad = np.clip(grad, 0, 255).astype(np.uint8)
    grad_color = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    ws = cv2.watershed(grad_color, markers.copy())

    rooms = []
    for i, (lbl, _) in enumerate(seeds):
        room_mask = ((ws == i + 2) & (mask > 0)).astype(np.uint8)
        area = room_mask.sum()
        if area >= min_room_px:
            rooms.append({'mask': room_mask, 'area_px': area})

    # Merge small rooms
    rooms = merge_small(rooms, min_area_px=int(3.0 / (res * res)))
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  Final rooms: {len(rooms)}")
    return rooms


def merge_small(rooms, min_area_px=1500):
    """Merge rooms smaller than threshold with largest neighbor."""
    changed = True
    while changed:
        changed = False
        small = [r for r in rooms if r['area_px'] < min_area_px]
        if not small:
            break
        for sr in small:
            dilated = cv2.dilate(sr['mask'], np.ones((7, 7), np.uint8))
            best = None
            best_overlap = 0
            for r in rooms:
                if r is sr:
                    continue
                overlap = (dilated & r['mask']).sum()
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = r
            if best and best_overlap > 0:
                best['mask'] = best['mask'] | sr['mask']
                best['area_px'] = best['mask'].sum()
                rooms.remove(sr)
                changed = True
                break
    return rooms


def extract_polygon_aligned(room_mask, transform):
    """Extract rectilinear polygon in axis-aligned space."""
    x_min, z_min, res = transform

    # Smooth
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, 0.015 * peri, True)

    pts_px = simplified.reshape(-1, 2).astype(float)
    pts = np.zeros((len(pts_px), 2))
    pts[:, 0] = pts_px[:, 0] * res + x_min
    pts[:, 1] = pts_px[:, 1] * res + z_min

    # Strong rectilinear snap — should work well in aligned space
    pts = rectilinear_snap(pts, angle_thresh=20)
    pts = remove_collinear(pts)
    return pts


def rectilinear_snap(pts, angle_thresh=20):
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
            length = math.sqrt(dx**2 + dz**2)
            if length < 0.05:
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
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if cross > thresh:
            cleaned.append(pts[i])
    return np.array(cleaned) if len(cleaned) >= 3 else pts


def polygon_area(pts):
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
                })
    return doors


def render_floorplan(rooms, doors, transform, pca_angle, pca_center, output_path, title="v38"):
    """Render floor plan in ORIGINAL coordinates (rotate polygons back)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    colors = ['#E8E8E8', '#F0F0F0', '#E0E0E0', '#F5F5F5', '#EBEBEB',
              '#E3E3E3', '#F2F2F2', '#EDEDED']

    for i, room in enumerate(rooms):
        poly = room.get('polygon_orig')
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

    x_min, z_min, res = transform
    for door in doors:
        cx, cy = door['pos_px']
        # Convert door position back to original coords
        dp = np.array([[cx * res + x_min, cy * res + z_min]])
        dp_orig = rotate_points(dp, -pca_angle, pca_center)
        ax.plot(dp_orig[0, 0], dp_orig[0, 1], 's', color='brown', markersize=8, zorder=5)

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


def render_debug(density, mask, wall_thin, barrier, rooms, transform, output_path):
    """Debug with aligned views."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(np.log1p(density), cmap='hot', origin='lower')
    axes[0, 0].set_title('Log Density (PCA-aligned)')

    axes[0, 1].imshow(wall_thin, cmap='gray', origin='lower')
    axes[0, 1].set_title('Wall Mask (thin)')

    axes[0, 2].imshow(barrier, cmap='gray', origin='lower')
    axes[0, 2].set_title('Wall Barrier (thick)')

    interior = ((mask > 0) & (barrier == 0)).astype(np.uint8)
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    axes[1, 0].imshow(dist, cmap='jet', origin='lower')
    axes[1, 0].set_title('Distance Transform')

    room_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                   (255, 255, 100), (255, 100, 255), (100, 255, 255)]
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, room in enumerate(rooms):
        vis[room['mask'] > 0] = room_colors[i % len(room_colors)]
    axes[1, 1].imshow(vis, origin='lower')
    axes[1, 1].set_title(f'Room Masks ({len(rooms)} rooms)')

    # Polygons in aligned space
    x_min, z_min, res = transform
    axes[1, 2].imshow(np.log1p(density), cmap='gray', origin='lower', alpha=0.5)
    for i, room in enumerate(rooms):
        poly = room.get('polygon')
        if poly is None or len(poly) < 3:
            continue
        px = (poly[:, 0] - x_min) / res
        pz = (poly[:, 1] - z_min) / res
        c = np.array(room_colors[i % len(room_colors)]) / 255.0
        axes[1, 2].plot(np.append(px, px[0]), np.append(pz, pz[0]), '-', color=c, linewidth=2)
        axes[1, 2].fill(np.append(px, px[0]), np.append(pz, pz[0]), color=c, alpha=0.2)
    axes[1, 2].set_title('Polygons (aligned)')

    plt.suptitle('v38 PCA-Aligned Debug', fontsize=14)
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
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v38_pca_aligned'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    verts = np.array(mesh.vertices)
    xz = verts[:, [0, 2]]  # XZ plane

    print("Step 1: PCA axis alignment...")
    pca_angle = find_principal_angle(xz)
    pca_center = xz.mean(axis=0)
    xz_aligned = rotate_points(xz, pca_angle, pca_center)
    print(f"  Rotated by {np.degrees(pca_angle):.1f}° to axis-align")

    print("Step 2: Density image (aligned)...")
    density, transform = mesh_to_density(xz_aligned, args.resolution)
    print(f"  Size: {density.shape}")

    print("Step 3: Apartment mask...")
    mask = get_apartment_mask(density)
    res = args.resolution
    print(f"  Area: {mask.sum() * res * res:.1f} m²")

    print("Step 4: Wall detection (aligned)...")
    wall_thin, barrier = detect_walls_gradient(density, mask)
    barrier_pct = barrier.sum() / max(mask.sum(), 1) * 100
    print(f"  Barrier coverage: {barrier_pct:.1f}%")

    print("Step 5: Room finding...")
    min_room_px = int(3.0 / (res * res))
    rooms = find_rooms_flood(barrier, wall_thin, mask, min_room_px, res)

    print("Step 6: Polygon extraction (aligned space)...")
    for i, room in enumerate(rooms):
        poly = extract_polygon_aligned(room['mask'], transform)
        if poly is not None:
            area = polygon_area(poly)
            rtype = classify_room(poly, area)
            room['polygon'] = poly
            room['area_m2'] = round(area, 1)
            room['type'] = rtype
            room['vertices'] = len(poly)

            # Rotate polygon back to original coordinates
            poly_orig = rotate_points(poly, -pca_angle, pca_center)
            room['polygon_orig'] = poly_orig
            print(f"  Room {i+1}: {area:.1f}m², {len(poly)}v, type={rtype}")
        else:
            room['polygon'] = None
            room['polygon_orig'] = None
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
    doors = detect_doors(rooms_valid)
    print(f"  {len(doors)} doors")

    print("Step 8: Rendering...")
    mesh_name = mesh_path.stem

    render_floorplan(rooms_valid, doors, transform, pca_angle, pca_center,
                     out_dir / f"v38_{mesh_name}_plan.png",
                     f"v38 PCA-Aligned — {mesh_name}")

    render_debug(density, mask, wall_thin, barrier, rooms_valid, transform,
                 out_dir / f"v38_{mesh_name}_debug.png")

    # Save results
    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v38_pca_aligned',
        'pca_angle_deg': round(np.degrees(pca_angle), 1),
        'rooms': [],
        'doors': len(doors),
        'total_area_m2': round(total_area, 1)
    }
    for room in rooms_valid:
        poly_orig = room.get('polygon_orig')
        results['rooms'].append({
            'name': room.get('name', 'unknown'),
            'area_m2': room.get('area_m2', 0),
            'type': room.get('type', 'unknown'),
            'vertices': room.get('vertices', 0),
            'polygon': poly_orig.tolist() if poly_orig is not None else None
        })

    json_path = out_dir / f"v38_{mesh_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n=== Summary ===")
    print(f"  PCA rotation: {np.degrees(pca_angle):.1f}°")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

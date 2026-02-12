#!/usr/bin/env python3
"""
mesh2plan v42 - Edge Contour Partition

NEW approach: Use Canny edges on normal-filtered wall density to find wall contours,
then use those contours as barriers for flood-fill room segmentation.

Key difference from v41b: Instead of watershed (which under-segmented to 4 rooms),
use the actual edge contours as hard barriers. The normal-filtered density gives
clean wall signal; Canny should produce connected wall lines that partition space.

Pipeline:
1. Normal-filtered wall density (from v41b - proven cleaner)
2. Canny edge detection on wall density → binary edge image
3. Morphological closing to connect nearby edges into continuous walls
4. Dilate edges to create wall barriers
5. Flood fill from distance-transform peaks of barrier-free regions → rooms
6. Per-room polygon extraction with Hough angle snap

Why this should work better:
- Canny on WALL-ONLY density (not all-vertex) avoids furniture noise
- Edge closing bridges doorway gaps to create room-enclosing barriers
- Flood fill is binary (wall/not-wall) — no watershed gradient ambiguity
- Distance transform peaks naturally find room centers
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
    normals = mesh.face_normals
    centroids = mesh.triangles_center
    y_comp = np.abs(normals[:, 1])
    wall_mask = y_comp < normal_thresh
    wall_strength = 1.0 - y_comp
    wall_centroids = centroids[wall_mask]
    wall_weights = wall_strength[wall_mask]
    
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


def create_wall_barriers(wall_density, mask):
    """
    Use Canny edges on wall density + morphological ops to create wall barriers.
    Returns: binary barrier image where 1 = wall.
    """
    d = wall_density.copy()
    d[mask == 0] = 0
    
    # Normalize wall density to 0-255
    masked = d[mask > 0]
    if len(masked) == 0 or masked.max() == 0:
        return np.zeros_like(mask)
    
    d_norm = np.zeros_like(d)
    d_norm[mask > 0] = d[mask > 0] / masked.max() * 255
    d_uint8 = d_norm.astype(np.uint8)
    
    # Gaussian blur to smooth noise
    d_smooth = cv2.GaussianBlur(d_uint8, (5, 5), 1.5)
    
    # Canny edge detection with multiple thresholds, combine
    edges_low = cv2.Canny(d_smooth, 20, 60)
    edges_high = cv2.Canny(d_smooth, 40, 120)
    
    # Use Canny edges as primary signal — they're thin and clean on wall density
    # Augment with strong threshold pixels
    p85 = np.percentile(masked[masked > 0], 85) if len(masked[masked > 0]) > 0 else 1
    thresh_strong = ((d >= p85) & (mask > 0)).astype(np.uint8) * 255
    
    # Thin Canny edges + strong threshold peaks
    combined = edges_low | thresh_strong
    
    # Targeted directional closing ONLY along dominant wall angles
    # This bridges gaps along actual wall directions, not random angles
    barrier = combined.copy()
    
    # Detect dominant angles from Hough on the combined signal
    lines = cv2.HoughLinesP(combined, 1, np.pi/180, threshold=20, 
                            minLineLength=15, maxLineGap=5)
    wall_angles_deg = set()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
            # Snap to nearest 10°
            wall_angles_deg.add(round(a / 10) * 10)
    
    if not wall_angles_deg:
        wall_angles_deg = {0, 30, 60, 90, 120, 150}
    
    for angle_deg in wall_angles_deg:
        klen = 30  # ~0.6m bridging
        k = np.zeros((klen, klen), dtype=np.uint8)
        cx, cy = klen // 2, klen // 2
        rad = math.radians(angle_deg)
        for t in range(-klen//2, klen//2 + 1):
            x = int(cx + t * math.cos(rad))
            y = int(cy + t * math.sin(rad))
            if 0 <= x < klen and 0 <= y < klen:
                k[y, x] = 1
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
        barrier = barrier | closed
    
    # Minimal dilation — just 1px to ensure connectivity
    barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8), iterations=1)
    
    # Mask to apartment
    barrier = barrier & (mask * 255)
    
    barrier_frac = (barrier > 0).sum() / mask.sum() if mask.sum() > 0 else 0
    print(f"  Barrier coverage: {barrier_frac*100:.1f}% of apartment")
    
    return (barrier > 0).astype(np.uint8)


def find_rooms_flood(barrier, mask, min_room_m2=2.0, res=0.02):
    """
    Find rooms by flood filling non-barrier areas within the apartment mask.
    Uses distance transform to find seed points (room centers = far from walls).
    """
    interior = ((mask > 0) & (barrier == 0)).astype(np.uint8)
    
    # Connected components of interior = candidate rooms
    n_labels, labels = cv2.connectedComponents(interior)
    print(f"  Connected components (raw): {n_labels - 1}")
    
    min_px = int(min_room_m2 / (res * res))
    rooms = []
    for lbl in range(1, n_labels):
        room_mask = (labels == lbl).astype(np.uint8)
        area_px = room_mask.sum()
        if area_px >= min_px:
            rooms.append({'mask': room_mask, 'area_px': area_px})
    
    rooms.sort(key=lambda r: r['area_px'], reverse=True)
    print(f"  Rooms after area filter (>{min_room_m2}m²): {len(rooms)}")
    
    # Now expand rooms into barrier regions using dilation
    # Each room grows into adjacent barrier pixels, splitting walls between rooms
    all_rooms_mask = np.zeros_like(mask, dtype=np.int32)
    for i, room in enumerate(rooms):
        all_rooms_mask[room['mask'] > 0] = i + 1
    
    # Iterative dilation: each room grows by 1px at a time
    for _ in range(20):  # enough to fill ~0.4m of wall thickness
        for i, room in enumerate(rooms):
            dilated = cv2.dilate(room['mask'], np.ones((3, 3), np.uint8), iterations=1)
            # Only expand into unclaimed barrier pixels within apartment
            expand = (dilated > 0) & (all_rooms_mask == 0) & (mask > 0)
            room['mask'][expand] = 1
            room['area_px'] = room['mask'].sum()
            all_rooms_mask[expand] = i + 1
    
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
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
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


def render_debug(wall_density, all_density, mask, barrier, rooms, 
                 angles, transform, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].imshow(np.log1p(wall_density), cmap='hot', origin='lower')
    axes[0,0].set_title('Wall-only density (log)')
    
    # Wall density normalized
    d = wall_density.copy()
    d[mask == 0] = 0
    masked = d[mask > 0]
    if len(masked[masked > 0]) > 0:
        d_norm = d / masked.max() * 255
    else:
        d_norm = d
    d_uint8 = d_norm.astype(np.uint8)
    d_smooth = cv2.GaussianBlur(d_uint8, (5, 5), 1.5)
    edges = cv2.Canny(d_smooth, 20, 60)
    axes[0,1].imshow(edges, cmap='gray', origin='lower')
    axes[0,1].set_title('Canny edges on wall density')
    
    axes[0,2].imshow(barrier * 255, cmap='gray', origin='lower')
    axes[0,2].set_title(f'Wall barrier ({(barrier > 0).sum() / max(mask.sum(), 1) * 100:.1f}% coverage)')
    
    # Interior (non-barrier)
    interior = ((mask > 0) & (barrier == 0)).astype(np.uint8)
    axes[1,0].imshow(interior * 255, cmap='gray', origin='lower')
    axes[1,0].set_title('Interior (non-barrier)')
    
    # Room masks colored
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
    
    plt.suptitle('v42 Edge Contour Partition — Debug', fontsize=14)
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
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v42_edge_contour_partition'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    res = args.resolution

    print("\nStep 1: Create wall-only density image...")
    wall_density, all_density, transform = create_wall_density(mesh, res)

    print("\nStep 2: Apartment mask...")
    mask = get_apartment_mask(all_density)
    print(f"  Apartment area: {mask.sum() * res * res:.1f} m²")

    print("\nStep 3: Dominant wall angles...")
    angles = detect_dominant_angles(wall_density, mask)

    print("\nStep 4: Create wall barriers from Canny edges...")
    barrier = create_wall_barriers(wall_density, mask)

    print("\nStep 5: Flood fill rooms from barrier-free regions...")
    rooms = find_rooms_flood(barrier, mask, min_room_m2=2.0, res=res)

    print("\nStep 6: Polygon extraction with angle snap...")
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

    print("\nStep 7: Doors...")
    doors = detect_doors(rooms_valid)
    print(f"  {len(doors)} doors")

    print("\nStep 8: Rendering...")
    mesh_name = mesh_path.stem
    render_floorplan(rooms_valid, doors, transform, angles,
                     out_dir / f"v42_{mesh_name}_plan.png",
                     f"v42 Edge Contour Partition — {mesh_name}")
    render_debug(wall_density, all_density, mask, barrier, rooms_valid,
                 angles, transform, out_dir / f"v42_{mesh_name}_debug.png")

    shutil.copy2(out_dir / f"v42_{mesh_name}_plan.png",
                 Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')

    total_area = sum(r.get('area_m2', 0) for r in rooms_valid)
    results = {
        'approach': 'v42_edge_contour_partition',
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
    with open(out_dir / f"v42_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== v42 Summary ===")
    print(f"  Dominant angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    for r in results['rooms']:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {results['total_area_m2']}m², {len(doors)} doors")


if __name__ == '__main__':
    main()

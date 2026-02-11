#!/usr/bin/env python3
"""
mesh2plan v27g - Watershed + Distance Transform Room Detection

Strategy:
1. Project mid-height vertices to XZ, axis-align via dominant angle
2. Build density image at 2cm cells
3. Create room mask (threshold, morph close, fill holes)
4. Distance transform → room centers are far from walls
5. Find seeds as local maxima of distance transform (>0.5m threshold)
6. Watershed from seeds using negative distance as terrain
7. Hallway detection: narrow basins connecting 2+ rooms
8. Extract room polygons from basin contours, snap to Hough walls
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndlabel


# ─── Shared utilities ───

def detect_up_axis(mesh):
    ranges = [np.ptp(mesh.vertices[:, i]) for i in range(3)]
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
    return np.argmin(ranges), ['X','Y','Z'][np.argmin(ranges)]


def project_vertices(mesh, up_axis_idx):
    v = mesh.vertices
    if up_axis_idx == 1: return v[:, 0], v[:, 2]
    elif up_axis_idx == 2: return v[:, 0], v[:, 1]
    return v[:, 1], v[:, 2]


def find_dominant_angle(rx, rz, cell=0.02):
    x_min, x_max = rx.min(), rx.max()
    z_min, z_max = rz.min(), rz.max()
    nx = int((x_max - x_min) / cell) + 1
    nz = int((z_max - z_min) / cell) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell).astype(int), 0, nx - 1)
    zi = np.clip(((rz - z_min) / cell).astype(int), 0, nz - 1)
    np.add.at(img, (zi, xi), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    mask = mag > np.percentile(mag, 80)
    folded = ang[mask] % 90
    hist, bins = np.histogram(folded, bins=90, range=(0, 90))
    peak = np.argmax(hist)
    return (bins[peak] + bins[peak + 1]) / 2


def build_density_image(rx, rz, cell_size=0.02, margin=0.3):
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx - 1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz - 1)
    np.add.at(img, (zi, xi), 1)
    return img, x_min, z_min, cell_size


def hough_wall_positions(density_img, x_min, z_min, cell_size, nms_dist=0.25):
    """Detect dominant wall lines using projection histograms."""
    nz_img, nx_img = density_img.shape
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)

    # Project along rows (→ X walls) and columns (→ Z walls)
    proj_x = smoothed.sum(axis=0)  # sum each column → profile along X
    proj_z = smoothed.sum(axis=1)  # sum each row → profile along Z

    def find_peaks_nms(profile, origin, cs, min_dist_cells):
        # Smooth profile
        from scipy.ndimage import uniform_filter1d
        prof = uniform_filter1d(profile.astype(float), size=5)
        # Find local maxima
        local_max = maximum_filter(prof, size=max(3, min_dist_cells)) == prof
        # Threshold: above mean + 0.5*std
        threshold = prof.mean() + 0.3 * prof.std()
        peaks = np.where(local_max & (prof > threshold))[0]
        # Convert to world coords
        positions = origin + peaks * cs
        # NMS
        if len(positions) == 0:
            return np.array([])
        strengths = prof[peaks]
        order = np.argsort(-strengths)
        kept = []
        used = set()
        for i in order:
            pos = positions[i]
            if any(abs(pos - k) < nms_dist for k in kept):
                continue
            kept.append(pos)
        return np.sort(kept)

    min_dist = int(nms_dist / cell_size)
    x_walls = find_peaks_nms(proj_x, x_min, cell_size, min_dist)
    z_walls = find_peaks_nms(proj_z, z_min, cell_size, min_dist)
    return x_walls, z_walls


# ─── v27g core ───

def build_room_mask(density_img, cell_size):
    """Build binary room mask: occupied interior space."""
    occupied = (density_img > 1).astype(np.uint8)
    # Morphological close to bridge small gaps
    k_size = max(3, int(0.15 / cell_size)) | 1  # ~15cm kernel, ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    # Fill holes
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    # Keep only largest connected component
    lbl, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, lbl, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        filled = (lbl == largest).astype(np.uint8)
    return filled


def find_room_seeds(dist_transform, cell_size, min_dist_m=0.5, min_sep_m=1.0):
    """Find room seeds as local maxima of distance transform above threshold."""
    min_dist_px = int(min_dist_m / cell_size)
    min_sep_px = max(3, int(min_sep_m / cell_size))

    # Local max filter with large neighborhood
    local_max = maximum_filter(dist_transform, size=min_sep_px)
    is_max = (dist_transform == local_max) & (dist_transform >= min_dist_px)

    # Label connected components of maxima
    lbl, n = ndimage.label(is_max)
    seeds = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        cy, cx = int(ys.mean()), int(xs.mean())
        val = dist_transform[cy, cx]
        seeds.append((cy, cx, val))

    # Sort by distance value (largest rooms first)
    seeds.sort(key=lambda s: -s[2])
    print(f"  Found {len(seeds)} seed candidates (min_dist={min_dist_m}m)")
    for i, (sy, sx, sv) in enumerate(seeds):
        print(f"    Seed {i}: pixel ({sx},{sy}), dist={sv * cell_size:.2f}m")
    return seeds


def watershed_from_seeds(dist_transform, room_mask, seeds, cell_size):
    """Run watershed segmentation from seeds on negative distance transform."""
    nz, nx = dist_transform.shape
    markers = np.zeros((nz, nx), dtype=np.int32)

    # Place seed markers
    seed_radius = max(2, int(0.1 / cell_size))  # 10cm radius seeds
    for i, (sy, sx, _) in enumerate(seeds):
        cv2.circle(markers, (sx, sy), seed_radius, int(i + 1), -1)

    # Background marker outside room mask
    bg_label = len(seeds) + 1
    markers[room_mask == 0] = bg_label

    # Prepare terrain for watershed: use negative distance (rooms are basins)
    # Normalize to 8-bit for watershed
    if dist_transform.max() > 0:
        inv_dist = dist_transform.max() - dist_transform
        inv_norm = (inv_dist / inv_dist.max() * 255).clip(0, 255).astype(np.uint8)
    else:
        inv_norm = np.zeros_like(dist_transform, dtype=np.uint8)

    img_color = cv2.cvtColor(inv_norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)

    # Clean up: remove background, set boundaries to 0
    result = markers.copy()
    result[result == -1] = 0   # watershed boundaries
    result[result == bg_label] = 0  # background
    return result


def split_hallways_from_rooms(labeled, dist_transform, room_mask, seeds, cell_size,
                               hallway_max_half_width_m=0.65):
    """Post-process watershed: carve hallway using skeleton + distance analysis.
    
    Strategy:
    1. Compute skeleton of room mask
    2. Find skeleton pixels where distance transform < threshold (narrow passages)
    3. Grow these narrow skeleton pixels by the local distance value → hallway region
    4. Keep only connected components touching 2+ basins
    """
    from skimage.morphology import skeletonize
    n_seeds = len(seeds)
    thresh_px = hallway_max_half_width_m / cell_size
    
    # Skeleton of room mask
    skel = skeletonize(room_mask > 0).astype(np.uint8)
    print(f"    Skeleton: {np.sum(skel)} pixels")
    
    # Narrow skeleton = skeleton pixels where dist < threshold (hallway spine)
    narrow_skel = (skel > 0) & (dist_transform < thresh_px) & (dist_transform > 0)
    narrow_skel = narrow_skel.astype(np.uint8)
    print(f"    Narrow skeleton: {np.sum(narrow_skel)} pixels")
    
    # Connect nearby narrow skeleton segments
    k_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    narrow_connected = cv2.dilate(narrow_skel, k_connect)
    narrow_connected = cv2.morphologyEx(narrow_connected, cv2.MORPH_CLOSE, 
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    
    # Grow narrow skeleton by threshold distance to create hallway region
    # Use distance transform from narrow skeleton
    grow_dist = cv2.distanceTransform(1 - narrow_connected, cv2.DIST_L2, 5)
    # Each pixel is in hallway if it's closer to narrow skeleton than to wall
    hallway_mask = ((grow_dist < dist_transform) & (room_mask > 0) & 
                    (dist_transform < thresh_px)).astype(np.uint8)
    
    # Also include the ridge itself
    ridge = ((labeled == 0) & (room_mask > 0)).astype(np.uint8)
    # Grow ridge slightly
    ridge_grown = cv2.dilate(ridge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    hallway_mask = hallway_mask | (ridge_grown & (dist_transform < thresh_px).astype(np.uint8) & room_mask)
    
    # Clean up
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hallway_mask = cv2.morphologyEx(hallway_mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    hallway_mask = cv2.morphologyEx(hallway_mask, cv2.MORPH_CLOSE, k_close)
    
    total_area = np.sum(hallway_mask) * cell_size * cell_size
    print(f"    Hallway mask area: {total_area:.1f}m²")
    
    # Label components
    corr_lbl, n_corr = ndimage.label(hallway_mask)
    
    new_labeled = labeled.copy()
    next_label = n_seeds + 1
    hallway_info = []
    
    for i in range(1, n_corr + 1):
        region = (corr_lbl == i)
        area_m2 = np.sum(region) * cell_size * cell_size
        if area_m2 < 1.0:
            continue
        
        # Check which basins it touches
        touching = set()
        dilated = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8))
        for lbl in np.unique(labeled[dilated > 0]):
            if 0 < lbl <= n_seeds:
                touching.add(int(lbl))
        
        region_dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        max_width = region_dist.max() * 2 * cell_size
        
        print(f"    Corridor {i}: area={area_m2:.1f}m² max_w={max_width:.2f}m touches={touching}")
        
        if len(touching) >= 2 and area_m2 >= 1.5:
            new_labeled[region] = next_label
            hallway_info.append({
                'label': next_label,
                'area_m2': area_m2,
                'max_width': max_width,
                'connects': touching,
            })
            next_label += 1
    
    return new_labeled, hallway_info


def classify_basins(labeled, hallway_labels, seeds, cell_size):
    """Classify all basins (original + carved hallways)."""
    all_labels = set(np.unique(labeled)) - {0, -1}
    hallway_label_set = {h['label'] for h in hallway_labels}
    
    room_info = []
    for label in sorted(all_labels):
        mask = (labeled == label).astype(np.uint8)
        area_px = np.sum(mask)
        if area_px == 0:
            continue
        area_m2 = area_px * cell_size * cell_size
        if area_m2 < 0.5:
            continue

        basin_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_width = basin_dist.max() * 2 * cell_size
        median_width = np.median(basin_dist[basin_dist > 0]) * 2 * cell_size if np.any(basin_dist > 0) else 0

        rows, cols = np.where(mask > 0)
        bbox_h = (rows.max() - rows.min() + 1) * cell_size
        bbox_w = (cols.max() - cols.min() + 1) * cell_size
        aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 0.01)

        is_hallway = label in hallway_label_set

        # Find seed if this is an original basin
        seed = None
        dist_val = 0
        for i, (sy, sx, sv) in enumerate(seeds):
            if i + 1 == label:
                seed = (sy, sx)
                dist_val = sv * cell_size
                break

        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        neighbors = set()
        for lbl in np.unique(labeled[dilated > 0]):
            if lbl > 0 and lbl != label:
                neighbors.add(int(lbl))

        room_info.append({
            'label': int(label),
            'seed': seed,
            'dist_val': float(dist_val),
            'area_m2': float(area_m2),
            'max_width': float(max_width),
            'median_width': float(median_width),
            'aspect': float(aspect),
            'is_hallway': bool(is_hallway),
            'neighbors': neighbors,
        })
        kind = "HALLWAY" if is_hallway else "ROOM"
        print(f"    Basin {label}: {kind} area={area_m2:.1f}m² med_w={median_width:.2f}m "
              f"aspect={aspect:.1f} neighbors={neighbors}")
    return room_info


def extract_basin_polygon(mask, x_min, z_min, cell_size, x_walls, z_walls, epsilon_m=0.25):
    """Extract polygon from basin mask, snap to wall positions, force axis-alignment."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # Largest contour
    contour = max(contours, key=cv2.contourArea)

    # Douglas-Peucker simplification
    eps_px = epsilon_m / cell_size
    simplified = cv2.approxPolyDP(contour, eps_px, True)

    # Convert to world coordinates
    pts = []
    for pt in simplified:
        px, py = pt[0]
        wx = x_min + px * cell_size
        wz = z_min + py * cell_size
        pts.append([wx, wz])

    if len(pts) < 3:
        return pts

    # Snap to nearest wall positions
    def snap_to_walls(val, walls, max_snap=0.3):
        if len(walls) == 0:
            return val
        dists = np.abs(walls - val)
        idx = np.argmin(dists)
        if dists[idx] < max_snap:
            return float(walls[idx])
        return val

    snapped = []
    for x, z in pts:
        sx = snap_to_walls(x, x_walls)
        sz = snap_to_walls(z, z_walls)
        snapped.append([sx, sz])

    # Force axis-alignment: each edge is horizontal or vertical
    aligned = make_rectilinear(snapped, cell_size)

    # Remove tiny jogs
    cleaned = remove_tiny_jogs(aligned, min_len=0.3)

    return cleaned


def make_rectilinear(pts, cell_size):
    """Force polygon to be rectilinear by inserting staircase points."""
    if len(pts) < 3:
        return pts
    result = []
    n = len(pts)
    for i in range(n):
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        result.append(cur[:])
        dx = abs(nxt[0] - cur[0])
        dz = abs(nxt[1] - cur[1])
        # If diagonal, add corner point
        if dx > 0.05 and dz > 0.05:
            # Choose staircase direction: go horizontal first
            result.append([nxt[0], cur[1]])
    return result


def remove_tiny_jogs(pts, min_len=0.3):
    """Remove tiny jog edges from a polygon."""
    if len(pts) < 4:
        return pts
    changed = True
    while changed:
        changed = False
        new_pts = []
        i = 0
        while i < len(pts):
            p = pts[i]
            nxt = pts[(i + 1) % len(pts)]
            edge_len = abs(p[0] - nxt[0]) + abs(p[1] - nxt[1])
            if edge_len < min_len and len(pts) > 4:
                # Merge: skip this vertex, average into next
                mid = [(p[0] + nxt[0]) / 2, (p[1] + nxt[1]) / 2]
                new_pts.append(mid)
                i += 2
                changed = True
            else:
                new_pts.append(p)
                i += 1
        pts = new_pts
        if len(pts) < 4:
            break
    return pts


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3:
        return 0
    area = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    return abs(area) / 2


ROOM_COLORS = [
    '#4A90D9', '#E8834A', '#67B868', '#C75B8F', '#8B6CC1',
    '#D4A843', '#4ABFBF', '#D96060', '#7B8FD4', '#A0C75B',
]


def analyze_mesh(mesh_file):
    print(f"\n{'='*60}")
    print(f"v27g: Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    up_idx, up_name = detect_up_axis(mesh)
    up_coords = mesh.vertices[:, up_idx]
    up_min, up_range = up_coords.min(), np.ptp(up_coords)
    x_raw, z_raw = project_vertices(mesh, up_idx)
    hmask = (up_coords >= up_min + up_range * 0.15) & (up_coords <= up_min + up_range * 0.85)
    x_mid, z_mid = x_raw[hmask], z_raw[hmask]

    angle = find_dominant_angle(x_mid, z_mid)
    angle_rad = angle * math.pi / 180
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    print(f"  Rotation: {angle:.1f}°, {up_name}-up")

    cell_size = 0.02
    print(f"  Building density image (cell={cell_size}m)...")
    density_img, img_x_min, img_z_min, cs = build_density_image(rx, rz, cell_size=cell_size)

    # Step 3: Room mask
    print("  Building room mask...")
    room_mask = build_room_mask(density_img, cs)
    room_area = np.sum(room_mask) * cs * cs
    print(f"    Room mask area: {room_area:.1f} m²")

    # Step 4: Distance transform
    print("  Computing distance transform...")
    dist_transform = cv2.distanceTransform(room_mask, cv2.DIST_L2, 5)
    max_dist_m = dist_transform.max() * cs
    print(f"    Max distance: {max_dist_m:.2f}m")

    # Step 5: Find seeds — room centers only (high threshold)
    print("  Finding room seeds...")
    seeds = find_room_seeds(dist_transform, cs, min_dist_m=0.5, min_sep_m=1.5)
    if len(seeds) < 2 and max_dist_m > 1.0:
        print("  Retrying with lower threshold...")
        seeds = find_room_seeds(dist_transform, cs, min_dist_m=0.3, min_sep_m=1.0)

    if len(seeds) == 0:
        print("  WARNING: No seeds found, falling back to single room")
        # Use center of mass as single seed
        ys, xs = np.where(room_mask > 0)
        if len(ys) > 0:
            seeds = [(int(ys.mean()), int(xs.mean()), dist_transform[int(ys.mean()), int(xs.mean())])]

    # Step 6: Watershed
    print("  Running watershed...")
    labeled = watershed_from_seeds(dist_transform, room_mask, seeds, cs)

    # Step 7: Split hallways from rooms (only if multiple rooms)
    hallway_info = []
    if len(seeds) >= 2:
        print("  Splitting hallways from rooms...")
        labeled, hallway_info = split_hallways_from_rooms(labeled, dist_transform, room_mask, seeds, cs)
    else:
        print("  Single room detected, skipping hallway detection")
    print("  Classifying basins...")
    basin_info = classify_basins(labeled, hallway_info, seeds, cs)

    # Step 8: Hough wall positions for snapping
    print("  Detecting wall positions...")
    x_walls, z_walls = hough_wall_positions(density_img, img_x_min, img_z_min, cs)
    print(f"    X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"    Z-walls: {[f'{w:.2f}' for w in z_walls]}")

    # Step 9: Extract polygons
    print("  Extracting room polygons...")
    rooms = []
    for info in basin_info:
        if info['area_m2'] < 0.5:
            continue
        mask = (labeled == info['label']).astype(np.uint8)
        poly = extract_basin_polygon(mask, img_x_min, img_z_min, cs, x_walls, z_walls)
        poly_area = compute_polygon_area(poly) if len(poly) >= 3 else 0

        kind = "Hallway" if info['is_hallway'] else "Room"
        idx = len(rooms) + 1
        name = f"{kind} {idx}" if not info['is_hallway'] else "Hallway"

        rooms.append({
            'label': info['label'],
            'polygon_rot': poly,
            'area': info['area_m2'],
            'poly_area': poly_area,
            'name': name,
            'is_hallway': info['is_hallway'],
            'max_width': info['max_width'],
            'median_width': info['median_width'],
        })
        print(f"    {name}: {info['area_m2']:.1f}m² (poly: {poly_area:.1f}m², {len(poly)} verts)")

    total_area = sum(r['area'] for r in rooms)

    # Fine edge map for overlay
    density_fine, fx_min, fz_min, fcs = build_density_image(rx, rz, cell_size=0.01)
    from scipy import ndimage as ndi

    print(f"\n=== v27g Summary ===")
    print(f"  Spaces: {len(rooms)} ({sum(1 for r in rooms if not r['is_hallway'])} rooms + {sum(1 for r in rooms if r['is_hallway'])} hallways)")
    print(f"  Total area: {total_area:.1f} m²")
    for r in rooms:
        print(f"    {r['name']}: {r['area']:.1f} m²")

    return {
        'rooms': rooms, 'total_area': total_area,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'density_img': density_img, 'dist_transform': dist_transform,
        'room_mask': room_mask, 'labeled': labeled,
        'seeds': seeds, 'basin_info': basin_info,
        'x_walls': x_walls, 'z_walls': z_walls,
        'img_origin': (img_x_min, img_z_min, cs),
        'fine_density': density_fine,
        'fine_origin': (fx_min, fz_min, fcs),
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    ix_min, iz_min, cs = results['img_origin']
    fx_min, fz_min, fcs = results['fine_origin']

    # Panel 1: Density image
    density = results['density_img']
    d_display = density.copy()
    if d_display.max() > 0:
        d_display = (d_display / np.percentile(d_display[d_display > 0], 95)).clip(0, 1)
    axes[0].imshow(d_display, cmap='hot', origin='lower',
                   extent=[ix_min, ix_min + density.shape[1] * cs,
                           iz_min, iz_min + density.shape[0] * cs])
    axes[0].set_title('1. Density Image (2cm)', color='white', fontsize=14)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Z (m)')

    # Panel 2: Distance transform
    dist = results['dist_transform'] * cs  # convert to meters
    axes[1].imshow(dist, cmap='magma', origin='lower',
                   extent=[ix_min, ix_min + density.shape[1] * cs,
                           iz_min, iz_min + density.shape[0] * cs])
    # Mark seeds
    for sy, sx, sv in results['seeds']:
        wx = ix_min + sx * cs
        wz = iz_min + sy * cs
        axes[1].plot(wx, wz, 'c*', markersize=12, markeredgecolor='white')
        axes[1].annotate(f'{sv * cs:.2f}m', (wx, wz), color='cyan', fontsize=8,
                        textcoords='offset points', xytext=(5, 5))
    axes[1].set_title('2. Distance Transform + Seeds', color='white', fontsize=14)
    axes[1].set_xlabel('X (m)')

    # Panel 3: Watershed segmentation
    ax2 = axes[2]
    labeled = results['labeled']
    seg_img = np.zeros((*labeled.shape, 3), dtype=np.float32)
    for i, room in enumerate(results['rooms']):
        ch = ROOM_COLORS[i % len(ROOM_COLORS)]
        r, g, b = int(ch[1:3], 16) / 255, int(ch[3:5], 16) / 255, int(ch[5:7], 16) / 255
        seg_img[labeled == room['label']] = [r, g, b]
    ax2.imshow(seg_img, origin='lower',
               extent=[ix_min, ix_min + density.shape[1] * cs,
                       iz_min, iz_min + density.shape[0] * cs])
    for room in results['rooms']:
        mask = labeled == room['label']
        rows, cols = np.where(mask)
        if len(rows) > 0:
            cx = ix_min + np.mean(cols) * cs
            cz = iz_min + np.mean(rows) * cs
            ax2.text(cx, cz, f"{room['name']}\n{room['area']:.1f}m²",
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    n_rooms = sum(1 for r in results['rooms'] if not r['is_hallway'])
    n_hall = sum(1 for r in results['rooms'] if r['is_hallway'])
    ax2.set_title(f'3. Watershed ({n_rooms}R + {n_hall}H)', color='white', fontsize=14)
    ax2.set_xlabel('X (m)')

    # Panel 4: Edge + polygon overlay
    ax3 = axes[3]
    ax3.set_aspect('equal')
    ax3.set_facecolor('black')
    # Show fine density as background
    fine = results['fine_density']
    if fine.max() > 0:
        f_display = (fine / np.percentile(fine[fine > 0], 95)).clip(0, 1)
    else:
        f_display = fine
    ax3.imshow(f_display, cmap='gray', origin='lower', alpha=0.4,
               extent=[fx_min, fx_min + fine.shape[1] * fcs,
                       fz_min, fz_min + fine.shape[0] * fcs])
    # Draw wall lines
    x_walls = results['x_walls']
    z_walls = results['z_walls']
    ext_z = [iz_min, iz_min + density.shape[0] * cs]
    ext_x = [ix_min, ix_min + density.shape[1] * cs]
    for xw in x_walls:
        ax3.axvline(xw, color='yellow', alpha=0.3, linewidth=0.5)
    for zw in z_walls:
        ax3.axhline(zw, color='yellow', alpha=0.3, linewidth=0.5)

    # Draw polygons
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3:
            continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs = [p[0] for p in pc]
        zs = [p[1] for p in pc]
        ax3.fill(xs, zs, color=color, alpha=0.15)
        ax3.plot(xs, zs, color=color, linewidth=2.5, alpha=0.9)
    ax3.set_title('4. Walls + Polygon Overlay', color='white', fontsize=14)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')

    # Panel 5: Final floor plan
    ax4 = axes[4]
    ax4.set_aspect('equal')
    ax4.set_facecolor('#1a1a2e')
    all_x, all_z = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3:
            continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs = [p[0] for p in pc]
        zs = [p[1] for p in pc]
        ax4.fill(xs, zs, color=color, alpha=0.3)
        all_x.extend(xs)
        all_z.extend(zs)
        # Draw walls
        for j in range(len(poly)):
            k = (j + 1) % len(poly)
            ax4.plot([poly[j][0], poly[k][0]], [poly[j][1], poly[k][1]],
                    color='white', linewidth=3, solid_capstyle='round')
            # Dimension label on longer edges
            edge_len = math.sqrt((poly[k][0] - poly[j][0])**2 + (poly[k][1] - poly[j][1])**2)
            if edge_len > 0.5:
                mx = (poly[j][0] + poly[k][0]) / 2
                mz = (poly[j][1] + poly[k][1]) / 2
                ax4.text(mx, mz, f'{edge_len:.1f}m', ha='center', va='center',
                        color='#aaaaaa', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
        # Room label
        cx = sum(p[0] for p in poly) / len(poly)
        cz = sum(p[1] for p in poly) / len(poly)
        label_text = f"{room['name']}\n{room['area']:.1f}m²"
        ax4.text(cx, cz, label_text,
                ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))

    ax4.set_title(f'5. v27g Floor Plan — {len(results["rooms"])} spaces, {results["total_area"]:.1f}m²',
                  color='white', fontsize=14)
    ax4.grid(True, alpha=0.15, color='gray')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Z (m)')
    if all_x:
        m = 0.5
        ax4.set_xlim(min(all_x) - m, max(all_x) + m)
        ax4.set_ylim(min(all_z) - m, max(all_z) + m)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27g_watershed_distxform',
            'num_rooms': sum(1 for r in results['rooms'] if not r['is_hallway']),
            'num_hallways': sum(1 for r in results['rooms'] if r['is_hallway']),
            'num_spaces': len(results['rooms']),
            'total_area_m2': round(results['total_area'], 1),
        },
        'rooms': [{
            'name': r['name'],
            'area_m2': round(r['area'], 1),
            'is_hallway': bool(r['is_hallway']),
            'polygon': [[round(p[0], 3), round(p[1], 3)] for p in r['polygon_rot']],
        } for r in results['rooms']],
        'walls': {
            'x_positions': [round(w, 3) for w in results['x_walls']],
            'z_positions': [round(w, 3) for w in results['z_walls']],
        }
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27g - Watershed + Distance Transform Rooms')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27g/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27g_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v57c - Hybrid Cross-Section: Raster Wall Detection + Vector Room Partition

Combines:
- Cross-section slicing → rasterize segments → clean wall image (like v56b)
- Hough line detection on wall image → dominant wall lines (like v52)
- Shapely polygonize for room partition (like v57) → clean vector rooms

Key insight: Rasterization is great for noise filtering (tiny mesh triangulation
segments become coherent wall pixels). But room partition should be vector-based
for clean straight walls.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import cv2
from scipy import ndimage
from shapely.geometry import LineString, Polygon, MultiPoint, Point
from shapely.ops import polygonize, unary_union


def slice_mesh_to_image(mesh, heights, resolution=0.02):
    """Slice mesh at multiple heights, rasterize to binary image."""
    all_points = []
    for h in heights:
        try:
            segs = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, h, 0])
            if len(segs) > 0:
                pts = segs[:, :, [0, 2]].reshape(-1, 2)
                all_points.append(pts)
        except:
            pass
    
    if not all_points:
        return None, None, None
    
    pts = np.concatenate(all_points, axis=0)
    
    # Compute bounds
    x_min, z_min = pts.min(axis=0) - 0.5
    x_max, z_max = pts.max(axis=0) + 0.5
    
    # Create image
    w = int((x_max - x_min) / resolution) + 1
    h = int((z_max - z_min) / resolution) + 1
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Rasterize
    px = ((pts[:, 0] - x_min) / resolution).astype(int)
    py = ((pts[:, 1] - z_min) / resolution).astype(int)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    img[py, px] = 255
    
    # Transform: pixel (px, py) → world (x_min + px*res, z_min + py*res)
    transform = {'x_min': x_min, 'z_min': z_min, 'resolution': resolution}
    return img, pts, transform


def pixel_to_world(px, py, transform):
    x = transform['x_min'] + px * transform['resolution']
    z = transform['z_min'] + py * transform['resolution']
    return x, z


def world_to_pixel(x, z, transform):
    px = (x - transform['x_min']) / transform['resolution']
    py = (z - transform['z_min']) / transform['resolution']
    return px, py


def detect_wall_lines(wall_img, transform, angle_tol=15, min_line_length=30, max_line_gap=15):
    """Detect wall lines using Hough transform on binary wall image."""
    # Dilate slightly to connect nearby pixels
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(wall_img, kernel, iterations=1)
    
    # Standard Hough
    lines = cv2.HoughLines(dilated, rho=1, theta=np.pi/180, threshold=40)
    
    if lines is None:
        return [], 0, 0
    
    lines = lines[:, 0, :]  # (n, 2) - rho, theta
    print(f"  Raw Hough lines: {len(lines)}")
    
    # Find dominant angle pair from theta histogram
    thetas_deg = np.degrees(lines[:, 1])
    
    # Weighted histogram
    hist, bin_edges = np.histogram(thetas_deg, bins=180, range=(0, 180))
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3, mode='wrap')
    
    peak1_idx = np.argmax(hist_smooth)
    angle1 = bin_edges[peak1_idx] + 0.5
    
    # Suppress near peak1
    hist_masked = hist_smooth.copy()
    for i in range(180):
        diff = min(abs(i - peak1_idx), 180 - abs(i - peak1_idx))
        if diff < 20:
            hist_masked[i] = 0
    peak2_idx = np.argmax(hist_masked)
    angle2 = bin_edges[peak2_idx] + 0.5
    
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1
    
    print(f"  Dominant angles: {angle1:.0f}° and {angle2:.0f}°")
    
    # Filter lines to dominant angles
    filtered = []
    for rho, theta in lines:
        td = np.degrees(theta)
        diff1 = min(abs(td - angle1), 180 - abs(td - angle1))
        diff2 = min(abs(td - angle2), 180 - abs(td - angle2))
        if diff1 < angle_tol or diff2 < angle_tol:
            filtered.append((rho, theta))
    
    print(f"  Filtered to dominant angles: {len(filtered)} lines")
    return filtered, angle1, angle2


def score_wall_line(rho, theta, wall_img, sample_length=None):
    """Score a wall line by how much wall density it passes through."""
    h, w = wall_img.shape
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Generate points along line within image bounds
    points = []
    for t in range(-max(h, w), max(h, w)):
        x = int(rho * cos_t + t * (-sin_t))
        y = int(rho * sin_t + t * cos_t)
        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))
    
    if not points:
        return 0, 0
    
    # Count wall pixels along line
    wall_count = sum(1 for x, y in points if wall_img[y, x] > 0)
    total = len(points)
    
    # Also compute max consecutive run
    max_run = 0
    current_run = 0
    for x, y in points:
        if wall_img[y, x] > 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    
    return wall_count, max_run


def cluster_and_select_walls(lines, wall_img, angle1, angle2, 
                              min_gap_px=12, max_walls_per_family=6):
    """Cluster lines by rho, score by wall density, select best walls."""
    
    # Separate by angle family
    fam1 = []
    fam2 = []
    for rho, theta in lines:
        td = np.degrees(theta)
        diff1 = min(abs(td - angle1), 180 - abs(td - angle1))
        diff2 = min(abs(td - angle2), 180 - abs(td - angle2))
        if diff1 < diff2:
            fam1.append((rho, theta))
        else:
            fam2.append((rho, theta))
    
    selected = []
    for fam_name, fam, angle in [("Fam1", fam1, angle1), ("Fam2", fam2, angle2)]:
        if not fam:
            continue
        
        # Sort by rho
        fam.sort(key=lambda x: x[0])
        rhos = np.array([r for r, t in fam])
        
        # Cluster by rho gap
        clusters = []
        current = [fam[0]]
        for i in range(1, len(fam)):
            if rhos[i] - rhos[i-1] > min_gap_px:
                clusters.append(current)
                current = []
            current.append(fam[i])
        clusters.append(current)
        
        # For each cluster, pick representative and score
        wall_candidates = []
        for cluster in clusters:
            # Average rho and theta
            avg_rho = np.mean([r for r, t in cluster])
            avg_theta = np.mean([t for r, t in cluster])
            
            density, max_run = score_wall_line(avg_rho, avg_theta, wall_img)
            score = density * np.sqrt(max_run + 1)
            wall_candidates.append((avg_rho, avg_theta, score, density, max_run))
        
        # Sort by score, take top N
        wall_candidates.sort(key=lambda x: x[2], reverse=True)
        top = wall_candidates[:max_walls_per_family]
        
        print(f"  {fam_name} ({angle:.0f}°): {len(clusters)} clusters → {len(top)} walls")
        for r, t, s, d, mr in top:
            print(f"    ρ={r:.0f}, score={s:.0f} (density={d}, run={mr})")
        
        selected.extend([(r, t) for r, t, s, d, mr in top])
    
    return selected


def hough_line_to_shapely(rho, theta, transform, extent=500):
    """Convert Hough line (pixel space) to Shapely LineString (world space)."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Point on line in pixel coords
    x0 = rho * cos_t
    y0 = rho * sin_t
    
    # Direction along line
    dx = -sin_t
    dy = cos_t
    
    # Two endpoints far apart
    px1, py1 = x0 - extent * dx, y0 - extent * dy
    px2, py2 = x0 + extent * dx, y0 + extent * dy
    
    # Convert to world coords
    wx1, wz1 = pixel_to_world(px1, py1, transform)
    wx2, wz2 = pixel_to_world(px2, py2, transform)
    
    return LineString([(wx1, wz1), (wx2, wz2)])


def make_apartment_boundary(wall_img, transform, threshold=10):
    """Create apartment boundary polygon from wall image."""
    # Dilate walls heavily to merge
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(wall_img, kernel, iterations=3)
    
    # Fill interior
    filled = ndimage.binary_fill_holes(dilated > 0).astype(np.uint8) * 255
    
    # Erode back
    filled = cv2.erode(filled, np.ones((15, 15), np.uint8), iterations=2)
    
    # Find largest contour
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    
    # Simplify and convert to world coords
    epsilon = 5.0
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    coords = []
    for pt in approx[:, 0]:
        wx, wz = pixel_to_world(pt[0], pt[1], transform)
        coords.append((wx, wz))
    
    if len(coords) < 3:
        return None
    
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def classify_room(area, perimeter):
    """Classify room by area and compactness."""
    compactness = perimeter ** 2 / (4 * np.pi * area) if area > 0 else 0
    if compactness > 2.5 and area < 8:
        return "Hallway"
    if area > 8:
        return "Room"
    elif area > 4:
        return "Bathroom"
    elif area > 2:
        return "Closet"
    else:
        return "Closet"


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v57c')
    parser.add_argument('mesh_path', help='Path to mesh file')
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--resolution', type=float, default=0.02, help='Raster resolution (m/px)')
    parser.add_argument('--min-room-area', type=float, default=1.5)
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path('results/v57c_hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    y_floor = mesh.vertices[:, 1].min()
    print(f"  Floor at Y={y_floor:.2f}")
    
    # Multi-height slicing
    heights = np.linspace(y_floor + 0.8, y_floor + 1.5, 8)
    print(f"\nSlicing at 8 heights ({heights[0]:.2f} to {heights[-1]:.2f})...")
    wall_img, pts, transform = slice_mesh_to_image(mesh, heights, resolution=args.resolution)
    
    if wall_img is None:
        print("ERROR: No cross-section found")
        return
    
    print(f"  Image size: {wall_img.shape[1]}x{wall_img.shape[0]}")
    print(f"  Wall pixels: {np.count_nonzero(wall_img)}")
    
    # Also create normal-filtered wall density for better wall detection
    print("\nCreating normal-filtered wall density...")
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.5  # Horizontal normals = walls
    wall_faces = np.where(wall_mask)[0]
    print(f"  Wall faces: {len(wall_faces)}/{len(mesh.faces)} ({100*len(wall_faces)/len(mesh.faces):.0f}%)")
    
    # Wall face centroids projected to XZ
    centroids = mesh.triangles_center[wall_faces]
    cx = centroids[:, 0]
    cz = centroids[:, 2]
    
    # Rasterize wall density
    res = transform['resolution']
    xmin = transform['x_min']
    zmin = transform['z_min']
    h, w = wall_img.shape
    
    density = np.zeros((h, w), dtype=np.float32)
    dpx = ((cx - xmin) / res).astype(int)
    dpy = ((cz - zmin) / res).astype(int)
    valid = (dpx >= 0) & (dpx < w) & (dpy >= 0) & (dpy < h)
    for px, py in zip(dpx[valid], dpy[valid]):
        density[py, px] += 1
    
    # Combine cross-section + wall density
    # Normalize density
    if density.max() > 0:
        density_norm = (density / np.percentile(density[density > 0], 95) * 255).clip(0, 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(wall_img)
    
    combined_wall = np.maximum(wall_img, density_norm)
    
    # Detect wall lines
    print("\nDetecting wall lines...")
    lines, angle1, angle2 = detect_wall_lines(combined_wall, transform)
    
    if not lines:
        print("ERROR: No wall lines detected")
        return
    
    # Score and select walls
    print("\nScoring and selecting walls...")
    selected_walls = cluster_and_select_walls(lines, combined_wall, angle1, angle2,
                                                min_gap_px=12)
    print(f"  Selected {len(selected_walls)} walls total")
    
    # Create apartment boundary
    print("\nCreating apartment boundary...")
    boundary = make_apartment_boundary(combined_wall, transform)
    if boundary is None:
        print("ERROR: Could not create boundary")
        return
    print(f"  Boundary area: {boundary.area:.1f}m²")
    
    # Convert walls to Shapely lines, clip to boundary
    wall_lines_shapely = []
    for rho, theta in selected_walls:
        line = hough_line_to_shapely(rho, theta, transform)
        clipped = line.intersection(boundary)
        if not clipped.is_empty:
            wall_lines_shapely.append(clipped)
    
    print(f"  {len(wall_lines_shapely)} wall lines after clipping")
    
    # Partition into rooms
    print("\nPartitioning into rooms...")
    all_lines = wall_lines_shapely + [boundary.boundary]
    merged = unary_union(all_lines)
    result_polys = list(polygonize(merged))
    
    # Filter rooms
    rooms = []
    for poly in result_polys:
        if poly.area < args.min_room_area:
            continue
        if boundary.contains(poly.centroid) or boundary.intersection(poly).area > 0.5 * poly.area:
            rooms.append(poly)
    
    rooms.sort(key=lambda r: r.area, reverse=True)
    total_area = sum(r.area for r in rooms)
    
    print(f"  Found {len(rooms)} rooms, total {total_area:.1f}m²")
    
    room_data = []
    for i, room in enumerate(rooms):
        area = room.area
        perim = room.length
        name = classify_room(area, perim)
        n_verts = len(room.exterior.coords) - 1
        cx, cy = room.centroid.x, room.centroid.y
        compactness = perim ** 2 / (4 * np.pi * area) if area > 0 else 0
        print(f"  {name}: {area:.1f}m² ({n_verts}v, compact={compactness:.1f})")
        room_data.append({'name': name, 'area': area, 'vertices': n_verts,
                         'centroid': [cx, cy], 'compactness': compactness})
    
    # === PLOTTING ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel 1: Raw cross-section
    ax = axes[0, 0]
    ax.set_title('Raw Cross-Section')
    ax.imshow(wall_img, cmap='gray', origin='lower')
    ax.set_aspect('equal')
    
    # Panel 2: Wall density (normal-filtered)
    ax = axes[0, 1]
    ax.set_title('Wall Density (normal-filtered)')
    ax.imshow(density_norm, cmap='hot', origin='lower')
    ax.set_aspect('equal')
    
    # Panel 3: Combined wall image
    ax = axes[0, 2]
    ax.set_title('Combined Wall Image')
    ax.imshow(combined_wall, cmap='gray', origin='lower')
    ax.set_aspect('equal')
    
    # Panel 4: Detected wall lines on image
    ax = axes[1, 0]
    ax.set_title(f'Wall Lines ({len(selected_walls)} walls)')
    ax.imshow(combined_wall, cmap='gray', origin='lower', alpha=0.5)
    for rho, theta in selected_walls:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0 = rho * cos_t
        y0 = rho * sin_t
        dx = -sin_t * 500
        dy = cos_t * 500
        ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], 'r-', lw=1.5)
    ax.set_xlim(0, wall_img.shape[1])
    ax.set_ylim(0, wall_img.shape[0])
    ax.set_aspect('equal')
    
    # Panel 5: Boundary + wall lines (world coords)
    ax = axes[1, 1]
    ax.set_title('Boundary + Walls (world)')
    ax.set_aspect('equal')
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', lw=2)
    for line in wall_lines_shapely:
        if hasattr(line, 'geoms'):
            for g in line.geoms:
                x, y = g.xy
                ax.plot(x, y, 'r-', lw=1.5)
        else:
            x, y = line.xy
            ax.plot(x, y, 'r-', lw=1.5)
    
    # Panel 6: Room polygons
    ax = axes[1, 2]
    colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    ax.set_title(f'v57c — {len(rooms)} rooms, {total_area:.1f}m²\nAngles: {angle1:.0f}°, {angle2:.0f}°')
    ax.set_aspect('equal')
    
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', lw=2)
    
    for i, room in enumerate(rooms):
        x, y = room.exterior.xy
        ax.fill(x, y, color=colors[i % len(colors)], alpha=0.6)
        ax.plot(x, y, 'k-', lw=1.5)
        cx, cy = room.centroid.x, room.centroid.y
        name = room_data[i]['name']
        area = room_data[i]['area']
        ax.text(cx, cy, f'{name}\n{area:.1f}m²', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Scale bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([xlim[0] + 0.5, xlim[0] + 1.5], [ylim[0] + 0.3, ylim[0] + 0.3], 'k-', lw=3)
    ax.text(xlim[0] + 1.0, ylim[0] + 0.15, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'floorplan.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'floorplan.png'}")
    
    # Save JSON
    result = {
        'version': 'v57c_hybrid_xsection',
        'angles': [angle1, angle2],
        'num_walls': len(selected_walls),
        'num_rooms': len(rooms),
        'total_area': total_area,
        'rooms': room_data
    }
    with open(output_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()

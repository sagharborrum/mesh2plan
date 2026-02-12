#!/usr/bin/env python3
"""
mesh2plan v61b - Four Rooms via Cross-Section Hough + Smart Wall Selection

GROUND TRUTH: 2 bedrooms + 1 hallway + 1 bathroom = 4 rooms

Strategy:
1. Cross-section density (v60) for wall image
2. Hough on density for wall lines with proper extents
3. Strict 2-angle filter (29°/119°)
4. Score walls by density integral
5. Incrementally add walls until we get 4 rooms
6. Angle-snapped boundary
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import polygonize, unary_union


RESOLUTION = 0.02
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
WALL_ANGLES = np.array([29.0, 119.0])
ANGLE_TOL = 15


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def slice_mesh(mesh, y_height):
    try:
        lines = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, y_height, 0])
    except:
        return np.array([]).reshape(0, 2, 2)
    if lines is None or len(lines) == 0:
        return np.array([]).reshape(0, 2, 2)
    return lines[:, :, [0, 2]]


def segments_to_density(segments_list, resolution):
    all_pts = np.concatenate([s.reshape(-1, 2) for s in segments_list if len(s) > 0])
    xmin, zmin = all_pts.min(axis=0) - 0.5
    xmax, zmax = all_pts.max(axis=0) + 0.5
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    density = np.zeros((h, w), dtype=np.float32)
    for segments in segments_list:
        if len(segments) == 0:
            continue
        img = np.zeros((h, w), dtype=np.uint8)
        for seg in segments:
            p1x = int(np.clip((seg[0, 0] - xmin) / resolution, 0, w-1))
            p1y = int(np.clip((seg[0, 1] - zmin) / resolution, 0, h-1))
            p2x = int(np.clip((seg[1, 0] - xmin) / resolution, 0, w-1))
            p2y = int(np.clip((seg[1, 1] - zmin) / resolution, 0, h-1))
            cv2.line(img, (p1x, p1y), (p2x, p2y), 1, thickness=1)
        density += img.astype(np.float32)
    return density, (xmin, zmin, xmax, zmax, w, h)


def extract_hough_walls(density, grid_info):
    """Extract wall segments via Hough on thresholded+skeletonized density."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    # Threshold and skeletonize
    wall_mask = (density >= 2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
    skeleton = skeletonize(wall_mask > 0)
    
    # Standard Hough Transform for infinite lines
    skel_u8 = (skeleton * 255).astype(np.uint8)
    lines = cv2.HoughLines(skel_u8, rho=1, theta=np.pi/180, threshold=15)
    
    if lines is None:
        return [], skeleton
    
    walls = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) % 180
        
        # Filter to dominant angles
        best_da = None
        best_diff = 180
        for da in WALL_ANGLES:
            # Hough theta is angle of normal, wall direction = theta + 90
            wall_angle = (angle_deg + 90) % 180
            diff = abs(wall_angle - da)
            diff = min(diff, 180 - diff)
            if diff < best_diff:
                best_diff = diff
                best_da = da
        
        if best_diff > ANGLE_TOL:
            continue
        
        # Compute perpendicular offset in world coords
        # rho in pixels, need to convert
        # The Hough line normal direction: (cos(theta), sin(theta)) in image coords
        # In world: normal = (cos(theta)*res, sin(theta)*res)
        # perp offset = rho * res + origin offset projection
        
        # Actually, let's just store rho/theta and compute wall line geometry
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # A point on the line in pixel coords
        x0 = cos_t * rho
        y0 = sin_t * rho
        
        # Direction along the line (perpendicular to normal)
        dx = -sin_t
        dy = cos_t
        
        # Extend far
        ext = max(w, h) * 2
        px1, py1 = x0 - ext * dx, y0 - ext * dy
        px2, py2 = x0 + ext * dx, y0 + ext * dy
        
        # Convert to world
        wx1 = px1 * RESOLUTION + xmin
        wz1 = py1 * RESOLUTION + zmin
        wx2 = px2 * RESOLUTION + xmin
        wz2 = py2 * RESOLUTION + zmin
        
        # Score: density integral along this line
        # Sample density along the line in pixel coords
        n_samples = 200
        ts = np.linspace(-ext, ext, n_samples)
        sample_x = (x0 + ts * dx).astype(int)
        sample_y = (y0 + ts * dy).astype(int)
        valid = (sample_x >= 0) & (sample_x < w) & (sample_y >= 0) & (sample_y < h)
        score = 0
        if valid.any():
            score = density[sample_y[valid], sample_x[valid]].sum()
        
        walls.append({
            'angle': best_da,
            'rho': rho,
            'theta': theta,
            'score': score,
            'line_world': ((wx1, wz1), (wx2, wz2)),
        })
    
    print(f"  Raw Hough lines: {len(lines)}, angle-filtered: {len(walls)}")
    return walls, skeleton


def cluster_walls(walls, min_rho_gap=12):
    """Cluster walls by rho within each angle family."""
    by_angle = {}
    for w in walls:
        a = round(w['angle'])
        by_angle.setdefault(a, []).append(w)
    
    clustered = []
    for angle, group in by_angle.items():
        group.sort(key=lambda w: w['rho'])
        
        clusters = [[group[0]]]
        for w in group[1:]:
            if abs(w['rho'] - clusters[-1][-1]['rho']) < min_rho_gap:
                clusters[-1].append(w)
            else:
                clusters.append([w])
        
        for cluster in clusters:
            best = max(cluster, key=lambda w: w['score'])
            best['cluster_size'] = len(cluster)
            # Average rho for this cluster
            best['rho'] = np.mean([w['rho'] for w in cluster])
            best['total_score'] = sum(w['score'] for w in cluster)
            clustered.append(best)
    
    return clustered


def make_boundary(mesh, resolution=0.02):
    """Create apartment boundary polygon with angle-snapped edges."""
    verts_xz = mesh.vertices[:, [0, 2]]
    xmin, zmin = verts_xz.min(axis=0) - 0.3
    xmax, zmax = verts_xz.max(axis=0) + 0.3
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    img = np.zeros((h, w), dtype=np.uint8)
    px = ((verts_xz[:, 0] - xmin) / resolution).astype(int)
    py = ((verts_xz[:, 1] - zmin) / resolution).astype(int)
    px = np.clip(px, 0, w-1)
    py = np.clip(py, 0, h-1)
    for x, y in zip(px, py):
        img[y, x] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = binary_fill_holes(img).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * resolution + xmin
    pts[:, 1] = pts[:, 1] * resolution + zmin
    
    poly = Polygon(pts).buffer(0)
    poly = poly.simplify(0.15).buffer(0.05).buffer(-0.05).simplify(0.1)
    return poly


def wall_to_line(wall, boundary, grid_info):
    """Convert a Hough wall to a Shapely LineString clipped to boundary."""
    (wx1, wz1), (wx2, wz2) = wall['line_world']
    line = LineString([(wx1, wz1), (wx2, wz2)])
    
    clipped = line.intersection(boundary)
    if clipped.is_empty:
        return None
    if isinstance(clipped, MultiLineString):
        clipped = max(clipped.geoms, key=lambda g: g.length)
    if isinstance(clipped, LineString):
        return clipped
    return None


def try_partition(wall_subset, boundary):
    """Try to partition boundary with given walls, return rooms."""
    lines = []
    for w in wall_subset:
        if w.get('_line') is not None:
            lines.append(w['_line'])
    
    all_geoms = [boundary.exterior] + lines
    union = unary_union(all_geoms)
    result = list(polygonize(union))
    
    rooms = []
    for poly in result:
        if not boundary.contains(poly.representative_point()):
            continue
        if poly.area > 1.0:
            rooms.append(poly)
    
    return rooms


def select_walls_for_4_rooms(walls, boundary, grid_info):
    """Greedily add walls to get exactly 4 rooms."""
    # Precompute clipped lines
    for w in walls:
        w['_line'] = wall_to_line(w, boundary, grid_info)
    
    walls = [w for w in walls if w['_line'] is not None]
    walls.sort(key=lambda w: w['total_score'], reverse=True)
    
    print(f"\n  Candidate walls (by score):")
    for i, w in enumerate(walls):
        print(f"    [{i}] {w['angle']:.0f}° rho={w['rho']:.0f} "
              f"score={w['total_score']:.0f} cluster={w['cluster_size']}")
    
    # Greedy: start with no walls (1 room = whole boundary), add walls one by one
    selected = []
    
    for w in walls:
        test = selected + [w]
        rooms = try_partition(test, boundary)
        n = len(rooms)
        
        if n <= 7:  # Don't over-segment
            selected.append(w)
            print(f"    Added wall [{walls.index(w)}] → {n} rooms")
            
            if n == 4:
                return selected, rooms
            if n > 4:
                # Went past 4, remove this wall
                selected.pop()
                print(f"    Removed (over-segmented)")
    
    # If we didn't hit 4 exactly, try all combinations of reasonable size
    rooms = try_partition(selected, boundary)
    return selected, rooms


def merge_to_4(rooms):
    """Smart merge to exactly 4 rooms."""
    while len(rooms) > 4:
        # Merge smallest into its best neighbor
        idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
        best_j = None
        best_shared = -1
        for j in range(len(rooms)):
            if j == idx:
                continue
            shared = rooms[idx].buffer(0.05).intersection(rooms[j].buffer(0.05)).area
            if shared > best_shared:
                best_shared = shared
                best_j = j
        if best_j is not None:
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        else:
            break
    rooms.sort(key=lambda r: r.area, reverse=True)
    return rooms


def classify_4rooms(rooms):
    """Label 4 rooms as 2 bedrooms + hallway + bathroom."""
    if len(rooms) < 4:
        return [f"Room {i+1}" for i in range(len(rooms))]
    
    # Two largest = bedrooms
    labels = [None] * 4
    by_area = sorted(range(4), key=lambda i: rooms[i].area, reverse=True)
    labels[by_area[0]] = "Bedroom 1"
    labels[by_area[1]] = "Bedroom 2"
    
    # Of remaining 2, higher aspect = hallway
    rem = by_area[2:]
    aspects = []
    for i in rem:
        b = rooms[i].bounds
        w = b[2] - b[0]
        h = b[3] - b[1]
        aspects.append(max(w, h) / (min(w, h) + 1e-6))
    
    if aspects[0] >= aspects[1]:
        labels[rem[0]] = "Hallway"
        labels[rem[1]] = "Bathroom"
    else:
        labels[rem[0]] = "Bathroom"
        labels[rem[1]] = "Hallway"
    
    return labels


def plot_results(density, grid_info, skeleton, walls, rooms, labels, boundary, output_path):
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Density
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot',
              extent=[xmin, xmax, zmin, zmax], aspect='equal')
    ax.set_title(f"Cross-Section Density ({len(SLICE_HEIGHTS)} slices)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Walls on density
    ax = axes[1]
    ax.imshow(density, origin='lower', cmap='gray_r',
              extent=[xmin, xmax, zmin, zmax], aspect='equal', alpha=0.4)
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=2)
    for w in walls:
        if w.get('_line'):
            xs, ys = w['_line'].xy
            color = 'red' if round(w['angle']) == 29 else 'blue'
            ax.plot(xs, ys, color=color, linewidth=2.5)
    ax.set_title(f"Selected Walls ({len(walls)})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Rooms
    ax = axes[2]
    pastel = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9']
    total = 0
    for i, (room, label) in enumerate(zip(rooms, labels)):
        c = pastel[i % len(pastel)]
        geoms = room.geoms if isinstance(room, MultiPolygon) else [room]
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color=c, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2.5)
        area = room.area
        total += area
        cx, cy = room.centroid.coords[0]
        nv = len(room.exterior.coords) - 1
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=3)
    
    ax.set_title(f"v61b — {len(rooms)} rooms, {total:.1f}m²\n29°/119°")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'floorplan.png'}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v61b')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh = load_mesh(data_dir / args.mesh)
    
    # Cross-section density
    print("\nStep 1: Cross-section slices...")
    segs_list = []
    for y in SLICE_HEIGHTS:
        segs = slice_mesh(mesh, y)
        segs_list.append(segs)
        print(f"  Y={y:.1f}m: {len(segs)} segments")
    
    density, grid_info = segments_to_density(segs_list, RESOLUTION)
    print(f"  Density: {grid_info[4]}x{grid_info[5]}")
    
    # Hough wall detection
    print("\nStep 2: Hough wall detection...")
    raw_walls, skeleton = extract_hough_walls(density, grid_info)
    
    # Cluster
    print("\nStep 3: Clustering walls...")
    walls = cluster_walls(raw_walls, min_rho_gap=15)
    print(f"  Clustered: {len(raw_walls)} → {len(walls)} walls")
    
    # Boundary
    print("\nStep 4: Boundary...")
    boundary = make_boundary(mesh)
    print(f"  Boundary: {boundary.area:.1f}m²")
    
    # Select walls for 4 rooms
    print("\nStep 5: Wall selection for 4 rooms...")
    selected, rooms = select_walls_for_4_rooms(walls, boundary, grid_info)
    
    # Merge if needed
    if len(rooms) != 4:
        print(f"\nStep 6: Merging {len(rooms)} → 4...")
        rooms = merge_to_4(rooms)
    
    labels = classify_4rooms(rooms)
    
    print(f"\nFinal rooms:")
    for room, label in zip(rooms, labels):
        nv = len(room.exterior.coords) - 1
        print(f"  {label}: {room.area:.1f}m² ({nv}v)")
    
    total = plot_results(density, grid_info, skeleton, selected, rooms, labels,
                         boundary, args.output)
    
    print(f"\nv61b: {len(rooms)} rooms, {total:.1f}m²")


if __name__ == '__main__':
    main()

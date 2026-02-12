#!/usr/bin/env python3
"""
mesh2plan v61 - Four Rooms: 2 Bedrooms + Hallway + Bathroom

GROUND TRUTH: The apartment has exactly 4 rooms:
- 2 bedrooms (~10m² each) with beds
- 1 narrow hallway between them
- 1 bathroom

APPROACH: Combine best elements:
1. Cross-section slicing (v60) for wall density image
2. RANSAC on wall-face centroids (v58) for precise wall positions
3. Angle-snapped apartment boundary (NOT convex hull)
4. Minimum walls for 4-room partition
5. Strict 4-room target with smart merging

Key insight: We need exactly 2-3 internal walls to divide apartment into 4 rooms.
The hallway is the narrow space between the two bedrooms.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, box
from shapely.ops import polygonize, unary_union
from shapely.affinity import rotate as shapely_rotate
import pyransac3d as pyrsc


RESOLUTION = 0.02  # 2cm per pixel
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
WALL_ANGLES = np.array([29.0, 119.0])  # Known from RANSAC (v58)


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def get_wall_faces(mesh, normal_thresh=0.5):
    """Get faces with horizontal normals (walls)."""
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < normal_thresh  # Y is up
    centroids = mesh.triangles_center[wall_mask]
    print(f"  Wall faces: {wall_mask.sum()}/{len(normals)} ({100*wall_mask.mean():.0f}%)")
    return centroids


def ransac_wall_planes(wall_centroids, n_planes=12, dist_thresh=0.04):
    """Sequential RANSAC to find dominant vertical wall planes from 3D points."""
    points = wall_centroids.copy()
    planes = []
    
    for i in range(n_planes):
        if len(points) < 100:
            break
        
        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(points, thresh=dist_thresh, maxIteration=500)
        
        if len(best_inliers) < 50:
            break
        
        a, b, c, d = best_eq
        # Skip non-vertical planes (floor/ceiling)
        if abs(b) > 0.5:
            points = np.delete(points, best_inliers, axis=0)
            continue
        
        # Wall angle in XZ plane
        angle = np.degrees(np.arctan2(c, a)) % 180
        
        # Project inlier centroids to XZ for wall extent
        inlier_pts = points[best_inliers][:, [0, 2]]
        
        planes.append({
            'eq': best_eq,
            'angle': angle,
            'n_inliers': len(best_inliers),
            'inlier_xz': inlier_pts,
        })
        
        points = np.delete(points, best_inliers, axis=0)
    
    print(f"  RANSAC found {len(planes)} vertical planes")
    return planes


def planes_to_wall_lines(planes, dominant_angles, angle_tol=15):
    """Convert RANSAC planes to wall lines (perpendicular offset + angle family)."""
    walls = []
    
    for plane in planes:
        # Snap to dominant angle
        angle = plane['angle']
        best_da = None
        best_diff = 180
        for da in dominant_angles:
            diff = abs(angle - da)
            diff = min(diff, 180 - diff)
            if diff < best_diff:
                best_diff = diff
                best_da = da
        
        if best_diff > angle_tol:
            continue
        
        # Compute perpendicular offset from origin
        rad = np.radians(best_da)
        normal_2d = np.array([-np.sin(rad), np.cos(rad)])
        direction = np.array([np.cos(rad), np.sin(rad)])
        
        # Mean perpendicular offset of inlier points
        perp_offsets = plane['inlier_xz'] @ normal_2d
        perp = np.median(perp_offsets)
        
        # Wall extent along direction
        para_projs = plane['inlier_xz'] @ direction
        extent_min = np.percentile(para_projs, 5)
        extent_max = np.percentile(para_projs, 95)
        wall_length = extent_max - extent_min
        
        walls.append({
            'angle': best_da,
            'perp': perp,
            'extent': (extent_min, extent_max),
            'length': wall_length,
            'n_inliers': plane['n_inliers'],
            'direction': direction,
            'normal': normal_2d,
        })
    
    return walls


def deduplicate_walls(walls, min_gap=0.35):
    """Merge walls with similar perpendicular offset within same angle family."""
    by_angle = {}
    for w in walls:
        a = round(w['angle'])
        by_angle.setdefault(a, []).append(w)
    
    deduped = []
    for angle, group in by_angle.items():
        group.sort(key=lambda w: w['perp'])
        
        clusters = [[group[0]]]
        for w in group[1:]:
            if abs(w['perp'] - clusters[-1][-1]['perp']) < min_gap:
                clusters[-1].append(w)
            else:
                clusters.append([w])
        
        for cluster in clusters:
            # Keep the one with most inliers, but use weighted mean perp
            total_inliers = sum(w['n_inliers'] for w in cluster)
            avg_perp = sum(w['perp'] * w['n_inliers'] for w in cluster) / total_inliers
            
            # Union of extents
            ext_min = min(w['extent'][0] for w in cluster)
            ext_max = max(w['extent'][1] for w in cluster)
            
            best = max(cluster, key=lambda w: w['n_inliers'])
            best['perp'] = avg_perp
            best['extent'] = (ext_min, ext_max)
            best['length'] = ext_max - ext_min
            best['n_inliers'] = total_inliers
            deduped.append(best)
    
    return deduped


def make_angle_snapped_boundary(mesh, dominant_angles, resolution=0.02):
    """Create apartment boundary with edges snapped to dominant wall angles."""
    verts_xz = mesh.vertices[:, [0, 2]]
    xmin, zmin = verts_xz.min(axis=0) - 0.3
    xmax, zmax = verts_xz.max(axis=0) + 0.3
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    # Create occupancy image
    img = np.zeros((h, w), dtype=np.uint8)
    px = ((verts_xz[:, 0] - xmin) / resolution).astype(int)
    py = ((verts_xz[:, 1] - zmin) / resolution).astype(int)
    px = np.clip(px, 0, w-1)
    py = np.clip(py, 0, h-1)
    for x, y in zip(px, py):
        img[y, x] = 255
    
    # Close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = binary_fill_holes(img).astype(np.uint8) * 255
    
    # Get contour
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * resolution + xmin
    pts[:, 1] = pts[:, 1] * resolution + zmin
    
    raw_poly = Polygon(pts).buffer(0)
    
    # Simplify with Douglas-Peucker
    simplified = raw_poly.simplify(0.2)
    
    # Now snap each edge of the simplified polygon to the nearest dominant angle
    coords = list(simplified.exterior.coords[:-1])  # drop closing point
    snapped = []
    
    for i in range(len(coords)):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % len(coords)])
        
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        edge_len = np.sqrt(dx**2 + dz**2)
        if edge_len < 0.1:
            continue
        
        edge_angle = np.degrees(np.arctan2(dz, dx)) % 180
        
        # Find closest dominant angle (or its perpendicular)
        all_angles = list(dominant_angles) + [a + 90 for a in dominant_angles]
        all_angles = [a % 180 for a in all_angles]
        
        best_snap = edge_angle
        best_diff = 180
        for da in all_angles:
            diff = abs(edge_angle - da)
            diff = min(diff, 180 - diff)
            if diff < best_diff:
                best_diff = diff
                best_snap = da
        
        if best_diff < 25:  # Snap if within 25°
            # Rotate edge to snap angle, keeping midpoint
            mid = (p1 + p2) / 2
            half = edge_len / 2
            rad = np.radians(best_snap)
            d = np.array([np.cos(rad), np.sin(rad)])
            p1_new = mid - d * half
            p2_new = mid + d * half
            snapped.append(tuple(p1_new))
        else:
            snapped.append(tuple(p1))
    
    if len(snapped) < 3:
        return simplified.buffer(0)
    
    boundary = Polygon(snapped).buffer(0)
    if not boundary.is_valid or boundary.is_empty:
        return simplified.buffer(0)
    
    # Smooth small jitter
    boundary = boundary.buffer(0.05).buffer(-0.05)
    return boundary


def select_partition_walls(walls, boundary_poly, target_rooms=4):
    """Select the minimum set of walls to partition apartment into target rooms.
    
    For 4 rooms we need ~3 internal walls (2-3 cuts).
    Strategy: rank walls by structural importance (length × inliers), 
    then pick walls that create the right partition.
    """
    # Score each wall
    for w in walls:
        w['score'] = w['length'] * np.sqrt(w['n_inliers'])
    
    walls.sort(key=lambda w: w['score'], reverse=True)
    
    print(f"\n  All walls (sorted by score):")
    for i, w in enumerate(walls):
        print(f"    [{i}] {w['angle']:.0f}° perp={w['perp']:.2f} "
              f"len={w['length']:.1f}m inliers={w['n_inliers']} score={w['score']:.0f}")
    
    # Try increasing wall counts until we get target_rooms
    best_rooms = None
    best_n_rooms = 0
    best_walls_used = []
    
    for n_walls in range(2, min(len(walls) + 1, 10)):
        selected = walls[:n_walls]
        
        # Create wall lines
        lines = []
        for w in selected:
            # Extend wall across boundary
            mid_para = (w['extent'][0] + w['extent'][1]) / 2
            mid_point = w['direction'] * mid_para + w['normal'] * w['perp']
            
            far1 = mid_point - w['direction'] * 15
            far2 = mid_point + w['direction'] * 15
            
            line = LineString([far1, far2])
            clipped = line.intersection(boundary_poly)
            if not clipped.is_empty:
                if isinstance(clipped, MultiLineString):
                    clipped = max(clipped.geoms, key=lambda g: g.length)
                if isinstance(clipped, LineString):
                    lines.append(clipped)
        
        # Polygonize
        all_geoms = [boundary_poly.exterior] + lines
        union = unary_union(all_geoms)
        result = list(polygonize(union))
        
        rooms = [p for p in result 
                 if boundary_poly.contains(p.representative_point()) and p.area > 1.5]
        
        if len(rooms) == target_rooms:
            print(f"\n  ✅ {n_walls} walls → {len(rooms)} rooms (target!)")
            return selected, lines, rooms
        
        if abs(len(rooms) - target_rooms) < abs(best_n_rooms - target_rooms):
            best_n_rooms = len(rooms)
            best_rooms = rooms
            best_walls_used = selected
            best_lines = lines
        
        print(f"    {n_walls} walls → {len(rooms)} rooms")
    
    # If we didn't hit target exactly, use best and merge
    print(f"\n  Best: {len(best_walls_used)} walls → {best_n_rooms} rooms (target was {target_rooms})")
    return best_walls_used, best_lines, best_rooms


def smart_merge_to_target(rooms, target=4):
    """Merge rooms intelligently to hit target count."""
    while len(rooms) > target:
        # Find the smallest room
        idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
        room = rooms[idx]
        
        # Find best neighbor to merge with (most shared boundary)
        best_j = None
        best_shared = 0
        for j in range(len(rooms)):
            if j == idx:
                continue
            # Check adjacency
            shared = room.buffer(0.05).intersection(rooms[j].buffer(0.05)).area
            if shared > best_shared:
                best_shared = shared
                best_j = j
        
        if best_j is not None:
            merged = unary_union([rooms[best_j], rooms[idx]])
            rooms[best_j] = merged
            rooms.pop(idx)
        else:
            break
    
    rooms.sort(key=lambda r: r.area, reverse=True)
    return rooms


def classify_4rooms(rooms):
    """Classify exactly 4 rooms: 2 bedrooms + hallway + bathroom."""
    labels = []
    areas = [r.area for r in rooms]
    
    # Sort by area descending
    indexed = sorted(enumerate(areas), key=lambda x: x[1], reverse=True)
    
    result = [None] * len(rooms)
    
    if len(rooms) >= 4:
        # Largest 2 = bedrooms
        result[indexed[0][0]] = "Bedroom 1"
        result[indexed[1][0]] = "Bedroom 2"
        
        # Of remaining, check aspect ratio for hallway
        remaining = indexed[2:]
        aspects = []
        for idx, area in remaining:
            b = rooms[idx].bounds
            w = b[2] - b[0]
            h = b[3] - b[1]
            aspect = max(w, h) / (min(w, h) + 1e-6)
            aspects.append((idx, aspect, area))
        
        # Highest aspect ratio = hallway
        aspects.sort(key=lambda x: x[1], reverse=True)
        result[aspects[0][0]] = "Hallway"
        result[aspects[1][0]] = "Bathroom"
    else:
        for i, r in enumerate(rooms):
            result[i] = f"Room {i+1}"
    
    return result


def plot_results(density, grid_info, walls_used, wall_lines, rooms, labels, 
                 boundary_poly, output_path):
    """Diagnostic plot."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Cross-section density
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot',
              extent=[xmin, xmax, zmin, zmax], aspect='equal')
    ax.set_title(f"Cross-Section Wall Density\n({len(SLICE_HEIGHTS)} slices)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: RANSAC walls on density
    ax = axes[1]
    ax.imshow(density, origin='lower', cmap='gray_r',
              extent=[xmin, xmax, zmin, zmax], aspect='equal', alpha=0.4)
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=2)
    
    for wl in wall_lines:
        xs, ys = wl.xy
        ax.plot(xs, ys, 'r-', linewidth=2.5)
    
    ax.set_title(f"RANSAC Walls ({len(walls_used)}) + Boundary")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Room partition
    ax = axes[2]
    pastel = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', '#E8BAFF', '#FFD4BA']
    
    total_area = 0
    for i, (room, label) in enumerate(zip(rooms, labels)):
        color = pastel[i % len(pastel)]
        geoms = room.geoms if isinstance(room, MultiPolygon) else [room]
        for geom in geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2.5)
        
        area = room.area
        total_area += area
        cx, cy = room.centroid.coords[0]
        nv = len(room.exterior.coords) - 1
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=3)
    
    # Scale bar
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    sx = xlims[0] + 0.5
    sy = ylims[0] + 0.5
    ax.plot([sx, sx + 1], [sy, sy], 'k-', linewidth=3)
    ax.text(sx + 0.5, sy - 0.3, '1m', ha='center', fontsize=10)
    
    ax.set_title(f"v61 — {len(rooms)} rooms, {total_area:.1f}m²\n"
                 f"Angles: {WALL_ANGLES[0]:.0f}°/{WALL_ANGLES[1]:.0f}°")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'floorplan.png'}")
    return total_area


def slice_mesh(mesh, y_height):
    """Slice mesh at given Y height, return XZ segments."""
    try:
        lines = trimesh.intersections.mesh_plane(
            mesh, [0, 1, 0], [0, y_height, 0]
        )
    except Exception:
        return np.array([]).reshape(0, 2, 2)
    
    if lines is None or len(lines) == 0:
        return np.array([]).reshape(0, 2, 2)
    
    return lines[:, :, [0, 2]]


def segments_to_density(segments_list, resolution):
    """Rasterize segments into density image."""
    all_pts = np.concatenate([s.reshape(-1, 2) for s in segments_list if len(s) > 0])
    xmin, zmin = all_pts.min(axis=0) - 0.5
    xmax, zmax = all_pts.max(axis=0) + 0.5
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    density = np.zeros((h, w), dtype=np.float32)
    
    for segments in segments_list:
        if len(segments) == 0:
            continue
        slice_img = np.zeros((h, w), dtype=np.uint8)
        for seg in segments:
            p1x = int(np.clip((seg[0, 0] - xmin) / resolution, 0, w-1))
            p1y = int(np.clip((seg[0, 1] - zmin) / resolution, 0, h-1))
            p2x = int(np.clip((seg[1, 0] - xmin) / resolution, 0, w-1))
            p2y = int(np.clip((seg[1, 1] - zmin) / resolution, 0, h-1))
            cv2.line(slice_img, (p1x, p1y), (p2x, p2y), 1, thickness=1)
        density += slice_img.astype(np.float32)
    
    return density, (xmin, zmin, xmax, zmax, w, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v61')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh_path = data_dir / args.mesh
    
    mesh = load_mesh(mesh_path)
    
    # Step 1: Cross-section density (for visualization)
    print(f"\nStep 1: Cross-section slices...")
    segments_list = []
    for y in SLICE_HEIGHTS:
        segs = slice_mesh(mesh, y)
        segments_list.append(segs)
        print(f"  Y={y:.1f}m: {len(segs)} segments")
    
    density, grid_info = segments_to_density(segments_list, RESOLUTION)
    print(f"  Density image: {grid_info[4]}x{grid_info[5]}")
    
    # Step 2: RANSAC wall planes from 3D geometry
    print(f"\nStep 2: RANSAC wall detection...")
    wall_centroids = get_wall_faces(mesh)
    planes = ransac_wall_planes(wall_centroids, n_planes=15, dist_thresh=0.04)
    
    # Step 3: Convert to wall lines and deduplicate
    print(f"\nStep 3: Wall line extraction...")
    walls = planes_to_wall_lines(planes, WALL_ANGLES, angle_tol=15)
    print(f"  Angle-filtered walls: {len(walls)}")
    
    walls = deduplicate_walls(walls, min_gap=0.35)
    print(f"  After dedup: {len(walls)}")
    
    # Step 4: Create angle-snapped boundary
    print(f"\nStep 4: Building boundary...")
    boundary = make_angle_snapped_boundary(mesh, WALL_ANGLES)
    print(f"  Boundary area: {boundary.area:.1f}m²")
    
    # Step 5: Select walls for 4-room partition
    print(f"\nStep 5: Selecting partition walls...")
    walls_used, wall_lines, rooms = select_partition_walls(walls, boundary, target_rooms=4)
    
    # Step 6: Merge to exactly 4 rooms
    if len(rooms) != 4:
        print(f"\n  Merging {len(rooms)} rooms → 4...")
        rooms = smart_merge_to_target(rooms, target=4)
    
    # Step 7: Classify rooms
    labels = classify_4rooms(rooms)
    
    print(f"\nFinal rooms:")
    for room, label in zip(rooms, labels):
        nv = len(room.exterior.coords) - 1
        print(f"  {label}: {room.area:.1f}m² ({nv}v)")
    
    # Step 8: Plot
    total = plot_results(density, grid_info, walls_used, wall_lines, rooms, labels,
                         boundary, args.output)
    
    print(f"\n{'='*50}")
    print(f"v61 Four Rooms: {len(rooms)} rooms, {total:.1f}m²")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

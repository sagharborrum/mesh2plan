#!/usr/bin/env python3
"""
mesh2plan v58 - RANSAC Wall Planes → Room Polygons

APPROACH: Extract vertical planar surfaces from the mesh using sequential RANSAC.
Each detected plane = a wall. Project wall planes to XZ as lines. Use the line
arrangement (Shapely polygonize) to find room polygons.

Key difference from v53: 
- Filter to wall faces FIRST (face normal filtering from v41b)
- Use face centroids weighted by face area for better RANSAC
- Project planes to XZ lines with thickness
- Use Shapely for proper planar subdivision
- Clip to apartment boundary

Pipeline:
1. Load mesh → filter wall faces by normal
2. Sequential RANSAC on wall-face vertices → dominant vertical planes
3. Each plane → line in XZ (strike direction + offset)
4. Cluster lines by angle → 2 dominant families
5. Extend lines to apartment boundary
6. Polygonize line arrangement → room cells
7. Merge small cells, classify rooms
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from scipy.cluster.hierarchy import fcluster, linkage
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.ops import polygonize, unary_union
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors


def extract_wall_points(mesh, normal_thresh=0.5):
    """Extract points belonging to wall faces (horizontal normals)."""
    normals = mesh.face_normals
    # Wall faces: normal is mostly horizontal (small Y component)
    wall_mask = np.abs(normals[:, 1]) < normal_thresh
    
    wall_faces = mesh.faces[wall_mask]
    wall_areas = mesh.area_faces[wall_mask]
    
    # Get centroids of wall faces, weighted by area
    wall_centroids = mesh.triangles_center[wall_mask]
    
    # Use centroids (much fewer than raw vertices) for RANSAC
    wall_verts_idx = np.unique(wall_faces.flatten())
    wall_verts = mesh.vertices[wall_verts_idx]
    
    # Downsample: use centroids instead of raw vertices for speed
    # Weight by area for representative sampling
    if len(wall_centroids) > 50000:
        # Random sample weighted by area
        probs = wall_areas / wall_areas.sum()
        chosen = np.random.choice(len(wall_centroids), 50000, replace=False, p=probs)
        wall_verts = wall_centroids[chosen]
    else:
        wall_verts = wall_centroids
    
    print(f"  Wall faces: {wall_mask.sum()}/{len(normals)} ({100*wall_mask.mean():.0f}%)")
    print(f"  Wall vertices: {len(wall_verts)}")
    
    return wall_verts, wall_centroids, wall_areas


def ransac_vertical_plane(points, n_iter=1000, dist_thresh=0.05):
    """
    RANSAC to find a vertical plane in 3D points.
    Vertical plane: normal has no Y component (nx, 0, nz).
    Returns: (normal_xz, d, inlier_mask) where nx*x + nz*z = d
    """
    best_inliers = None
    best_count = 0
    best_normal = None
    best_d = None
    
    xz = points[:, [0, 2]]  # Project to XZ for 2D line fitting
    n = len(xz)
    
    for _ in range(n_iter):
        # Sample 2 points → define a line in XZ
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = xz[idx]
        
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        
        # Normal to line in XZ
        normal = np.array([-direction[1], direction[0]]) / length
        d = np.dot(normal, p1)
        
        # Distances from all points to this line
        dists = np.abs(np.dot(xz, normal) - d)
        inlier_mask = dists < dist_thresh
        count = inlier_mask.sum()
        
        if count > best_count:
            best_count = count
            best_inliers = inlier_mask
            best_normal = normal
            best_d = d
    
    if best_inliers is None:
        return None, None, None
    
    # Refit on inliers
    inlier_xz = xz[best_inliers]
    # SVD for best-fit line
    centroid = inlier_xz.mean(axis=0)
    centered = inlier_xz - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    normal = np.array([-direction[1], direction[0]])
    if normal[0] < 0:
        normal = -normal
    d = np.dot(normal, centroid)
    
    # Recompute inliers with refined plane
    dists = np.abs(np.dot(xz, normal) - d)
    inlier_mask = dists < dist_thresh
    
    return normal, d, inlier_mask


def sequential_ransac(points, max_planes=15, min_inliers=500, 
                      n_iter=2000, dist_thresh=0.04):
    """Sequential RANSAC: find planes one by one, removing inliers each time."""
    remaining = points.copy()
    remaining_mask = np.ones(len(points), dtype=bool)
    planes = []
    
    for i in range(max_planes):
        if len(remaining) < min_inliers:
            break
        
        normal, d, inlier_mask = ransac_vertical_plane(
            remaining, n_iter=n_iter, dist_thresh=dist_thresh
        )
        
        if normal is None or inlier_mask.sum() < min_inliers:
            break
        
        # Compute extent of inliers along wall direction
        direction = np.array([normal[1], -normal[0]])  # Perpendicular to normal
        inlier_xz = remaining[inlier_mask][:, [0, 2]]
        projections = np.dot(inlier_xz, direction)
        extent = projections.max() - projections.min()
        
        angle = np.degrees(np.arctan2(normal[1], normal[0])) % 180
        
        planes.append({
            'normal': normal,
            'd': d,
            'n_inliers': inlier_mask.sum(),
            'extent': extent,
            'angle': angle,
            'direction': direction,
            'proj_range': (projections.min(), projections.max()),
            'centroid_xz': inlier_xz.mean(axis=0),
        })
        
        print(f"  Plane {i}: angle={angle:.1f}°, d={d:.2f}m, "
              f"inliers={inlier_mask.sum()}, extent={extent:.2f}m")
        
        # Remove inliers
        remaining = remaining[~inlier_mask]
    
    return planes


def cluster_planes_by_angle(planes, angle_tol=15):
    """Group planes into angle families."""
    angles = np.array([p['angle'] for p in planes])
    
    # Circular distance for angles in [0, 180)
    n = len(angles)
    if n <= 1:
        return {0: list(range(n))}
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = abs(angles[i] - angles[j])
            d = min(d, 180 - d)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    condensed = dist_matrix[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=angle_tol, criterion='distance')
    
    families = {}
    for i, label in enumerate(labels):
        families.setdefault(label, []).append(i)
    
    return families


def planes_to_lines(planes, families, boundary_poly, extension=2.0):
    """Convert planes to extended line segments clipped to apartment boundary."""
    lines = []
    
    for fam_id, indices in families.items():
        fam_angles = [planes[i]['angle'] for i in indices]
        mean_angle = np.mean(fam_angles)
        
        for idx in indices:
            p = planes[idx]
            direction = p['direction']
            centroid = p['centroid_xz']
            
            # Extend line well beyond apartment
            t_min, t_max = p['proj_range']
            margin = extension
            p1 = centroid + direction * (t_min - margin - np.dot(centroid, direction))
            p2 = centroid + direction * (t_max + margin - np.dot(centroid, direction))
            
            # Extend line far beyond apartment (will be clipped to boundary)
            p1_xz = centroid - direction * 20
            p2_xz = centroid + direction * 20
            
            line = LineString([p1_xz, p2_xz])
            
            # Clip to boundary
            clipped = line.intersection(boundary_poly)
            if clipped.is_empty:
                continue
            
            lines.append({
                'line': clipped if isinstance(clipped, LineString) else line,
                'family': fam_id,
                'angle': p['angle'],
                'd': p['d'],
                'n_inliers': p['n_inliers'],
                'extent': p['extent'],
            })
    
    return lines


def deduplicate_walls(lines, min_gap=0.3):
    """Remove duplicate walls (same family, close perpendicular offset)."""
    if not lines:
        return lines
    
    # Group by family
    families = {}
    for l in lines:
        families.setdefault(l['family'], []).append(l)
    
    kept = []
    for fam_id, fam_lines in families.items():
        # Sort by d (perpendicular offset)
        fam_lines.sort(key=lambda x: x['d'])
        
        merged = [fam_lines[0]]
        for l in fam_lines[1:]:
            if abs(l['d'] - merged[-1]['d']) < min_gap:
                # Keep the one with more inliers
                if l['n_inliers'] > merged[-1]['n_inliers']:
                    merged[-1] = l
            else:
                merged.append(l)
        
        kept.extend(merged)
    
    return kept


def make_apartment_boundary(mesh, normal_thresh=0.5, resolution=0.02):
    """Create apartment boundary polygon from mesh XZ projection."""
    verts = mesh.vertices[:, [0, 2]]
    
    # Create density image
    xmin, zmin = verts.min(axis=0) - 0.5
    xmax, zmax = verts.max(axis=0) + 0.5
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    img = np.zeros((h, w), dtype=np.float32)
    px = ((verts[:, 0] - xmin) / resolution).astype(int)
    py = ((verts[:, 1] - zmin) / resolution).astype(int)
    px = np.clip(px, 0, w-1)
    py = np.clip(py, 0, h-1)
    
    for x, y in zip(px, py):
        img[y, x] += 1
    
    import cv2
    # Threshold and morphological close
    binary = (img > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (xmin, zmin, resolution)
    
    largest = max(contours, key=cv2.contourArea)
    
    # Convert back to world coordinates
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * resolution + xmin
    pts[:, 1] = pts[:, 1] * resolution + zmin
    
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    
    # Simplify boundary - smooth out jagged edges
    poly = poly.simplify(0.15, preserve_topology=True)
    poly = poly.buffer(0.05).buffer(-0.05)  # Smooth small notches
    poly = poly.simplify(0.1, preserve_topology=True)
    
    return poly, (xmin, zmin, resolution)


def build_room_polygons(wall_lines, boundary_poly, min_area=1.5):
    """Use wall lines to partition the boundary into room polygons."""
    # Collect all lines including boundary
    all_lines = [boundary_poly.exterior]
    for wl in wall_lines:
        line = wl['line']
        if isinstance(line, LineString):
            all_lines.append(line)
    
    # Polygonize
    result = list(polygonize(unary_union(all_lines)))
    
    # Filter: keep only polygons inside boundary and above min area
    rooms = []
    for poly in result:
        if not boundary_poly.contains(poly.representative_point()):
            continue
        area = poly.area
        if area < min_area:
            continue
        rooms.append(poly)
    
    # Sort by area descending
    rooms.sort(key=lambda p: p.area, reverse=True)
    
    return rooms


def merge_small_rooms(rooms, min_area=2.0, max_rooms=7):
    """Merge rooms smaller than min_area into their largest neighbor."""
    from shapely.ops import unary_union
    
    while True:
        small = [i for i, r in enumerate(rooms) if r.area < min_area]
        if not small:
            break
        
        # Merge smallest room into its largest neighbor
        idx = min(small, key=lambda i: rooms[i].area)
        room = rooms[idx]
        
        best_neighbor = None
        best_shared = 0
        for j, other in enumerate(rooms):
            if j == idx:
                continue
            shared = room.boundary.intersection(other.boundary).length
            if shared > best_shared:
                best_shared = shared
                best_neighbor = j
        
        if best_neighbor is not None:
            rooms[best_neighbor] = unary_union([rooms[best_neighbor], rooms[idx]])
            rooms.pop(idx)
        else:
            break
    
    # If still too many rooms, merge smallest iteratively
    while len(rooms) > max_rooms:
        idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
        room = rooms[idx]
        
        best_neighbor = None
        best_shared = 0
        for j, other in enumerate(rooms):
            if j == idx:
                continue
            shared = room.boundary.intersection(other.boundary).length
            if shared > best_shared:
                best_shared = shared
                best_neighbor = j
        
        if best_neighbor is not None:
            rooms[best_neighbor] = unary_union([rooms[best_neighbor], rooms[idx]])
            rooms.pop(idx)
        else:
            break
    
    return rooms


def classify_room(poly, all_rooms):
    """Classify room by area and aspect ratio."""
    area = poly.area
    bounds = poly.bounds
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    aspect = max(w, h) / (min(w, h) + 1e-6)
    
    if area > 8:
        return "Room"
    elif area > 5:
        return "Room" 
    elif aspect > 2.5:
        return "Hallway"
    elif area > 3:
        return "Bathroom"
    else:
        return "Closet"


def plot_results(mesh, planes, wall_lines, rooms, boundary_poly, 
                 output_path, grid_info):
    """Create diagnostic + result plot."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: RANSAC planes on mesh projection
    ax = axes[0]
    verts_xz = mesh.vertices[:, [0, 2]]
    ax.scatter(verts_xz[::10, 0], verts_xz[::10, 1], s=0.1, c='lightgray', alpha=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, p in enumerate(planes):
        direction = p['direction']
        centroid = p['centroid_xz']
        half = p['extent'] / 2 + 0.5
        p1 = centroid - direction * half
        p2 = centroid + direction * half
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                c=colors[i % 10], linewidth=2, 
                label=f"P{i}: {p['angle']:.0f}° ({p['n_inliers']})")
    
    ax.set_title(f"RANSAC Planes ({len(planes)} found)")
    ax.legend(fontsize=6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Wall lines on boundary
    ax = axes[1]
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=1.5, alpha=0.5)
    
    for wl in wall_lines:
        line = wl['line']
        if isinstance(line, LineString):
            lx, ly = line.xy
            ax.plot(lx, ly, linewidth=2, 
                    label=f"Fam{wl['family']} {wl['angle']:.0f}°")
    
    ax.set_title(f"Wall Lines ({len(wall_lines)} walls)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Room polygons
    ax = axes[2]
    
    pastel_colors = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', 
                     '#E8BAFF', '#FFD4BA', '#BAF0FF']
    
    total_area = 0
    for i, room in enumerate(rooms):
        color = pastel_colors[i % len(pastel_colors)]
        if isinstance(room, MultiPolygon):
            for geom in room.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, color=color, alpha=0.6)
                ax.plot(xs, ys, 'k-', linewidth=2)
        else:
            xs, ys = room.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2)
        
        area = room.area
        total_area += area
        label = classify_room(room, rooms)
        cx, cy = room.centroid.coords[0]
        n_verts = len(room.exterior.coords) - 1
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw boundary
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=3)
    
    # Scale bar
    ax.plot([-4, -3], [-4.5, -4.5], 'k-', linewidth=3)
    ax.text(-3.5, -4.8, '1m', ha='center', fontsize=10)
    
    fam_angles = set()
    for wl in wall_lines:
        fam_angles.add(f"{wl['angle']:.0f}°")
    
    ax.set_title(f"v58 — {len(rooms)} rooms, {total_area:.1f}m²\n"
                 f"Angles: {', '.join(sorted(fam_angles))}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'floorplan.png'}")
    
    return total_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v58')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--max-planes', type=int, default=15)
    parser.add_argument('--dist-thresh', type=float, default=0.04)
    parser.add_argument('--min-inliers', type=int, default=500)
    parser.add_argument('--min-gap', type=float, default=0.35)
    parser.add_argument('--min-room-area', type=float, default=2.0)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh_path = data_dir / args.mesh
    
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    # Step 1: Extract wall points
    print("\nStep 1: Extracting wall points...")
    wall_verts, wall_centroids, wall_areas = extract_wall_points(mesh)
    
    # Step 2: Sequential RANSAC
    print(f"\nStep 2: Sequential RANSAC (max {args.max_planes} planes)...")
    planes = sequential_ransac(
        wall_verts, 
        max_planes=args.max_planes,
        min_inliers=args.min_inliers,
        n_iter=2000,
        dist_thresh=args.dist_thresh
    )
    print(f"  Found {len(planes)} planes")
    
    # Step 3: Cluster by angle
    print("\nStep 3: Clustering planes by angle...")
    families = cluster_planes_by_angle(planes, angle_tol=15)
    for fam_id, indices in families.items():
        angles = [planes[i]['angle'] for i in indices]
        print(f"  Family {fam_id}: {len(indices)} planes, "
              f"angles {[f'{a:.0f}°' for a in angles]}")
    
    # Keep only 2 largest families (perpendicular wall pairs)
    fam_sizes = {k: len(v) for k, v in families.items()}
    top_fams = sorted(fam_sizes, key=fam_sizes.get, reverse=True)[:2]
    filtered_families = {k: families[k] for k in top_fams}
    
    kept_planes = []
    for fam_id, indices in filtered_families.items():
        for idx in indices:
            kept_planes.append(planes[idx])
    
    print(f"  Keeping {len(kept_planes)} planes from top 2 families")
    
    # Step 4: Create apartment boundary
    print("\nStep 4: Creating apartment boundary...")
    boundary_poly, grid_info = make_apartment_boundary(mesh)
    print(f"  Boundary area: {boundary_poly.area:.1f}m²")
    
    # Step 5: Convert planes to wall lines
    print("\nStep 5: Converting to wall lines...")
    # Re-cluster kept planes
    families2 = cluster_planes_by_angle(kept_planes, angle_tol=15)
    wall_lines = planes_to_lines(kept_planes, families2, boundary_poly, extension=3.0)
    print(f"  {len(wall_lines)} wall lines before dedup")
    
    # Deduplicate
    wall_lines = deduplicate_walls(wall_lines, min_gap=args.min_gap)
    print(f"  {len(wall_lines)} wall lines after dedup (gap={args.min_gap}m)")
    
    for wl in wall_lines:
        print(f"    Fam{wl['family']} {wl['angle']:.0f}° d={wl['d']:.2f} "
              f"inliers={wl['n_inliers']} extent={wl['extent']:.1f}m")
    
    # Step 6: Build room polygons
    print("\nStep 6: Building room polygons...")
    rooms = build_room_polygons(wall_lines, boundary_poly, min_area=1.0)
    print(f"  {len(rooms)} raw room polygons")
    
    # Step 7: Merge small rooms
    rooms = merge_small_rooms(rooms, min_area=args.min_room_area, max_rooms=7)
    print(f"  {len(rooms)} rooms after merge")
    
    for i, room in enumerate(rooms):
        label = classify_room(room, rooms)
        n_verts = len(room.exterior.coords) - 1
        print(f"    {label}: {room.area:.1f}m² ({n_verts}v)")
    
    # Step 8: Plot
    print("\nStep 8: Plotting...")
    total_area = plot_results(
        mesh, planes, wall_lines, rooms, boundary_poly, 
        args.output, grid_info
    )
    
    print(f"\n{'='*50}")
    print(f"v58 RANSAC Rooms: {len(rooms)} rooms, {total_area:.1f}m²")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

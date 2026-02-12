#!/usr/bin/env python3
"""
mesh2plan v60c - RANSAC + Cross-Section Validation

Combines v58's RANSAC wall detection with cross-section slicing for wall extent.
- RANSAC on wall-face normals → wall positions (infinite lines)  
- Cross-section density → where walls actually exist (finite segments)
- Use cross-section to TRIM RANSAC lines to actual wall extents

This solves v58's main problem (all walls extend across full apartment)
while keeping its strength (accurate wall positions from 3D geometry).
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.cluster.hierarchy import fcluster, linkage
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
import matplotlib.colors as mcolors


RESOLUTION = 0.02
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
MIN_ROOM_AREA = 2.0


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def extract_wall_points(mesh, normal_thresh=0.5):
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < normal_thresh
    wall_centroids = mesh.triangles_center[wall_mask]
    wall_areas = mesh.area_faces[wall_mask]
    
    if len(wall_centroids) > 50000:
        probs = wall_areas / wall_areas.sum()
        chosen = np.random.choice(len(wall_centroids), 50000, replace=False, p=probs)
        pts = wall_centroids[chosen]
    else:
        pts = wall_centroids
    
    print(f"  Wall faces: {wall_mask.sum()}/{len(normals)}")
    return pts


def ransac_line_2d(xz, n_iter=2000, dist_thresh=0.04):
    best_inliers = None
    best_count = 0
    best_normal = None
    best_d = None
    n = len(xz)
    
    for _ in range(n_iter):
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = xz[idx]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6: continue
        normal = np.array([-direction[1], direction[0]]) / length
        d = np.dot(normal, p1)
        dists = np.abs(np.dot(xz, normal) - d)
        mask = dists < dist_thresh
        count = mask.sum()
        if count > best_count:
            best_count = count
            best_inliers = mask
            best_normal = normal
            best_d = d
    
    if best_inliers is None:
        return None, None, None, None
    
    inlier_xz = xz[best_inliers]
    centroid = inlier_xz.mean(axis=0)
    centered = inlier_xz - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    normal = np.array([-direction[1], direction[0]])
    if normal[0] < 0: normal = -normal
    d = np.dot(normal, centroid)
    
    dists = np.abs(np.dot(xz, normal) - d)
    mask = dists < dist_thresh
    
    # Compute extent
    projections = np.dot(xz[mask][:, [0, 1]], direction)
    
    return normal, d, mask, (projections.min(), projections.max(), direction, centroid)


def sequential_ransac(points, max_planes=15, min_inliers=500, dist_thresh=0.04):
    remaining_xz = points[:, [0, 2]].copy()
    planes = []
    
    for i in range(max_planes):
        if len(remaining_xz) < min_inliers:
            break
        
        normal, d, mask, extent_info = ransac_line_2d(remaining_xz, dist_thresh=dist_thresh)
        if normal is None or mask.sum() < min_inliers:
            break
        
        angle = np.degrees(np.arctan2(normal[1], normal[0])) % 180
        proj_min, proj_max, direction, centroid = extent_info
        
        planes.append({
            'normal': normal, 'd': d, 'n_inliers': mask.sum(),
            'angle': angle, 'direction': direction, 'centroid': centroid,
            'proj_range': (proj_min, proj_max),
            'extent': proj_max - proj_min,
        })
        print(f"  Plane {i}: {angle:.0f}° d={d:.2f} inliers={mask.sum()} extent={proj_max-proj_min:.1f}m")
        
        remaining_xz = remaining_xz[~mask]
    
    return planes


def cluster_and_filter(planes, angle_tol=15):
    """Keep only 2 dominant angle families."""
    angles = np.array([p['angle'] for p in planes])
    n = len(angles)
    if n <= 1:
        return planes
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = abs(angles[i] - angles[j])
            d = min(d, 180 - d)
            dist[i,j] = dist[j,i] = d
    
    condensed = dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=angle_tol, criterion='distance')
    
    families = {}
    for i, lab in enumerate(labels):
        families.setdefault(lab, []).append(i)
    
    # Keep top 2 families
    top = sorted(families, key=lambda k: len(families[k]), reverse=True)[:2]
    kept = []
    for k in top:
        for idx in families[k]:
            kept.append(planes[idx])
    
    return kept


def deduplicate(planes, min_gap=0.35):
    """Remove duplicate parallel walls."""
    if not planes: return planes
    
    angle_groups = {}
    for p in planes:
        a = round(p['angle'])
        angle_groups.setdefault(a, []).append(p)
    
    # Merge nearby angles
    merged_groups = {}
    used = set()
    for a in sorted(angle_groups):
        if a in used: continue
        group = list(angle_groups[a])
        for b in sorted(angle_groups):
            if b in used or b == a: continue
            diff = abs(a - b)
            diff = min(diff, 180 - diff)
            if diff < 5:
                group.extend(angle_groups[b])
                used.add(b)
        used.add(a)
        merged_groups[a] = group
    
    kept = []
    for a, group in merged_groups.items():
        group.sort(key=lambda p: p['d'])
        deduped = [group[0]]
        for p in group[1:]:
            if abs(p['d'] - deduped[-1]['d']) < min_gap:
                if p['n_inliers'] > deduped[-1]['n_inliers']:
                    deduped[-1] = p
            else:
                deduped.append(p)
        kept.extend(deduped)
    
    return kept


def cross_section_density(mesh, resolution=RESOLUTION):
    """Build cross-section density image for wall validation."""
    v = mesh.vertices
    xmin, xmax = v[:,0].min()-0.5, v[:,0].max()+0.5
    zmin, zmax = v[:,2].min()-0.5, v[:,2].max()+0.5
    w = int((xmax-xmin)/resolution)
    h = int((zmax-zmin)/resolution)
    density = np.zeros((h, w), dtype=np.float32)
    
    for yh in SLICE_HEIGHTS:
        try:
            lines = trimesh.intersections.mesh_plane(mesh, [0,1,0], [0,yh,0])
        except: continue
        if lines is None or len(lines)==0: continue
        s = np.zeros((h,w), dtype=np.uint8)
        segs = lines[:,:,[0,2]]
        for seg in segs:
            p1x = np.clip(int((seg[0,0]-xmin)/resolution),0,w-1)
            p1y = np.clip(int((seg[0,1]-zmin)/resolution),0,h-1)
            p2x = np.clip(int((seg[1,0]-xmin)/resolution),0,w-1)
            p2y = np.clip(int((seg[1,1]-zmin)/resolution),0,h-1)
            cv2.line(s,(p1x,p1y),(p2x,p2y),1,thickness=2)
        density += s.astype(np.float32)
    
    return density, (xmin, zmin, xmax, zmax, w, h)


def trim_wall_by_density(plane, density, grid_info, resolution=RESOLUTION, 
                         threshold=2, min_run=0.3):
    """Trim a RANSAC wall line to only where cross-section density confirms it."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    direction = plane['direction']
    normal = plane['normal']
    d = plane['d']
    
    # Sample points along the wall line
    proj_min, proj_max = plane['proj_range']
    margin = 0.5
    proj_range = np.arange(proj_min - margin, proj_max + margin, resolution)
    
    # For each sample, check density in a small perpendicular band
    wall_present = []
    centroid = plane['centroid']
    
    for t in proj_range:
        # Point on wall line at projection t
        pt = centroid + direction * (t - np.dot(centroid, direction))
        # Adjust to be exactly on the line (d = normal . pt)
        pt = pt + normal * (d - np.dot(normal, pt))
        
        # Convert to pixel
        px = int((pt[0] - xmin) / resolution)
        py = int((pt[1] - zmin) / resolution)
        
        # Check density in 3px perpendicular band
        has_wall = False
        for offset in range(-2, 3):
            nx = px + int(offset * normal[0])
            ny = py + int(offset * normal[1])
            if 0 <= nx < w and 0 <= ny < h:
                if density[ny, nx] >= threshold:
                    has_wall = True
                    break
        
        wall_present.append(has_wall)
    
    wall_present = np.array(wall_present)
    
    # Find continuous runs of True (wall segments)
    segments = []
    in_run = False
    run_start = 0
    
    for i, present in enumerate(wall_present):
        if present and not in_run:
            run_start = i
            in_run = True
        elif not present and in_run:
            run_length = (i - run_start) * resolution
            if run_length >= min_run:
                t_start = proj_range[run_start]
                t_end = proj_range[i-1]
                segments.append((t_start, t_end))
            in_run = False
    
    if in_run:
        run_length = (len(wall_present) - run_start) * resolution
        if run_length >= min_run:
            segments.append((proj_range[run_start], proj_range[-1]))
    
    # Convert to LineStrings
    lines = []
    for t_start, t_end in segments:
        p1 = centroid + direction * (t_start - np.dot(centroid, direction))
        p1 = p1 + normal * (d - np.dot(normal, p1))
        p2 = centroid + direction * (t_end - np.dot(centroid, direction))
        p2 = p2 + normal * (d - np.dot(normal, p2))
        lines.append(LineString([p1, p2]))
    
    return lines


def make_boundary(mesh, resolution=0.02):
    verts = mesh.vertices[:, [0, 2]]
    xmin, zmin = verts.min(axis=0) - 0.5
    xmax, zmax = verts.max(axis=0) + 0.5
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    img = np.zeros((h, w), dtype=np.uint8)
    px = ((verts[:,0] - xmin) / resolution).astype(int)
    py = ((verts[:,1] - zmin) / resolution).astype(int)
    np.clip(px, 0, w-1, out=px)
    np.clip(py, 0, h-1, out=py)
    for x, y in zip(px, py):
        img[y, x] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * resolution + xmin
    pts[:, 1] = pts[:, 1] * resolution + zmin
    
    poly = Polygon(pts).buffer(0)
    poly = poly.simplify(0.15).buffer(0.05).buffer(-0.05).simplify(0.1)
    return poly


def build_rooms_extended(walls_trimmed, walls_full, boundary, min_area=1.0):
    """Try trimmed walls first; if not enough rooms, extend some walls."""
    # First try with trimmed walls
    all_lines = [boundary.exterior]
    for wl in walls_trimmed:
        all_lines.append(wl)
    
    union = unary_union(all_lines)
    result = list(polygonize(union))
    rooms = [p for p in result if boundary.contains(p.representative_point()) and p.area >= min_area]
    
    if len(rooms) >= 4:
        return rooms
    
    # Not enough rooms with trimmed walls — extend walls to boundary
    print("  Extending walls to boundary for better partition...")
    all_lines = [boundary.exterior]
    for wl in walls_full:
        all_lines.append(wl)
    
    union = unary_union(all_lines)
    result = list(polygonize(union))
    rooms = [p for p in result if boundary.contains(p.representative_point()) and p.area >= min_area]
    
    return rooms


def merge_small_rooms(rooms, min_area=2.0, max_rooms=7):
    while True:
        small = [i for i, r in enumerate(rooms) if r.area < min_area]
        if not small: break
        idx = min(small, key=lambda i: rooms[i].area)
        best_j, best_shared = None, 0
        for j, other in enumerate(rooms):
            if j == idx: continue
            shared = rooms[idx].boundary.intersection(other.boundary).length
            if shared > best_shared:
                best_shared = shared
                best_j = j
        if best_j is not None:
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        else: break
    
    while len(rooms) > max_rooms:
        idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
        best_j, best_shared = None, 0
        for j, other in enumerate(rooms):
            if j == idx: continue
            shared = rooms[idx].boundary.intersection(other.boundary).length
            if shared > best_shared:
                best_shared = shared
                best_j = j
        if best_j is not None:
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        else: break
    
    return rooms


def classify_room(poly):
    area = poly.area
    bounds = poly.bounds
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    aspect = max(w, h) / (min(w, h) + 1e-6)
    if area > 8: return "Room"
    elif area > 5: return "Room"
    elif aspect > 2.5: return "Hallway"
    elif area > 3: return "Bathroom"
    else: return "Closet"


def plot_results(density, grid_info, planes, trimmed_walls, rooms, boundary, output_path):
    xmin, zmin, xmax, zmax, w, h = grid_info
    extent = [xmin, xmax, zmin, zmax]
    
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    
    # Panel 1: Cross-section density
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=extent, aspect='equal')
    ax.set_title(f"Cross-Section Density ({len(SLICE_HEIGHTS)} slices)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: RANSAC planes on density
    ax = axes[1]
    ax.imshow(density, origin='lower', cmap='gray_r', extent=extent, aspect='equal', alpha=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, p in enumerate(planes):
        d = p['direction']
        c = p['centroid']
        half = p['extent'] / 2 + 0.5
        p1 = c - d * half
        p2 = c + d * half
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=colors[i%10], linewidth=2,
                label=f"{p['angle']:.0f}° d={p['d']:.1f} ({p['n_inliers']})")
    ax.set_title(f"RANSAC Walls ({len(planes)})")
    ax.legend(fontsize=5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Trimmed walls
    ax = axes[2]
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=1.5, alpha=0.5)
    for wl in trimmed_walls:
        lx, ly = wl.xy
        ax.plot(lx, ly, 'r-', linewidth=2.5)
    ax.set_title(f"Cross-Section Trimmed Walls ({len(trimmed_walls)})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Rooms
    ax = axes[3]
    pastel = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', '#E8BAFF', '#FFD4BA', '#BAF0FF']
    total_area = 0
    for i, room in enumerate(rooms):
        color = pastel[i % len(pastel)]
        geoms = room.geoms if isinstance(room, MultiPolygon) else [room]
        for geom in geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2)
        area = room.area
        total_area += area
        label = classify_room(room)
        cx, cy = room.centroid.coords[0]
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=3)
    
    ax.plot([-4, -3], [-5, -5], 'k-', linewidth=3)
    ax.text(-3.5, -5.3, '1m', ha='center', fontsize=10)
    
    angles = sorted(set(round(p['angle']) for p in planes))
    ax.set_title(f"v60c — {len(rooms)} rooms, {total_area:.1f}m²\nAngles: {angles}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'floorplan.png'}")
    return total_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v60c')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh = load_mesh(data_dir / args.mesh)
    
    # Step 1: RANSAC wall detection
    print("\nStep 1: RANSAC wall detection...")
    wall_pts = extract_wall_points(mesh)
    planes = sequential_ransac(wall_pts, max_planes=15, min_inliers=500, dist_thresh=0.04)
    
    # Step 2: Filter to 2 angle families
    print("\nStep 2: Angle clustering...")
    planes = cluster_and_filter(planes)
    planes = deduplicate(planes, min_gap=0.35)
    print(f"  {len(planes)} walls after dedup")
    for p in planes:
        print(f"    {p['angle']:.0f}° d={p['d']:.2f} inliers={p['n_inliers']} extent={p['extent']:.1f}m")
    
    # Step 3: Cross-section density
    print("\nStep 3: Cross-section density...")
    density, grid_info = cross_section_density(mesh)
    
    # Step 4: Trim walls using cross-section
    print("\nStep 4: Trimming walls with cross-section data...")
    trimmed_walls = []
    for p in planes:
        segments = trim_wall_by_density(p, density, grid_info, threshold=1, min_run=0.3)
        trimmed_walls.extend(segments)
        print(f"    {p['angle']:.0f}° d={p['d']:.2f}: {len(segments)} segments")
    
    print(f"  Total trimmed segments: {len(trimmed_walls)}")
    
    # Step 5: Create boundary
    print("\nStep 5: Boundary...")
    boundary = make_boundary(mesh)
    print(f"  Boundary: {boundary.area:.1f}m²")
    
    # Step 6: Build full-length walls for fallback
    full_walls = []
    for p in planes:
        d = p['direction']
        c = p['centroid']
        p1 = c - d * 15
        p2 = c + d * 15
        line = LineString([p1, p2]).intersection(boundary)
        if not line.is_empty and isinstance(line, LineString):
            full_walls.append(line)
    
    # Step 7: Build rooms
    print("\nStep 6: Building rooms...")
    rooms = build_rooms_extended(trimmed_walls, full_walls, boundary, min_area=1.0)
    rooms.sort(key=lambda p: p.area, reverse=True)
    print(f"  Raw rooms: {len(rooms)}")
    
    rooms = merge_small_rooms(rooms, min_area=MIN_ROOM_AREA, max_rooms=7)
    print(f"  Final: {len(rooms)}")
    for r in rooms:
        print(f"    {classify_room(r)}: {r.area:.1f}m² ({len(r.exterior.coords)-1}v)")
    
    # Step 8: Plot
    total = plot_results(density, grid_info, planes, trimmed_walls, rooms, boundary, args.output)
    print(f"\n{'='*50}")
    print(f"v60c: {len(rooms)} rooms, {total:.1f}m²")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

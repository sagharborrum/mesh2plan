#!/usr/bin/env python3
"""
mesh2plan v53 - RANSAC Planar Wall Extraction

FUNDAMENTALLY DIFFERENT from v48-v52 (Hough on density images).
Works directly on 3D mesh geometry:

1. Extract wall faces (horizontal normals) from mesh
2. RANSAC to find large planar surfaces among wall faces
3. Each plane = a wall segment. Project wall planes to XZ (top-down).
4. Each vertical wall plane becomes a LINE in 2D (the wall's footprint).
5. Cluster lines by angle → 2 perpendicular families.
6. Find intersections → room corners → room polygons.

Key advantage: works in 3D metric space, no rasterization/resolution loss.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import math
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict
import shutil


def extract_wall_faces(mesh):
    """Get faces with horizontal normals (walls)."""
    normals = mesh.face_normals
    # Wall = normal mostly horizontal (small Y component)
    horiz = np.abs(normals[:, 1])  # Y is up in this mesh
    wall_mask = horiz < 0.3  # strict: nearly horizontal normal
    return wall_mask


def ransac_plane(points, n_iter=1000, thresh=0.02):
    """RANSAC to fit a plane to 3D points. Returns (normal, d, inlier_mask)."""
    best_inliers = None
    best_count = 0
    best_normal = None
    best_d = None
    n = len(points)
    if n < 3:
        return None, None, np.zeros(n, dtype=bool)
    
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)
        
        dists = np.abs(points @ normal + d)
        inliers = dists < thresh
        count = inliers.sum()
        
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d
    
    return best_normal, best_d, best_inliers


def sequential_ransac(points, min_points=500, max_planes=20, n_iter=2000, thresh=0.02):
    """Find multiple planes by sequential RANSAC. Subsamples for speed."""
    planes = []
    remaining = np.ones(len(points), dtype=bool)
    
    for iteration in range(max_planes):
        pts = points[remaining]
        if len(pts) < min_points:
            break
        
        # Subsample for RANSAC speed, then validate on full set
        if len(pts) > 5000:
            sample_idx = np.random.choice(len(pts), 5000, replace=False)
            sample_pts = pts[sample_idx]
        else:
            sample_pts = pts
            sample_idx = np.arange(len(pts))
        
        normal, d, _ = ransac_plane(sample_pts, n_iter=n_iter, thresh=thresh)
        if normal is None:
            break
        
        # Validate on ALL remaining points
        dists = np.abs(pts @ normal + d)
        inlier_mask = dists < thresh
        
        if inlier_mask.sum() < min_points:
            break
        
        # Map inlier mask back to full array
        full_inlier = np.zeros(len(points), dtype=bool)
        remaining_idx = np.where(remaining)[0]
        full_inlier[remaining_idx[inlier_mask]] = True
        
        planes.append({
            'normal': normal,
            'd': d,
            'inlier_mask': full_inlier,
            'n_points': inlier_mask.sum()
        })
        
        remaining[full_inlier] = False
        print(f"    Plane {iteration}: {inlier_mask.sum()} inliers, normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})")
    
    return planes


def plane_to_2d_line(normal, d):
    """Convert a vertical 3D plane to a 2D line in XZ.
    
    A vertical plane has normal ≈ (nx, 0, nz).
    In XZ top-down view, this becomes line: nx*x + nz*z + d = 0
    → parametric: angle = atan2(nz, nx), offset = -d / sqrt(nx²+nz²)
    """
    nx, ny, nz = normal
    horiz_norm = math.sqrt(nx**2 + nz**2)
    if horiz_norm < 0.1:
        return None  # Not a vertical plane
    
    # Normalize to XZ
    nx2 = nx / horiz_norm
    nz2 = nz / horiz_norm
    d2 = d / horiz_norm
    
    angle = math.atan2(nz2, nx2)  # normal direction
    # Wall direction is perpendicular to normal
    wall_angle = angle + math.pi / 2
    offset = -d2  # signed distance from origin
    
    return {
        'angle': wall_angle,  # direction along wall
        'normal_angle': angle,
        'offset': offset,  # perpendicular distance from origin
        'nx': nx2, 'nz': nz2, 'd': d2
    }


def get_wall_extent(points_3d, normal, d, wall_line):
    """Get the extent (start/end) of a wall in its direction."""
    nx, nz = wall_line['nx'], wall_line['nz']
    # Project points onto wall direction
    wx, wz = -nz, nx  # wall direction (perpendicular to normal)
    
    projections = points_3d[:, 0] * wx + points_3d[:, 2] * wz
    
    return projections.min(), projections.max()


def cluster_by_angle(lines, angle_thresh=15):
    """Cluster wall lines by angle into families."""
    angles_deg = [math.degrees(l['angle']) % 180 for l in lines]
    
    # Simple clustering: find dominant angles
    families = []
    used = set()
    
    for i, a in enumerate(angles_deg):
        if i in used:
            continue
        family = [i]
        used.add(i)
        for j, b in enumerate(angles_deg):
            if j in used:
                continue
            diff = min(abs(a - b), 180 - abs(a - b))
            if diff < angle_thresh:
                family.append(j)
                used.add(j)
        families.append(family)
    
    # Sort by size, keep top 2
    families.sort(key=len, reverse=True)
    return families[:2]


def line_intersection_2d(l1, l2):
    """Find intersection of two 2D lines (nx*x + nz*z + d = 0)."""
    a1, b1, c1 = l1['nx'], l1['nz'], l1['d']
    a2, b2, c2 = l2['nx'], l2['nz'], l2['d']
    
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    
    x = (-c1 * b2 + c2 * b1) / det
    z = (-a1 * c2 + a2 * c1) / det
    return (x, z)


def build_room_polygons(walls_fam0, walls_fam1, apartment_mask_xz, bounds):
    """
    Given two families of parallel walls, their intersections form a grid.
    Each grid cell that overlaps the apartment interior is a room.
    """
    xmin, xmax, zmin, zmax = bounds
    
    # Sort walls within each family by offset
    walls_fam0.sort(key=lambda w: w['offset'])
    walls_fam1.sort(key=lambda w: w['offset'])
    
    rooms = []
    
    # For each pair of consecutive walls in family 0 × family 1
    for i in range(len(walls_fam0) - 1):
        for j in range(len(walls_fam1) - 1):
            w0a, w0b = walls_fam0[i], walls_fam0[i+1]
            w1a, w1b = walls_fam1[j], walls_fam1[j+1]
            
            # 4 corners from intersections
            corners = []
            for w0 in [w0a, w0b]:
                for w1 in [w1a, w1b]:
                    pt = line_intersection_2d(w0, w1)
                    if pt:
                        corners.append(pt)
            
            if len(corners) != 4:
                continue
            
            # Order corners as polygon
            cx = np.mean([c[0] for c in corners])
            cz = np.mean([c[1] for c in corners])
            corners.sort(key=lambda c: math.atan2(c[1]-cz, c[0]-cx))
            
            rooms.append({
                'corners': corners,
                'fam0_idx': (i, i+1),
                'fam1_idx': (j, j+1),
            })
    
    return rooms


def room_area(corners):
    """Shoelace formula for polygon area."""
    n = len(corners)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2


def room_overlaps_apartment(corners, mesh_points_xz, margin=0.3):
    """Check if room cell overlaps with actual apartment (has mesh points inside)."""
    from matplotlib.path import Path
    poly_path = Path(corners)
    inside = poly_path.contains_points(mesh_points_xz)
    # Need significant overlap
    return inside.sum() > 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path', help='Path to mesh file')
    parser.add_argument('--output-dir', default='results/v53_ransac_planes')
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(f"  {len(verts)} vertices, {len(faces)} faces")
    
    # Step 1: Extract wall faces
    print("\nStep 1: Extracting wall faces...")
    wall_mask = extract_wall_faces(mesh)
    n_wall = wall_mask.sum()
    print(f"  {n_wall}/{len(faces)} wall faces ({100*n_wall/len(faces):.0f}%)")
    
    # Get wall face centroids as point cloud
    wall_face_idx = np.where(wall_mask)[0]
    wall_centroids = mesh.triangles_center[wall_face_idx]
    print(f"  {len(wall_centroids)} wall centroids")
    
    # Step 2: Sequential RANSAC on wall centroids
    print("\nStep 2: RANSAC plane detection...")
    planes = sequential_ransac(
        wall_centroids, 
        min_points=300,  # minimum wall size
        max_planes=15,
        n_iter=3000,
        thresh=0.03  # 3cm tolerance
    )
    print(f"  Found {len(planes)} planes")
    
    # Step 3: Filter to vertical planes and convert to 2D lines
    print("\nStep 3: Converting to 2D wall lines...")
    wall_lines = []
    for i, p in enumerate(planes):
        # Check if plane is vertical (normal mostly horizontal)
        ny = abs(p['normal'][1])
        if ny > 0.3:
            print(f"  Plane {i}: {p['n_points']} pts, normal_y={ny:.2f} → SKIP (not vertical)")
            continue
        
        line = plane_to_2d_line(p['normal'], p['d'])
        if line is None:
            continue
        
        # Get wall extent
        inlier_pts = wall_centroids[p['inlier_mask']]
        t_min, t_max = get_wall_extent(inlier_pts, p['normal'], p['d'], line)
        line['t_min'] = t_min
        line['t_max'] = t_max
        line['n_points'] = p['n_points']
        line['plane_idx'] = i
        
        angle_deg = math.degrees(line['angle']) % 180
        print(f"  Plane {i}: {p['n_points']} pts, angle={angle_deg:.1f}°, offset={line['offset']:.2f}m, length={t_max-t_min:.1f}m")
        wall_lines.append(line)
    
    print(f"\n  {len(wall_lines)} vertical wall lines")
    
    if len(wall_lines) < 4:
        print("ERROR: Not enough wall lines found!")
        return
    
    # Step 4: Cluster by angle
    print("\nStep 4: Clustering by angle...")
    families = cluster_by_angle(wall_lines)
    
    for fi, fam in enumerate(families):
        angles = [math.degrees(wall_lines[i]['angle']) % 180 for i in fam]
        mean_angle = np.mean(angles)
        print(f"  Family {fi}: {len(fam)} walls, mean angle={mean_angle:.1f}°")
        for idx in fam:
            wl = wall_lines[idx]
            print(f"    Wall {idx}: {wl['n_points']} pts, offset={wl['offset']:.2f}m, length={wl['t_max']-wl['t_min']:.1f}m")
    
    if len(families) < 2:
        print("ERROR: Need at least 2 angle families!")
        return
    
    fam0_lines = [wall_lines[i] for i in families[0]]
    fam1_lines = [wall_lines[i] for i in families[1]]
    
    # Step 5: Build room grid from wall intersections
    print("\nStep 5: Building room grid...")
    
    # Sample mesh points in XZ for overlap checking
    sample_idx = np.random.choice(len(verts), min(10000, len(verts)), replace=False)
    mesh_xz = verts[sample_idx][:, [0, 2]]
    
    xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
    
    rooms = build_room_polygons(fam0_lines, fam1_lines, mesh_xz, (xmin, xmax, zmin, zmax))
    print(f"  {len(rooms)} grid cells")
    
    # Filter to cells that overlap apartment
    valid_rooms = []
    for r in rooms:
        area = room_area(r['corners'])
        if area < 1.0 or area > 25.0:
            continue
        if room_overlaps_apartment(r['corners'], mesh_xz):
            r['area'] = area
            valid_rooms.append(r)
    
    print(f"  {len(valid_rooms)} rooms overlapping apartment")
    
    # Classify rooms
    for r in valid_rooms:
        a = r['area']
        if a > 8:
            r['label'] = f"Room"
        elif a > 4:
            r['label'] = f"Hallway"
        elif a > 2:
            r['label'] = f"Bathroom"
        else:
            r['label'] = f"Closet"
    
    total_area = sum(r['area'] for r in valid_rooms)
    
    # Step 6: Render
    print(f"\nStep 6: Rendering {len(valid_rooms)} rooms, {total_area:.1f}m² total...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Left: wall lines + planes overlay
    ax = axes[0]
    ax.set_title('RANSAC Wall Planes (XZ projection)')
    ax.scatter(verts[::10, 0], verts[::10, 2], s=0.1, c='#ddd', alpha=0.3)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(wall_lines)))
    for i, wl in enumerate(wall_lines):
        # Draw wall line segment
        wx, wz = -wl['nz'], wl['nx']  # wall direction
        cx = -wl['d'] * wl['nx']  # point on line
        cz = -wl['d'] * wl['nz']
        
        p1 = (cx + wx * wl['t_min'], cz + wz * wl['t_min'])
        p2 = (cx + wx * wl['t_max'], cz + wz * wl['t_max'])
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=colors[i], linewidth=2, 
                label=f"W{i}: {wl['n_points']}pts")
    
    # Draw plane inlier points
    for i, p in enumerate(planes):
        if i >= len(colors):
            break
        inlier_pts = wall_centroids[p['inlier_mask']]
        ax.scatter(inlier_pts[::5, 0], inlier_pts[::5, 2], s=1, color=colors[min(i, len(colors)-1)], alpha=0.3)
    
    ax.set_aspect('equal')
    ax.legend(fontsize=6, loc='upper left')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Right: room polygons
    ax = axes[1]
    ax.set_title(f'v53 RANSAC Planes — {len(valid_rooms)} rooms, {total_area:.1f}m²')
    
    # Draw mesh outline
    ax.scatter(verts[::10, 0], verts[::10, 2], s=0.1, c='#eee', alpha=0.3)
    
    room_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(valid_rooms), 1)))
    for i, r in enumerate(valid_rooms):
        corners = r['corners']
        xs = [c[0] for c in corners] + [corners[0][0]]
        zs = [c[1] for c in corners] + [corners[0][1]]
        
        ax.fill(xs, zs, color=room_colors[i], alpha=0.4)
        ax.plot(xs, zs, 'k-', linewidth=1.5)
        
        # Double-line walls
        for j in range(len(corners)):
            k = (j + 1) % len(corners)
            dx = corners[k][0] - corners[j][0]
            dz = corners[k][1] - corners[j][1]
            length = math.sqrt(dx**2 + dz**2)
            if length < 0.01:
                continue
            nx_w = -dz / length * 0.08
            nz_w = dx / length * 0.08
            ax.plot([corners[j][0]+nx_w, corners[k][0]+nx_w],
                    [corners[j][1]+nz_w, corners[k][1]+nz_w], 'k-', linewidth=0.5)
            ax.plot([corners[j][0]-nx_w, corners[k][0]-nx_w],
                    [corners[j][1]-nz_w, corners[k][1]-nz_w], 'k-', linewidth=0.5)
        
        cx = np.mean([c[0] for c in corners])
        cz = np.mean([c[1] for c in corners])
        ax.text(cx, cz, f"{r['label']}\n{r['area']:.1f}m²", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw all wall lines extended
    for wl in wall_lines:
        wx, wz = -wl['nz'], wl['nx']
        cx = -wl['d'] * wl['nx']
        cz = -wl['d'] * wl['nz']
        p1 = (cx + wx * wl['t_min'], cz + wz * wl['t_min'])
        p2 = (cx + wx * wl['t_max'], cz + wz * wl['t_max'])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='red', linewidth=0.5, alpha=0.5)
    
    # Scale bar
    ax.plot([xmin + 0.3, xmin + 1.3], [zmin + 0.3, zmin + 0.3], 'k-', linewidth=3)
    ax.text(xmin + 0.8, zmin + 0.5, '1m', ha='center', fontsize=8)
    
    # Info box
    angles_str = ', '.join([f"{math.degrees(wall_lines[families[fi][0]]['angle'])%180:.0f}°" for fi in range(len(families))])
    ax.text(0.02, 0.98, f"Wall angles: {angles_str}\n{len(valid_rooms)} rooms, {total_area:.1f}m²\n{len(wall_lines)} RANSAC planes",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    out_path = out / 'floorplan.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    
    # Save data
    summary = {
        'version': 'v53_ransac_planes',
        'n_planes': len(planes),
        'n_wall_lines': len(wall_lines),
        'n_rooms': len(valid_rooms),
        'total_area_m2': round(total_area, 1),
        'wall_angles': [round(math.degrees(wall_lines[families[fi][0]]['angle']) % 180, 1) for fi in range(len(families))],
        'rooms': [{
            'label': r['label'],
            'area_m2': round(r['area'], 1),
            'vertices': len(r['corners']),
            'corners': [(round(c[0], 2), round(c[1], 2)) for c in r['corners']]
        } for r in valid_rooms]
    }
    
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDone! {len(valid_rooms)} rooms, {total_area:.1f}m²")
    for r in valid_rooms:
        print(f"  {r['label']}: {r['area']:.1f}m² ({len(r['corners'])}v)")


if __name__ == '__main__':
    main()

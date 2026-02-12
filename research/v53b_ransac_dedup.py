#!/usr/bin/env python3
"""
mesh2plan v53b - RANSAC Planes with Deduplication

v53 found correct wall angles (119°/29°) and real wall positions, but
RANSAC detected multiple planes for each physical wall (e.g., 4 planes at
offset ~0.5m). v53b merges close parallel planes → clean wall set → proper rooms.

Pipeline:
1. RANSAC → 15 vertical planes (same as v53)
2. Cluster into 2 angle families
3. MERGE planes within 0.4m offset → unique walls
4. Grid intersection → room cells
5. Filter by apartment overlap
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from pathlib import Path
import argparse
import json
import math
from scipy.spatial import ConvexHull


def extract_wall_faces(mesh):
    normals = mesh.face_normals
    horiz = np.abs(normals[:, 1])
    return horiz < 0.3


def ransac_plane(points, n_iter=1000, thresh=0.02):
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
    planes = []
    remaining = np.ones(len(points), dtype=bool)
    
    for iteration in range(max_planes):
        pts = points[remaining]
        if len(pts) < min_points:
            break
        
        if len(pts) > 5000:
            sample_idx = np.random.choice(len(pts), 5000, replace=False)
            sample_pts = pts[sample_idx]
        else:
            sample_pts = pts
        
        normal, d, _ = ransac_plane(sample_pts, n_iter=n_iter, thresh=thresh)
        if normal is None:
            break
        
        dists = np.abs(pts @ normal + d)
        inlier_mask = dists < thresh
        
        if inlier_mask.sum() < min_points:
            break
        
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
        print(f"    Plane {iteration}: {inlier_mask.sum()} inliers")
    
    return planes


def plane_to_2d_line(normal, d):
    nx, ny, nz = normal
    horiz_norm = math.sqrt(nx**2 + nz**2)
    if horiz_norm < 0.1:
        return None
    nx2 = nx / horiz_norm
    nz2 = nz / horiz_norm
    d2 = d / horiz_norm
    wall_angle = math.atan2(nz2, nx2) + math.pi / 2
    offset = -d2
    return {
        'angle': wall_angle,
        'offset': offset,
        'nx': nx2, 'nz': nz2, 'd': d2
    }


def get_wall_extent(points_3d, wall_line):
    wx, wz = -wall_line['nz'], wall_line['nx']
    projections = points_3d[:, 0] * wx + points_3d[:, 2] * wz
    return projections.min(), projections.max()


def merge_close_walls(walls, min_offset_gap=0.4):
    """Merge walls with similar offsets (same physical wall detected multiple times)."""
    if not walls:
        return []
    
    walls_sorted = sorted(walls, key=lambda w: w['offset'])
    merged = [walls_sorted[0]]
    
    for w in walls_sorted[1:]:
        if abs(w['offset'] - merged[-1]['offset']) < min_offset_gap:
            # Merge: keep the one with more points, combine extent
            if w['n_points'] > merged[-1]['n_points']:
                w['t_min'] = min(w['t_min'], merged[-1]['t_min'])
                w['t_max'] = max(w['t_max'], merged[-1]['t_max'])
                w['n_points'] += merged[-1]['n_points']
                merged[-1] = w
            else:
                merged[-1]['t_min'] = min(w['t_min'], merged[-1]['t_min'])
                merged[-1]['t_max'] = max(w['t_max'], merged[-1]['t_max'])
                merged[-1]['n_points'] += w['n_points']
        else:
            merged.append(w)
    
    return merged


def cluster_by_angle(lines, angle_thresh=15):
    angles_deg = [math.degrees(l['angle']) % 180 for l in lines]
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
    families.sort(key=len, reverse=True)
    return families[:2]


def line_intersection_2d(l1, l2):
    a1, b1, c1 = l1['nx'], l1['nz'], l1['d']
    a2, b2, c2 = l2['nx'], l2['nz'], l2['d']
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    z = (-a1 * c2 + a2 * c1) / det
    return (x, z)


def room_area(corners):
    n = len(corners)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', default='results/v53b_ransac_dedup')
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(f"  {len(verts)} verts, {len(faces)} faces")
    
    # Step 1: Wall faces
    wall_mask = extract_wall_faces(mesh)
    wall_centroids = mesh.triangles_center[wall_mask]
    print(f"  {len(wall_centroids)} wall centroids")
    
    # Step 2: RANSAC
    print("\nRANSAC...")
    planes = sequential_ransac(wall_centroids, min_points=300, max_planes=15, n_iter=3000, thresh=0.03)
    print(f"  {len(planes)} planes")
    
    # Step 3: Convert to 2D lines
    wall_lines = []
    for i, p in enumerate(planes):
        if abs(p['normal'][1]) > 0.3:
            continue
        line = plane_to_2d_line(p['normal'], p['d'])
        if line is None:
            continue
        inlier_pts = wall_centroids[p['inlier_mask']]
        t_min, t_max = get_wall_extent(inlier_pts, line)
        line['t_min'] = t_min
        line['t_max'] = t_max
        line['n_points'] = p['n_points']
        wall_lines.append(line)
    
    print(f"  {len(wall_lines)} vertical wall lines")
    
    # Step 4: Cluster by angle
    families = cluster_by_angle(wall_lines)
    fam0_raw = [wall_lines[i] for i in families[0]]
    fam1_raw = [wall_lines[i] for i in families[1]]
    
    mean_angle0 = np.mean([math.degrees(w['angle']) % 180 for w in fam0_raw])
    mean_angle1 = np.mean([math.degrees(w['angle']) % 180 for w in fam1_raw])
    print(f"\n  Family 0: {len(fam0_raw)} walls at ~{mean_angle0:.0f}°")
    for w in sorted(fam0_raw, key=lambda x: x['offset']):
        print(f"    offset={w['offset']:.2f}m, {w['n_points']}pts, len={w['t_max']-w['t_min']:.1f}m")
    print(f"  Family 1: {len(fam1_raw)} walls at ~{mean_angle1:.0f}°")
    for w in sorted(fam1_raw, key=lambda x: x['offset']):
        print(f"    offset={w['offset']:.2f}m, {w['n_points']}pts, len={w['t_max']-w['t_min']:.1f}m")
    
    # Step 5: MERGE close parallel walls
    print("\nMerging close walls (gap < 0.4m)...")
    fam0 = merge_close_walls(fam0_raw, min_offset_gap=0.4)
    fam1 = merge_close_walls(fam1_raw, min_offset_gap=0.4)
    
    print(f"  Family 0: {len(fam0_raw)} → {len(fam0)} unique walls")
    for w in fam0:
        print(f"    offset={w['offset']:.2f}m, {w['n_points']}pts")
    print(f"  Family 1: {len(fam1_raw)} → {len(fam1)} unique walls")
    for w in fam1:
        print(f"    offset={w['offset']:.2f}m, {w['n_points']}pts")
    
    if len(fam0) < 2 or len(fam1) < 2:
        print("ERROR: Not enough unique walls!")
        return
    
    # Step 6: Build room grid
    print("\nBuilding room grid...")
    fam0.sort(key=lambda w: w['offset'])
    fam1.sort(key=lambda w: w['offset'])
    
    # Sample mesh XZ for overlap check
    sample_idx = np.random.choice(len(verts), min(20000, len(verts)), replace=False)
    mesh_xz = verts[sample_idx][:, [0, 2]]
    
    rooms = []
    for i in range(len(fam0) - 1):
        for j in range(len(fam1) - 1):
            corners = []
            for w0 in [fam0[i], fam0[i+1]]:
                for w1 in [fam1[j], fam1[j+1]]:
                    pt = line_intersection_2d(w0, w1)
                    if pt:
                        corners.append(pt)
            
            if len(corners) != 4:
                continue
            
            cx = np.mean([c[0] for c in corners])
            cz = np.mean([c[1] for c in corners])
            corners.sort(key=lambda c: math.atan2(c[1]-cz, c[0]-cx))
            
            area = room_area(corners)
            if area < 0.5 or area > 25:
                continue
            
            # Check overlap with apartment
            try:
                poly_path = MplPath(corners)
                inside = poly_path.contains_points(mesh_xz)
                # Require significant mesh coverage
                coverage = inside.sum() / max(1, (area / 0.02))  # points per m²
                if inside.sum() < 20:
                    continue
            except:
                continue
            
            rooms.append({
                'corners': corners,
                'area': area,
                'mesh_points': inside.sum(),
                'fam0': (i, i+1),
                'fam1': (j, j+1),
            })
    
    # Classify
    for r in rooms:
        a = r['area']
        if a > 8:
            r['label'] = "Room"
        elif a > 4:
            r['label'] = "Hallway"
        elif a > 2:
            r['label'] = "Bathroom"
        else:
            r['label'] = "Closet"
    
    total_area = sum(r['area'] for r in rooms)
    print(f"  {len(rooms)} rooms, {total_area:.1f}m²")
    for r in rooms:
        print(f"  {r['label']}: {r['area']:.1f}m² ({r['mesh_points']} pts)")
    
    # Step 7: Render
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Left: wall lines
    ax = axes[0]
    ax.set_title(f'RANSAC Walls (merged): {len(fam0)}+{len(fam1)} unique walls')
    ax.scatter(verts[::10, 0], verts[::10, 2], s=0.1, c='#ddd', alpha=0.3)
    
    for i, w in enumerate(fam0):
        wx, wz = -w['nz'], w['nx']
        cx_w = -w['d'] * w['nx']
        cz_w = -w['d'] * w['nz']
        p1 = (cx_w + wx * w['t_min'], cz_w + wz * w['t_min'])
        p2 = (cx_w + wx * w['t_max'], cz_w + wz * w['t_max'])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='blue', linewidth=2,
                label=f"F0-{i}: off={w['offset']:.2f}" if i < 8 else None)
    
    for i, w in enumerate(fam1):
        wx, wz = -w['nz'], w['nx']
        cx_w = -w['d'] * w['nx']
        cz_w = -w['d'] * w['nz']
        p1 = (cx_w + wx * w['t_min'], cz_w + wz * w['t_min'])
        p2 = (cx_w + wx * w['t_max'], cz_w + wz * w['t_max'])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='red', linewidth=2,
                label=f"F1-{i}: off={w['offset']:.2f}" if i < 8 else None)
    
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    # Right: rooms
    ax = axes[1]
    ax.set_title(f'v53b RANSAC Dedup — {len(rooms)} rooms, {total_area:.1f}m²')
    ax.scatter(verts[::10, 0], verts[::10, 2], s=0.1, c='#eee', alpha=0.3)
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    for i, r in enumerate(rooms):
        corners = r['corners']
        xs = [c[0] for c in corners] + [corners[0][0]]
        zs = [c[1] for c in corners] + [corners[0][1]]
        ax.fill(xs, zs, color=colors[i], alpha=0.5)
        ax.plot(xs, zs, 'k-', linewidth=2)
        
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
    
    # Scale bar
    xmin = verts[:, 0].min()
    zmin = verts[:, 2].min()
    ax.plot([xmin + 0.3, xmin + 1.3], [zmin + 0.3, zmin + 0.3], 'k-', linewidth=3)
    ax.text(xmin + 0.8, zmin + 0.5, '1m', ha='center', fontsize=8)
    
    ax.text(0.02, 0.98, f"Wall angles: {mean_angle0:.0f}°, {mean_angle1:.0f}°\n"
            f"{len(rooms)} rooms, {total_area:.1f}m²\n"
            f"{len(fam0)}+{len(fam1)} unique walls (from {len(wall_lines)} RANSAC planes)",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    out_path = out / 'floorplan.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")
    
    summary = {
        'version': 'v53b_ransac_dedup',
        'n_ransac_planes': len(wall_lines),
        'n_unique_walls': [len(fam0), len(fam1)],
        'wall_angles': [round(mean_angle0, 1), round(mean_angle1, 1)],
        'n_rooms': len(rooms),
        'total_area_m2': round(total_area, 1),
        'rooms': [{'label': r['label'], 'area_m2': round(r['area'], 1)} for r in rooms]
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

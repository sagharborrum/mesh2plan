#!/usr/bin/env python3
"""
mesh2plan v61e - Four Rooms: combo search with quality scoring

Use top-10 walls, try all 3-5 wall combinations, score by:
- Must produce exactly 4 rooms
- Bedroom sizes should be ~10m² each (minimize variance)
- Total area should be close to boundary area
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import polygonize, unary_union
from itertools import combinations


RESOLUTION = 0.02
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
WALL_ANGLES = np.array([29.0, 119.0])
ANGLE_TOL = 18


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def slice_mesh(mesh, y):
    try:
        lines = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, y, 0])
    except:
        return np.array([]).reshape(0, 2, 2)
    if lines is None or len(lines) == 0:
        return np.array([]).reshape(0, 2, 2)
    return lines[:, :, [0, 2]]


def segments_to_density(segs_list, res):
    all_pts = np.concatenate([s.reshape(-1, 2) for s in segs_list if len(s) > 0])
    xmin, zmin = all_pts.min(axis=0) - 0.5
    xmax, zmax = all_pts.max(axis=0) + 0.5
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    density = np.zeros((h, w), dtype=np.float32)
    for segs in segs_list:
        if len(segs) == 0: continue
        img = np.zeros((h, w), dtype=np.uint8)
        for seg in segs:
            p1x = int(np.clip((seg[0,0]-xmin)/res, 0, w-1))
            p1y = int(np.clip((seg[0,1]-zmin)/res, 0, h-1))
            p2x = int(np.clip((seg[1,0]-xmin)/res, 0, w-1))
            p2y = int(np.clip((seg[1,1]-zmin)/res, 0, h-1))
            cv2.line(img, (p1x, p1y), (p2x, p2y), 1, 1)
        density += img.astype(np.float32)
    return density, (xmin, zmin, xmax, zmax, w, h)


def extract_walls(density, grid_info):
    xmin, zmin, xmax, zmax, w, h = grid_info
    wall_mask = (density >= 2).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    wall_mask = cv2.dilate(wall_mask, k, 1)
    skel = skeletonize(wall_mask > 0)
    
    # Also lower threshold
    wall_mask2 = (density >= 1).astype(np.uint8)
    wall_mask2 = cv2.dilate(wall_mask2, k, 1)
    skel2 = skeletonize(wall_mask2 > 0)
    combined = skel | skel2
    
    lines = probabilistic_hough_line(combined, threshold=6, line_length=10, line_gap=10)
    
    segments = []
    for (x1,y1),(x2,y2) in lines:
        wx1 = x1*RESOLUTION+xmin; wz1 = y1*RESOLUTION+zmin
        wx2 = x2*RESOLUTION+xmin; wz2 = y2*RESOLUTION+zmin
        dx = wx2-wx1; dz = wz2-wz1
        length = np.sqrt(dx**2+dz**2)
        if length < 0.3: continue
        
        angle = np.degrees(np.arctan2(dz, dx)) % 180
        best_da = min(WALL_ANGLES, key=lambda da: min(abs(angle-da), 180-abs(angle-da)))
        diff = abs(angle - best_da); diff = min(diff, 180-diff)
        if diff > ANGLE_TOL: continue
        
        mid = np.array([(wx1+wx2)/2, (wz1+wz2)/2])
        rad = np.radians(best_da)
        d = np.array([np.cos(rad), np.sin(rad)])
        n = np.array([-np.sin(rad), np.cos(rad)])
        half = length/2
        
        # Density score
        ts = np.linspace(-half, half, max(int(length/RESOLUTION), 10))
        pts = mid + ts[:,None] * d
        px = ((pts[:,0]-xmin)/RESOLUTION).astype(int)
        py = ((pts[:,1]-zmin)/RESOLUTION).astype(int)
        v = (px>=0)&(px<w)&(py>=0)&(py<h)
        dscore = float(density[py[v], px[v]].sum()) if v.any() else 0
        
        segments.append({
            'p1': mid-d*half, 'p2': mid+d*half,
            'angle': best_da, 'length': length, 'mid': mid,
            'perp': float(np.dot(mid, n)), 'direction': d, 'normal': n,
            'dscore': dscore,
        })
    
    return segments, skel


def merge_dedup(segments, max_perp=0.2, max_gap=0.6, dedup_gap=0.4):
    if not segments: return []
    by_a = {}
    for s in segments:
        by_a.setdefault(round(s['angle']), []).append(s)
    
    merged = []
    for angle, grp in by_a.items():
        grp.sort(key=lambda s: s['perp'])
        d = grp[0]['direction']; n = grp[0]['normal']
        
        clusters = [[grp[0]]]
        for s in grp[1:]:
            if abs(s['perp']-clusters[-1][-1]['perp']) < max_perp:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        
        for cl in clusters:
            projs = [(min(np.dot(s['p1'],d), np.dot(s['p2'],d)),
                       max(np.dot(s['p1'],d), np.dot(s['p2'],d))) for s in cl]
            projs.sort()
            intervals = [list(projs[0])]
            for a, b in projs[1:]:
                if a <= intervals[-1][1]+max_gap: intervals[-1][1] = max(intervals[-1][1], b)
                else: intervals.append([a, b])
            
            mp = np.mean([s['perp'] for s in cl])
            td = sum(s['dscore'] for s in cl)
            
            for a, b in intervals:
                l = b-a; mid = d*(a+b)/2 + n*mp
                merged.append({
                    'p1': mid-d*(l/2), 'p2': mid+d*(l/2),
                    'angle': angle, 'length': l, 'mid': mid,
                    'perp': mp, 'direction': d, 'normal': n,
                    'n_segs': len(cl), 'dscore': td,
                })
    
    # Dedup
    by_a2 = {}
    for s in merged:
        by_a2.setdefault(round(s['angle']), []).append(s)
    deduped = []
    for angle, grp in by_a2.items():
        grp.sort(key=lambda s: s['perp'])
        keep = [grp[0]]
        for s in grp[1:]:
            if abs(s['perp']-keep[-1]['perp']) < dedup_gap:
                if s['dscore'] > keep[-1]['dscore']: keep[-1] = s
            else: keep.append(s)
        deduped.extend(keep)
    
    return deduped


def make_boundary(mesh, grid_info):
    xmin, zmin, xmax, zmax, w, h = grid_info
    verts = mesh.vertices[:, [0,2]]
    img = np.zeros((h, w), dtype=np.uint8)
    px = ((verts[:,0]-xmin)/RESOLUTION).astype(int)
    py = ((verts[:,1]-zmin)/RESOLUTION).astype(int)
    px = np.clip(px, 0, w-1); py = np.clip(py, 0, h-1)
    for x, y in zip(px, py): img[y, x] = 255
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    img = binary_fill_holes(img).astype(np.uint8) * 255
    
    # Moderate erosion
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.erode(img, k2, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k2)
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0]*RESOLUTION+xmin
    pts[:, 1] = pts[:, 1]*RESOLUTION+zmin
    poly = Polygon(pts).buffer(0).simplify(0.12)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def extend_wall(seg, boundary, ext=15.0):
    d = seg['direction']; mid = seg['mid']
    line = LineString([mid-d*ext, mid+d*ext])
    cl = line.intersection(boundary)
    if cl.is_empty: return None
    if isinstance(cl, MultiLineString):
        cl = max(cl.geoms, key=lambda g: g.length)
    return cl if isinstance(cl, LineString) else None


def try_walls(walls, boundary, min_area=1.5):
    """Try a set of walls, return resulting rooms."""
    lines = [boundary.exterior]
    for w in walls:
        ext = extend_wall(w, boundary)
        if ext: lines.append(ext)
    union = unary_union(lines)
    result = list(polygonize(union))
    rooms = [p for p in result if boundary.contains(p.representative_point()) and p.area > min_area]
    rooms.sort(key=lambda r: r.area, reverse=True)
    return rooms


def score_partition(rooms):
    """Score a 4-room partition. Lower = better.
    Prefer: two ~10m² bedrooms, one ~4m² hallway, one ~3m² bathroom."""
    if len(rooms) != 4:
        return float('inf')
    
    areas = sorted([r.area for r in rooms], reverse=True)
    
    # Two largest should be ~10m² each
    bed_penalty = abs(areas[0] - 10) + abs(areas[1] - 10)
    
    # No room should be > 15m²
    if areas[0] > 15:
        bed_penalty += (areas[0] - 15) * 5
    
    # Two smallest should be 2-6m²
    small_penalty = 0
    for a in areas[2:]:
        if a < 2:
            small_penalty += (2 - a) * 3
        elif a > 8:
            small_penalty += (a - 8) * 2
    
    # Total coverage — closer to boundary area is better
    total = sum(areas)
    coverage_penalty = abs(total - 35) * 0.5
    
    return bed_penalty + small_penalty + coverage_penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v61e')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh = load_mesh(data_dir / args.mesh)
    print(f"Loaded: {len(mesh.vertices)} verts")
    
    # Cross-section density
    segs_list = [slice_mesh(mesh, y) for y in SLICE_HEIGHTS]
    for y, s in zip(SLICE_HEIGHTS, segs_list):
        print(f"  Y={y}: {len(s)} segs")
    density, grid_info = segments_to_density(segs_list, RESOLUTION)
    
    # Walls
    raw, skel = extract_walls(density, grid_info)
    walls = merge_dedup(raw)
    walls.sort(key=lambda w: w['length']*w['dscore'], reverse=True)
    
    print(f"\nWalls ({len(walls)}):")
    for i, w in enumerate(walls):
        print(f"  [{i}] {w['angle']:.0f}° perp={w['perp']:.2f} len={w['length']:.1f}m dscore={w['dscore']:.0f}")
    
    # Boundary
    boundary = make_boundary(mesh, grid_info)
    print(f"\nBoundary: {boundary.area:.1f}m²")
    
    # Search for best 4-room partition from top-12 walls
    top = walls[:min(12, len(walls))]
    
    best = None
    best_score = float('inf')
    
    for n in range(2, min(7, len(top)+1)):
        for combo in combinations(range(len(top)), n):
            selected = [top[i] for i in combo]
            rooms = try_walls(selected, boundary)
            if len(rooms) == 4:
                s = score_partition(rooms)
                if s < best_score:
                    best_score = s
                    best = (selected, rooms)
                    areas = sorted([r.area for r in rooms], reverse=True)
                    print(f"  Combo {combo}: {[f'{a:.1f}' for a in areas]} score={s:.1f}")
    
    if best is None:
        # Try with merge
        print("\nNo exact 4-room found. Using all walls + merge...")
        rooms = try_walls(top[:8], boundary)
        print(f"  {len(rooms)} rooms with top-8 walls")
        while len(rooms) > 4:
            idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
            best_j = max(range(len(rooms)), 
                        key=lambda j: rooms[idx].buffer(0.1).intersection(rooms[j].buffer(0.1)).area if j != idx else -1)
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        rooms.sort(key=lambda r: r.area, reverse=True)
        best = (top[:8], rooms)
    
    used, rooms = best
    
    # Classify
    if len(rooms) == 4:
        by_area = sorted(range(4), key=lambda i: rooms[i].area, reverse=True)
        labels = [None]*4
        labels[by_area[0]] = "Bedroom 1"
        labels[by_area[1]] = "Bedroom 2"
        rem = by_area[2:]
        aspects = [(i, max(rooms[i].bounds[2]-rooms[i].bounds[0], rooms[i].bounds[3]-rooms[i].bounds[1]) / 
                    (min(rooms[i].bounds[2]-rooms[i].bounds[0], rooms[i].bounds[3]-rooms[i].bounds[1])+1e-6)) 
                   for i in rem]
        aspects.sort(key=lambda x: x[1], reverse=True)
        labels[aspects[0][0]] = "Hallway"
        labels[aspects[1][0]] = "Bathroom"
    else:
        labels = [f"Room {i+1}" for i in range(len(rooms))]
    
    # Plot
    xmin, zmin, xmax, zmax, w, h = grid_info
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    ax.set_title("Cross-Section Density")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=2)
    for w_seg in used:
        ext = extend_wall(w_seg, boundary)
        if ext:
            xs, ys = ext.xy
            color = 'red' if round(w_seg['angle'])==29 else 'blue'
            ax.plot(xs, ys, color=color, linewidth=2.5)
    ax.set_title(f"Selected Walls ({len(used)})")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9']
    total = 0
    for i, (room, label) in enumerate(zip(rooms, labels)):
        c = pastel[i%len(pastel)]
        geoms = room.geoms if isinstance(room, MultiPolygon) else [room]
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color=c, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2.5)
        area = room.area; total += area
        cx, cy = room.centroid.coords[0]
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center',
                fontsize=9, fontweight='bold')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=3)
    ax.set_title(f"v61e — {len(rooms)} rooms, {total:.1f}m²")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}")
    
    print(f"\nFinal:")
    for room, label in zip(rooms, labels):
        print(f"  {label}: {room.area:.1f}m² ({len(room.exterior.coords)-1}v)")
    print(f"Total: {total:.1f}m², score: {best_score:.1f}")


if __name__ == '__main__':
    main()

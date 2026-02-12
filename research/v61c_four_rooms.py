#!/usr/bin/env python3
"""
mesh2plan v61c - Four Rooms: v60 pipeline + iterative wall selection

Uses v60's proven cross-section + probabilistic Hough + angle snap pipeline,
but with smarter wall selection:
1. Extract all walls (v60 style)  
2. Start with strongest walls, add one at a time
3. Stop when we reach 4 rooms
4. If we overshoot, binary search for the right wall set
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
MIN_SEG_LEN = 0.4


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


def extract_wall_segments(density, grid_info):
    """v60-style: threshold → skeleton → probabilistic Hough → angle snap → merge."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    wall_mask = (density >= 2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
    skeleton = skeletonize(wall_mask > 0)
    
    lines = probabilistic_hough_line(skeleton, threshold=8, line_length=15, line_gap=8)
    print(f"  Raw Hough segments: {len(lines)}")
    
    segments = []
    for (x1, y1), (x2, y2) in lines:
        wx1 = x1 * RESOLUTION + xmin
        wz1 = y1 * RESOLUTION + zmin
        wx2 = x2 * RESOLUTION + xmin
        wz2 = y2 * RESOLUTION + zmin
        
        dx = wx2 - wx1
        dz = wz2 - wz1
        length = np.sqrt(dx**2 + dz**2)
        if length < MIN_SEG_LEN:
            continue
        
        angle = np.degrees(np.arctan2(dz, dx)) % 180
        best_da = None
        best_diff = 180
        for da in WALL_ANGLES:
            diff = abs(angle - da)
            diff = min(diff, 180 - diff)
            if diff < best_diff:
                best_diff = diff
                best_da = da
        
        if best_diff > ANGLE_TOL:
            continue
        
        mid = np.array([(wx1+wx2)/2, (wz1+wz2)/2])
        rad = np.radians(best_da)
        d = np.array([np.cos(rad), np.sin(rad)])
        n = np.array([-np.sin(rad), np.cos(rad)])
        half = length / 2
        
        segments.append({
            'p1': mid - d * half,
            'p2': mid + d * half,
            'angle': best_da,
            'length': length,
            'mid': mid,
            'perp': np.dot(mid, n),
            'direction': d,
            'normal': n,
        })
    
    print(f"  Angle-filtered: {len(segments)}")
    return segments, skeleton


def merge_and_dedup(segments, max_perp=0.15, max_gap=0.5, min_dedup_gap=0.35):
    """Merge collinear segments, then deduplicate parallel walls."""
    if not segments:
        return []
    
    by_angle = {}
    for s in segments:
        a = round(s['angle'])
        by_angle.setdefault(a, []).append(s)
    
    merged = []
    for angle, group in by_angle.items():
        group.sort(key=lambda s: s['perp'])
        
        # Cluster by perp offset
        clusters = [[group[0]]]
        for s in group[1:]:
            if abs(s['perp'] - clusters[-1][-1]['perp']) < max_perp:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        
        d = group[0]['direction']
        n = group[0]['normal']
        
        for cluster in clusters:
            projs = []
            for s in cluster:
                p1 = np.dot(s['p1'], d)
                p2 = np.dot(s['p2'], d)
                projs.append((min(p1, p2), max(p1, p2)))
            projs.sort()
            
            intervals = [list(projs[0])]
            for start, end in projs[1:]:
                if start <= intervals[-1][1] + max_gap:
                    intervals[-1][1] = max(intervals[-1][1], end)
                else:
                    intervals.append([start, end])
            
            mean_perp = np.mean([s['perp'] for s in cluster])
            
            for start, end in intervals:
                length = end - start
                mid_para = (start + end) / 2
                mid = d * mid_para + n * mean_perp
                
                merged.append({
                    'p1': mid - d * (length/2),
                    'p2': mid + d * (length/2),
                    'angle': angle,
                    'length': length,
                    'mid': mid,
                    'perp': mean_perp,
                    'direction': d,
                    'normal': n,
                    'n_segs': len(cluster),
                })
    
    # Dedup parallel walls
    by_angle2 = {}
    for s in merged:
        a = round(s['angle'])
        by_angle2.setdefault(a, []).append(s)
    
    deduped = []
    for angle, group in by_angle2.items():
        group.sort(key=lambda s: s['perp'])
        keep = [group[0]]
        for s in group[1:]:
            if abs(s['perp'] - keep[-1]['perp']) < min_dedup_gap:
                if s['length'] > keep[-1]['length']:
                    keep[-1] = s
            else:
                keep.append(s)
        deduped.extend(keep)
    
    return deduped


def make_boundary(mesh, resolution=0.02):
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


def extend_to_boundary(seg, boundary, ext=15.0):
    d = seg['direction']
    mid = seg['mid']
    line = LineString([mid - d * ext, mid + d * ext])
    clipped = line.intersection(boundary)
    if clipped.is_empty:
        return None
    if isinstance(clipped, MultiLineString):
        clipped = max(clipped.geoms, key=lambda g: g.length)
    return clipped if isinstance(clipped, LineString) else None


def count_rooms(wall_lines, boundary, min_area=1.5):
    """Count rooms from wall lines + boundary."""
    all_geoms = [boundary.exterior] + [wl for wl in wall_lines if wl is not None]
    union = unary_union(all_geoms)
    result = list(polygonize(union))
    rooms = [p for p in result if boundary.contains(p.representative_point()) and p.area > min_area]
    return rooms


def find_4room_partition(walls, boundary):
    """Find the wall subset that gives exactly 4 rooms."""
    # Pre-extend all walls
    for w in walls:
        w['_ext'] = extend_to_boundary(w, boundary)
    
    walls = [w for w in walls if w['_ext'] is not None]
    
    # Score walls: length × n_segs (structural importance)
    for w in walls:
        w['score'] = w['length'] * np.sqrt(w.get('n_segs', 1))
    walls.sort(key=lambda w: w['score'], reverse=True)
    
    print(f"\n  All walls ({len(walls)}):")
    for i, w in enumerate(walls):
        print(f"    [{i}] {w['angle']:.0f}° perp={w['perp']:.2f} len={w['length']:.1f}m "
              f"segs={w.get('n_segs',1)} score={w['score']:.1f}")
    
    # Strategy 1: Greedy add, tracking room count
    best_4 = None
    
    # Try all combinations of 3-5 walls from top-10
    top = walls[:min(12, len(walls))]
    
    for n in range(2, min(7, len(top) + 1)):
        for combo in combinations(range(len(top)), n):
            lines = [top[i]['_ext'] for i in combo]
            rooms = count_rooms(lines, boundary, min_area=1.5)
            if len(rooms) == 4:
                total = sum(r.area for r in rooms)
                if best_4 is None or total > best_4[1]:
                    best_4 = ([top[i] for i in combo], total, rooms)
    
    if best_4:
        used, total, rooms = best_4
        print(f"\n  ✅ Found 4-room partition with {len(used)} walls, {total:.1f}m²")
        return used, rooms
    
    # Fallback: greedy + merge
    print("\n  No exact 4-room partition found, using greedy + merge...")
    selected = []
    best_rooms = count_rooms([], boundary, min_area=1.5)
    
    for w in walls:
        test_lines = [s['_ext'] for s in selected + [w]]
        rooms = count_rooms(test_lines, boundary, min_area=1.5)
        if len(rooms) > len(best_rooms) and len(rooms) <= 6:
            selected.append(w)
            best_rooms = rooms
            print(f"    Added [{walls.index(w)}] → {len(rooms)} rooms")
    
    return selected, best_rooms


def merge_to_4(rooms):
    while len(rooms) > 4:
        idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
        best_j = None
        best_shared = -1
        for j in range(len(rooms)):
            if j == idx:
                continue
            shared = rooms[idx].buffer(0.1).intersection(rooms[j].buffer(0.1)).area
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
    if len(rooms) < 4:
        return [f"Room {i+1}" for i in range(len(rooms))]
    by_area = sorted(range(4), key=lambda i: rooms[i].area, reverse=True)
    labels = [None] * 4
    labels[by_area[0]] = "Bedroom 1"
    labels[by_area[1]] = "Bedroom 2"
    
    rem = by_area[2:]
    aspects = []
    for i in rem:
        b = rooms[i].bounds
        w = b[2] - b[0]
        h = b[3] - b[1]
        aspects.append((i, max(w, h) / (min(w, h) + 1e-6)))
    aspects.sort(key=lambda x: x[1], reverse=True)
    labels[aspects[0][0]] = "Hallway"
    labels[aspects[1][0]] = "Bathroom"
    return labels


def plot_results(density, grid_info, skeleton, walls, rooms, labels, boundary, output_path):
    xmin, zmin, xmax, zmax, w, h = grid_info
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    
    # Panel 1: Density
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot',
              extent=[xmin, xmax, zmin, zmax], aspect='equal')
    ax.set_title(f"Cross-Section Density")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Skeleton + segments
    ax = axes[1]
    ax.imshow(skeleton, origin='lower', cmap='gray',
              extent=[xmin, xmax, zmin, zmax], aspect='equal', alpha=0.4)
    for w_seg in walls:
        ax.plot([w_seg['p1'][0], w_seg['p2'][0]], [w_seg['p1'][1], w_seg['p2'][1]],
                'b-', linewidth=1.5, alpha=0.7)
    ax.set_title(f"All Walls ({len(walls)})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Selected walls extended
    ax = axes[2]
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=2)
    for w_seg in walls:
        if w_seg.get('_ext'):
            xs, ys = w_seg['_ext'].xy
            color = 'red' if round(w_seg['angle']) == 29 else 'blue'
            ax.plot(xs, ys, color=color, linewidth=2.5)
    ax.set_title(f"Extended Walls")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Rooms
    ax = axes[3]
    pastel = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', '#E8BAFF', '#FFD4BA']
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
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center',
                fontsize=9, fontweight='bold')
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=3)
    
    ax.set_title(f"v61c — {len(rooms)} rooms, {total:.1f}m²\n29°/119°")
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
    parser.add_argument('--output', default='output_v61c')
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
    
    # Wall extraction
    print("\nStep 2: Wall extraction...")
    raw_segs, skeleton = extract_wall_segments(density, grid_info)
    
    # Merge + dedup
    print("\nStep 3: Merge + dedup...")
    walls = merge_and_dedup(raw_segs, max_perp=0.15, max_gap=0.5, min_dedup_gap=0.35)
    print(f"  Final walls: {len(walls)}")
    
    # Boundary
    print("\nStep 4: Boundary...")
    boundary = make_boundary(mesh)
    print(f"  Boundary: {boundary.area:.1f}m²")
    
    # Find 4-room partition
    print("\nStep 5: Finding 4-room partition...")
    used, rooms = find_4room_partition(walls, boundary)
    
    if len(rooms) != 4:
        print(f"\nStep 6: Merging {len(rooms)} → 4...")
        rooms = merge_to_4(rooms)
    
    labels = classify_4rooms(rooms)
    
    print(f"\nFinal:")
    for room, label in zip(rooms, labels):
        nv = len(room.exterior.coords) - 1
        print(f"  {label}: {room.area:.1f}m² ({nv}v)")
    
    total = plot_results(density, grid_info, skeleton, used, rooms, labels, boundary, args.output)
    print(f"\nv61c: {len(rooms)} rooms, {total:.1f}m²")


if __name__ == '__main__':
    main()

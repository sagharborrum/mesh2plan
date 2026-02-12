#!/usr/bin/env python3
"""
mesh2plan v62d - Trimmed boundary

v62c worked well but boundary has scan artifacts (tall chimney). 
Fix: use density to define apartment extent, not raw vertex mask.
Boundary = convex-ish hull of dense regions only.
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
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import polygonize, unary_union
from itertools import combinations


RESOLUTION = 0.02
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
WALL_ANGLE = 29.0


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def rotate_points(pts, angle_deg, center=None):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if center is not None: pts = pts - center
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: rotated += center
    return rotated


def slice_mesh(mesh, y):
    try:
        lines = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, y, 0])
    except: return np.array([]).reshape(0, 2, 2)
    if lines is None or len(lines) == 0: return np.array([]).reshape(0, 2, 2)
    return lines[:, :, [0, 2]]


def build_all(mesh, angle_deg):
    pts_xz = mesh.vertices[:, [0, 2]]
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.3
    xmax, zmax = rot_verts.max(axis=0) + 0.3
    w = int((xmax-xmin)/RESOLUTION); h = int((zmax-zmin)/RESOLUTION)
    
    density = np.zeros((h, w), dtype=np.float32)
    for y in SLICE_HEIGHTS:
        segs = slice_mesh(mesh, y)
        if len(segs) == 0: continue
        flat = segs.reshape(-1, 2)
        rot = rotate_points(flat, -angle_deg, center).reshape(-1, 2, 2)
        img = np.zeros((h, w), dtype=np.uint8)
        for seg in rot:
            p1x = int(np.clip((seg[0,0]-xmin)/RESOLUTION, 0, w-1))
            p1y = int(np.clip((seg[0,1]-zmin)/RESOLUTION, 0, h-1))
            p2x = int(np.clip((seg[1,0]-xmin)/RESOLUTION, 0, w-1))
            p2y = int(np.clip((seg[1,1]-zmin)/RESOLUTION, 0, h-1))
            cv2.line(img, (p1x, p1y), (p2x, p2y), 1, 1)
        density += img.astype(np.float32)
    
    grid = (xmin, zmin, xmax, zmax, w, h, center)
    return density, grid


def find_wall_peaks(density, grid, orientation):
    xmin, zmin, xmax, zmax, w, h, center = grid
    if orientation == 'h':
        profile = density.sum(axis=1)
        axis_min = zmin
    else:
        profile = density.sum(axis=0)
        axis_min = xmin
    
    kernel = np.ones(3) / 3
    profile = np.convolve(profile, kernel, mode='same')
    threshold = np.percentile(profile[profile > 0], 50) if (profile > 0).any() else 0
    
    peaks = []
    for i in range(2, len(profile)-2):
        if profile[i] > threshold and profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
            pos = i * RESOLUTION + axis_min
            if not peaks or abs(pos - peaks[-1][0]) > 0.25:
                peaks.append((pos, float(profile[i])))
    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks


def make_boundary(density, grid, h_walls, v_walls):
    """Build boundary from density mask — only where walls exist."""
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    # Use density-based mask (where cross-sections show structure)
    # This naturally excludes scan artifacts outside the apartment
    dmask = (density >= 1).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dmask = cv2.morphologyEx(dmask, cv2.MORPH_CLOSE, k)
    dmask = binary_fill_holes(dmask).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dmask = cv2.erode(dmask, k2, iterations=2)
    
    contours, _ = cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
    pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
    
    poly = Polygon(pts).buffer(0).simplify(0.4)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    print(f"  Density mask boundary: {poly.area:.1f}m², {len(poly.exterior.coords)-1}v")
    
    # Snap vertices to H/V wall positions, then make edges H/V
    h_pos = [p for p, s in h_walls]
    v_pos = [p for p, s in v_walls]
    
    coords = np.array(poly.exterior.coords[:-1])
    snapped = []
    for x, y in coords:
        # Snap to nearest wall if close
        if v_pos:
            sv = min(v_pos, key=lambda p: abs(p-x))
            if abs(sv-x) < 0.4: x = sv
        if h_pos:
            sh = min(h_pos, key=lambda p: abs(p-y))
            if abs(sh-y) < 0.4: y = sh
        snapped.append((x, y))
    
    # Make edges H or V by inserting corners
    result = []
    n = len(snapped)
    for i in range(n):
        x1, y1 = snapped[i]
        x2, y2 = snapped[(i+1) % n]
        result.append((x1, y1))
        if abs(x2-x1) > 0.1 and abs(y2-y1) > 0.1:
            result.append((x2, y1))  # H then V corner
    
    result.append(result[0])
    poly = Polygon(result).buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    # Remove collinear
    coords = list(poly.exterior.coords[:-1])
    cleaned = []
    n = len(coords)
    for i in range(n):
        p = coords[(i-1)%n]; c = coords[i]; nn = coords[(i+1)%n]
        cross = abs((c[0]-p[0])*(nn[1]-p[1]) - (c[1]-p[1])*(nn[0]-p[0]))
        if cross > 0.01:
            cleaned.append(c)
    if len(cleaned) >= 3:
        poly = Polygon(cleaned).buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    
    print(f"  Snapped boundary: {poly.area:.1f}m², {len(poly.exterior.coords)-1}v")
    return poly


def partition(boundary, h_walls, v_walls):
    minx, miny, maxx, maxy = boundary.bounds
    margin = 0.3
    
    h_int = [(p, s) for p, s in h_walls if miny+margin < p < maxy-margin][:8]
    v_int = [(p, s) for p, s in v_walls if minx+margin < p < maxx-margin][:8]
    
    print(f"\nInterior walls: {len(h_int)}H + {len(v_int)}V")
    for o, ws in [('H', h_int), ('V', v_int)]:
        for p, s in ws: print(f"  {o} {p:.2f} str={s:.0f}")
    
    all_w = []
    for p, s in h_int:
        line = LineString([(minx-1,p),(maxx+1,p)]).intersection(boundary)
        if not line.is_empty: all_w.append(('h', p, s, line))
    for p, s in v_int:
        line = LineString([(p,miny-1),(p,maxy+1)]).intersection(boundary)
        if not line.is_empty: all_w.append(('v', p, s, line))
    
    def try_combo(idxs):
        lines = [boundary.exterior]
        for i in idxs: lines.append(all_w[i][3])
        polys = list(polygonize(unary_union(lines)))
        return sorted([p for p in polys if boundary.contains(p.representative_point()) and p.area > 1.5],
                      key=lambda r: r.area, reverse=True)
    
    def score(rooms):
        if len(rooms) != 4: return float('inf')
        a = sorted([r.area for r in rooms], reverse=True)
        bed = abs(a[0]-10.5) + abs(a[1]-10)
        if a[0] > 14: bed += (a[0]-14)*5
        sm = sum(max(0,2-x)*3 + max(0,x-7)*2 for x in a[2:])
        return bed + sm + abs(sum(a)-35)*0.3
    
    best = None; best_score = float('inf')
    for n in range(2, min(7, len(all_w)+1)):
        for combo in combinations(range(len(all_w)), n):
            rooms = try_combo(combo)
            if len(rooms) == 4:
                s = score(rooms)
                if s < best_score:
                    best_score = s; best = (combo, rooms)
                    a = sorted([r.area for r in rooms], reverse=True)
                    desc = [f"{all_w[i][0]}{all_w[i][1]:.1f}" for i in combo]
                    print(f"  ✓ [{','.join(desc)}] → {[f'{x:.1f}' for x in a]} sc={s:.1f}")
    
    if best is None:
        print("Fallback: merge")
        rooms = try_combo(range(min(5, len(all_w))))
        while len(rooms) > 4:
            i = min(range(len(rooms)), key=lambda i: rooms[i].area)
            j = max((j for j in range(len(rooms)) if j!=i),
                    key=lambda j: rooms[i].buffer(0.1).intersection(rooms[j].buffer(0.1)).area)
            rooms[j] = unary_union([rooms[j], rooms[i]]); rooms.pop(i)
        rooms.sort(key=lambda r: r.area, reverse=True)
        best = ([], rooms)
    return best


def classify(rooms):
    labels = [None]*len(rooms)
    by_area = sorted(range(len(rooms)), key=lambda i: rooms[i].area, reverse=True)
    if len(rooms) >= 4:
        labels[by_area[0]] = "Bedroom 1"; labels[by_area[1]] = "Bedroom 2"
        rem = by_area[2:]
        asps = [(i, max(rooms[i].bounds[2]-rooms[i].bounds[0], rooms[i].bounds[3]-rooms[i].bounds[1]) /
                 (min(rooms[i].bounds[2]-rooms[i].bounds[0], rooms[i].bounds[3]-rooms[i].bounds[1])+1e-6)) for i in rem]
        asps.sort(key=lambda x: x[1], reverse=True)
        labels[asps[0][0]] = "Hallway"; labels[asps[1][0]] = "Bathroom"
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v62d')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    print(f"Loaded: {len(mesh.vertices)} verts")
    
    density, grid = build_all(mesh, args.angle)
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    h_walls = find_wall_peaks(density, grid, 'h')
    v_walls = find_wall_peaks(density, grid, 'v')
    print(f"Top H: {[(f'{p:.2f}',f'{s:.0f}') for p,s in h_walls[:8]]}")
    print(f"Top V: {[(f'{p:.2f}',f'{s:.0f}') for p,s in v_walls[:8]]}")
    
    boundary = make_boundary(density, grid, h_walls, v_walls)
    if boundary is None: print("No boundary"); return
    print(f"Boundary: {boundary.area:.1f}m², {len(boundary.exterior.coords)-1}v")
    
    combo, rooms = partition(boundary, h_walls, v_walls)
    labels = classify(rooms)
    
    def rot_back(poly):
        c = np.array(poly.exterior.coords)
        r = rotate_points(c, args.angle, np.zeros(2)) + center
        return Polygon(r)
    
    rooms_orig = [rot_back(r) for r in rooms]
    boundary_orig = rot_back(boundary)
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'cyan', linewidth=2)
    ax.set_title("Density + boundary"); ax.grid(True, alpha=0.3)
    
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9']
    total = sum(r.area for r in rooms)
    for panel, rms, bnd in [(1, rooms, boundary), (2, rooms_orig, boundary_orig)]:
        ax = axes[panel]
        for i, (rm, lbl) in enumerate(zip(rms, labels)):
            xs, ys = rm.exterior.xy
            ax.fill(xs, ys, color=pastel[i%4], alpha=0.5)
            ax.plot(xs, ys, 'k-', linewidth=2)
            cx, cy = rm.centroid.coords[0]
            ax.text(cx, cy, f"{lbl}\n{rooms[i].area:.1f}m²", ha='center', va='center', fontsize=9, fontweight='bold')
        bx, by = bnd.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=2.5)
        ax.set_title(f"{'Rotated' if panel==1 else 'v62d'} — {len(rooms)} rooms, {total:.1f}m²")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}")
    for rm, lbl in zip(rooms, labels):
        print(f"  {lbl}: {rm.area:.1f}m² ({len(rm.exterior.coords)-1}v)")
    print(f"Total: {total:.1f}m²")


if __name__ == '__main__':
    main()

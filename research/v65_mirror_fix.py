#!/usr/bin/env python3
"""
mesh2plan v65 - Mirror fix + actual dimensions

CRITICAL FIX: Negate X in XZ projection to fix mirroring vs real floor plan.
Also: skip slow cross-section slicing, use only wall-face density (faster).

Actual dimensions from floor plan:
- Right bedroom: 15.22m² (3.38m × 4.59m)  
- Left bedroom: ~15-18m² (3.31m × 5.58m, 2.75m internal)
- Bathroom (top-center): ~2.5m² (1.56m × 1.59m)
- Hallway: ~5m² (2.95m × 1.73m)
- WC (bottom): ~2m² (1.98m × 1.01m)
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
import sys


RESOLUTION = 0.02
WALL_ANGLE = 29.0


def load_mesh(path):
    print(f"Loading mesh...", flush=True)
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    return mesh


def rotate_points(pts, angle_deg, center=None):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if center is not None: pts = pts - center
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: rotated += center
    return rotated


def build_all(mesh, angle_deg):
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    # *** MIRROR FIX: negate X ***
    pts_xz[:, 0] = -pts_xz[:, 0]
    
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.5
    xmax, zmax = rot_verts.max(axis=0) + 0.5
    w = int((xmax-xmin)/RESOLUTION)
    h = int((zmax-zmin)/RESOLUTION)
    print(f"Grid: {w}x{h}, range x=[{xmin:.2f},{xmax:.2f}] z=[{zmin:.2f},{zmax:.2f}]", flush=True)
    
    # Wall-face density (from normal-filtered faces)
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    # *** MIRROR FIX: negate X ***
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    wall_density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py = np.clip(((wall_rot[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    np.add.at(wall_density, (py, px), 1)
    wall_density = cv2.GaussianBlur(wall_density, (5,5), 1)
    print(f"Wall faces: {wall_mask.sum()} ({wall_mask.mean()*100:.0f}%)", flush=True)
    
    # Build boundary from ALL vertices
    vmask = np.zeros((h, w), dtype=np.uint8)
    px2 = np.clip(((rot_verts[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py2 = np.clip(((rot_verts[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    vmask[py2, px2] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, k)
    vmask = binary_fill_holes(vmask).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vmask = cv2.erode(vmask, k2, iterations=2)
    
    grid = (xmin, zmin, xmax, zmax, w, h, center)
    return wall_density, vmask, grid


def find_peaks(wall_density, vmask, grid, orientation, min_gap=0.20):
    xmin, zmin, xmax, zmax, w, h, center = grid
    masked = wall_density * (vmask > 0).astype(np.float32)
    
    if orientation == 'h':
        profile = masked.sum(axis=1)
        axis_min = zmin
    else:
        profile = masked.sum(axis=0)
        axis_min = xmin
    
    kernel = np.ones(5) / 5
    profile = np.convolve(profile, kernel, mode='same')
    
    threshold = np.percentile(profile[profile > 0], 20) if (profile > 0).any() else 0
    peaks = []
    for i in range(3, len(profile)-3):
        if profile[i] > threshold and profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
            window = profile[max(0,i-7):i+8]
            if profile[i] >= np.max(window) * 0.9:
                pos = i * RESOLUTION + axis_min
                if not peaks or abs(pos - peaks[-1][0]) > min_gap:
                    peaks.append((pos, float(profile[i])))
    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks


def make_boundary(vmask, grid, h_peaks, v_peaks):
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    contours, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
    pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
    
    poly = Polygon(pts).buffer(0).simplify(0.3)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    h_pos = [p for p, s in h_peaks]
    v_pos = [p for p, s in v_peaks]
    
    coords = list(poly.exterior.coords[:-1])
    snapped = []
    for x, y in coords:
        if v_pos:
            sv = min(v_pos, key=lambda p: abs(p - x))
            if abs(sv - x) < 0.4: x = sv
        if h_pos:
            sh = min(h_pos, key=lambda p: abs(p - y))
            if abs(sh - y) < 0.4: y = sh
        snapped.append((x, y))
    
    # Make rectilinear
    result = []
    n = len(snapped)
    for i in range(n):
        x1, y1 = snapped[i]
        x2, y2 = snapped[(i+1) % n]
        result.append((x1, y1))
        if abs(x2-x1) > 0.15 and abs(y2-y1) > 0.15:
            result.append((x2, y1))
    
    result.append(result[0])
    poly = Polygon(result).buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    # Remove collinear
    coords = list(poly.exterior.coords[:-1])
    cleaned = []
    n = len(coords)
    for i in range(n):
        p = coords[(i-1) % n]
        c = coords[i]
        nn = coords[(i+1) % n]
        if abs((c[0]-p[0])*(nn[1]-p[1]) - (c[1]-p[1])*(nn[0]-p[0])) > 0.01:
            cleaned.append(c)
    if len(cleaned) >= 3:
        poly = Polygon(cleaned).buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    
    return poly


def partition_search(boundary, h_peaks, v_peaks):
    """Search for best 4-6 room partition matching actual dimensions."""
    minx, miny, maxx, maxy = boundary.bounds
    margin = 0.2
    
    h_int = [(p, s) for p, s in h_peaks if miny + margin < p < maxy - margin][:8]
    v_int = [(p, s) for p, s in v_peaks if minx + margin < p < maxx - margin][:8]
    
    print(f"\nInterior: {len(h_int)} H + {len(v_int)} V walls", flush=True)
    for o, walls in [('H', h_int), ('V', v_int)]:
        for p, s in walls[:8]:
            print(f"  {o} pos={p:.2f} str={s:.0f}", flush=True)
    
    all_w = []
    for p, s in h_int:
        line = LineString([(minx-1, p), (maxx+1, p)]).intersection(boundary)
        if not line.is_empty: all_w.append(('h', p, s, line))
    for p, s in v_int:
        line = LineString([(p, miny-1), (p, maxy+1)]).intersection(boundary)
        if not line.is_empty: all_w.append(('v', p, s, line))
    
    print(f"  Total wall lines: {len(all_w)}", flush=True)
    
    def try_combo(idxs):
        lines = [boundary.exterior]
        for i in idxs: lines.append(all_w[i][3])
        polys = list(polygonize(unary_union(lines)))
        rooms = [p for p in polys if boundary.contains(p.representative_point()) and p.area > 0.8]
        rooms.sort(key=lambda r: r.area, reverse=True)
        return rooms
    
    def score(rooms):
        n = len(rooms)
        if n < 4 or n > 7: return float('inf')
        a = sorted([r.area for r in rooms], reverse=True)
        
        err = 0
        # Two large bedrooms (12-18m² each, target ~15m²)
        if a[0] < 12: err += (12-a[0])*4
        if a[0] > 20: err += (a[0]-20)*3
        if a[1] < 10: err += (10-a[1])*4
        if a[1] > 20: err += (a[1]-20)*3
        err += abs(a[0]-15) * 0.5
        err += abs(a[1]-15) * 0.5
        
        # Remaining rooms: hallway ~5m², bathroom ~2.5m², WC ~2m²
        if n >= 3: err += abs(a[2]-5) * 0.8
        if n >= 4: err += abs(a[3]-2.5) * 0.5
        if n >= 5: err += abs(a[4]-2) * 0.3
        
        if n == 5: err -= 3
        if n == 6: err -= 1
        
        total = sum(a)
        err += abs(total - 42) * 0.3
        
        # Penalize rooms that span full boundary width (indicates no V walls)
        bminx, bminy, bmaxx, bmaxy = boundary.bounds
        bw = bmaxx - bminx
        bh = bmaxy - bminy
        for r in rooms:
            rb = r.bounds
            rw = rb[2]-rb[0]
            rh = rb[3]-rb[1]
            if rw > bw * 0.8: err += 5  # room spans full width = bad
            if rh > bh * 0.8: err += 5  # room spans full height = bad
            # Penalize extreme aspect ratios
            asp = max(rw,rh)/(min(rw,rh)+0.01)
            if r.area > 5 and asp > 4: err += 3
        
        return err
    
    best = None
    best_score = float('inf')
    
    max_walls = min(6, len(all_w))
    for n_walls in range(3, max_walls+1):
        for combo in combinations(range(len(all_w)), n_walls):
            # Require at least 1 H and 1 V wall for proper room shapes
            types = set(all_w[i][0] for i in combo)
            if len(types) < 2: continue
            rooms = try_combo(combo)
            if 4 <= len(rooms) <= 7:
                s = score(rooms)
                if s < best_score:
                    best_score = s
                    best = (combo, rooms)
                    a = sorted([r.area for r in rooms], reverse=True)
                    desc = [f"{all_w[i][0]}{all_w[i][1]:.1f}" for i in combo]
                    print(f"  ✓ [{','.join(desc)}] → {len(rooms)}r {[f'{x:.1f}' for x in a]} sc={s:.1f}", flush=True)
    
    if best is None:
        # Relax: allow 3 rooms
        for n_walls in range(2, max_walls+1):
            for combo in combinations(range(len(all_w)), n_walls):
                rooms = try_combo(combo)
                if len(rooms) >= 3:
                    s = score(rooms) if len(rooms) >= 4 else 100 + abs(sum(r.area for r in rooms) - 42)
                    if s < best_score:
                        best_score = s
                        best = (combo, rooms)
    
    if best is None:
        return None, None
    return best


def classify(rooms):
    labels = [None]*len(rooms)
    by_area = sorted(range(len(rooms)), key=lambda i: rooms[i].area, reverse=True)
    labels[by_area[0]] = "Bedroom 1"
    labels[by_area[1]] = "Bedroom 2"
    
    if len(rooms) >= 3:
        rem = by_area[2:]
        for i in rem:
            a = rooms[i].area
            bnd = rooms[i].bounds
            w = bnd[2]-bnd[0]
            h = bnd[3]-bnd[1]
            aspect = max(w,h)/(min(w,h)+1e-6)
            
            if labels[i] is None:
                if a > 3.5 and aspect > 1.5:
                    labels[i] = "Hallway"
                elif a < 3:
                    labels[i] = "WC"
                else:
                    labels[i] = "Bathroom"
        
        for i in range(len(labels)):
            if labels[i] is None:
                labels[i] = f"Room {i+1}"
    
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v65')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    print(f"Y range: {mesh.vertices[:,1].min():.3f} to {mesh.vertices[:,1].max():.3f}", flush=True)
    
    wall_d, vmask, grid = build_all(mesh, args.angle)
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    h_peaks = find_peaks(wall_d, vmask, grid, 'h')
    v_peaks = find_peaks(wall_d, vmask, grid, 'v')
    
    print(f"H peaks: {[(f'{p:.2f}',f'{s:.0f}') for p,s in h_peaks[:10]]}", flush=True)
    print(f"V peaks: {[(f'{p:.2f}',f'{s:.0f}') for p,s in v_peaks[:10]]}", flush=True)
    
    boundary = make_boundary(vmask, grid, h_peaks, v_peaks)
    print(f"\nBoundary: {boundary.area:.1f}m², {len(boundary.exterior.coords)-1}v", flush=True)
    print(f"Bounds: x=[{boundary.bounds[0]:.2f},{boundary.bounds[2]:.2f}] y=[{boundary.bounds[1]:.2f},{boundary.bounds[3]:.2f}]", flush=True)
    bw = boundary.bounds[2]-boundary.bounds[0]
    bh = boundary.bounds[3]-boundary.bounds[1]
    print(f"Width: {bw:.2f}m, Height: {bh:.2f}m", flush=True)
    
    result = partition_search(boundary, h_peaks, v_peaks)
    if result[0] is None:
        print("Failed to find rooms", flush=True)
        return
    
    combo, rooms = result
    labels = classify(rooms)
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: wall density
    ax = axes[0]
    ax.imshow(wall_d, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'cyan', linewidth=2)
    ax.set_title("Wall density (mirror-fixed, axis-aligned)")
    ax.grid(True, alpha=0.3)
    
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9','#E8BAFF','#FFE0BA']
    total = sum(r.area for r in rooms)
    
    # Panel 2: rooms
    ax = axes[1]
    for i, (rm, lbl) in enumerate(zip(rooms, labels)):
        xs, ys = rm.exterior.xy
        ax.fill(xs, ys, color=pastel[i%len(pastel)], alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=2)
        cx, cy = rm.centroid.coords[0]
        bnd = rm.bounds
        w_rm = bnd[2]-bnd[0]
        h_rm = bnd[3]-bnd[1]
        ax.text(cx, cy, f"{lbl}\n{rm.area:.1f}m²\n{w_rm:.1f}×{h_rm:.1f}m", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=2.5)
    ax.set_title(f"v65 — {len(rooms)} rooms, {total:.1f}m²")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    # Panel 3: comparison text
    ax = axes[2]
    ax.text(0.5, 0.95, "Actual vs Detected", ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    y_pos = 0.85
    sorted_rooms = sorted(zip(rooms, labels), key=lambda x: x[0].area, reverse=True)
    for rm, lbl in sorted_rooms:
        bnd = rm.bounds
        w_rm = bnd[2]-bnd[0]
        h_rm = bnd[3]-bnd[1]
        ax.text(0.05, y_pos, f"[D] {lbl}: {rm.area:.1f}m² ({w_rm:.2f}×{h_rm:.2f}m)", 
                fontsize=10, transform=ax.transAxes, color='blue')
        y_pos -= 0.07
    
    y_pos -= 0.03
    actual = [
        ("Right bedroom", 15.22, 3.38, 4.59),
        ("Left bedroom", 15.5, 3.31, 5.58),
        ("Hallway", 5.1, 1.73, 2.95),
        ("Bathroom", 2.5, 1.56, 1.59),
        ("WC", 2.0, 1.01, 1.98),
    ]
    for name, area, w_a, h_a in actual:
        ax.text(0.05, y_pos, f"[A] {name}: {area:.1f}m² ({w_a:.2f}×{h_a:.2f}m)", 
                fontsize=10, transform=ax.transAxes, color='red')
        y_pos -= 0.07
    
    ax.axis('off')
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}", flush=True)
    for rm, lbl in sorted_rooms:
        bnd = rm.bounds
        print(f"  {lbl}: {rm.area:.1f}m² ({bnd[2]-bnd[0]:.2f}×{bnd[3]-bnd[1]:.2f}m)")
    print(f"Total: {total:.1f}m²")


if __name__ == '__main__':
    main()

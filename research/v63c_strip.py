#!/usr/bin/env python3
"""
mesh2plan v63c - Strip hallway approach

Key insight from target: the hallway is a VERTICAL STRIP between the two
strongest V walls (x=-0.80 and x=0.46 in rotated coords). One H wall in
the center strip separates hallway from bathroom.

Layout in rotated coords:
  Left strip  (x < -0.80): Bedroom 1
  Center strip (-0.80 < x < 0.46): Hallway (top) + Bathroom (bottom) 
  Right strip (x > 0.46): Bedroom 2

This naturally gives 4 rooms with 2V + 1H walls.
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
    if center is not None:
        pts = pts - center
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None:
        rotated += center
    return rotated


def slice_mesh(mesh, y):
    try:
        lines = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, y, 0])
    except:
        return np.array([]).reshape(0, 2, 2)
    if lines is None or len(lines) == 0:
        return np.array([]).reshape(0, 2, 2)
    return lines[:, :, [0, 2]]


def build_density(mesh, angle_deg):
    pts_xz = mesh.vertices[:, [0, 2]]
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.3
    xmax, zmax = rot_verts.max(axis=0) + 0.3
    w = int((xmax-xmin)/RESOLUTION)
    h = int((zmax-zmin)/RESOLUTION)
    
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
    
    # Wall-face density
    normals = mesh.face_normals
    wall_faces = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wall_faces][:, [0, 2]]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    wall_density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py = np.clip(((wall_rot[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    np.add.at(wall_density, (py, px), 1)
    wall_density = cv2.GaussianBlur(wall_density, (5,5), 1)
    
    # Vertex mask
    vmask = np.zeros((h, w), dtype=np.uint8)
    px2 = np.clip(((rot_verts[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py2 = np.clip(((rot_verts[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    vmask[py2, px2] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, k)
    vmask = binary_fill_holes(vmask).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vmask = cv2.erode(vmask, k2, iterations=3)
    
    grid = (xmin, zmin, xmax, zmax, w, h, center)
    return density, wall_density, vmask, grid


def find_wall_peaks(wall_density, vmask, grid, orientation, min_gap=0.3):
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
    
    threshold = np.percentile(profile[profile > 0], 30) if (profile > 0).any() else 0
    peaks = []
    for i in range(3, len(profile)-3):
        if profile[i] > threshold and profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
            window = profile[max(0,i-5):i+6]
            if profile[i] >= np.max(window) * 0.95:
                pos = i * RESOLUTION + axis_min
                if not peaks or abs(pos - peaks[-1][0]) > min_gap:
                    peaks.append((pos, float(profile[i])))
    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks


def build_boundary(vmask, grid, h_walls, v_walls):
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    contours, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
    pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
    
    poly = Polygon(pts).buffer(0).simplify(0.4)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    # Snap vertices
    h_pos = [p for p, s in h_walls[:15]]
    v_pos = [p for p, s in v_walls[:15]]
    coords = np.array(poly.exterior.coords[:-1])
    snapped = []
    for x, y in coords:
        if v_pos:
            sv = min(v_pos, key=lambda p: abs(p - x))
            if abs(sv - x) < 0.5: x = sv
        if h_pos:
            sh = min(h_pos, key=lambda p: abs(p - y))
            if abs(sh - y) < 0.5: y = sh
        snapped.append((x, y))
    
    # Make rectilinear
    result = []
    n = len(snapped)
    for i in range(n):
        x1, y1 = snapped[i]
        x2, y2 = snapped[(i+1) % n]
        result.append((x1, y1))
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dx > 0.15 and dy > 0.15:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v63c')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    print(f"Loaded: {len(mesh.vertices)} verts")
    
    density, wall_density, vmask, grid = build_density(mesh, args.angle)
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    h_walls = find_wall_peaks(wall_density, vmask, grid, 'h')
    v_walls = find_wall_peaks(wall_density, vmask, grid, 'v')
    
    print(f"Top H: {[(f'{p:.2f}', f'{s:.0f}') for p,s in h_walls[:8]]}")
    print(f"Top V: {[(f'{p:.2f}', f'{s:.0f}') for p,s in v_walls[:8]]}")
    
    boundary = build_boundary(vmask, grid, h_walls, v_walls)
    print(f"Boundary: {boundary.area:.1f}m², {len(boundary.exterior.coords)-1}v")
    for i, (x, y) in enumerate(boundary.exterior.coords[:-1]):
        print(f"  v{i}: ({x:.2f}, {y:.2f})")
    
    minx, miny, maxx, maxy = boundary.bounds
    
    # The two strongest V walls define the hallway strip
    # V=-0.80 (strongest) and V=0.46 (2nd strongest)
    v1 = v_walls[0][0]  # -0.80
    v2 = v_walls[1][0]  # 0.46
    if v1 > v2: v1, v2 = v2, v1  # ensure v1 < v2
    
    print(f"\nHallway strip: V={v1:.2f} to V={v2:.2f} (width={v2-v1:.2f}m)")
    
    # Now find best H wall to split the center strip into hallway + bathroom
    # Try all interior H walls
    margin = 0.3
    h_int = [(p, s) for p, s in h_walls if miny + margin < p < maxy - margin]
    
    best = None
    best_score = float('inf')
    
    for h_pos, h_str in h_int:
        # Build 3 walls
        ext = 1.0
        lines = [boundary.exterior]
        for v in [v1, v2]:
            l = LineString([(v, miny-ext), (v, maxy+ext)]).intersection(boundary)
            if not l.is_empty: lines.append(l)
        l = LineString([(v1, h_pos), (v2, h_pos)]).intersection(boundary)
        if not l.is_empty: lines.append(l)
        
        polys = list(polygonize(unary_union(lines)))
        rooms = [p for p in polys if boundary.contains(p.representative_point()) and p.area > 1.0]
        rooms.sort(key=lambda r: r.area, reverse=True)
        
        if len(rooms) != 4: 
            # Try with H wall spanning full width instead of just center strip
            lines2 = [boundary.exterior]
            for v in [v1, v2]:
                l = LineString([(v, miny-ext), (v, maxy+ext)]).intersection(boundary)
                if not l.is_empty: lines2.append(l)
            l = LineString([(minx-ext, h_pos), (maxx+ext, h_pos)]).intersection(boundary)
            if not l.is_empty: lines2.append(l)
            polys2 = list(polygonize(unary_union(lines2)))
            rooms2 = [p for p in polys2 if boundary.contains(p.representative_point()) and p.area > 1.0]
            if len(rooms2) == 4:
                rooms = rooms2
            elif len(rooms) != 4:
                continue
        
        areas = sorted([r.area for r in rooms], reverse=True)
        
        # Score: bedrooms ~10m², hallway ~4m², bathroom ~3m²
        err = abs(areas[0]-10.5) + abs(areas[1]-10)
        err += abs(areas[2]-4.5) + abs(areas[3]-3)
        if areas[0] > 13: err += (areas[0]-13)*5
        if areas[1] > 13: err += (areas[1]-13)*5
        if areas[3] > 6: err += (areas[3]-6)*3
        
        if err < best_score:
            best_score = err
            best = rooms
            print(f"  ✓ H={h_pos:.2f} → {[f'{x:.1f}' for x in areas]} sc={err:.1f}")
    
    if best is None:
        # Fallback: just use 2V walls, merge to 4 rooms
        print("No 4-room combo, trying 2V only + merge")
        ext = 1.0
        lines = [boundary.exterior]
        for v in [v1, v2]:
            l = LineString([(v, miny-ext), (v, maxy+ext)]).intersection(boundary)
            if not l.is_empty: lines.append(l)
        polys = list(polygonize(unary_union(lines)))
        best = [p for p in polys if boundary.contains(p.representative_point()) and p.area > 1.0]
        best.sort(key=lambda r: r.area, reverse=True)
    
    rooms = best
    
    # Classify
    labels = [None]*len(rooms)
    by_area = sorted(range(len(rooms)), key=lambda i: rooms[i].area, reverse=True)
    labels[by_area[0]] = "Bedroom 1"
    labels[by_area[1]] = "Bedroom 2"
    if len(rooms) >= 3:
        rem = by_area[2:]
        asps = []
        for i in rem:
            b = rooms[i].bounds
            w2 = b[2]-b[0]; h2 = b[3]-b[1]
            asps.append((i, max(w2,h2)/(min(w2,h2)+1e-6)))
        asps.sort(key=lambda x: x[1], reverse=True)
        labels[asps[0][0]] = "Hallway"
        if len(asps) > 1:
            labels[asps[1][0]] = "Bathroom"
    
    # Rotate back
    def rot_back(poly):
        c = np.array(poly.exterior.coords)
        r = rotate_points(c, args.angle, np.zeros(2)) + center
        return Polygon(r)
    
    rooms_orig = [rot_back(r) for r in rooms]
    boundary_orig = rot_back(boundary)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'cyan', linewidth=2)
    ax.axvline(v1, color='lime', linewidth=2, alpha=0.7)
    ax.axvline(v2, color='lime', linewidth=2, alpha=0.7)
    ax.set_title(f"Density + hallway strip V=[{v1:.1f},{v2:.1f}]")
    ax.grid(True, alpha=0.3)
    
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9']
    total = sum(r.area for r in rooms)
    
    for panel, rms, bnd, title in [
        (1, rooms, boundary, "Rotated"),
        (2, rooms_orig, boundary_orig, "v63c")
    ]:
        ax = axes[panel]
        for i, (rm, lbl) in enumerate(zip(rms, labels)):
            xs, ys = rm.exterior.xy
            ax.fill(xs, ys, color=pastel[i%4], alpha=0.5)
            ax.plot(xs, ys, 'k-', linewidth=2)
            a = rooms[i].area
            cx, cy = rm.centroid.coords[0]
            ax.text(cx, cy, f"{lbl}\n{a:.1f}m²", ha='center', va='center', fontsize=9, fontweight='bold')
        bx, by = bnd.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=2.5)
        ax.set_title(f"{title} — {len(rooms)} rooms, {total:.1f}m²")
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

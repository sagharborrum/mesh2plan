#!/usr/bin/env python3
"""
mesh2plan v63b - Direct room construction from wall positions

Instead of combinatorial wall search, directly construct rooms:
1. Rotate to axis-align
2. Build clean boundary from mesh vertices
3. Find strongest V wall (bedroom divider) and H walls
4. Cut boundary with: 1 V wall + 1-2 H walls → 4 rooms
5. The V wall divides left/right bedrooms
6. H wall(s) separate hallway and bathroom at the top

Key insight: the apartment has ONE main vertical divider and ONE main
horizontal divider. That gives us 4 quadrants = 4 rooms.
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
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import polygonize, unary_union, split
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
    """Build cross-section density + wall-normal density in rotated frame."""
    pts_xz = mesh.vertices[:, [0, 2]]
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.3
    xmax, zmax = rot_verts.max(axis=0) + 0.3
    w = int((xmax-xmin)/RESOLUTION)
    h = int((zmax-zmin)/RESOLUTION)
    
    # Cross-section density
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
    
    # Vertex mask for boundary
    vmask = np.zeros((h, w), dtype=np.uint8)
    px = np.clip(((rot_verts[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py = np.clip(((rot_verts[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    vmask[py, px] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, k)
    vmask = binary_fill_holes(vmask).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vmask = cv2.erode(vmask, k2, iterations=3)
    
    grid = (xmin, zmin, xmax, zmax, w, h, center)
    return density, wall_density, vmask, grid, rot_verts


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
    return peaks, profile


def build_boundary(vmask, grid):
    """Build a clean rectilinear boundary from vertex mask."""
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    contours, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
    pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
    
    poly = Polygon(pts).buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    # Simplify aggressively
    poly = poly.simplify(0.5)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    return poly


def snap_boundary_to_walls(boundary, h_walls, v_walls):
    """Snap boundary vertices to nearest wall positions and make rectilinear."""
    coords = np.array(boundary.exterior.coords[:-1])
    
    # Collect all strong wall positions
    h_pos = [p for p, s in h_walls[:15]]
    v_pos = [p for p, s in v_walls[:15]]
    
    snapped = []
    for x, y in coords:
        # Snap to nearest wall
        if v_pos:
            sv = min(v_pos, key=lambda p: abs(p - x))
            if abs(sv - x) < 0.5: x = sv
        if h_pos:
            sh = min(h_pos, key=lambda p: abs(p - y))
            if abs(sh - y) < 0.5: y = sh
        snapped.append((x, y))
    
    # Make rectilinear: insert corner at diagonal edges
    result = []
    n = len(snapped)
    for i in range(n):
        x1, y1 = snapped[i]
        x2, y2 = snapped[(i+1) % n]
        result.append((x1, y1))
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dx > 0.15 and dy > 0.15:
            # Insert intermediate corner (horizontal-first)
            result.append((x2, y1))
    
    result.append(result[0])
    poly = Polygon(result).buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    # Remove collinear vertices
    coords = list(poly.exterior.coords[:-1])
    cleaned = []
    n = len(coords)
    for i in range(n):
        p = coords[(i-1) % n]
        c = coords[i]
        nn = coords[(i+1) % n]
        cross = abs((c[0]-p[0])*(nn[1]-p[1]) - (c[1]-p[1])*(nn[0]-p[0]))
        if cross > 0.01:
            cleaned.append(c)
    
    if len(cleaned) >= 3:
        poly = Polygon(cleaned).buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def cut_rooms(boundary, v_wall, h_wall):
    """Cut boundary with one V and one H wall to get 4 rooms."""
    minx, miny, maxx, maxy = boundary.bounds
    ext = 1.0
    
    v_line = LineString([(v_wall, miny-ext), (v_wall, maxy+ext)])
    h_line = LineString([(minx-ext, h_wall), (maxx+ext, h_wall)])
    
    # Clip to boundary
    v_clipped = v_line.intersection(boundary)
    h_clipped = h_line.intersection(boundary)
    
    # Polygonize
    all_lines = [boundary.exterior, v_clipped, h_clipped]
    polys = list(polygonize(unary_union(all_lines)))
    rooms = [p for p in polys if boundary.contains(p.representative_point()) and p.area > 1.0]
    rooms.sort(key=lambda r: r.area, reverse=True)
    return rooms


def try_3wall_cut(boundary, v_wall, h_wall1, h_wall2):
    """Cut with 1V + 2H walls."""
    minx, miny, maxx, maxy = boundary.bounds
    ext = 1.0
    
    lines = [boundary.exterior]
    for v in ([v_wall] if not isinstance(v_wall, list) else v_wall):
        l = LineString([(v, miny-ext), (v, maxy+ext)]).intersection(boundary)
        if not l.is_empty: lines.append(l)
    for h in [h_wall1, h_wall2]:
        if h is not None:
            l = LineString([(minx-ext, h), (maxx+ext, h)]).intersection(boundary)
            if not l.is_empty: lines.append(l)
    
    polys = list(polygonize(unary_union(lines)))
    rooms = [p for p in polys if boundary.contains(p.representative_point()) and p.area > 1.0]
    rooms.sort(key=lambda r: r.area, reverse=True)
    return rooms


def classify(rooms):
    labels = [None]*len(rooms)
    by_area = sorted(range(len(rooms)), key=lambda i: rooms[i].area, reverse=True)
    labels[by_area[0]] = "Bedroom 1"
    labels[by_area[1]] = "Bedroom 2"
    if len(rooms) >= 3:
        rem = by_area[2:]
        asps = []
        for i in rem:
            b = rooms[i].bounds
            w = b[2]-b[0]; h = b[3]-b[1]
            asps.append((i, max(w,h)/(min(w,h)+1e-6)))
        asps.sort(key=lambda x: x[1], reverse=True)
        labels[asps[0][0]] = "Hallway"
        if len(asps) > 1:
            labels[asps[1][0]] = "Bathroom"
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v63b')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    density, wall_density, vmask, grid, rot_verts = build_density(mesh, args.angle)
    xmin, zmin, xmax, zmax, w, h, center = grid
    
    h_walls, h_prof = find_wall_peaks(wall_density, vmask, grid, 'h')
    v_walls, v_prof = find_wall_peaks(wall_density, vmask, grid, 'v')
    
    print(f"\nTop H walls: {[(f'{p:.2f}', f'{s:.0f}') for p,s in h_walls[:10]]}")
    print(f"Top V walls: {[(f'{p:.2f}', f'{s:.0f}') for p,s in v_walls[:10]]}")
    
    # Build boundary
    boundary = build_boundary(vmask, grid)
    boundary = snap_boundary_to_walls(boundary, h_walls, v_walls)
    print(f"\nBoundary: {boundary.area:.1f}m², {len(boundary.exterior.coords)-1}v")
    
    # Try all combos of 1V + 1-2H walls
    minx, miny, maxx, maxy = boundary.bounds
    margin = 0.3
    h_int = [p for p, s in h_walls if miny + margin < p < maxy - margin][:8]
    v_int = [p for p, s in v_walls if minx + margin < p < maxx - margin][:8]
    
    print(f"\nInterior: {len(v_int)} V walls, {len(h_int)} H walls")
    
    best = None
    best_score = float('inf')
    
    def score(rooms):
        if len(rooms) != 4: return float('inf')
        areas = sorted([r.area for r in rooms], reverse=True)
        # Target: 10.5, 10, 4, 3 (total ~27.5)
        err = abs(areas[0]-10.5) + abs(areas[1]-10)
        err += abs(areas[2]-4.5) + abs(areas[3]-3)
        # Hard penalties
        if areas[0] > 13: err += (areas[0]-13)*5
        if areas[1] > 13: err += (areas[1]-13)*5
        if areas[3] > 5.5: err += (areas[3]-5.5)*3
        if areas[3] < 1.5: err += (1.5-areas[3])*5
        return err
    
    # 1V + 1H
    for v in v_int:
        for h in h_int:
            rooms = cut_rooms(boundary, v, h)
            if len(rooms) == 4:
                s = score(rooms)
                if s < best_score:
                    best_score = s
                    best = rooms
                    a = sorted([r.area for r in rooms], reverse=True)
                    print(f"  ✓ V={v:.2f} H={h:.2f} → {[f'{x:.1f}' for x in a]} sc={s:.1f}")
    
    # 1V + 2H  
    for v in v_int:
        for i, h1 in enumerate(h_int):
            for h2 in h_int[i+1:]:
                rooms = try_3wall_cut(boundary, v, h1, h2)
                if len(rooms) == 4:
                    s = score(rooms)
                    if s < best_score:
                        best_score = s
                        best = rooms
                        a = sorted([r.area for r in rooms], reverse=True)
                        print(f"  ✓ V={v:.2f} H={h1:.2f},{h2:.2f} → {[f'{x:.1f}' for x in a]} sc={s:.1f}")
    
    # 2V + 1H
    for i, v1 in enumerate(v_int):
        for v2 in v_int[i+1:]:
            for h in h_int:
                rooms = try_3wall_cut(boundary, [v1, v2], h, None)
                if len(rooms) == 4:
                    s = score(rooms)
                    if s < best_score:
                        best_score = s
                        best = rooms
                        a = sorted([r.area for r in rooms], reverse=True)
                        print(f"  ✓ V={v1:.2f},{v2:.2f} H={h:.2f} → {[f'{x:.1f}' for x in a]} sc={s:.1f}")
    
    if best is None:
        print("No 4-room combo found!")
        return
    
    rooms = best
    labels = classify(rooms)
    
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
    # Show wall density
    combined = density + wall_density * 0.5
    ax.imshow(combined, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'cyan', linewidth=2)
    ax.set_title("Density + boundary")
    ax.grid(True, alpha=0.3)
    
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9']
    total = sum(r.area for r in rooms)
    
    for panel, rms, bnd, title in [
        (1, rooms, boundary, "Rotated"),
        (2, rooms_orig, boundary_orig, "v63b")
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

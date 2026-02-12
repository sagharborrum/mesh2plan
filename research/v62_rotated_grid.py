#!/usr/bin/env python3
"""
mesh2plan v62 - Rotated Grid Partition

KEY INSIGHT: Rotate everything by -29° so walls become axis-aligned (H/V).
Then find H/V wall lines, build rectangular rooms, rotate back.

This gives perfectly straight walls at the correct angles.
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
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import rotate as shapely_rotate


RESOLUTION = 0.02
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
WALL_ANGLE = 29.0  # degrees — dominant wall direction


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def rotate_points(pts, angle_deg, center=None):
    """Rotate 2D points by angle_deg around center."""
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


def build_density_rotated(mesh, angle_deg, res=RESOLUTION):
    """Build cross-section density in rotated coordinate system."""
    # Get all cross-section segments
    all_segs = []
    for y in SLICE_HEIGHTS:
        segs = slice_mesh(mesh, y)
        if len(segs) > 0:
            all_segs.append(segs)
            print(f"  Y={y}: {len(segs)} segs")
    
    if not all_segs:
        return None, None
    
    # Flatten all segment endpoints
    all_pts = np.concatenate([s.reshape(-1, 2) for s in all_segs])
    center = all_pts.mean(axis=0)
    
    # Rotate all segments
    rotated_segs = []
    for segs in all_segs:
        flat = segs.reshape(-1, 2)
        rot = rotate_points(flat, -angle_deg, center)
        rotated_segs.append(rot.reshape(-1, 2, 2))
    
    # Build density image
    all_rot = np.concatenate([s.reshape(-1, 2) for s in rotated_segs])
    xmin, zmin = all_rot.min(axis=0) - 0.3
    xmax, zmax = all_rot.max(axis=0) + 0.3
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    
    density = np.zeros((h, w), dtype=np.float32)
    for segs in rotated_segs:
        img = np.zeros((h, w), dtype=np.uint8)
        for seg in segs:
            p1x = int(np.clip((seg[0,0]-xmin)/res, 0, w-1))
            p1y = int(np.clip((seg[0,1]-zmin)/res, 0, h-1))
            p2x = int(np.clip((seg[1,0]-xmin)/res, 0, w-1))
            p2y = int(np.clip((seg[1,1]-zmin)/res, 0, h-1))
            cv2.line(img, (p1x, p1y), (p2x, p2y), 1, 1)
        density += img.astype(np.float32)
    
    grid_info = (xmin, zmin, xmax, zmax, w, h, center)
    return density, grid_info


def find_wall_lines(density, grid_info, orientation='h', min_votes=15):
    """Find wall lines along one axis using projection profile.
    
    For H walls: project density onto Y axis (sum across X)
    For V walls: project density onto X axis (sum across Y)
    
    Peaks in projection = wall positions.
    """
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    
    if orientation == 'h':
        # Horizontal walls = constant Y. Project onto Y axis.
        profile = density.sum(axis=1)  # sum across columns (X)
        axis_min, axis_res = zmin, RESOLUTION
    else:
        # Vertical walls = constant X. Project onto X axis.  
        profile = density.sum(axis=0)  # sum across rows (Y)
        axis_min, axis_res = xmin, RESOLUTION
    
    # Smooth
    kernel = np.ones(5) / 5
    profile = np.convolve(profile, kernel, mode='same')
    
    # Find peaks: local maxima above threshold
    threshold = np.percentile(profile[profile > 0], 70) if (profile > 0).any() else 0
    peaks = []
    for i in range(2, len(profile)-2):
        if profile[i] > threshold and profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
            # Not too close to existing peaks
            pos = i * axis_res + axis_min
            if not peaks or abs(pos - peaks[-1][0]) > 0.3:
                peaks.append((pos, profile[i]))
    
    # Sort by strength
    peaks.sort(key=lambda p: p[1], reverse=True)
    
    return peaks


def find_wall_extents(density, grid_info, wall_pos, orientation, threshold=1.0):
    """Find the extent (start, end) of a wall along its length.
    
    For H wall at y: scan along x to find where density > threshold.
    For V wall at x: scan along y to find where density > threshold.
    """
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    
    if orientation == 'h':
        # Wall at fixed y, scan x
        py = int((wall_pos - zmin) / RESOLUTION)
        py = np.clip(py, 0, h-1)
        # Sample a band of ±2 pixels
        band = density[max(0,py-2):min(h,py+3), :].max(axis=0)
        vals = np.where(band >= threshold)[0]
        if len(vals) == 0:
            return None
        x_start = vals[0] * RESOLUTION + xmin
        x_end = vals[-1] * RESOLUTION + xmin
        return (x_start, x_end)
    else:
        # Wall at fixed x, scan y
        px = int((wall_pos - xmin) / RESOLUTION)
        px = np.clip(px, 0, w-1)
        band = density[:, max(0,px-2):min(w,px+3)].max(axis=1)
        vals = np.where(band >= threshold)[0]
        if len(vals) == 0:
            return None
        y_start = vals[0] * RESOLUTION + zmin
        y_end = vals[-1] * RESOLUTION + zmin
        return (y_start, y_end)


def build_boundary_rotated(mesh, grid_info, angle_deg):
    """Build apartment boundary in rotated space from mesh vertices."""
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    
    # Project ALL mesh vertices into rotated frame
    pts_xz = mesh.vertices[:, [0, 2]]
    rotated = rotate_points(pts_xz, -angle_deg, center)
    
    # Build mask from vertices
    mask = np.zeros((h, w), dtype=np.uint8)
    px = ((rotated[:,0]-xmin)/RESOLUTION).astype(int)
    py = ((rotated[:,1]-zmin)/RESOLUTION).astype(int)
    px = np.clip(px, 0, w-1); py = np.clip(py, 0, h-1)
    mask[py, px] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = binary_fill_holes(mask).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, k2, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
    pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
    
    # Use simplified contour (not aggressive H/V snap which destroys geometry)
    poly = Polygon(pts).buffer(0).simplify(0.15)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    print(f"  Raw boundary: {poly.area:.1f}m², {len(poly.exterior.coords)-1}v")
    
    # Now snap each edge to H or V
    coords = np.array(poly.exterior.coords[:-1])  # drop closing vertex
    snapped = snap_boundary_hv(coords)
    
    return snapped, mask


def snap_boundary_hv(coords):
    """Snap polygon edges to H/V. Each edge becomes either H or V.
    Insert corner vertices where direction changes."""
    n = len(coords)
    new_pts = []
    
    for i in range(n):
        p1 = coords[i]
        p2 = coords[(i+1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if abs(dx) >= abs(dy):
            # Mostly horizontal — snap to H
            # First point, then step H then V
            new_pts.append(p1.copy())
            if abs(dy) > 0.1:  # need a corner
                new_pts.append(np.array([p2[0], p1[1]]))
        else:
            # Mostly vertical — snap to V
            new_pts.append(p1.copy())
            if abs(dx) > 0.1:
                new_pts.append(np.array([p1[0], p2[1]]))
    
    if len(new_pts) < 3:
        return Polygon(coords)
    
    new_pts.append(new_pts[0].copy())
    poly = Polygon(new_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    
    # Simplify to remove redundant vertices
    poly = poly.simplify(0.05)
    print(f"  Snapped boundary: {poly.area:.1f}m², {len(poly.exterior.coords)-1}v")
    return poly


def snap_contour_to_hv(pts, min_seg_len=0.3):
    """Snap a contour's edges to be either horizontal or vertical.
    
    Walk the contour. For each edge, if more horizontal → make H, else → make V.
    Then clean up: merge consecutive same-direction segments, remove tiny ones.
    """
    n = len(pts)
    new_pts = [pts[0].copy()]
    
    for i in range(1, n):
        prev = new_pts[-1]
        curr = pts[i]
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        
        if dx > dy:
            # Horizontal: keep x, snap y to prev
            new_pts.append(np.array([curr[0], prev[1]]))
        else:
            # Vertical: keep y, snap x to prev
            new_pts.append(np.array([prev[0], curr[1]]))
    
    # Close
    new_pts.append(new_pts[0].copy())
    
    # Remove degenerate (zero-length) segments
    cleaned = [new_pts[0]]
    for i in range(1, len(new_pts)):
        if np.linalg.norm(new_pts[i] - cleaned[-1]) > 0.05:
            cleaned.append(new_pts[i])
    
    # Merge consecutive segments in same direction
    if len(cleaned) < 3:
        return Polygon(pts)  # fallback
    
    final = [cleaned[0]]
    for i in range(1, len(cleaned)-1):
        p, c, n_pt = final[-1], cleaned[i], cleaned[i+1]
        # If p→c and c→n are both H or both V, skip c
        pc_h = abs(c[1]-p[1]) < 0.05
        cn_h = abs(n_pt[1]-c[1]) < 0.05
        pc_v = abs(c[0]-p[0]) < 0.05
        cn_v = abs(n_pt[0]-c[0]) < 0.05
        if (pc_h and cn_h) or (pc_v and cn_v):
            continue  # skip this vertex
        final.append(c)
    final.append(final[0])
    
    poly = Polygon(final)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def partition_rooms(boundary, h_walls, v_walls, grid_info, density):
    """Partition boundary using H and V wall lines.
    
    Try combinations of walls to get exactly 4 rooms with good sizes.
    """
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    minx, miny, maxx, maxy = boundary.bounds
    
    print(f"\nH walls: {len(h_walls)}")
    for i, (pos, strength) in enumerate(h_walls[:10]):
        ext = find_wall_extents(density, grid_info, pos, 'h')
        ext_str = f" extent={ext[0]:.1f}..{ext[1]:.1f}" if ext else ""
        print(f"  H[{i}] y={pos:.2f} strength={strength:.0f}{ext_str}")
    
    print(f"V walls: {len(v_walls)}")
    for i, (pos, strength) in enumerate(v_walls[:10]):
        ext = find_wall_extents(density, grid_info, pos, 'v')
        ext_str = f" extent={ext[0]:.1f}..{ext[1]:.1f}" if ext else ""
        print(f"  V[{i}] x={pos:.2f} strength={strength:.0f}{ext_str}")
    
    from itertools import combinations
    from shapely.geometry import LineString
    from shapely.ops import polygonize
    
    # Build candidate wall lines clipped to boundary
    def make_wall_line(pos, orientation):
        if orientation == 'h':
            line = LineString([(minx-1, pos), (maxx+1, pos)])
        else:
            line = LineString([(pos, miny-1), (pos, maxy+1)])
        clipped = line.intersection(boundary)
        if clipped.is_empty:
            return None
        return clipped
    
    # Take top walls
    h_cands = h_walls[:8]
    v_cands = v_walls[:8]
    
    all_walls = []
    for pos, strength in h_cands:
        line = make_wall_line(pos, 'h')
        if line:
            all_walls.append(('h', pos, strength, line))
    for pos, strength in v_cands:
        line = make_wall_line(pos, 'v')
        if line:
            all_walls.append(('v', pos, strength, line))
    
    print(f"\nCandidate walls: {len(all_walls)}")
    
    def try_combo(indices):
        lines = [boundary.exterior]
        for idx in indices:
            lines.append(all_walls[idx][3])
        union = unary_union(lines)
        polys = list(polygonize(union))
        rooms = [p for p in polys if boundary.contains(p.representative_point()) and p.area > 1.5]
        rooms.sort(key=lambda r: r.area, reverse=True)
        return rooms
    
    def score(rooms):
        if len(rooms) != 4:
            return float('inf')
        areas = sorted([r.area for r in rooms], reverse=True)
        # Two bedrooms ~10m², one hallway ~4-5m², one bathroom ~3-4m²
        bed_pen = abs(areas[0]-10.5) + abs(areas[1]-10)
        if areas[0] > 15: bed_pen += (areas[0]-15)*5
        small_pen = 0
        for a in areas[2:]:
            if a < 2: small_pen += (2-a)*3
            elif a > 8: small_pen += (a-8)*2
        total = sum(areas)
        cov_pen = abs(total - 35) * 0.3
        return bed_pen + small_pen + cov_pen
    
    best = None
    best_score = float('inf')
    n_walls = len(all_walls)
    
    for n in range(2, min(7, n_walls+1)):
        for combo in combinations(range(n_walls), n):
            rooms = try_combo(combo)
            if len(rooms) == 4:
                s = score(rooms)
                if s < best_score:
                    best_score = s
                    best = (combo, rooms)
                    areas = sorted([r.area for r in rooms], reverse=True)
                    wdesc = [f"{all_walls[i][0]}{all_walls[i][1]:.1f}" for i in combo]
                    print(f"  [{','.join(wdesc)}] → {[f'{a:.1f}' for a in areas]} score={s:.1f}")
    
    if best is None:
        print("No 4-room found, using all walls + merge")
        rooms = try_combo(range(min(6, n_walls)))
        while len(rooms) > 4:
            idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
            others = [(j, rooms[idx].buffer(0.1).intersection(rooms[j].buffer(0.1)).area) 
                       for j in range(len(rooms)) if j != idx]
            best_j = max(others, key=lambda x: x[1])[0]
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        rooms.sort(key=lambda r: r.area, reverse=True)
        best = (list(range(min(6, n_walls))), rooms)
    
    return best


def classify_rooms(rooms):
    """Classify 4 rooms: 2 bedrooms, 1 hallway, 1 bathroom."""
    labels = [None] * len(rooms)
    areas = [(i, rooms[i].area) for i in range(len(rooms))]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    if len(rooms) >= 4:
        labels[areas[0][0]] = "Bedroom 1"
        labels[areas[1][0]] = "Bedroom 2"
        # Of remaining two, more elongated = hallway
        rem = areas[2:]
        aspects = []
        for idx, _ in rem:
            r = rooms[idx]
            bx = r.bounds[2] - r.bounds[0]
            by = r.bounds[3] - r.bounds[1]
            asp = max(bx, by) / (min(bx, by) + 1e-6)
            aspects.append((idx, asp))
        aspects.sort(key=lambda x: x[1], reverse=True)
        labels[aspects[0][0]] = "Hallway"
        labels[aspects[1][0]] = "Bathroom"
    else:
        for i in range(len(rooms)):
            labels[i] = f"Room {i+1}"
    
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v62')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh = load_mesh(data_dir / args.mesh)
    print(f"Loaded: {len(mesh.vertices)} verts, Y range: {mesh.vertices[:,1].min():.2f} to {mesh.vertices[:,1].max():.2f}")
    
    # Build density in rotated frame
    print(f"\nBuilding density (rotated by -{args.angle}°)...")
    density, grid_info = build_density_rotated(mesh, args.angle)
    if density is None:
        print("ERROR: no density"); return
    
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    print(f"Grid: {w}x{h}, range X=[{xmin:.1f},{xmax:.1f}] Z=[{zmin:.1f},{zmax:.1f}]")
    
    # Build boundary in rotated space
    print("\nBuilding boundary...")
    boundary, mask = build_boundary_rotated(mesh, grid_info, args.angle)
    if boundary is None:
        print("ERROR: no boundary"); return
    print(f"Boundary: {boundary.area:.1f}m², {len(boundary.exterior.coords)-1}v")
    
    # Find H and V wall lines
    print("\nFinding walls...")
    h_walls = find_wall_lines(density, grid_info, 'h')
    v_walls = find_wall_lines(density, grid_info, 'v')
    
    # Partition
    print("\nPartitioning...")
    (combo, rooms) = partition_rooms(boundary, h_walls, v_walls, grid_info, density)
    labels = classify_rooms(rooms)
    
    # Now rotate everything back for display
    def rotate_poly_back(poly, angle_deg, ctr):
        coords = np.array(poly.exterior.coords)
        rot = rotate_points(coords, angle_deg, np.zeros(2))
        # Need to translate: rotated frame was centered on `ctr` in original space
        # Actually the center was used during rotation TO rotated frame,
        # so rotating back should use the same center but positive angle
        rot_back = rotate_points(coords, angle_deg, None)
        # Add center back
        rot_back += ctr
        return Polygon(rot_back)
    
    rooms_orig = [rotate_poly_back(r, args.angle, center) for r in rooms]
    boundary_orig = rotate_poly_back(boundary, args.angle, center)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: density in rotated frame
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'cyan', linewidth=1.5, alpha=0.7)
    ax.set_title(f"Density (rotated -{args.angle}°)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: rooms in rotated frame (clean H/V)
    ax = axes[1]
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9']
    total = 0
    for i, (room, label) in enumerate(zip(rooms, labels)):
        c = pastel[i%len(pastel)]
        xs, ys = room.exterior.xy
        ax.fill(xs, ys, color=c, alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=2)
        total += room.area
        cx, cy = room.centroid.coords[0]
        ax.text(cx, cy, f"{label}\n{room.area:.1f}m²", ha='center', va='center', fontsize=9, fontweight='bold')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=2.5)
    ax.set_title(f"Rotated frame — {len(rooms)} rooms, {total:.1f}m²")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    # Panel 3: rooms rotated back to original orientation
    ax = axes[2]
    for i, (room, label) in enumerate(zip(rooms_orig, labels)):
        c = pastel[i%len(pastel)]
        xs, ys = room.exterior.xy
        ax.fill(xs, ys, color=c, alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=2)
        cx, cy = room.centroid.coords[0]
        ax.text(cx, cy, f"{label}\n{rooms[i].area:.1f}m²", ha='center', va='center', fontsize=9, fontweight='bold')
    bx, by = boundary_orig.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=2.5)
    ax.set_title(f"v62 — {len(rooms)} rooms, {total:.1f}m²")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}")
    
    print(f"\nFinal:")
    for room, label in zip(rooms, labels):
        print(f"  {label}: {room.area:.1f}m² ({len(room.exterior.coords)-1}v)")
    print(f"Total: {total:.1f}m²")


if __name__ == '__main__':
    main()

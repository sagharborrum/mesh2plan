#!/usr/bin/env python3
"""
mesh2plan v62b - Clean Boundary from Wall Lines

v62 showed rotating works. Now: build boundary from outermost H/V walls,
not from contour snapping (which creates staircases).

Approach:
1. Rotate by -29° to axis-align
2. Build density, find H/V wall peaks
3. Outermost walls = boundary edges → clean rectilinear boundary
4. Inner walls partition into 4 rooms
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
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
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


def build_rotated_data(mesh, angle_deg):
    """Build density and boundary mask in rotated coordinate system."""
    pts_xz = mesh.vertices[:, [0, 2]]
    center = pts_xz.mean(axis=0)
    rotated_verts = rotate_points(pts_xz, -angle_deg, center)
    
    # Collect cross-section segments
    all_segs = []
    for y in SLICE_HEIGHTS:
        segs = slice_mesh(mesh, y)
        if len(segs) > 0:
            flat = segs.reshape(-1, 2)
            rot = rotate_points(flat, -angle_deg, center)
            all_segs.append(rot.reshape(-1, 2, 2))
    
    # Grid from all rotated vertices
    all_pts = rotated_verts
    xmin, zmin = all_pts.min(axis=0) - 0.3
    xmax, zmax = all_pts.max(axis=0) + 0.3
    w = int((xmax - xmin) / RESOLUTION)
    h = int((zmax - zmin) / RESOLUTION)
    
    # Density from cross-sections
    density = np.zeros((h, w), dtype=np.float32)
    for segs in all_segs:
        img = np.zeros((h, w), dtype=np.uint8)
        for seg in segs:
            p1x = int(np.clip((seg[0,0]-xmin)/RESOLUTION, 0, w-1))
            p1y = int(np.clip((seg[0,1]-zmin)/RESOLUTION, 0, h-1))
            p2x = int(np.clip((seg[1,0]-xmin)/RESOLUTION, 0, w-1))
            p2y = int(np.clip((seg[1,1]-zmin)/RESOLUTION, 0, h-1))
            cv2.line(img, (p1x, p1y), (p2x, p2y), 1, 1)
        density += img.astype(np.float32)
    
    # Boundary mask from ALL vertices
    vmask = np.zeros((h, w), dtype=np.uint8)
    px = ((rotated_verts[:,0]-xmin)/RESOLUTION).astype(int)
    py = ((rotated_verts[:,1]-zmin)/RESOLUTION).astype(int)
    px = np.clip(px, 0, w-1); py = np.clip(py, 0, h-1)
    vmask[py, px] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, k)
    vmask = binary_fill_holes(vmask).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vmask = cv2.erode(vmask, k2, iterations=3)
    
    grid_info = (xmin, zmin, xmax, zmax, w, h, center)
    return density, vmask, grid_info


def find_wall_peaks(density, vmask, grid_info, orientation='h'):
    """Find wall peaks from density projection, masked to apartment interior."""
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    
    # Mask density to apartment
    masked = density * (vmask > 0).astype(np.float32)
    
    if orientation == 'h':
        profile = masked.sum(axis=1)
        axis_min = zmin
    else:
        profile = masked.sum(axis=0)
        axis_min = xmin
    
    # Smooth
    kernel = np.ones(3) / 3
    profile = np.convolve(profile, kernel, mode='same')
    
    # Peaks
    threshold = np.percentile(profile[profile > 0], 60) if (profile > 0).any() else 0
    peaks = []
    for i in range(2, len(profile)-2):
        if profile[i] > threshold and profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
            pos = i * RESOLUTION + axis_min
            if not peaks or abs(pos - peaks[-1][0]) > 0.25:
                peaks.append((pos, float(profile[i])))
    
    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks


def build_clean_boundary(h_walls, v_walls, vmask, grid_info):
    """Build boundary from outermost walls.
    
    Find the outermost H and V walls that enclose the apartment.
    The apartment shape in the target is NOT a simple rectangle — it has 
    a notch/step. So we need the mask shape to guide which walls form the boundary.
    """
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    
    # Get approximate bounding rect from mask
    contours, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
    pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
    mask_poly = Polygon(pts).buffer(0)
    
    # Find outermost walls in each direction
    h_sorted = sorted(h_walls, key=lambda w: w[0])  # by position
    v_sorted = sorted(v_walls, key=lambda w: w[0])
    
    # Bottom H wall = lowest strong wall near mask bottom
    mask_bounds = mask_poly.bounds  # minx, miny, maxx, maxy
    
    # For boundary: pick the outermost strong walls
    # H: need bottom and top
    # V: need left and right
    h_positions = [p for p, s in h_sorted if s > 20]
    v_positions = [p for p, s in v_sorted if s > 20]
    
    if len(h_positions) < 2 or len(v_positions) < 2:
        return mask_poly.simplify(0.2)
    
    y_bot = h_positions[0]   # bottommost
    y_top = h_positions[-1]  # topmost  
    x_left = v_positions[0]
    x_right = v_positions[-1]
    
    print(f"  Outer walls: x=[{x_left:.2f}, {x_right:.2f}] y=[{y_bot:.2f}, {y_top:.2f}]")
    
    # The apartment is NOT rectangular — look at the target geometry.
    # It has a step: the top portion is narrower (doesn't extend as far left).
    # Let's detect this from the mask.
    
    # Sample the mask at different Y levels to find the X extent
    extents = []
    for row_y in np.arange(y_bot, y_top, 0.2):
        py = int((row_y - zmin) / RESOLUTION)
        if 0 <= py < h:
            row = vmask[py, :]
            nz = np.where(row > 0)[0]
            if len(nz) > 10:
                xl = nz[0] * RESOLUTION + xmin
                xr = nz[-1] * RESOLUTION + xmin
                extents.append((row_y, xl, xr))
    
    if not extents:
        return Polygon([(x_left, y_bot), (x_right, y_bot), (x_right, y_top), (x_left, y_top)])
    
    # Find if there's a step — the left extent changes significantly at some Y
    # Group into "wide" and "narrow" sections
    mid_y = (y_bot + y_top) / 2
    bottom_extents = [(y, xl, xr) for y, xl, xr in extents if y < mid_y]
    top_extents = [(y, xl, xr) for y, xl, xr in extents if y >= mid_y]
    
    if bottom_extents and top_extents:
        bot_left = np.median([xl for _, xl, _ in bottom_extents])
        top_left = np.median([xl for _, xl, _ in top_extents])
        bot_right = np.median([xr for _, _, xr in bottom_extents])
        top_right = np.median([xr for _, _, xr in top_extents])
        
        # Snap to nearest wall positions
        def snap_to_wall(val, positions, tol=0.5):
            best = min(positions, key=lambda p: abs(p - val))
            return best if abs(best - val) < tol else val
        
        bot_left = snap_to_wall(bot_left, v_positions)
        top_left = snap_to_wall(top_left, v_positions)
        bot_right = snap_to_wall(bot_right, v_positions)
        top_right = snap_to_wall(top_right, v_positions)
        
        # Detect step in top section — check for a narrower top 
        # Look for H wall near the step position
        step_y_candidates = [p for p, s in h_walls if y_bot + 0.5 < p < y_top - 0.5 and s > 20]
        
        if abs(bot_left - top_left) > 0.3 or abs(bot_right - top_right) > 0.3:
            # There IS a step — build L-shaped or stepped boundary
            # Find best step Y from H walls
            if step_y_candidates:
                step_y = min(step_y_candidates, key=lambda p: abs(p - mid_y))
            else:
                step_y = mid_y
            
            print(f"  Step detected at y={step_y:.2f}: bot_left={bot_left:.2f}, top_left={top_left:.2f}")
            
            # Build stepped polygon (CCW)
            # Bottom-left → bottom-right → step-right → top-right → top-left → step-left → back
            if top_left > bot_left:  # top is narrower on left
                pts = [
                    (bot_left, y_bot),
                    (bot_right, y_bot),
                    (top_right, y_bot),  # might be same as bot_right
                    (top_right, y_top),
                    (top_left, y_top),
                    (top_left, step_y),
                    (bot_left, step_y),
                ]
            else:  # top extends further left
                pts = [
                    (bot_left, y_bot),
                    (bot_right, y_bot),
                    (bot_right, step_y),
                    (top_right, step_y),
                    (top_right, y_top),
                    (top_left, y_top),
                    (top_left, y_bot),
                ]
            
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            return poly
    
    # Simple rectangle
    return Polygon([(x_left, y_bot), (x_right, y_bot), (x_right, y_top), (x_left, y_top)])


def partition_rooms(boundary, h_walls, v_walls, density, vmask, grid_info):
    """Try wall combos to get exactly 4 rooms."""
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    minx, miny, maxx, maxy = boundary.bounds
    
    # Filter to interior walls (not on boundary edges)
    margin = 0.3
    h_interior = [(p, s) for p, s in h_walls if miny + margin < p < maxy - margin]
    v_interior = [(p, s) for p, s in v_walls if minx + margin < p < maxx - margin]
    
    print(f"\nInterior H walls: {len(h_interior)}")
    for i, (pos, s) in enumerate(h_interior[:8]):
        print(f"  H[{i}] y={pos:.2f} str={s:.0f}")
    print(f"Interior V walls: {len(v_interior)}")
    for i, (pos, s) in enumerate(v_interior[:8]):
        print(f"  V[{i}] y={pos:.2f} str={s:.0f}")
    
    def make_line(pos, orient):
        if orient == 'h':
            return LineString([(minx - 1, pos), (maxx + 1, pos)])
        else:
            return LineString([(pos, miny - 1), (pos, maxy + 1)])
    
    all_walls = []
    for pos, s in h_interior[:8]:
        line = make_line(pos, 'h')
        cl = line.intersection(boundary)
        if not cl.is_empty:
            all_walls.append(('h', pos, s, cl))
    for pos, s in v_interior[:8]:
        line = make_line(pos, 'v')
        cl = line.intersection(boundary)
        if not cl.is_empty:
            all_walls.append(('v', pos, s, cl))
    
    print(f"\nCandidate interior walls: {len(all_walls)}")
    for i, (o, p, s, _) in enumerate(all_walls):
        print(f"  [{i}] {o} pos={p:.2f} str={s:.0f}")
    
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
        if len(rooms) != 4: return float('inf')
        areas = sorted([r.area for r in rooms], reverse=True)
        # Two bedrooms ~10m², hallway ~4m², bathroom ~3m²
        bed = abs(areas[0]-10.5) + abs(areas[1]-10)
        if areas[0] > 14: bed += (areas[0]-14)*5
        small = sum(max(0, 2-a)*3 + max(0, a-7)*2 for a in areas[2:])
        total = sum(areas)
        cov = abs(total - 35) * 0.3
        return bed + small + cov
    
    best = None
    best_score = float('inf')
    nw = len(all_walls)
    
    for n in range(2, min(7, nw+1)):
        for combo in combinations(range(nw), n):
            rooms = try_combo(combo)
            if len(rooms) == 4:
                s = score(rooms)
                if s < best_score:
                    best_score = s
                    best = (combo, rooms)
                    areas = sorted([r.area for r in rooms], reverse=True)
                    wdesc = [f"{all_walls[i][0]}{all_walls[i][1]:.1f}" for i in combo]
                    print(f"  ✓ [{','.join(wdesc)}] → {[f'{a:.1f}' for a in areas]} score={s:.1f}")
    
    if best is None:
        print("No 4-room found, trying merge from all walls...")
        rooms = try_combo(range(min(5, nw)))
        while len(rooms) > 4:
            idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
            others = [(j, rooms[idx].buffer(0.1).intersection(rooms[j].buffer(0.1)).area) 
                       for j in range(len(rooms)) if j != idx]
            best_j = max(others, key=lambda x: x[1])[0]
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        rooms.sort(key=lambda r: r.area, reverse=True)
        best = (list(range(min(5, nw))), rooms)
    
    return best


def classify_rooms(rooms):
    labels = [None] * len(rooms)
    areas = [(i, rooms[i].area) for i in range(len(rooms))]
    areas.sort(key=lambda x: x[1], reverse=True)
    if len(rooms) >= 4:
        labels[areas[0][0]] = "Bedroom 1"
        labels[areas[1][0]] = "Bedroom 2"
        rem = areas[2:]
        aspects = []
        for idx, _ in rem:
            r = rooms[idx]
            bx = r.bounds[2] - r.bounds[0]
            by = r.bounds[3] - r.bounds[1]
            aspects.append((idx, max(bx,by)/(min(bx,by)+1e-6)))
        aspects.sort(key=lambda x: x[1], reverse=True)
        labels[aspects[0][0]] = "Hallway"
        labels[aspects[1][0]] = "Bathroom"
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v62b')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh = load_mesh(data_dir / args.mesh)
    print(f"Loaded: {len(mesh.vertices)} verts")
    
    print(f"\nBuilding rotated data (-{args.angle}°)...")
    density, vmask, grid_info = build_rotated_data(mesh, args.angle)
    xmin, zmin, xmax, zmax, w, h, center = grid_info
    print(f"Grid: {w}x{h}")
    
    # Find wall peaks
    h_walls = find_wall_peaks(density, vmask, grid_info, 'h')
    v_walls = find_wall_peaks(density, vmask, grid_info, 'v')
    
    print(f"\nH wall peaks ({len(h_walls)}):")
    for i, (p, s) in enumerate(h_walls[:10]):
        print(f"  [{i}] y={p:.2f} str={s:.0f}")
    print(f"V wall peaks ({len(v_walls)}):")
    for i, (p, s) in enumerate(v_walls[:10]):
        print(f"  [{i}] x={p:.2f} str={s:.0f}")
    
    # Build clean boundary from outermost walls
    print("\nBuilding boundary...")
    boundary = build_clean_boundary(h_walls, v_walls, vmask, grid_info)
    if boundary is None:
        print("ERROR: no boundary"); return
    print(f"Boundary: {boundary.area:.1f}m², {len(boundary.exterior.coords)-1}v")
    
    # Partition
    combo, rooms = partition_rooms(boundary, h_walls, v_walls, density, vmask, grid_info)
    labels = classify_rooms(rooms)
    
    # Rotate back
    rooms_orig = []
    for r in rooms:
        coords = np.array(r.exterior.coords)
        rot = rotate_points(coords, args.angle, np.zeros(2)) + center
        rooms_orig.append(Polygon(rot))
    bcoords = np.array(boundary.exterior.coords)
    brot = rotate_points(bcoords, args.angle, np.zeros(2)) + center
    boundary_orig = Polygon(brot)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: density + boundary
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=[xmin,xmax,zmin,zmax], aspect='equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'cyan', linewidth=2)
    ax.set_title(f"Density + boundary")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: rooms in rotated frame
    ax = axes[1]
    pastel = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9']
    total = 0
    for i, (room, label) in enumerate(zip(rooms, labels)):
        c = pastel[i%4]
        xs, ys = room.exterior.xy
        ax.fill(xs, ys, color=c, alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=2)
        total += room.area
        cx, cy = room.centroid.coords[0]
        ax.text(cx, cy, f"{label}\n{room.area:.1f}m²", ha='center', va='center', fontsize=9, fontweight='bold')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=2.5)
    ax.set_title(f"Rotated — {len(rooms)} rooms, {total:.1f}m²")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    # Panel 3: rotated back
    ax = axes[2]
    for i, (room, label) in enumerate(zip(rooms_orig, labels)):
        c = pastel[i%4]
        xs, ys = room.exterior.xy
        ax.fill(xs, ys, color=c, alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=2)
        cx, cy = room.centroid.coords[0]
        ax.text(cx, cy, f"{label}\n{rooms[i].area:.1f}m²", ha='center', va='center', fontsize=9, fontweight='bold')
    bx, by = boundary_orig.exterior.xy
    ax.plot(bx, by, 'k-', linewidth=2.5)
    ax.set_title(f"v62b — {len(rooms)} rooms, {total:.1f}m²")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}")
    
    for room, label in zip(rooms, labels):
        print(f"  {label}: {room.area:.1f}m² ({len(room.exterior.coords)-1}v)")
    print(f"Total: {total:.1f}m²")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v60b - Cross-Section Flood Fill

Simpler than v60: cross-section density → wall mask → flood fill rooms → vectorize.
No skeleton, no Hough. The cross-section contour IS the wall shape directly.

Pipeline:
1. Multi-slice cross-sections → density image
2. Threshold density → wall mask (binary)
3. Morphological close to seal small gaps
4. Invert → interior = rooms, walls = barriers
5. Connected components → each component = room seed
6. Extract room contours → simplify → snap edges to 29°/119°
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.ndimage import binary_fill_holes, label as scipy_label
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


RESOLUTION = 0.02
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]
DOMINANT_ANGLES = [29.0, 119.0]  # Known from RANSAC
MIN_ROOM_AREA = 2.0


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def multi_slice_density(mesh, resolution=RESOLUTION):
    """Cross-section density from multiple horizontal slices."""
    verts = mesh.vertices
    xmin, xmax = verts[:, 0].min() - 0.5, verts[:, 0].max() + 0.5
    zmin, zmax = verts[:, 2].min() - 0.5, verts[:, 2].max() + 0.5
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    density = np.zeros((h, w), dtype=np.float32)
    
    for y_height in SLICE_HEIGHTS:
        try:
            lines = trimesh.intersections.mesh_plane(
                mesh, [0, 1, 0], [0, y_height, 0]
            )
        except Exception:
            continue
        
        if lines is None or len(lines) == 0:
            continue
        
        slice_img = np.zeros((h, w), dtype=np.uint8)
        segments_xz = lines[:, :, [0, 2]]
        
        for seg in segments_xz:
            p1x = int((seg[0, 0] - xmin) / resolution)
            p1y = int((seg[0, 1] - zmin) / resolution)
            p2x = int((seg[1, 0] - xmin) / resolution)
            p2y = int((seg[1, 1] - zmin) / resolution)
            p1x, p1y = np.clip(p1x, 0, w-1), np.clip(p1y, 0, h-1)
            p2x, p2y = np.clip(p2x, 0, w-1), np.clip(p2y, 0, h-1)
            cv2.line(slice_img, (p1x, p1y), (p2x, p2y), 1, thickness=2)
        
        density += slice_img.astype(np.float32)
        print(f"  Y={y_height:.1f}m: {len(lines)} segments")
    
    print(f"  Density: {w}x{h}, max={density.max():.0f}")
    return density, (xmin, zmin, xmax, zmax, w, h)


def extract_rooms(density, grid_info, wall_threshold=2, close_size=15, 
                  open_size=5, min_area_px=None):
    """Extract room polygons from density image via flood fill."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    if min_area_px is None:
        min_area_px = int(MIN_ROOM_AREA / (RESOLUTION ** 2))
    
    # Wall mask: pixels where cross-section appears in >= threshold slices
    wall_mask = (density >= wall_threshold).astype(np.uint8) * 255
    
    # Thicken walls slightly
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.dilate(wall_mask, dilate_k, iterations=1)
    
    # Close gaps in walls (doors, scanning artifacts)
    # Use directional closing along dominant angles
    for angle in DOMINANT_ANGLES:
        rad = np.radians(angle)
        dx = np.cos(rad)
        dz = np.sin(rad)
        # Create angled kernel
        ksize = close_size
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        cx, cy = ksize // 2, ksize // 2
        for t in range(-ksize//2, ksize//2 + 1):
            px = int(cx + t * dx)
            py = int(cy + t * dz)
            if 0 <= px < ksize and 0 <= py < ksize:
                kernel[py, px] = 1
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    
    # Also close with a small circular kernel for general gaps
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, close_k)
    
    # Create apartment mask (fill the boundary)
    apt_mask = wall_mask.copy()
    # Flood fill from edges to find exterior
    flood = np.zeros((h+2, w+2), dtype=np.uint8)
    cv2.floodFill(apt_mask, flood, (0, 0), 255)
    # Interior = NOT exterior AND NOT wall
    interior = (apt_mask == 0).astype(np.uint8) * 255
    
    # Remove wall pixels from interior
    interior[wall_mask > 0] = 0
    
    # Open to remove thin connections through doorways
    if open_size > 0:
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, open_k)
    
    # Connected components = rooms
    labeled, n_labels = scipy_label(interior > 0)
    print(f"  Connected components: {n_labels}")
    
    rooms = []
    for i in range(1, n_labels + 1):
        component = (labeled == i).astype(np.uint8) * 255
        area_px = component.sum() / 255
        area_m2 = area_px * (RESOLUTION ** 2)
        
        if area_m2 < MIN_ROOM_AREA:
            continue
        
        # Extract contour
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 3:
            continue
        
        # Convert to world coordinates
        pts = cnt.reshape(-1, 2).astype(float)
        pts[:, 0] = pts[:, 0] * RESOLUTION + xmin
        pts[:, 1] = pts[:, 1] * RESOLUTION + zmin
        
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # Simplify
        poly = poly.simplify(0.08, preserve_topology=True)
        
        rooms.append(poly)
    
    rooms.sort(key=lambda p: p.area, reverse=True)
    print(f"  Rooms above {MIN_ROOM_AREA}m²: {len(rooms)}")
    
    return rooms, wall_mask, interior, labeled


def snap_polygon_to_angles(poly, angles, tolerance=0.2):
    """Snap polygon edges to dominant wall angles."""
    coords = list(poly.exterior.coords[:-1])  # Remove closing coord
    if len(coords) < 3:
        return poly
    
    snapped = []
    angle_rads = [np.radians(a) for a in angles]
    directions = [(np.cos(r), np.sin(r)) for r in angle_rads]
    
    # For each vertex, project to maintain snapped edges
    for i in range(len(coords)):
        p = np.array(coords[i])
        snapped.append(p)
    
    # Snap each edge to closest dominant angle
    new_coords = [np.array(coords[0])]
    for i in range(1, len(coords)):
        p_prev = new_coords[-1]
        p_curr = np.array(coords[i])
        
        edge = p_curr - p_prev
        edge_len = np.linalg.norm(edge)
        if edge_len < 0.05:
            continue
        
        edge_angle = np.degrees(np.arctan2(edge[1], edge[0])) % 180
        
        # Find closest dominant angle
        best_da = None
        best_diff = 180
        for da in angles:
            diff = abs(edge_angle - da)
            diff = min(diff, 180 - diff)
            if diff < best_diff:
                best_diff = diff
                best_da = da
        
        if best_diff < 25:  # Only snap if close enough
            # Project endpoint to maintain dominant angle
            rad = np.radians(best_da)
            direction = np.array([np.cos(rad), np.sin(rad)])
            
            # Project edge onto direction
            proj = np.dot(edge, direction)
            if proj < 0:
                direction = -direction
                proj = -proj
            
            # Keep perpendicular component from original
            perp = np.array([-direction[1], direction[0]])
            perp_comp = np.dot(edge, perp)
            
            # Snapped endpoint: move along direction, discard perpendicular drift
            new_p = p_prev + direction * proj
            new_coords.append(new_p)
        else:
            new_coords.append(p_curr)
    
    if len(new_coords) < 3:
        return poly
    
    try:
        snapped_poly = Polygon(new_coords)
        if snapped_poly.is_valid and snapped_poly.area > 0.5:
            return snapped_poly
    except:
        pass
    
    return poly


def classify_room(poly):
    area = poly.area
    bounds = poly.bounds
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    aspect = max(w, h) / (min(w, h) + 1e-6)
    
    if area > 8:
        return "Room"
    elif area > 5:
        return "Room"
    elif aspect > 2.5:
        return "Hallway"
    elif area > 3:
        return "Bathroom"
    else:
        return "Closet"


def plot_results(density, grid_info, wall_mask, interior, rooms, output_path):
    xmin, zmin, xmax, zmax, w, h = grid_info
    extent = [xmin, xmax, zmin, zmax]
    
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    
    # Panel 1: Cross-section density
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot', extent=extent, aspect='equal')
    ax.set_title(f"Cross-Section Density ({len(SLICE_HEIGHTS)} slices)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Wall mask
    ax = axes[1]
    ax.imshow(wall_mask, origin='lower', cmap='gray', extent=extent, aspect='equal')
    ax.set_title("Wall Mask (closed)")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Interior / rooms
    ax = axes[2]
    ax.imshow(interior, origin='lower', cmap='gray', extent=extent, aspect='equal')
    ax.set_title("Interior (rooms)")
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Room polygons
    ax = axes[3]
    pastel = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', '#E8BAFF', '#FFD4BA', '#BAF0FF']
    
    total_area = 0
    for i, room in enumerate(rooms):
        color = pastel[i % len(pastel)]
        geoms = room.geoms if isinstance(room, MultiPolygon) else [room]
        for geom in geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2)
        
        area = room.area
        total_area += area
        label = classify_room(room)
        cx, cy = room.centroid.coords[0]
        nv = len(room.exterior.coords) - 1
        ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    ax.set_title(f"v60b — {len(rooms)} rooms, {total_area:.1f}m²\n"
                 f"Angles: {DOMINANT_ANGLES[0]:.0f}°, {DOMINANT_ANGLES[1]:.0f}°")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Scale bar
    ax.plot([-4, -3], [-5, -5], 'k-', linewidth=3)
    ax.text(-3.5, -5.3, '1m', ha='center', fontsize=10)
    
    plt.tight_layout()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'floorplan.png'}")
    return total_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v60b')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--wall-thresh', type=int, default=2)
    parser.add_argument('--close-size', type=int, default=20)
    parser.add_argument('--open-size', type=int, default=7)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh_path = data_dir / args.mesh
    
    mesh = load_mesh(mesh_path)
    
    print("\nStep 1: Multi-slice cross-sections...")
    density, grid_info = multi_slice_density(mesh)
    
    print("\nStep 2: Extracting rooms via flood fill...")
    rooms, wall_mask, interior, labeled = extract_rooms(
        density, grid_info, 
        wall_threshold=args.wall_thresh,
        close_size=args.close_size,
        open_size=args.open_size
    )
    
    # Snap polygons to dominant angles
    print("\nStep 3: Snapping to dominant angles...")
    snapped_rooms = []
    for room in rooms:
        snapped = snap_polygon_to_angles(room, DOMINANT_ANGLES)
        snapped_rooms.append(snapped)
        label = classify_room(snapped)
        nv = len(snapped.exterior.coords) - 1
        print(f"  {label}: {snapped.area:.1f}m² ({nv}v)")
    
    print("\nStep 4: Plotting...")
    total = plot_results(density, grid_info, wall_mask, interior, snapped_rooms, args.output)
    
    print(f"\n{'='*50}")
    print(f"v60b: {len(snapped_rooms)} rooms, {total:.1f}m²")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

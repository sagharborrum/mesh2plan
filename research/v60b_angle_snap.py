#!/usr/bin/env python3
"""
mesh2plan v60b - Contour Rooms + Angle-Snapped Polygons

Same room extraction as v60 but with proper angle-constrained polygon fitting.
Each room contour is simplified into a polygon where ALL edges are aligned
to one of the two dominant wall angles.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           label, binary_closing, binary_opening, gaussian_filter)
from skimage.measure import find_contours, approximate_polygon
from skimage.segmentation import watershed
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union
from shapely.affinity import rotate


RESOLUTION = 0.02
SLICE_HEIGHTS = np.arange(-1.8, -0.5, 0.05)
WALL_CLOSE_RADIUS = 25
MIN_ROOM_AREA = 2.0
WALL_NORMAL_THRESH = 0.5
DENSITY_PCT = 60


def load_mesh(path):
    print(f"Loading mesh: {path}")
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def wall_density_image(mesh, resolution):
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < WALL_NORMAL_THRESH
    wall_faces = mesh.faces[wall_mask]
    wall_verts = mesh.vertices[wall_faces]
    centroids = wall_verts.mean(axis=1)
    areas = mesh.area_faces[wall_mask]
    
    xs, zs = centroids[:, 0], centroids[:, 2]
    xmin, xmax = xs.min() - 0.2, xs.max() + 0.2
    zmin, zmax = zs.min() - 0.2, zs.max() + 0.2
    
    w = int((xmax - xmin) / resolution) + 1
    h = int((zmax - zmin) / resolution) + 1
    
    density = np.zeros((h, w), dtype=np.float64)
    xi = ((xs - xmin) / resolution).astype(int).clip(0, w-1)
    zi = ((zs - zmin) / resolution).astype(int).clip(0, h-1)
    np.add.at(density, (zi, xi), areas)
    
    print(f"  Wall density: {w}x{h}, wall faces: {wall_mask.sum()}/{len(mesh.faces)}")
    return density, (xmin, zmin)


def get_apartment_mask(mesh, resolution, origin, shape):
    verts = mesh.vertices
    xmin, zmin = origin
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    xi = ((verts[:, 0] - xmin) / resolution).astype(int).clip(0, w-1)
    zi = ((verts[:, 2] - zmin) / resolution).astype(int).clip(0, h-1)
    mask[zi, xi] = True
    mask = binary_dilation(mask, iterations=5)
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=5)
    return mask


def detect_dominant_angles(mesh):
    normals = mesh.face_normals
    areas = mesh.area_faces
    wall_mask = np.abs(normals[:, 1]) < 0.5
    wall_normals = normals[wall_mask]
    wall_areas = areas[wall_mask]
    
    angles = np.degrees(np.arctan2(wall_normals[:, 2], wall_normals[:, 0])) % 180
    hist, bins = np.histogram(angles, bins=180, range=(0, 180), weights=wall_areas)
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')
    from scipy.signal import find_peaks
    peaks, props = find_peaks(hist_smooth, distance=20, height=0)
    
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
        wall_angles = sorted([(bins[p] + 90) % 180 for p in top2])
    else:
        wall_angles = [30, 120]
    
    print(f"  Dominant wall angles: {wall_angles[0]:.0f}° and {wall_angles[1]:.0f}°")
    return wall_angles


def extract_rooms(density, apt_mask, resolution):
    """Wall density → binary wall mask → invert → rooms → watershed expand."""
    nonzero = density[density > 0]
    if len(nonzero) == 0:
        return None, None
    
    thresh = np.percentile(nonzero, DENSITY_PCT)
    wall_mask = density >= thresh
    wall_mask = wall_mask & apt_mask
    
    # Thicken walls
    wall_mask = binary_dilation(wall_mask, iterations=2)
    
    # Directional closing
    from skimage.morphology import disk as sk_disk
    for angle in [29, 119]:
        rad = np.radians(angle)
        length = WALL_CLOSE_RADIUS
        struct = np.zeros((2*length+1, 2*length+1), dtype=bool)
        for t in np.linspace(-length, length, 4*length+1):
            r = int(round(length + t * np.sin(rad)))
            c = int(round(length + t * np.cos(rad)))
            if 0 <= r < 2*length+1 and 0 <= c < 2*length+1:
                struct[r, c] = True
        wall_mask = binary_closing(wall_mask, structure=struct)
    
    wall_mask = binary_closing(wall_mask, structure=sk_disk(5))
    wall_mask = binary_opening(wall_mask, iterations=2)
    
    # Interior = apartment minus walls
    interior = apt_mask & ~wall_mask
    labeled, n_rooms = label(interior)
    
    # Filter by area
    rooms = []
    for i in range(1, n_rooms + 1):
        area = (labeled == i).sum() * resolution * resolution
        if area >= MIN_ROOM_AREA:
            rooms.append((i, area))
    
    print(f"  Pre-expansion: {len(rooms)} rooms, {sum(a for _,a in rooms):.1f}m²")
    
    # Watershed expand into wall area
    gradient = gaussian_filter(density, sigma=2)
    gradient = gradient / (gradient.max() + 1e-10)
    
    markers = np.zeros_like(labeled)
    for rid, _ in rooms:
        markers[labeled == rid] = rid
    
    expanded = watershed(gradient, markers=markers, mask=apt_mask)
    
    new_rooms = []
    for rid, _ in rooms:
        area = (expanded == rid).sum() * resolution * resolution
        if area >= MIN_ROOM_AREA:
            new_rooms.append((rid, area))
    
    print(f"  Post-expansion: {len(new_rooms)} rooms, {sum(a for _,a in new_rooms):.1f}m²")
    
    return (expanded, new_rooms), wall_mask


def fit_angle_constrained_polygon(contour_coords, angles_deg, min_edge=0.3):
    """
    Fit a polygon to contour where all edges are at one of the dominant angles.
    
    Strategy:
    1. Rotate contour so angle[0] is horizontal
    2. Fit axis-aligned bounding polygon (rectilinear)
    3. Rotate back
    
    For L-shapes: use convex hull corners as guide points, then fit rectilinear
    polygon in rotated space.
    """
    if len(contour_coords) < 4:
        return None
    
    angle0 = angles_deg[0]
    
    # Rotate so dominant angle is horizontal
    rad = np.radians(-angle0)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    coords = np.array(contour_coords)
    cx, cy = coords.mean(axis=0)
    
    # Rotate around centroid
    rotated = np.zeros_like(coords)
    rotated[:, 0] = (coords[:, 0] - cx) * cos_a - (coords[:, 1] - cy) * sin_a
    rotated[:, 1] = (coords[:, 0] - cx) * sin_a + (coords[:, 1] - cy) * cos_a
    
    # In rotated space, fit a rectilinear polygon
    # Start with bounding box
    xmin_r, xmax_r = rotated[:, 0].min(), rotated[:, 0].max()
    ymin_r, ymax_r = rotated[:, 1].min(), rotated[:, 1].max()
    
    # Try to detect L-shape or notch by dividing into grid cells
    # and checking which cells are inside the contour
    from shapely.geometry import Polygon as SPoly, Point as SPoint
    
    original_poly = SPoly(contour_coords)
    if not original_poly.is_valid:
        original_poly = original_poly.buffer(0)
    
    # Grid resolution for rectilinear fitting
    grid_res = 0.2  # 20cm
    nx = max(int((xmax_r - xmin_r) / grid_res), 1)
    ny = max(int((ymax_r - ymin_r) / grid_res), 1)
    
    grid = np.zeros((ny, nx), dtype=bool)
    
    # Unrotate grid cell centers to check containment
    rad_back = np.radians(angle0)
    cos_b, sin_b = np.cos(rad_back), np.sin(rad_back)
    
    for iy in range(ny):
        for ix in range(nx):
            # Rotated coordinates of cell center
            rx = xmin_r + (ix + 0.5) * grid_res
            ry = ymin_r + (iy + 0.5) * grid_res
            
            # Unrotate back to world
            wx = rx * cos_b - ry * sin_b + cx
            wy = rx * sin_b + ry * cos_b + cy
            
            grid[iy, ix] = original_poly.contains(SPoint(wx, wy))
    
    # Now extract rectilinear contour of the grid
    # Expand grid cells to polygon
    cell_polys = []
    for iy in range(ny):
        for ix in range(nx):
            if grid[iy, ix]:
                x0 = xmin_r + ix * grid_res
                x1 = x0 + grid_res
                y0 = ymin_r + iy * grid_res
                y1 = y0 + grid_res
                cell_polys.append(SPoly([
                    (x0, y0), (x1, y0), (x1, y1), (x0, y1)
                ]))
    
    if not cell_polys:
        return None
    
    rectilinear = unary_union(cell_polys)
    if rectilinear.is_empty:
        return None
    
    # Simplify the rectilinear polygon
    if isinstance(rectilinear, MultiPolygon):
        rectilinear = max(rectilinear.geoms, key=lambda g: g.area)
    
    rectilinear = rectilinear.simplify(grid_res * 0.3)
    
    # Rotate back to world coordinates
    rect_coords = list(rectilinear.exterior.coords)
    world_coords = []
    for rx, ry in rect_coords:
        wx = rx * cos_b - ry * sin_b + cx
        wy = rx * sin_b + ry * cos_b + cy
        world_coords.append((wx, wy))
    
    result = Polygon(world_coords)
    if not result.is_valid:
        result = result.buffer(0)
    
    return result


def classify_room(area, aspect):
    if area > 8: return "Room"
    elif area > 5:
        return "Hallway" if aspect > 2.5 else "Room"
    elif area > 3:
        return "Hallway" if aspect > 2 else "Bathroom"
    else:
        return "Closet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', help='Path to mesh')
    parser.add_argument('-o', '--output', default='v60b_angle_snap.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    wall_angles = detect_dominant_angles(mesh)
    density, origin = wall_density_image(mesh, RESOLUTION)
    apt_mask = get_apartment_mask(mesh, RESOLUTION, origin, density.shape)
    
    result, wall_mask = extract_rooms(density, apt_mask, RESOLUTION)
    if result is None:
        print("No rooms found")
        return
    
    expanded, rooms = result
    
    # Vectorize each room with angle-constrained polygons
    xmin, zmin = origin
    room_polys = []
    
    print(f"\n=== RESULTS ===")
    total = 0
    
    for rid, area in rooms:
        room_mask = (expanded == rid).astype(float)
        contours = find_contours(room_mask, 0.5)
        if not contours:
            room_polys.append(None)
            continue
        
        contour = max(contours, key=len)
        coords = [(c[1] * RESOLUTION + xmin, c[0] * RESOLUTION + zmin) for c in contour]
        
        poly = fit_angle_constrained_polygon(coords, wall_angles)
        
        if poly is None or poly.is_empty or poly.area < 1.0:
            # Fallback to simplified contour
            poly = Polygon(coords).simplify(0.15)
        
        room_polys.append(poly)
        
        bounds = poly.bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        aspect = max(w, h) / (min(w, h) + 0.01)
        nv = len(poly.exterior.coords) - 1
        label_text = classify_room(poly.area, aspect)
        total += poly.area
        print(f"  {label_text}: {poly.area:.1f}m² ({nv}v)")
    
    print(f"  Total: {total:.1f}m²")
    print(f"  Angles: {wall_angles[0]:.0f}° and {wall_angles[1]:.0f}°")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    h, w = density.shape
    extent = [xmin, xmin + w*RESOLUTION, zmin, zmin + h*RESOLUTION]
    
    # Panel 1: Wall density
    ax = axes[0]
    display = np.where(density > 0, density, np.nan)
    ax.imshow(display.T, origin='lower', extent=extent, cmap='hot', aspect='equal')
    ax.set_title("Wall Density")
    ax.set_aspect('equal')
    
    # Panel 2: Labeled rooms (expanded)
    ax = axes[1]
    pastel = ['#AEC6CF', '#FFB3BA', '#BAFFC9', '#FFFFBA', '#E8BAFF', '#FFD1A4',
              '#C4E1FF', '#FFC4E1']
    vis = np.ones((*expanded.shape, 3))
    for idx, (rid, _) in enumerate(rooms):
        c = matplotlib.colors.to_rgb(pastel[idx % len(pastel)])
        vis[expanded == rid] = c
    vis[wall_mask & apt_mask] = [0.3, 0.3, 0.3]
    ax.imshow(np.transpose(vis, (1,0,2)), origin='lower', extent=extent, aspect='equal')
    ax.set_title(f"Expanded Rooms ({len(rooms)})")
    ax.set_aspect('equal')
    
    # Panel 3: Angle-snapped polygons
    ax = axes[2]
    total_area = 0
    for idx, (poly, (rid, _)) in enumerate(zip(room_polys, rooms)):
        if poly is None:
            continue
        total_area += poly.area
        
        bounds = poly.bounds
        w_r = bounds[2] - bounds[0]
        h_r = bounds[3] - bounds[1]
        aspect = max(w_r, h_r) / (min(w_r, h_r) + 0.01)
        label_text = classify_room(poly.area, aspect)
        
        color = pastel[idx % len(pastel)]
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=color, alpha=0.6)
        ax.plot(xs, ys, 'k-', linewidth=2)
        
        cx, cy = poly.centroid.x, poly.centroid.y
        nv = len(poly.exterior.coords) - 1
        ax.text(cx, cy, f"{label_text}\n{poly.area:.1f}m²\n({nv}v)",
                ha='center', va='center', fontsize=8, weight='bold')
    
    # Boundary
    bnd_contours = find_contours(apt_mask.astype(float), 0.5)
    if bnd_contours:
        bnd = max(bnd_contours, key=len)
        bx = [c[1]*RESOLUTION+xmin for c in bnd]
        by = [c[0]*RESOLUTION+zmin for c in bnd]
        ax.plot(bx, by, 'k--', linewidth=1)
    
    ax.set_title(f"v60b — {sum(1 for p in room_polys if p)} rooms, {total_area:.1f}m²\nAngles: {wall_angles[0]:.0f}°, {wall_angles[1]:.0f}°")
    ax.set_aspect('equal')
    
    # Scale bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.3, ylim[0]+0.3], 'k-', linewidth=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.1, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"  Saved: {args.output}")
    plt.close()


if __name__ == '__main__':
    main()

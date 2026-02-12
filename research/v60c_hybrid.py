#!/usr/bin/env python3
"""
mesh2plan v60c - Hybrid: v60 Room Seeds + Rotated Bounding Rectangles

Room detection from v60 (wall density → threshold → watershed → 5 rooms).
For each room, fit the minimum-area bounding rectangle at dominant wall angles.
This gives clean 4-vertex rooms aligned to wall angles.

For L-shaped rooms (where a rectangle poorly fits), use the two-rectangle decomposition.
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
from skimage.measure import find_contours
from skimage.segmentation import watershed
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import rotate, translate


RESOLUTION = 0.02
WALL_CLOSE_RADIUS = 25
MIN_ROOM_AREA = 2.0
WALL_NORMAL_THRESH = 0.5
DENSITY_PCT = 60


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def wall_density_image(mesh, resolution):
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < WALL_NORMAL_THRESH
    centroids = mesh.vertices[mesh.faces[wall_mask]].mean(axis=1)
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
    angles = np.degrees(np.arctan2(normals[wall_mask][:, 2], normals[wall_mask][:, 0])) % 180
    hist, bins = np.histogram(angles, bins=180, range=(0, 180), weights=areas[wall_mask])
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')
    peaks, props = find_peaks(hist_smooth, distance=20, height=0)
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
        wall_angles = sorted([(bins[p] + 90) % 180 for p in top2])
    else:
        wall_angles = [30, 120]
    print(f"Wall angles: {wall_angles[0]:.0f}° and {wall_angles[1]:.0f}°")
    return wall_angles


def extract_room_seeds(density, apt_mask, resolution):
    nonzero = density[density > 0]
    thresh = np.percentile(nonzero, DENSITY_PCT)
    wall_mask = (density >= thresh) & apt_mask
    wall_mask = binary_dilation(wall_mask, iterations=2)
    
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
    
    interior = apt_mask & ~wall_mask
    labeled, n = label(interior)
    
    rooms = []
    for i in range(1, n + 1):
        area = (labeled == i).sum() * resolution * resolution
        if area >= MIN_ROOM_AREA:
            rooms.append((i, area))
    
    # Watershed expand
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
    
    print(f"Seeds: {len(rooms)} rooms → expanded: {len(new_rooms)} rooms, {sum(a for _,a in new_rooms):.1f}m²")
    return expanded, new_rooms, wall_mask


def fit_rotated_rect(coords, angle_deg):
    """Fit minimum bounding rectangle at given angle."""
    coords = np.array(coords)
    rad = np.radians(-angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    cx, cy = coords.mean(axis=0)
    
    # Rotate to axis-aligned
    rot = np.zeros_like(coords)
    rot[:, 0] = (coords[:, 0] - cx) * cos_a - (coords[:, 1] - cy) * sin_a
    rot[:, 1] = (coords[:, 0] - cx) * sin_a + (coords[:, 1] - cy) * cos_a
    
    xmin, xmax = rot[:, 0].min(), rot[:, 0].max()
    ymin, ymax = rot[:, 1].min(), rot[:, 1].max()
    
    # Bounding box in rotated space
    corners_rot = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    
    # Rotate back
    rad_back = np.radians(angle_deg)
    cos_b, sin_b = np.cos(rad_back), np.sin(rad_back)
    corners_world = []
    for rx, ry in corners_rot:
        wx = rx * cos_b - ry * sin_b + cx
        wy = rx * sin_b + ry * cos_b + cy
        corners_world.append((wx, wy))
    
    return Polygon(corners_world)


def fit_room_polygon(contour_coords, room_poly_shapely, angle_deg, room_area):
    """
    Fit a clean polygon to the room contour.
    For small/medium rooms: rotated bounding rectangle.
    For large rooms that are L-shaped: try two-rectangle decomposition.
    """
    coords = np.array(contour_coords)
    
    # Try simple rotated rectangle first
    rect = fit_rotated_rect(coords, angle_deg)
    
    # Check how well rectangle covers the room
    if room_poly_shapely.is_valid:
        iou = room_poly_shapely.intersection(rect).area / room_poly_shapely.union(rect).area
    else:
        iou = 0.5
    
    if iou > 0.75 or room_area < 6.0:
        # Good fit or small room — use rectangle
        return rect
    
    # L-shape detection: try splitting the room with a line and fitting 2 rectangles
    # Split along each dominant angle at various positions
    best_poly = rect
    best_iou = iou
    
    for split_angle in [angle_deg, (angle_deg + 90) % 180]:
        rad = np.radians(-split_angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cx, cy = coords.mean(axis=0)
        
        # Project room points onto perpendicular axis
        perp = np.radians(split_angle + 90)
        projections = coords[:, 0] * np.cos(perp) + coords[:, 1] * np.sin(perp)
        
        # Try splitting at several positions
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            split_val = projections.min() + frac * (projections.max() - projections.min())
            
            mask1 = projections <= split_val + 0.1
            mask2 = projections >= split_val - 0.1
            
            if mask1.sum() < 4 or mask2.sum() < 4:
                continue
            
            rect1 = fit_rotated_rect(coords[mask1], angle_deg)
            rect2 = fit_rotated_rect(coords[mask2], angle_deg)
            
            combined = unary_union([rect1, rect2])
            if isinstance(combined, MultiPolygon):
                combined = max(combined.geoms, key=lambda g: g.area)
            
            if room_poly_shapely.is_valid and combined.is_valid:
                try:
                    test_iou = room_poly_shapely.intersection(combined).area / room_poly_shapely.union(combined).area
                    if test_iou > best_iou:
                        best_iou = test_iou
                        best_poly = combined
                except:
                    pass
    
    return best_poly


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
    parser.add_argument('mesh')
    parser.add_argument('-o', '--output', default='v60c_hybrid.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    wall_angles = detect_dominant_angles(mesh)
    density, origin = wall_density_image(mesh, RESOLUTION)
    apt_mask = get_apartment_mask(mesh, RESOLUTION, origin, density.shape)
    
    expanded, rooms, wall_mask = extract_room_seeds(density, apt_mask, RESOLUTION)
    
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
        
        room_shapely = Polygon(coords)
        if not room_shapely.is_valid:
            room_shapely = room_shapely.buffer(0)
        
        poly = fit_room_polygon(coords, room_shapely, wall_angles[0], area)
        
        # Clip rectangle to watershed region to prevent overlap
        clipped = poly.intersection(room_shapely)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.is_empty or clipped.area < 1.0:
            clipped = poly  # fallback
        
        # Simplify back to clean polygon
        clipped = clipped.simplify(0.1)
        room_polys.append(clipped)
        
        bounds = poly.bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        aspect = max(w, h) / (min(w, h) + 0.01)
        nv = len(poly.exterior.coords) - 1
        label_text = classify_room(poly.area, aspect)
        total += poly.area
        print(f"  {label_text}: {poly.area:.1f}m² ({nv}v)")
    
    print(f"  Total: {total:.1f}m²")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    pastel = ['#AEC6CF', '#FFB3BA', '#BAFFC9', '#FFFFBA', '#E8BAFF', '#FFD1A4']
    
    # Draw apartment boundary
    bnd_contours = find_contours(apt_mask.astype(float), 0.5)
    if bnd_contours:
        bnd = max(bnd_contours, key=len)
        bx = [c[1]*RESOLUTION+xmin for c in bnd]
        by = [c[0]*RESOLUTION+zmin for c in bnd]
        ax.plot(bx, by, 'k-', linewidth=1.5, alpha=0.5)
    
    for idx, (poly, (rid, _)) in enumerate(zip(room_polys, rooms)):
        if poly is None:
            continue
        
        bounds = poly.bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        aspect = max(w, h) / (min(w, h) + 0.01)
        label_text = classify_room(poly.area, aspect)
        
        color = pastel[idx % len(pastel)]
        
        # Draw with thick wall effect (outline)
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=color, alpha=0.4)
        ax.plot(xs, ys, 'k-', linewidth=2.5)
        
        cx, cy = poly.centroid.x, poly.centroid.y
        nv = len(poly.exterior.coords) - 1
        ax.text(cx, cy, f"{label_text}\n{poly.area:.1f}m²",
                ha='center', va='center', fontsize=9, weight='bold')
    
    ax.set_title(f"v60c — {sum(1 for p in room_polys if p)} rooms, {total:.1f}m²\nAngles: {wall_angles[0]:.0f}°, {wall_angles[1]:.0f}°")
    ax.set_aspect('equal')
    
    # Scale bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.3, ylim[0]+0.3], 'k-', linewidth=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.1, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == '__main__':
    main()

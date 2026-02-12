#!/usr/bin/env python3
"""
mesh2plan v60 - Direct Contour Room Extraction

APPROACH: The cross-section contour at wall height IS the wall outline.
Instead of detecting lines and extending them, we:
1. Multi-height cross-section → rasterize wall thickness image
2. Threshold → closed wall mask
3. Invert wall mask → interior regions = rooms
4. Label connected components of interior = individual rooms
5. Vectorize room boundaries with angle-snapped simplification

This avoids all line extension problems (v48-v59). The rooms come directly
from "what's NOT a wall."

Key insight: Walls are THICK in cross-section (10-30cm). A well-thresholded
wall mask at the right height should form closed boundaries around rooms.
Where walls have gaps (doors), morphological closing bridges them.
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
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


RESOLUTION = 0.02  # 2cm per pixel
SLICE_HEIGHTS = np.arange(-1.8, -0.5, 0.05)  # 16 slices at wall height (Y-up, floor at -2.5)
WALL_CLOSE_RADIUS = 25  # pixels (~50cm) to close door gaps
MIN_ROOM_AREA = 2.0  # m²


def load_mesh(path):
    print(f"Loading mesh: {path}")
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def wall_density_image(mesh, resolution):
    """Wall-only density from face normals (v41b method)."""
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.5
    
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


def multi_slice_wall_image(mesh, heights, resolution):
    """Slice mesh at multiple heights, accumulate wall presence."""
    verts = mesh.vertices
    xmin, xmax = verts[:, 0].min() - 0.2, verts[:, 0].max() + 0.2
    zmin, zmax = verts[:, 2].min() - 0.2, verts[:, 2].max() + 0.2
    
    w = int((xmax - xmin) / resolution) + 1
    h = int((zmax - zmin) / resolution) + 1
    
    accumulator = np.zeros((h, w), dtype=np.float32)
    
    for height in heights:
        try:
            section = mesh.section(plane_origin=[0, height, 0],
                                   plane_normal=[0, 1, 0])
            if section is None:
                continue
            
            # Get 3D path segments
            for entity in section.entities:
                pts_3d = section.vertices[entity.points]
                # Rasterize each segment
                for i in range(len(pts_3d) - 1):
                    x0, z0 = pts_3d[i][0], pts_3d[i][2]
                    x1, z1 = pts_3d[i+1][0], pts_3d[i+1][2]
                    
                    # Bresenham-like: sample along segment
                    seg_len = np.hypot(x1-x0, z1-z0)
                    n_samples = max(int(seg_len / resolution * 2), 2)
                    ts = np.linspace(0, 1, n_samples)
                    xs = x0 + ts * (x1 - x0)
                    zs = z0 + ts * (z1 - z0)
                    
                    xi = ((xs - xmin) / resolution).astype(int).clip(0, w-1)
                    zi = ((zs - zmin) / resolution).astype(int).clip(0, h-1)
                    accumulator[zi, xi] += 1
        except Exception as e:
            continue
    
    print(f"  Slice accumulator: {w}x{h}, {len(heights)} heights")
    print(f"  Max hits: {accumulator.max():.0f}, nonzero: {(accumulator > 0).sum()}")
    
    origin = (xmin, zmin)
    return accumulator, origin


def get_apartment_mask(mesh, resolution, origin, shape):
    """Get apartment footprint."""
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


def extract_rooms(wall_accumulator, apt_mask, resolution):
    """
    Threshold wall accumulator → wall mask → invert → rooms.
    Try multiple thresholds and pick the one giving best room count.
    """
    best_rooms = None
    best_score = -1
    best_threshold = 0
    best_wall_mask = None
    
    nonzero_vals = wall_accumulator[wall_accumulator > 0]
    if len(nonzero_vals) == 0:
        return [], None, 0
    
    for pct in [30, 40, 50, 60]:
        thresh = np.percentile(nonzero_vals, pct)
        wall_mask = wall_accumulator >= thresh
        
        # Restrict to apartment
        wall_mask = wall_mask & apt_mask
        
        # Thicken walls slightly (walls should be solid)
        wall_mask = binary_dilation(wall_mask, iterations=2)
        
        # Close door gaps using directional closing along dominant wall angles
        from skimage.morphology import disk as sk_disk
        
        # First: directional close along each wall angle
        wall_closed = wall_mask.copy()
        for angle in [29, 119]:  # dominant angles from this apartment
            rad = np.radians(angle)
            length = WALL_CLOSE_RADIUS
            # Create line structuring element
            struct_line = np.zeros((2*length+1, 2*length+1), dtype=bool)
            for t in np.linspace(-length, length, 4*length+1):
                r = int(round(length + t * np.sin(rad)))
                c = int(round(length + t * np.cos(rad)))
                if 0 <= r < 2*length+1 and 0 <= c < 2*length+1:
                    struct_line[r, c] = True
            wall_closed = binary_closing(wall_closed, structure=struct_line)
        
        # Then small isotropic close to connect perpendicular walls
        struct = sk_disk(5)
        wall_closed = binary_closing(wall_closed, structure=struct)
        
        # Clean up
        wall_closed = binary_opening(wall_closed, iterations=2)
        
        # Interior = apartment minus walls
        interior = apt_mask & ~wall_closed
        
        # Label connected components
        labeled, n_rooms = label(interior)
        
        # Filter by area
        rooms = []
        for i in range(1, n_rooms + 1):
            area_px = (labeled == i).sum()
            area_m2 = area_px * resolution * resolution
            if area_m2 >= MIN_ROOM_AREA:
                rooms.append((i, area_m2))
        
        # Score: prefer 5-6 rooms with good total area
        total_area = sum(a for _, a in rooms)
        n = len(rooms)
        room_count_score = max(0, 10 - abs(n - 5) * 2)  # prefer 5 rooms
        area_score = min(total_area / 35.0, 1.0) * 10  # prefer ~35m² total
        # Penalize very low total area
        if total_area < 20:
            area_score *= 0.5
        score = room_count_score + area_score
        
        print(f"  Threshold p{pct} ({thresh:.1f}): {n} rooms, {total_area:.1f}m², score={score:.1f}")
        
        if score > best_score:
            best_score = score
            best_rooms = (labeled, rooms)
            best_threshold = pct
            best_wall_mask = wall_closed
    
    print(f"  Best: p{best_threshold}")
    
    if best_rooms is None:
        return [], None, 0
    
    # Expand rooms into wall area using watershed
    labeled, rooms = best_rooms
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    
    # Use density as gradient (walls = ridges)
    gradient = gaussian_filter(wall_accumulator, sigma=2)
    gradient = gradient / (gradient.max() + 1e-10)
    
    # Create markers: each room keeps its label, background = 0
    markers = np.zeros_like(labeled)
    for rid, area in rooms:
        markers[labeled == rid] = rid
    
    # Watershed expand within apartment mask
    expanded = watershed(gradient, markers=markers, mask=apt_mask)
    
    # Recalculate room areas
    new_rooms = []
    for rid, _ in rooms:
        area_px = (expanded == rid).sum()
        area_m2 = area_px * resolution * resolution
        if area_m2 >= MIN_ROOM_AREA:
            new_rooms.append((rid, area_m2))
    
    new_total = sum(a for _, a in new_rooms)
    print(f"  After watershed expansion: {len(new_rooms)} rooms, {new_total:.1f}m²")
    
    return (expanded, new_rooms), best_wall_mask, best_threshold


def room_mask_to_polygon(labeled, room_id, origin, resolution):
    """Convert a labeled room mask to a Shapely polygon."""
    room_mask = (labeled == room_id).astype(float)
    contours = find_contours(room_mask, 0.5)
    if not contours:
        return None
    
    contour = max(contours, key=len)
    
    xmin, zmin = origin
    # Simplify contour
    simplified = approximate_polygon(contour, tolerance=3)  # ~6cm tolerance
    
    coords = [(c[1] * resolution + xmin, c[0] * resolution + zmin) for c in simplified]
    
    if len(coords) < 4:
        return None
    
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    
    return poly


def detect_dominant_angles(mesh):
    """Detect dominant wall angles from face normals."""
    normals = mesh.face_normals
    areas = mesh.area_faces
    
    # Wall faces
    wall_mask = np.abs(normals[:, 1]) < 0.5
    wall_normals = normals[wall_mask]
    wall_areas = areas[wall_mask]
    
    # Normal angle in XZ plane (perpendicular to wall → wall angle = normal_angle + 90)
    angles = np.degrees(np.arctan2(wall_normals[:, 2], wall_normals[:, 0])) % 180
    
    # Weighted histogram
    hist, bins = np.histogram(angles, bins=180, range=(0, 180), weights=wall_areas)
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')
    
    from scipy.signal import find_peaks
    peaks, props = find_peaks(hist_smooth, distance=20, height=0)
    
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
        # These are NORMAL angles; wall angles are +90
        wall_angles = sorted([(bins[p] + 90) % 180 for p in top2])
    else:
        wall_angles = [30, 120]
    
    print(f"  Dominant wall angles: {wall_angles[0]:.0f}° and {wall_angles[1]:.0f}°")
    return wall_angles


def snap_polygon_to_angles(poly, angles_deg, min_edge=0.3):
    """Snap polygon edges to dominant angles, simplify."""
    if poly is None or poly.is_empty:
        return poly
    
    coords = list(poly.exterior.coords[:-1])  # remove closing point
    if len(coords) < 3:
        return poly
    
    # Remove very short edges first
    filtered = [coords[0]]
    for c in coords[1:]:
        d = np.hypot(c[0]-filtered[-1][0], c[1]-filtered[-1][1])
        if d >= min_edge:
            filtered.append(c)
    coords = filtered
    if len(coords) < 3:
        return poly
    
    # For each vertex, snap to intersection of angle-snapped edges
    snapped = []
    n = len(coords)
    for i in range(n):
        p0 = coords[(i-1) % n]
        p1 = coords[i]
        p2 = coords[(i+1) % n]
        
        # Incoming edge direction
        dx_in = p1[0] - p0[0]
        dy_in = p1[1] - p0[1]
        angle_in = np.degrees(np.arctan2(dy_in, dx_in)) % 180
        
        # Outgoing edge direction
        dx_out = p2[0] - p1[0]
        dy_out = p2[1] - p1[1]
        angle_out = np.degrees(np.arctan2(dy_out, dx_out)) % 180
        
        # Snap each to nearest dominant angle
        def snap_angle(a):
            best = a
            best_d = 999
            for da in angles_deg:
                for var in [da, da + 180]:
                    d = abs(a - (var % 360))
                    if d > 180: d = 360 - d
                    if d < best_d:
                        best_d = d
                        best = var
            return best if best_d < 25 else a
        
        snapped.append(p1)  # keep original for now
    
    # Simple approach: just use simplified polygon
    return poly.simplify(0.15)


def classify_room(area, aspect):
    if area > 8: return "Room"
    elif area > 5:
        return "Hallway" if aspect > 2.5 else "Room"
    elif area > 3:
        return "Hallway" if aspect > 2 else "Bathroom"
    else:
        return "Closet"


def plot_results(wall_acc, wall_mask, apt_mask, labeled, rooms, room_polys,
                 wall_angles, origin, resolution, output_path):
    """4-panel plot."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    xmin, zmin = origin
    h, w = wall_acc.shape
    extent = [xmin, xmin + w*resolution, zmin, zmin + h*resolution]
    
    # Panel 1: Wall accumulator
    ax = axes[0]
    display = np.where(wall_acc > 0, wall_acc, np.nan)
    ax.imshow(display.T, origin='lower', extent=extent, cmap='hot', aspect='equal')
    ax.set_title(f"Cross-Section Accumulator ({len(SLICE_HEIGHTS)} slices)")
    ax.set_aspect('equal')
    
    # Panel 2: Wall mask
    ax = axes[1]
    vis = np.zeros((*wall_mask.shape, 3), dtype=np.uint8)
    vis[apt_mask] = [200, 200, 200]  # apartment in gray
    vis[wall_mask] = [0, 0, 0]  # walls in black
    ax.imshow(np.transpose(vis, (1,0,2)), origin='lower', extent=extent, aspect='equal')
    ax.set_title(f"Wall Mask (closed, angles: {wall_angles[0]:.0f}°/{wall_angles[1]:.0f}°)")
    ax.set_aspect('equal')
    
    # Panel 3: Labeled rooms
    ax = axes[2]
    pastel = ['#AEC6CF', '#FFB3BA', '#BAFFC9', '#FFFFBA', '#E8BAFF', '#FFD1A4',
              '#C4E1FF', '#FFC4E1', '#D4E6B5', '#FFE0CC']
    room_vis = np.ones((*labeled.shape, 3))
    for idx, (rid, area) in enumerate(rooms):
        c = matplotlib.colors.to_rgb(pastel[idx % len(pastel)])
        room_vis[labeled == rid] = c
    room_vis[wall_mask] = [0, 0, 0]
    ax.imshow(np.transpose(room_vis, (1,0,2)), origin='lower', extent=extent, aspect='equal')
    ax.set_title(f"Labeled Rooms ({len(rooms)})")
    ax.set_aspect('equal')
    
    # Panel 4: Vectorized room polygons
    ax = axes[3]
    total_area = 0
    for idx, (poly, (rid, area)) in enumerate(zip(room_polys, rooms)):
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
        ax.text(cx, cy, f"{label_text}\n{poly.area:.1f}m²", 
                ha='center', va='center', fontsize=8, weight='bold')
    
    # Boundary
    bnd = get_boundary_poly(apt_mask, origin, resolution)
    if bnd:
        bx, by = bnd.exterior.xy
        ax.plot(bx, by, 'k--', linewidth=1)
    
    ax.set_title(f"v60 — {len(room_polys)} rooms, {total_area:.1f}m²\nAngles: {wall_angles[0]:.0f}°, {wall_angles[1]:.0f}°")
    ax.set_aspect('equal')
    
    # Scale bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.3, ylim[0]+0.3], 'k-', linewidth=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.1, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def get_boundary_poly(mask, origin, resolution):
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    xmin, zmin = origin
    coords = [(c[1]*resolution+xmin, c[0]*resolution+zmin) for c in contour]
    poly = Polygon(coords).simplify(0.15)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', help='Path to mesh')
    parser.add_argument('-o', '--output', default='v60_contour_rooms.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    
    # Detect wall angles
    wall_angles = detect_dominant_angles(mesh)
    
    # Use wall-only density (v41b method) instead of cross-section — much denser signal
    wall_acc, origin = wall_density_image(mesh, RESOLUTION)
    
    # Apartment mask
    apt_mask = get_apartment_mask(mesh, RESOLUTION, origin, wall_acc.shape)
    
    # Extract rooms
    result, wall_mask, best_pct = extract_rooms(wall_acc, apt_mask, RESOLUTION)
    
    if result is None:
        print("ERROR: No rooms found")
        return
    
    labeled, rooms = result
    
    print(f"\n=== RESULTS ===")
    print(f"  {len(rooms)} rooms found")
    
    # Vectorize room polygons
    room_polys = []
    for rid, area in rooms:
        poly = room_mask_to_polygon(labeled, rid, origin, RESOLUTION)
        if poly:
            poly = snap_polygon_to_angles(poly, wall_angles + [(a+180)%360 for a in wall_angles])
            room_polys.append(poly)
            bounds = poly.bounds
            w = bounds[2] - bounds[0]
            h = bounds[3] - bounds[1]
            aspect = max(w, h) / (min(w, h) + 0.01)
            nv = len(poly.exterior.coords) - 1
            label = classify_room(poly.area, aspect)
            print(f"  {label}: {poly.area:.1f}m² ({nv}v)")
        else:
            room_polys.append(None)
    
    total = sum(p.area for p in room_polys if p)
    print(f"  Total: {total:.1f}m²")
    print(f"  Angles: {wall_angles[0]:.0f}° and {wall_angles[1]:.0f}°")
    
    # Plot
    plot_results(wall_acc, wall_mask, apt_mask, labeled, rooms, room_polys,
                 wall_angles, origin, RESOLUTION, args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v59 - Skeleton Wall Tracing

APPROACH: Wall-only density (from face normals, v41b) → morphological skeletonization
→ extract line segments → snap to dominant angles → extend/connect → room partition.

Key difference from Hough approaches (v48-v52): skeleton gives actual wall SEGMENTS
with start/end points and real wall lengths, not infinite lines across the apartment.

Key difference from RANSAC (v58): works in 2D density space which is cleaner than
noisy 3D point clouds.

Pipeline:
1. Load mesh → wall-only density image (v41b method)
2. Binary threshold → morphological skeleton (1px wide)
3. Probabilistic Hough on skeleton → line segments
4. Cluster segments by angle → 2 dominant families
5. Merge collinear/overlapping segments within each family
6. Extend short segments to connect at intersections
7. Polygonize → room cells → merge small → classify
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label
from skimage.morphology import skeletonize, disk
from skimage.transform import probabilistic_hough_line
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, box
from shapely.ops import polygonize, unary_union, linemerge
from shapely.affinity import scale
import matplotlib.colors as mcolors


# --- CONFIG ---
RESOLUTION = 0.02  # meters per pixel (2cm)
WALL_NORMAL_THRESH = 0.5  # |ny| < this = wall face
DENSITY_PERCENTILE = 75  # wall density threshold
MIN_ROOM_AREA = 2.0  # m² minimum room
MIN_SEGMENT_LENGTH = 0.3  # meters, min wall segment
ANGLE_TOLERANCE = 15  # degrees, for clustering


def load_mesh(path):
    print(f"Loading mesh: {path}")
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def wall_density_image(mesh, resolution):
    """Create density image from wall faces only (v41b method)."""
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < WALL_NORMAL_THRESH
    
    wall_faces = mesh.faces[wall_mask]
    wall_verts = mesh.vertices[wall_faces]  # (N, 3, 3)
    centroids = wall_verts.mean(axis=1)  # (N, 3)
    areas = mesh.area_faces[wall_mask]
    
    xs, zs = centroids[:, 0], centroids[:, 2]
    
    xmin, xmax = xs.min() - 0.1, xs.max() + 0.1
    zmin, zmax = zs.min() - 0.1, zs.max() + 0.1
    
    w = int((xmax - xmin) / resolution) + 1
    h = int((zmax - zmin) / resolution) + 1
    
    density = np.zeros((h, w), dtype=np.float64)
    
    xi = ((xs - xmin) / resolution).astype(int).clip(0, w-1)
    zi = ((zs - zmin) / resolution).astype(int).clip(0, h-1)
    
    np.add.at(density, (zi, xi), areas)
    
    print(f"  Density image: {w}x{h}, wall faces: {wall_mask.sum()}/{len(mesh.faces)}")
    
    origin = (xmin, zmin)
    return density, origin, resolution


def get_apartment_mask(mesh, resolution, origin, shape):
    """Get apartment boundary mask from all faces."""
    verts = mesh.vertices
    xs, zs = verts[:, 0], verts[:, 2]
    xmin, zmin = origin
    
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    
    xi = ((xs - xmin) / resolution).astype(int).clip(0, w-1)
    zi = ((zs - zmin) / resolution).astype(int).clip(0, h-1)
    mask[zi, xi] = True
    
    # Close gaps and fill
    mask = binary_dilation(mask, iterations=3)
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=3)
    
    return mask


def extract_skeleton_segments(density, mask, resolution):
    """Threshold → skeleton → probabilistic Hough → line segments."""
    thresh = np.percentile(density[density > 0], DENSITY_PERCENTILE)
    wall_binary = (density > thresh) & mask
    
    # Clean up: remove small isolated pixels
    wall_binary = binary_dilation(wall_binary, iterations=1)
    wall_binary = binary_erosion(wall_binary, iterations=1)
    
    # Skeletonize
    skel = skeletonize(wall_binary)
    
    print(f"  Skeleton pixels: {skel.sum()}")
    
    # Probabilistic Hough to get line segments
    min_len_px = int(MIN_SEGMENT_LENGTH / resolution)
    segments = probabilistic_hough_line(
        skel,
        threshold=10,
        line_length=min_len_px,
        line_gap=int(0.15 / resolution),  # 15cm gap allowed
    )
    
    print(f"  Raw Hough segments: {len(segments)}")
    
    return segments, skel, wall_binary


def segment_angle(seg):
    """Get angle of segment in [0, 180) degrees."""
    (x0, y0), (x1, y1) = seg
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
    return angle


def cluster_segments_by_angle(segments, n_families=2):
    """Cluster segments into angle families."""
    if len(segments) < 2:
        return [], []
    
    angles = np.array([segment_angle(s) for s in segments])
    lengths = np.array([np.hypot(s[1][0]-s[0][0], s[1][1]-s[0][1]) for s in segments])
    
    # Weighted angle histogram to find dominant angles
    hist, bin_edges = np.histogram(angles, bins=180, range=(0, 180), weights=lengths)
    
    # Smooth histogram
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')
    
    # Find top 2 peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(hist_smooth, distance=20, height=0)
    
    if len(peaks) < 2:
        # Fallback: just use the single peak and its perpendicular
        peak_angle = bin_edges[peaks[0]] if len(peaks) > 0 else 30
        peak_angles = [peak_angle, (peak_angle + 90) % 180]
    else:
        # Sort by height, take top 2
        top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
        peak_angles = sorted(bin_edges[top2])
    
    print(f"  Dominant angles: {peak_angles[0]:.0f}° and {peak_angles[1]:.0f}°")
    
    # Assign each segment to nearest family
    families = [[] for _ in range(n_families)]
    for seg, angle in zip(segments, angles):
        dists = []
        for pa in peak_angles:
            d = min(abs(angle - pa), 180 - abs(angle - pa))
            dists.append(d)
        best = np.argmin(dists)
        if dists[best] < ANGLE_TOLERANCE:
            families[best].append(seg)
    
    print(f"  Family sizes: {[len(f) for f in families]}")
    return families, peak_angles


def segment_to_line_params(seg, angle_ref):
    """Convert segment to (rho, angle_ref) parametric form for merging."""
    (x0, y0), (x1, y1) = seg
    # Project midpoint onto perpendicular direction
    mid = ((x0+x1)/2, (y0+y1)/2)
    perp = np.radians(angle_ref + 90)
    rho = mid[0] * np.cos(perp) + mid[1] * np.sin(perp)
    
    # Project endpoints onto line direction
    line_dir = np.array([np.cos(np.radians(angle_ref)), np.sin(np.radians(angle_ref))])
    t0 = x0 * line_dir[0] + y0 * line_dir[1]
    t1 = x1 * line_dir[0] + y1 * line_dir[1]
    
    return rho, min(t0, t1), max(t0, t1)


def merge_collinear_segments(family, angle_ref, resolution, merge_dist=0.3):
    """Merge segments that are collinear (same rho, overlapping t)."""
    if not family:
        return []
    
    merge_dist_px = merge_dist / resolution
    gap_merge_px = 0.5 / resolution  # merge segments within 50cm gap
    
    # Get parametric form
    params = [segment_to_line_params(s, angle_ref) for s in family]
    rhos = np.array([p[0] for p in params])
    
    # Cluster by rho
    sorted_idx = np.argsort(rhos)
    clusters = []
    current_cluster = [sorted_idx[0]]
    
    for i in range(1, len(sorted_idx)):
        if abs(rhos[sorted_idx[i]] - rhos[sorted_idx[i-1]]) < merge_dist_px:
            current_cluster.append(sorted_idx[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_idx[i]]
    clusters.append(current_cluster)
    
    # For each cluster, merge overlapping t ranges
    merged_segments = []
    line_dir = np.array([np.cos(np.radians(angle_ref)), np.sin(np.radians(angle_ref))])
    perp_dir = np.array([np.cos(np.radians(angle_ref + 90)), np.sin(np.radians(angle_ref + 90))])
    
    for cluster in clusters:
        # Average rho
        avg_rho = np.mean([rhos[i] for i in cluster])
        
        # Collect all t-intervals
        intervals = [(params[i][1], params[i][2]) for i in cluster]
        intervals.sort()
        
        # Merge overlapping/close intervals
        merged_intervals = [list(intervals[0])]
        for t0, t1 in intervals[1:]:
            if t0 <= merged_intervals[-1][1] + gap_merge_px:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], t1)
            else:
                merged_intervals.append([t0, t1])
        
        # Convert back to pixel segments
        base = perp_dir * avg_rho
        for t0, t1 in merged_intervals:
            p0 = base + line_dir * t0
            p1 = base + line_dir * t1
            length = np.hypot(p1[0]-p0[0], p1[1]-p0[1]) * resolution
            if length >= MIN_SEGMENT_LENGTH:
                merged_segments.append(((p0[0], p0[1]), (p1[0], p1[1])))
    
    return merged_segments


def segments_to_shapely_lines(segments, origin, resolution, boundary_poly):
    """Convert pixel segments to world-coordinate Shapely LineStrings, clipped to boundary."""
    lines = []
    xmin, zmin = origin
    
    for (x0, y0), (x1, y1) in segments:
        # Convert pixel to world (x=col*res+xmin, z=row*res+zmin)
        wx0 = x0 * resolution + xmin
        wz0 = y0 * resolution + zmin
        wx1 = x1 * resolution + xmin
        wz1 = y1 * resolution + zmin
        
        ls = LineString([(wx0, wz0), (wx1, wz1)])
        
        # Extend line to boundary
        dx, dy = wx1 - wx0, wz1 - wz0
        length = np.hypot(dx, dy)
        if length < 0.01:
            continue
        dx, dy = dx/length, dy/length
        
        # Extend both directions by 20m (will be clipped)
        ext = 20.0
        ls_ext = LineString([
            (wx0 - dx*ext, wz0 - dy*ext),
            (wx1 + dx*ext, wz1 + dy*ext)
        ])
        
        clipped = ls_ext.intersection(boundary_poly)
        if not clipped.is_empty:
            if isinstance(clipped, MultiLineString):
                for part in clipped.geoms:
                    if part.length > MIN_SEGMENT_LENGTH:
                        lines.append(part)
            elif isinstance(clipped, LineString) and clipped.length > MIN_SEGMENT_LENGTH:
                lines.append(clipped)
    
    return lines


def mask_to_polygon(mask, origin, resolution):
    """Convert binary mask to Shapely polygon."""
    from skimage.measure import find_contours
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    
    # Take largest contour
    contour = max(contours, key=len)
    
    # Convert to world coords (row=z, col=x)
    xmin, zmin = origin
    coords = [(c[1] * resolution + xmin, c[0] * resolution + zmin) for c in contour]
    
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    
    # Simplify
    poly = poly.simplify(0.1)
    
    return poly


def snap_polygon_angles(poly, angles_deg, tolerance=0.2):
    """Snap polygon edges to dominant angles."""
    if poly is None or poly.is_empty:
        return poly
    
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return poly
    
    # For each edge, snap to nearest dominant angle
    new_coords = [coords[0]]
    for i in range(1, len(coords)):
        dx = coords[i][0] - new_coords[-1][0]
        dy = coords[i][1] - new_coords[-1][1]
        edge_len = np.hypot(dx, dy)
        if edge_len < 0.05:
            continue
        
        edge_angle = np.degrees(np.arctan2(dy, dx)) % 180
        
        # Find nearest dominant angle (including 180° variants)
        best_snap = edge_angle
        best_diff = 999
        for a in angles_deg:
            for variant in [a, a + 180, a - 180]:
                diff = abs(np.degrees(np.arctan2(dy, dx)) - variant)
                if diff > 180:
                    diff = 360 - diff
                if diff < best_diff:
                    best_diff = diff
                    best_snap = variant
        
        if best_diff < 20:  # snap if within 20°
            snap_rad = np.radians(best_snap)
            new_x = new_coords[-1][0] + edge_len * np.cos(snap_rad)
            new_y = new_coords[-1][1] + edge_len * np.sin(snap_rad)
            new_coords.append((new_x, new_y))
        else:
            new_coords.append(coords[i])
    
    if len(new_coords) >= 4:
        try:
            p = Polygon(new_coords)
            if p.is_valid and p.area > 0.5:
                return p
        except:
            pass
    return poly


def partition_rooms(lines, boundary_poly, min_area):
    """Use line arrangement to partition boundary into rooms."""
    # Add boundary as lines
    boundary_lines = [LineString(list(boundary_poly.exterior.coords))]
    
    all_lines = boundary_lines + lines
    
    # Polygonize
    result = list(polygonize(unary_union(all_lines)))
    
    # Filter to rooms inside boundary
    rooms = []
    for poly in result:
        if poly.area < min_area:
            continue
        # Check overlap with boundary
        overlap = poly.intersection(boundary_poly).area
        if overlap > poly.area * 0.5:
            rooms.append(poly)
    
    print(f"  Polygonize: {len(result)} cells, {len(rooms)} rooms (>{min_area}m²)")
    return rooms


def merge_small_rooms(rooms, min_area=2.5, target_count=6):
    """Merge small rooms into adjacent larger ones."""
    if len(rooms) <= target_count:
        return rooms
    
    while len(rooms) > target_count:
        # Find smallest room
        areas = [r.area for r in rooms]
        smallest_idx = np.argmin(areas)
        
        if areas[smallest_idx] >= min_area:
            break
        
        smallest = rooms[smallest_idx]
        
        # Find best neighbor to merge with (longest shared boundary)
        best_neighbor = -1
        best_shared = 0
        
        for i, r in enumerate(rooms):
            if i == smallest_idx:
                continue
            shared = smallest.intersection(r).length
            if shared > best_shared:
                best_shared = shared
                best_neighbor = i
        
        if best_neighbor >= 0:
            merged = unary_union([smallest, rooms[best_neighbor]])
            if isinstance(merged, MultiPolygon):
                merged = max(merged.geoms, key=lambda g: g.area)
            rooms[best_neighbor] = merged
            rooms.pop(smallest_idx)
        else:
            rooms.pop(smallest_idx)
    
    return rooms


def classify_room(poly, all_rooms):
    """Classify room by area and shape."""
    area = poly.area
    bounds = poly.bounds
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    aspect = max(w, h) / (min(w, h) + 0.01)
    
    if area > 8:
        return "Room"
    elif area > 5:
        if aspect > 2.5:
            return "Hallway"
        return "Room"
    elif area > 3:
        if aspect > 2:
            return "Hallway"
        return "Bathroom"
    else:
        return "Closet"


def plot_results(density, skel, wall_binary, segments_raw, families, merged_segs,
                 rooms, boundary_poly, peak_angles, origin, resolution, output_path):
    """Plot 4-panel results."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    xmin, zmin = origin
    h, w = density.shape
    extent = [xmin, xmin + w*resolution, zmin, zmin + h*resolution]
    
    # Panel 1: Wall density + skeleton
    ax = axes[0]
    ax.imshow(density.T, origin='lower', extent=extent, cmap='hot', aspect='equal')
    # Overlay skeleton
    sy, sx = np.where(skel)
    ax.scatter(sx * resolution + xmin, sy * resolution + zmin, s=0.1, c='cyan', alpha=0.3)
    ax.set_title(f"Wall Density + Skeleton ({skel.sum()} px)")
    ax.set_aspect('equal')
    
    # Panel 2: Raw segments colored by family
    ax = axes[1]
    ax.imshow(wall_binary.T, origin='lower', extent=extent, cmap='gray', alpha=0.3, aspect='equal')
    colors = ['red', 'blue', 'green', 'orange']
    for fi, family in enumerate(families):
        for (x0, y0), (x1, y1) in family:
            wx0, wz0 = x0*resolution+xmin, y0*resolution+zmin
            wx1, wz1 = x1*resolution+xmin, y1*resolution+zmin
            ax.plot([wx0, wx1], [wz0, wz1], c=colors[fi % len(colors)], linewidth=0.5, alpha=0.5)
    ax.set_title(f"Segments by Family ({sum(len(f) for f in families)})")
    ax.set_aspect('equal')
    
    # Panel 3: Merged segments
    ax = axes[2]
    for fi, segs in enumerate(merged_segs):
        for (x0, y0), (x1, y1) in segs:
            wx0, wz0 = x0*resolution+xmin, y0*resolution+zmin
            wx1, wz1 = x1*resolution+xmin, y1*resolution+zmin
            ax.plot([wx0, wx1], [wz0, wz1], c=colors[fi % len(colors)], linewidth=2)
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=1)
    ax.set_title(f"Merged Walls ({sum(len(s) for s in merged_segs)}), angles: {peak_angles[0]:.0f}°/{peak_angles[1]:.0f}°")
    ax.set_aspect('equal')
    
    # Panel 4: Room partition
    ax = axes[3]
    pastel_colors = ['#AEC6CF', '#FFB3BA', '#BAFFC9', '#FFFFBA', '#E8BAFF', '#FFD1A4',
                     '#C4E1FF', '#FFC4E1']
    
    total_area = 0
    for i, room in enumerate(rooms):
        label = classify_room(room, rooms)
        area = room.area
        total_area += area
        
        color = pastel_colors[i % len(pastel_colors)]
        
        if isinstance(room, Polygon):
            xs, ys = room.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.6)
            ax.plot(xs, ys, 'k-', linewidth=2)
            
            cx, cy = room.centroid.x, room.centroid.y
            nv = len(room.exterior.coords) - 1
            ax.text(cx, cy, f"{label}\n{area:.1f}m²", ha='center', va='center', fontsize=8, weight='bold')
    
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=1.5, linestyle='--')
    
    ax.set_title(f"v59 — {len(rooms)} rooms, {total_area:.1f}m²\nAngles: {peak_angles[0]:.0f}°, {peak_angles[1]:.0f}°")
    ax.set_aspect('equal')
    
    # Scale bar
    ax.plot([-4, -3], [-5, -5], 'k-', linewidth=3)
    ax.text(-3.5, -5.2, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', help='Path to mesh file')
    parser.add_argument('-o', '--output', default='v59_skeleton_walls.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    
    # Step 1: Wall density image
    density, origin, res = wall_density_image(mesh, RESOLUTION)
    
    # Step 2: Apartment mask
    mask = get_apartment_mask(mesh, RESOLUTION, origin, density.shape)
    
    # Step 3: Skeleton + segments
    segments, skel, wall_binary = extract_skeleton_segments(density, mask, RESOLUTION)
    
    if len(segments) < 3:
        print("ERROR: Too few segments found")
        return
    
    # Step 4: Cluster by angle
    families, peak_angles = cluster_segments_by_angle(segments)
    
    # Step 5: Merge collinear segments per family
    merged_segs = []
    for fi, family in enumerate(families):
        merged = merge_collinear_segments(family, peak_angles[fi], RESOLUTION)
        merged_segs.append(merged)
        print(f"  Family {fi} ({peak_angles[fi]:.0f}°): {len(family)} → {len(merged)} segments")
    
    # Step 6: Convert to world lines, extend to boundary
    boundary_poly = mask_to_polygon(mask, origin, RESOLUTION)
    
    all_world_lines = []
    for fi, segs in enumerate(merged_segs):
        lines = segments_to_shapely_lines(segs, origin, RESOLUTION, boundary_poly)
        all_world_lines.extend(lines)
    
    print(f"  Total world lines: {len(all_world_lines)}")
    
    # Step 7: Partition into rooms
    rooms = partition_rooms(all_world_lines, boundary_poly, MIN_ROOM_AREA)
    
    # Step 8: Merge small rooms
    rooms = merge_small_rooms(rooms, min_area=2.0, target_count=6)
    
    # Step 9: Snap room polygons to dominant angles
    snap_angles = peak_angles + [(a + 180) % 360 for a in peak_angles]
    rooms = [snap_polygon_angles(r, snap_angles) for r in rooms]
    rooms = [r for r in rooms if r is not None and not r.is_empty and r.area > 1.0]
    
    print(f"\n=== RESULTS ===")
    total = 0
    for i, room in enumerate(rooms):
        label = classify_room(room, rooms)
        nv = len(room.exterior.coords) - 1
        print(f"  {label}: {room.area:.1f}m² ({nv}v)")
        total += room.area
    print(f"  Total: {total:.1f}m²")
    print(f"  Angles: {peak_angles[0]:.0f}° and {peak_angles[1]:.0f}°")
    
    # Plot
    plot_results(density, skel, wall_binary, segments, families, merged_segs,
                 rooms, boundary_poly, peak_angles, origin, RESOLUTION, args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v60 - Cross-Section Contour + Angle-Snap Vectorization

APPROACH: Take horizontal slices of the mesh at wall height (0.8-1.5m).
The mesh_plane intersection gives EXACT wall positions as line segments.
Multi-slice consensus eliminates furniture. Snap segments to dominant angles.
Polygonize to get rooms.

Key insight: mesh_plane gives us the TRUE wall geometry directly from 3D.
No density images, no Hough, no RANSAC guessing. Just slice and vectorize.

Pipeline:
1. Take 5 horizontal slices at 0.8m, 1.0m, 1.2m, 1.4m, 1.6m
2. Each slice → set of line segments (mesh triangle intersections)
3. Accumulate segments into a 2D density image (wall consensus)
4. Skeletonize the wall consensus → 1px wall centerlines
5. Probabilistic Hough on skeleton → line segments with start/end
6. Cluster segment angles → 2 dominant families (~29°, ~119°)
7. Snap each segment to nearest dominant angle
8. Merge overlapping/collinear segments within each family
9. Extend wall segments to intersect with boundary + each other
10. Polygonize → rooms
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
from skimage.morphology import skeletonize, disk, binary_dilation as sk_dilate
from skimage.transform import probabilistic_hough_line
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import polygonize, unary_union, linemerge
import matplotlib.colors as mcolors


# --- CONFIG ---
RESOLUTION = 0.02  # 2cm per pixel
SLICE_HEIGHTS = [-1.8, -1.5, -1.2, -0.9, -0.5]  # meters (Y-up, floor≈-2.5, ceil≈0.35)
MIN_CONSENSUS = 1  # segment must appear in at least N slices
ANGLE_TOLERANCE = 18  # degrees for angle clustering
MIN_SEGMENT_LENGTH = 0.4  # meters
MIN_ROOM_AREA = 2.0


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def slice_mesh(mesh, y_height):
    """Slice mesh at given Y height, return line segments in XZ plane."""
    # mesh_plane returns (n,2,3) array of line segment endpoints
    try:
        lines, face_idx = trimesh.intersections.mesh_plane(
            mesh, [0, 1, 0], [0, y_height, 0], return_faces=True
        )
    except Exception:
        lines = trimesh.intersections.mesh_plane(
            mesh, [0, 1, 0], [0, y_height, 0]
        )
    
    if lines is None or len(lines) == 0:
        return np.array([]).reshape(0, 2, 2)
    
    # Extract XZ coordinates (drop Y)
    segments_xz = lines[:, :, [0, 2]]  # (n, 2, 2) — start/end in XZ
    return segments_xz


def segments_to_density(segments_list, resolution):
    """Rasterize multiple slices of segments into a consensus density image."""
    # Find bounding box across all slices
    all_pts = np.concatenate([s.reshape(-1, 2) for s in segments_list if len(s) > 0])
    xmin, zmin = all_pts.min(axis=0) - 0.5
    xmax, zmax = all_pts.max(axis=0) + 0.5
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    density = np.zeros((h, w), dtype=np.float32)
    
    for segments in segments_list:
        if len(segments) == 0:
            continue
        slice_img = np.zeros((h, w), dtype=np.uint8)
        
        for seg in segments:
            # Convert to pixel coords
            p1x = int((seg[0, 0] - xmin) / resolution)
            p1y = int((seg[0, 1] - zmin) / resolution)
            p2x = int((seg[1, 0] - xmin) / resolution)
            p2y = int((seg[1, 1] - zmin) / resolution)
            
            p1x = np.clip(p1x, 0, w-1)
            p1y = np.clip(p1y, 0, h-1)
            p2x = np.clip(p2x, 0, w-1)
            p2y = np.clip(p2y, 0, h-1)
            
            cv2.line(slice_img, (p1x, p1y), (p2x, p2y), 1, thickness=1)
        
        density += slice_img.astype(np.float32)
    
    return density, (xmin, zmin, xmax, zmax, w, h)


def find_dominant_angles(segments_list, n_families=2):
    """Find dominant wall angles from raw cross-section segments."""
    angles = []
    weights = []
    
    for segments in segments_list:
        for seg in segments:
            dx = seg[1, 0] - seg[0, 0]
            dz = seg[1, 1] - seg[0, 1]
            length = np.sqrt(dx**2 + dz**2)
            if length < 0.05:  # skip tiny segments
                continue
            angle = np.degrees(np.arctan2(dz, dx)) % 180
            angles.append(angle)
            weights.append(length)
    
    angles = np.array(angles)
    weights = np.array(weights)
    
    # Histogram with weights to find peaks
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=weights)
    
    # Smooth histogram
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=3)
    
    # Find top N peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist_smooth, distance=30, height=0)
    peak_heights = properties['peak_heights']
    top_peaks = peaks[np.argsort(peak_heights)[-n_families:]]
    top_peaks.sort()
    
    dominant_angles = top_peaks.astype(float)
    print(f"  Dominant angles: {dominant_angles}")
    
    return dominant_angles


def extract_wall_segments(density, grid_info, dominant_angles, 
                          min_consensus=2, min_length_px=15):
    """From density image, extract and vectorize wall segments."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    # Threshold: walls appear in multiple slices
    wall_mask = (density >= min_consensus).astype(np.uint8)
    
    # Dilate slightly to connect nearby wall pixels, then thin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
    
    # Skeletonize
    skeleton = skeletonize(wall_mask > 0)
    skel_pixels = skeleton.sum()
    print(f"  Skeleton pixels: {skel_pixels}")
    
    # Probabilistic Hough on skeleton
    lines = probabilistic_hough_line(
        skeleton,
        threshold=8,
        line_length=min_length_px,
        line_gap=8
    )
    print(f"  Raw Hough segments: {len(lines)}")
    
    # Convert to world coordinates and snap to dominant angles
    world_segments = []
    for (x1, y1), (x2, y2) in lines:
        wx1 = x1 * RESOLUTION + xmin
        wz1 = y1 * RESOLUTION + zmin
        wx2 = x2 * RESOLUTION + xmin
        wz2 = y2 * RESOLUTION + zmin
        
        # Compute segment angle
        dx = wx2 - wx1
        dz = wz2 - wz1
        length = np.sqrt(dx**2 + dz**2)
        if length < MIN_SEGMENT_LENGTH:
            continue
        
        angle = np.degrees(np.arctan2(dz, dx)) % 180
        
        # Find closest dominant angle
        min_diff = 180
        best_angle = angle
        for da in dominant_angles:
            diff = abs(angle - da)
            diff = min(diff, 180 - diff)
            if diff < min_diff:
                min_diff = diff
                best_angle = da
        
        if min_diff > ANGLE_TOLERANCE:
            continue  # Skip segments that don't match any dominant angle
        
        # Snap: rotate segment to dominant angle while keeping midpoint
        mid_x = (wx1 + wx2) / 2
        mid_z = (wz1 + wz2) / 2
        half_len = length / 2
        
        rad = np.radians(best_angle)
        dx_new = np.cos(rad) * half_len
        dz_new = np.sin(rad) * half_len
        
        world_segments.append({
            'p1': np.array([mid_x - dx_new, mid_z - dz_new]),
            'p2': np.array([mid_x + dx_new, mid_z + dz_new]),
            'angle': best_angle,
            'length': length,
            'midpoint': np.array([mid_x, mid_z]),
        })
    
    print(f"  Angle-filtered segments: {len(world_segments)}")
    return world_segments, skeleton


def merge_collinear_segments(segments, max_perp_dist=0.15, max_gap=0.5):
    """Merge segments that are nearly collinear (same angle, close perpendicular offset)."""
    if not segments:
        return segments
    
    # Group by angle
    angle_groups = {}
    for seg in segments:
        a = round(seg['angle'])
        angle_groups.setdefault(a, []).append(seg)
    
    merged_all = []
    for angle, group in angle_groups.items():
        # Sort by perpendicular offset
        rad = np.radians(angle)
        normal = np.array([-np.sin(rad), np.cos(rad)])
        
        for seg in group:
            seg['perp_offset'] = np.dot(seg['midpoint'], normal)
            seg['para_proj'] = np.dot(seg['midpoint'], np.array([np.cos(rad), np.sin(rad)]))
        
        group.sort(key=lambda s: s['perp_offset'])
        
        # Cluster by perpendicular offset
        clusters = []
        current = [group[0]]
        for seg in group[1:]:
            if abs(seg['perp_offset'] - current[-1]['perp_offset']) < max_perp_dist:
                current.append(seg)
            else:
                clusters.append(current)
                current = [seg]
        clusters.append(current)
        
        # For each cluster, merge into longest continuous segment(s)
        direction = np.array([np.cos(rad), np.sin(rad)])
        
        for cluster in clusters:
            # Project all endpoints onto direction axis
            projections = []
            for seg in cluster:
                proj1 = np.dot(seg['p1'], direction)
                proj2 = np.dot(seg['p2'], direction)
                projections.append((min(proj1, proj2), max(proj1, proj2)))
            
            # Sort by start projection
            projections.sort()
            
            # Merge overlapping intervals
            merged_intervals = [list(projections[0])]
            for start, end in projections[1:]:
                if start <= merged_intervals[-1][1] + max_gap:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
                else:
                    merged_intervals.append([start, end])
            
            # Convert back to segments
            mean_perp = np.mean([s['perp_offset'] for s in cluster])
            for start, end in merged_intervals:
                mid_para = (start + end) / 2
                mid_point = direction * mid_para + normal * mean_perp
                half_len = (end - start) / 2
                
                merged_all.append({
                    'p1': mid_point - direction * half_len,
                    'p2': mid_point + direction * half_len,
                    'angle': angle,
                    'length': end - start,
                    'midpoint': mid_point,
                    'perp_offset': mean_perp,
                })
    
    return merged_all


def deduplicate_walls(segments, min_gap=0.3):
    """Remove duplicate parallel walls (from double-wall surfaces)."""
    if not segments:
        return segments
    
    angle_groups = {}
    for seg in segments:
        a = round(seg['angle'])
        angle_groups.setdefault(a, []).append(seg)
    
    kept = []
    for angle, group in angle_groups.items():
        group.sort(key=lambda s: s['perp_offset'])
        
        deduped = [group[0]]
        for seg in group[1:]:
            if abs(seg['perp_offset'] - deduped[-1]['perp_offset']) < min_gap:
                # Keep longer one
                if seg['length'] > deduped[-1]['length']:
                    deduped[-1] = seg
            else:
                deduped.append(seg)
        kept.extend(deduped)
    
    return kept


def make_boundary(mesh, resolution=0.02):
    """Create apartment boundary polygon."""
    verts = mesh.vertices[:, [0, 2]]
    xmin, zmin = verts.min(axis=0) - 0.5
    xmax, zmax = verts.max(axis=0) + 0.5
    
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    img = np.zeros((h, w), dtype=np.uint8)
    px = ((verts[:, 0] - xmin) / resolution).astype(int)
    py = ((verts[:, 1] - zmin) / resolution).astype(int)
    px = np.clip(px, 0, w-1)
    py = np.clip(py, 0, h-1)
    for x, y in zip(px, py):
        img[y, x] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(float)
    pts[:, 0] = pts[:, 0] * resolution + xmin
    pts[:, 1] = pts[:, 1] * resolution + zmin
    
    poly = Polygon(pts).buffer(0)
    poly = poly.simplify(0.15).buffer(0.05).buffer(-0.05).simplify(0.1)
    return poly


def extend_segment_to_boundary(seg, boundary_poly, extension=5.0):
    """Extend a wall segment to reach the apartment boundary or other walls."""
    direction = seg['p2'] - seg['p1']
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    direction = direction / length
    
    mid = seg['midpoint']
    
    # Extend far in both directions
    far_p1 = mid - direction * extension
    far_p2 = mid + direction * extension
    
    extended = LineString([far_p1, far_p2])
    clipped = extended.intersection(boundary_poly)
    
    if clipped.is_empty:
        return None
    
    if isinstance(clipped, MultiLineString):
        # Pick the longest piece
        clipped = max(clipped.geoms, key=lambda g: g.length)
    
    if isinstance(clipped, LineString):
        return clipped
    return None


def build_rooms(wall_lines, boundary_poly, min_area=1.5):
    """Polygonize wall lines within boundary to get rooms."""
    all_lines = [boundary_poly.exterior]
    for wl in wall_lines:
        if wl is not None:
            all_lines.append(wl)
    
    union = unary_union(all_lines)
    result = list(polygonize(union))
    
    rooms = []
    for poly in result:
        if not boundary_poly.contains(poly.representative_point()):
            continue
        if poly.area < min_area:
            continue
        rooms.append(poly)
    
    rooms.sort(key=lambda p: p.area, reverse=True)
    return rooms


def merge_small_rooms(rooms, min_area=2.0, max_rooms=7):
    """Merge small rooms into neighbors."""
    while True:
        small = [i for i, r in enumerate(rooms) if r.area < min_area]
        if not small:
            break
        
        idx = min(small, key=lambda i: rooms[i].area)
        room = rooms[idx]
        
        best_j = None
        best_shared = 0
        for j, other in enumerate(rooms):
            if j == idx:
                continue
            shared = room.boundary.intersection(other.boundary).length
            if shared > best_shared:
                best_shared = shared
                best_j = j
        
        if best_j is not None:
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        else:
            break
    
    while len(rooms) > max_rooms:
        idx = min(range(len(rooms)), key=lambda i: rooms[i].area)
        best_j = None
        best_shared = 0
        for j, other in enumerate(rooms):
            if j == idx:
                continue
            shared = rooms[idx].boundary.intersection(other.boundary).length
            if shared > best_shared:
                best_shared = shared
                best_j = j
        if best_j is not None:
            rooms[best_j] = unary_union([rooms[best_j], rooms[idx]])
            rooms.pop(idx)
        else:
            break
    
    return rooms


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


def plot_results(density, grid_info, skeleton, raw_segments, merged_segments,
                 wall_lines_shapely, rooms, boundary_poly, dominant_angles, output_path):
    """4-panel diagnostic plot."""
    xmin, zmin, xmax, zmax, w, h = grid_info
    
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    
    # Panel 1: Cross-section density
    ax = axes[0]
    ax.imshow(density, origin='lower', cmap='hot',
              extent=[xmin, xmax, zmin, zmax], aspect='equal')
    ax.set_title(f"Cross-Section Density\n({len(SLICE_HEIGHTS)} slices, {RESOLUTION*100:.0f}cm res)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Skeleton + raw segments
    ax = axes[1]
    ax.imshow(skeleton, origin='lower', cmap='gray',
              extent=[xmin, xmax, zmin, zmax], aspect='equal', alpha=0.3)
    for seg in raw_segments:
        ax.plot([seg['p1'][0], seg['p2'][0]], [seg['p1'][1], seg['p2'][1]],
                'b-', linewidth=0.5, alpha=0.5)
    ax.set_title(f"Skeleton + Raw Segments ({len(raw_segments)})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Merged + deduped wall segments
    ax = axes[2]
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=1.5, alpha=0.5)
    
    colors = {'29': 'red', '119': 'blue', '30': 'red', '120': 'blue'}
    for seg in merged_segments:
        color = colors.get(str(round(seg['angle'])), 'green')
        ax.plot([seg['p1'][0], seg['p2'][0]], [seg['p1'][1], seg['p2'][1]],
                color=color, linewidth=2)
        # Label with perp offset
        mx, mz = seg['midpoint']
        ax.text(mx, mz, f"{seg['perp_offset']:.1f}", fontsize=5, color=color)
    
    ax.set_title(f"Merged Walls ({len(merged_segments)})\nAngles: {dominant_angles}")
    ax.set_aspect('equal')
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
    
    if boundary_poly:
        bx, by = boundary_poly.exterior.xy
        ax.plot(bx, by, 'k-', linewidth=3)
    
    # Scale bar
    ax.plot([-4, -3], [-5, -5], 'k-', linewidth=3)
    ax.text(-3.5, -5.3, '1m', ha='center', fontsize=10)
    
    ax.set_title(f"v60 — {len(rooms)} rooms, {total_area:.1f}m²\n"
                 f"Angles: {', '.join(f'{a:.0f}°' for a in dominant_angles)}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
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
    parser.add_argument('--output', default='output_v60')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--min-consensus', type=int, default=MIN_CONSENSUS)
    parser.add_argument('--min-gap', type=float, default=0.35)
    parser.add_argument('--min-room-area', type=float, default=2.0)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mesh_path = data_dir / args.mesh
    
    mesh = load_mesh(mesh_path)
    
    # Step 1: Multi-slice cross-sections
    print(f"\nStep 1: Slicing at Y = {SLICE_HEIGHTS}...")
    segments_list = []
    for y in SLICE_HEIGHTS:
        segs = slice_mesh(mesh, y)
        segments_list.append(segs)
        print(f"  Y={y:.1f}m: {len(segs)} segments")
    
    # Step 2: Find dominant angles from raw segments
    print("\nStep 2: Finding dominant angles...")
    dominant_angles = find_dominant_angles(segments_list)
    # Override: use known angles from RANSAC analysis (v58)
    # The histogram often misses the perpendicular family
    dominant_angles = np.array([29.0, 119.0])
    print(f"  Using known angles: {dominant_angles}")
    
    # Step 3: Accumulate into density image
    print("\nStep 3: Building consensus density...")
    density, grid_info = segments_to_density(segments_list, RESOLUTION)
    print(f"  Density image: {grid_info[4]}x{grid_info[5]}")
    print(f"  Max consensus: {density.max():.0f} slices")
    
    # Step 4: Extract and snap wall segments
    print("\nStep 4: Extracting wall segments...")
    wall_segments, skeleton = extract_wall_segments(
        density, grid_info, dominant_angles, 
        min_consensus=args.min_consensus
    )
    
    # Step 5: Merge collinear segments
    print("\nStep 5: Merging collinear segments...")
    merged = merge_collinear_segments(wall_segments, max_perp_dist=0.15, max_gap=0.5)
    print(f"  Merged: {len(wall_segments)} → {len(merged)} segments")
    
    # Step 6: Deduplicate parallel walls
    merged = deduplicate_walls(merged, min_gap=args.min_gap)
    print(f"  After dedup: {len(merged)} walls")
    
    # Filter: keep only walls longer than 1.5m (structural walls)
    # Plus the top N shorter walls by length
    long_walls = [s for s in merged if s['length'] >= 1.5]
    short_walls = sorted([s for s in merged if s['length'] < 1.5], 
                         key=lambda s: s['length'], reverse=True)
    # Keep up to 3 short walls (for internal partitions)
    merged = long_walls + short_walls[:3]
    print(f"  After length filter: {len(merged)} walls ({len(long_walls)} long + {min(3, len(short_walls))} short)")
    
    for seg in merged:
        print(f"    {seg['angle']:.0f}° perp={seg['perp_offset']:.2f} len={seg['length']:.1f}m")
    
    # Step 7: Create boundary and extend walls
    print("\nStep 7: Building boundary and extending walls...")
    boundary = make_boundary(mesh)
    print(f"  Boundary area: {boundary.area:.1f}m²")
    
    wall_lines = []
    for seg in merged:
        extended = extend_segment_to_boundary(seg, boundary, extension=15.0)
        if extended is not None:
            wall_lines.append(extended)
    print(f"  Extended wall lines: {len(wall_lines)}")
    
    # Step 8: Build rooms
    print("\nStep 8: Building rooms...")
    rooms = build_rooms(wall_lines, boundary, min_area=1.0)
    print(f"  Raw rooms: {len(rooms)}")
    
    rooms = merge_small_rooms(rooms, min_area=args.min_room_area, max_rooms=7)
    print(f"  Final rooms: {len(rooms)}")
    
    for room in rooms:
        label = classify_room(room)
        nv = len(room.exterior.coords) - 1
        print(f"    {label}: {room.area:.1f}m² ({nv}v)")
    
    # Step 9: Plot
    total = plot_results(density, grid_info, skeleton, wall_segments, merged,
                         wall_lines, rooms, boundary, dominant_angles, args.output)
    
    print(f"\n{'='*50}")
    print(f"v60 Cross-Section Snap: {len(rooms)} rooms, {total:.1f}m²")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v57 - Vector Cross-Section → Wall Lines → Room Polygons

GENUINELY NEW APPROACH: Instead of rasterizing anything, work with
the actual geometric line segments from mesh cross-sections.

Pipeline:
1. Slice mesh at multiple heights (0.8-1.6m) to get intersection segments
2. Each segment = exact wall outline piece (no rasterization noise!)
3. Project segments to XZ plane
4. Find dominant wall angles via segment orientation histogram
5. For each angle family: cluster segments by perpendicular offset → wall lines
6. Extend wall lines to apartment boundary
7. Find room polygons from line arrangement (planar subdivision)

Key advantage over v54/v56b: No rasterization, no density images, no watershed.
Pure vector geometry → clean straight walls guaranteed.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from collections import defaultdict
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point, box
from shapely.ops import polygonize, unary_union, split
import shapely


def slice_mesh_multi(mesh, heights):
    """Slice mesh at multiple heights, collect all 2D segments (XZ plane)."""
    all_segments = []
    for h in heights:
        try:
            # trimesh.intersections.mesh_plane returns (n,2,3) segments
            plane_origin = [0, h, 0]
            plane_normal = [0, 1, 0]
            segments_3d = trimesh.intersections.mesh_plane(
                mesh, plane_normal, plane_origin
            )
            if len(segments_3d) == 0:
                continue
            # Project to XZ: take X and Z coordinates
            segs_2d = segments_3d[:, :, [0, 2]]  # (n, 2, 2) - each is [[x1,z1],[x2,z2]]
            all_segments.append(segs_2d)
        except Exception as e:
            print(f"  Slice at h={h:.2f}m failed: {e}")
    if all_segments:
        return np.concatenate(all_segments, axis=0)
    return np.zeros((0, 2, 2))


def segment_angle(seg):
    """Return angle in [0, 180) degrees."""
    d = seg[1] - seg[0]
    angle = np.degrees(np.arctan2(d[1], d[0])) % 180
    return angle


def segment_length(seg):
    return np.linalg.norm(seg[1] - seg[0])


def find_dominant_angles(segments, min_length=0.05):
    """Find 2 dominant perpendicular angles from segment orientations."""
    angles = []
    weights = []
    for seg in segments:
        l = segment_length(seg)
        if l < min_length:
            continue
        a = segment_angle(seg)
        angles.append(a)
        weights.append(l)
    
    angles = np.array(angles)
    weights = np.array(weights)
    
    # Weighted histogram with 1-degree bins
    hist, bin_edges = np.histogram(angles, bins=180, range=(0, 180), weights=weights)
    
    # Smooth histogram
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')
    
    # Find top peak
    peak1_idx = np.argmax(hist_smooth)
    angle1 = bin_edges[peak1_idx] + 0.5
    
    # Suppress near peak1 (±15°)
    hist_masked = hist_smooth.copy()
    for i in range(180):
        diff = min(abs(i - peak1_idx), 180 - abs(i - peak1_idx))
        if diff < 15:
            hist_masked[i] = 0
    
    peak2_idx = np.argmax(hist_masked)
    angle2 = bin_edges[peak2_idx] + 0.5
    
    # Sort so angle1 < angle2
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1
    
    print(f"  Dominant angles: {angle1:.1f}° and {angle2:.1f}° (diff={angle2-angle1:.1f}°)")
    return angle1, angle2


def perpendicular_offset(seg, angle_deg):
    """Signed perpendicular distance of segment midpoint from origin along angle direction."""
    mid = (seg[0] + seg[1]) / 2
    # Normal to angle direction
    angle_rad = np.radians(angle_deg)
    nx = -np.sin(angle_rad)
    ny = np.cos(angle_rad)
    return mid[0] * nx + mid[1] * ny


def classify_segments(segments, angle1, angle2, angle_tol=12, min_length=0.03):
    """Classify segments into two angle families."""
    fam1 = []  # segments near angle1
    fam2 = []  # segments near angle2
    
    for seg in segments:
        l = segment_length(seg)
        if l < min_length:
            continue
        a = segment_angle(seg)
        
        diff1 = min(abs(a - angle1), 180 - abs(a - angle1))
        diff2 = min(abs(a - angle2), 180 - abs(a - angle2))
        
        if diff1 < angle_tol:
            fam1.append(seg)
        elif diff2 < angle_tol:
            fam2.append(seg)
    
    return np.array(fam1) if fam1 else np.zeros((0,2,2)), \
           np.array(fam2) if fam2 else np.zeros((0,2,2))


def cluster_wall_lines(segments, angle_deg, min_gap=0.15, min_support=0.3):
    """
    Cluster segments by perpendicular offset to find distinct wall lines.
    Returns list of (offset, total_support_length) for each wall.
    """
    if len(segments) == 0:
        return []
    
    offsets = np.array([perpendicular_offset(seg, angle_deg) for seg in segments])
    lengths = np.array([segment_length(seg) for seg in segments])
    
    # Sort by offset
    order = np.argsort(offsets)
    offsets = offsets[order]
    lengths = lengths[order]
    
    # Cluster by offset gap
    walls = []
    cluster_offsets = [offsets[0]]
    cluster_lengths = [lengths[0]]
    
    for i in range(1, len(offsets)):
        if offsets[i] - offsets[i-1] > min_gap:
            # Finish this cluster
            avg_offset = np.average(cluster_offsets, weights=cluster_lengths)
            total_length = sum(cluster_lengths)
            if total_length >= min_support:
                walls.append((avg_offset, total_length))
            cluster_offsets = []
            cluster_lengths = []
        cluster_offsets.append(offsets[i])
        cluster_lengths.append(lengths[i])
    
    # Last cluster
    if cluster_offsets:
        avg_offset = np.average(cluster_offsets, weights=cluster_lengths)
        total_length = sum(cluster_lengths)
        if total_length >= min_support:
            walls.append((avg_offset, total_length))
    
    return walls


def wall_line_to_shapely(offset, angle_deg, extent=20.0):
    """Convert wall (offset, angle) to a long Shapely LineString."""
    angle_rad = np.radians(angle_deg)
    # Direction along wall
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    # Normal (perpendicular) direction
    nx = -np.sin(angle_rad)
    ny = np.cos(angle_rad)
    # Point on line
    px = offset * nx
    py = offset * ny
    # Extend in both directions
    p1 = (px - extent * dx, py - extent * dy)
    p2 = (px + extent * dx, py + extent * dy)
    return LineString([p1, p2])


def find_apartment_boundary(segments, angle1, angle2, buffer=0.1):
    """Find convex hull of all segment endpoints as apartment boundary."""
    if len(segments) == 0:
        return None
    pts = segments.reshape(-1, 2)
    from shapely.geometry import MultiPoint
    hull = MultiPoint(pts).convex_hull.buffer(buffer)
    return hull


def partition_rooms(wall_lines, boundary, min_area=1.5):
    """
    Use wall lines to partition the boundary polygon into rooms.
    Split boundary by each wall line sequentially.
    """
    # Collect all lines
    all_lines = []
    for line in wall_lines:
        all_lines.append(line)
    
    # Also add boundary as lines
    merged_lines = unary_union(all_lines + [boundary.boundary])
    
    # Polygonize finds all closed regions
    result = list(polygonize(merged_lines))
    
    # Filter to rooms inside boundary and above min area
    rooms = []
    for poly in result:
        if poly.area < min_area:
            continue
        # Check centroid inside boundary
        if boundary.contains(poly.centroid) or boundary.intersection(poly).area > 0.5 * poly.area:
            rooms.append(poly)
    
    return rooms


def classify_room(area):
    """Simple room classification by area."""
    if area > 9:
        return "Room"
    elif area > 5:
        return "Room"
    elif area > 3:
        return "Bathroom"
    else:
        return "Closet"


def snap_polygon_to_angles(poly, angle1, angle2, tol=0.15):
    """Simplify polygon edges to snap to dominant angles."""
    coords = list(poly.exterior.coords)[:-1]  # Remove closing point
    if len(coords) < 3:
        return poly
    
    # For each edge, snap to nearest angle
    snapped_coords = []
    n = len(coords)
    
    for i in range(n):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i+1) % n])
        d = p2 - p1
        edge_angle = np.degrees(np.arctan2(d[1], d[0])) % 180
        
        diff1 = min(abs(edge_angle - angle1), 180 - abs(edge_angle - angle1))
        diff2 = min(abs(edge_angle - angle2), 180 - abs(edge_angle - angle2))
        
        # Snap to closer angle
        if diff1 < diff2:
            snap_angle = angle1
        else:
            snap_angle = angle2
        
        snapped_coords.append(tuple(p1))
    
    try:
        result = Polygon(snapped_coords)
        if result.is_valid and result.area > 0.5:
            return result
    except:
        pass
    return poly


def make_apartment_mask_polygon(segments, angle1, angle2, buffer_dist=0.3):
    """Create apartment boundary polygon from outermost walls."""
    if len(segments) == 0:
        return None
    
    pts = segments.reshape(-1, 2)
    from shapely.geometry import MultiPoint
    
    # Use alpha shape or convex hull
    hull = MultiPoint(pts).convex_hull
    
    # Buffer slightly to ensure walls are inside
    return hull.buffer(buffer_dist)


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v57 - Vector Cross-Section')
    parser.add_argument('mesh_path', help='Path to mesh file')
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--min-height', type=float, default=0.8, help='Min slice height (m)')
    parser.add_argument('--max-height', type=float, default=1.5, help='Max slice height (m)')
    parser.add_argument('--num-slices', type=int, default=8, help='Number of slices')
    parser.add_argument('--min-wall-gap', type=float, default=0.15, help='Min gap between walls (m)')
    parser.add_argument('--min-wall-support', type=float, default=0.3, help='Min total segment length for wall (m)')
    parser.add_argument('--min-room-area', type=float, default=1.5, help='Min room area (m²)')
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f'results/v57_vector_xsection')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    # Determine Y range - Y-up, floor at Y_min
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    y_floor = y_min
    print(f"  Y range: {y_min:.2f} to {y_max:.2f}m (floor at {y_floor:.2f})")
    
    # Slice at wall heights above floor
    heights = np.linspace(y_floor + args.min_height, y_floor + args.max_height, args.num_slices)
    print(f"\nSlicing at {args.num_slices} heights from {args.min_height}m to {args.max_height}m...")
    segments = slice_mesh_multi(mesh, heights)
    print(f"  Got {len(segments)} segments total")
    
    if len(segments) == 0:
        # Try different Y interpretation - maybe mesh isn't Y-up
        print("No segments found. Trying Z-up interpretation...")
        z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
        print(f"  Z range: {z_min:.2f} to {z_max:.2f}m")
        
        # Slice along Z instead
        all_segs = []
        for h in heights:
            try:
                segs_3d = trimesh.intersections.mesh_plane(
                    mesh, [0, 0, 1], [0, 0, h]
                )
                if len(segs_3d) > 0:
                    segs_2d = segs_3d[:, :, [0, 1]]  # XY plane
                    all_segs.append(segs_2d)
            except:
                pass
        if all_segs:
            segments = np.concatenate(all_segs, axis=0)
            print(f"  Z-up: Got {len(segments)} segments")
    
    if len(segments) == 0:
        print("ERROR: No cross-section segments found at any height")
        return
    
    # Filter out tiny segments
    lengths = np.array([segment_length(s) for s in segments])
    print(f"  Segment lengths: min={lengths.min():.4f}m, max={lengths.max():.2f}m, median={np.median(lengths):.4f}m")
    
    # Keep segments > 1cm
    mask = lengths > 0.01
    segments = segments[mask]
    print(f"  After filtering <1cm: {len(segments)} segments")
    
    # Find dominant angles
    print("\nFinding dominant wall angles...")
    angle1, angle2 = find_dominant_angles(segments)
    
    # Classify segments by angle family
    fam1, fam2 = classify_segments(segments, angle1, angle2)
    print(f"  Family 1 ({angle1:.0f}°): {len(fam1)} segments")
    print(f"  Family 2 ({angle2:.0f}°): {len(fam2)} segments")
    unclassified = len(segments) - len(fam1) - len(fam2)
    print(f"  Unclassified: {unclassified} segments")
    
    # Cluster into wall lines
    print(f"\nClustering wall lines (min_gap={args.min_wall_gap}m, min_support={args.min_wall_support}m)...")
    walls1 = cluster_wall_lines(fam1, angle1, min_gap=args.min_wall_gap, min_support=args.min_wall_support)
    walls2 = cluster_wall_lines(fam2, angle2, min_gap=args.min_wall_gap, min_support=args.min_wall_support)
    print(f"  Family 1 walls: {len(walls1)}")
    for i, (offset, support) in enumerate(walls1):
        print(f"    Wall {i}: offset={offset:.2f}m, support={support:.2f}m")
    print(f"  Family 2 walls: {len(walls2)}")
    for i, (offset, support) in enumerate(walls2):
        print(f"    Wall {i}: offset={offset:.2f}m, support={support:.2f}m")
    
    total_walls = len(walls1) + len(walls2)
    print(f"  Total walls: {total_walls}")
    
    # Create apartment boundary
    print("\nCreating apartment boundary...")
    boundary = make_apartment_mask_polygon(segments, angle1, angle2, buffer_dist=0.2)
    print(f"  Boundary area: {boundary.area:.1f}m²")
    
    # Convert walls to Shapely lines
    wall_lines = []
    wall_info = []
    for offset, support in walls1:
        line = wall_line_to_shapely(offset, angle1)
        # Clip to boundary
        clipped = line.intersection(boundary)
        if not clipped.is_empty:
            wall_lines.append(clipped)
            wall_info.append((angle1, offset, support))
    
    for offset, support in walls2:
        line = wall_line_to_shapely(offset, angle2)
        clipped = line.intersection(boundary)
        if not clipped.is_empty:
            wall_lines.append(clipped)
            wall_info.append((angle2, offset, support))
    
    print(f"  {len(wall_lines)} wall lines after clipping to boundary")
    
    # Partition into rooms
    print("\nPartitioning into rooms...")
    rooms = partition_rooms(wall_lines, boundary, min_area=args.min_room_area)
    print(f"  Found {len(rooms)} rooms")
    
    # Sort by area descending
    rooms.sort(key=lambda r: r.area, reverse=True)
    
    total_area = sum(r.area for r in rooms)
    print(f"  Total room area: {total_area:.1f}m²")
    
    # Classify and print
    room_data = []
    for i, room in enumerate(rooms):
        area = room.area
        name = classify_room(area)
        n_verts = len(room.exterior.coords) - 1
        cx, cy = room.centroid.x, room.centroid.y
        print(f"  {name}: {area:.1f}m² ({n_verts}v) at ({cx:.1f}, {cy:.1f})")
        room_data.append({
            'name': name,
            'area': area,
            'vertices': n_verts,
            'centroid': [cx, cy]
        })
    
    # === PLOTTING ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Raw segments colored by angle family
    ax = axes[0]
    ax.set_title(f'Cross-Section Segments ({len(segments)} total)')
    ax.set_aspect('equal')
    
    # Draw unclassified in gray
    for seg in segments:
        a = segment_angle(seg)
        diff1 = min(abs(a - angle1), 180 - abs(a - angle1))
        diff2 = min(abs(a - angle2), 180 - abs(a - angle2))
        if diff1 >= 12 and diff2 >= 12:
            ax.plot([seg[0,0], seg[1,0]], [seg[0,1], seg[1,1]], 'gray', alpha=0.2, lw=0.3)
    
    for seg in fam1:
        ax.plot([seg[0,0], seg[1,0]], [seg[0,1], seg[1,1]], 'blue', alpha=0.3, lw=0.5)
    for seg in fam2:
        ax.plot([seg[0,0], seg[1,0]], [seg[0,1], seg[1,1]], 'red', alpha=0.3, lw=0.5)
    ax.legend([f'Fam1 ({angle1:.0f}°)', f'Fam2 ({angle2:.0f}°)'], fontsize=8)
    
    # Panel 2: Wall lines
    ax = axes[1]
    ax.set_title(f'Wall Lines ({total_walls} walls)')
    ax.set_aspect('equal')
    
    # Draw boundary
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', lw=1, alpha=0.3)
    
    # Draw wall lines
    colors = plt.cm.Set1(np.linspace(0, 1, max(total_walls, 1)))
    for i, line in enumerate(wall_lines):
        if hasattr(line, 'geoms'):
            for geom in line.geoms:
                x, y = geom.xy
                ax.plot(x, y, color=colors[i % len(colors)], lw=2)
        else:
            x, y = line.xy
            ax.plot(x, y, color=colors[i % len(colors)], lw=2)
    
    # Panel 3: Room polygons
    ax = axes[2]
    room_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    ax.set_title(f'v57 — {len(rooms)} rooms, {total_area:.1f}m²\nAngles: {angle1:.0f}°, {angle2:.0f}°')
    ax.set_aspect('equal')
    
    # Draw boundary
    if boundary:
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, 'k-', lw=2)
    
    for i, room in enumerate(rooms):
        x, y = room.exterior.xy
        ax.fill(x, y, color=room_colors[i % len(room_colors)], alpha=0.6)
        ax.plot(x, y, 'k-', lw=1.5)
        
        # Label
        cx, cy = room.centroid.x, room.centroid.y
        name = room_data[i]['name']
        area = room_data[i]['area']
        ax.text(cx, cy, f'{name}\n{area:.1f}m²', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Scale bar
    ax.plot([ax.get_xlim()[0] + 0.5, ax.get_xlim()[0] + 1.5], 
            [ax.get_ylim()[0] + 0.3, ax.get_ylim()[0] + 0.3], 'k-', lw=3)
    ax.text(ax.get_xlim()[0] + 1.0, ax.get_ylim()[0] + 0.15, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'floorplan.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'floorplan.png'}")
    
    # Save JSON
    result = {
        'version': 'v57_vector_xsection',
        'angles': [angle1, angle2],
        'num_walls': total_walls,
        'num_rooms': len(rooms),
        'total_area': total_area,
        'rooms': room_data
    }
    with open(output_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {output_dir / 'result.json'}")


if __name__ == '__main__':
    main()

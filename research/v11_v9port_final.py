#!/usr/bin/env python3
"""
mesh2plan v9 Algorithm Port to Python - FINAL

Final version with strict boundary detection and rectilinear polygon construction.
Fixes area overshooting and over-sensitive gap detection.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
from pathlib import Path
import math


def detect_up_axis(mesh):
    """Detect the up-axis by finding which coordinate has reasonable room height."""
    vertices = mesh.vertices
    ranges = []
    axis_names = ['X', 'Y', 'Z']
    
    for i in range(3):
        coords = vertices[:, i]
        range_val = np.max(coords) - np.min(coords)
        ranges.append(range_val)
    
    # If Y has a reasonable range (1-4m) and isn't the largest, it's probably up
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    # If Z has a reasonable range and isn't the largest, it's probably up  
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
    else:
        # Fall back to smallest range
        min_idx = np.argmin(ranges)
        return min_idx, axis_names[min_idx]


def slice_at_height(triangles, height, up_axis):
    """Cut triangles with a horizontal plane at the given height along up_axis."""
    points = []
    
    for tri in triangles:
        edges = [
            [tri[0], tri[1]], 
            [tri[1], tri[2]], 
            [tri[2], tri[0]]
        ]
        
        hits = []
        for a, b in edges:
            # Check if edge crosses the horizontal plane
            if (a[up_axis] - height) * (b[up_axis] - height) < 0:
                # Linear interpolation to find intersection
                t = (height - a[up_axis]) / (b[up_axis] - a[up_axis])
                
                # Get the 2D coordinates in the horizontal plane
                if up_axis == 0:  # X-up: use Y,Z as horizontal
                    x = a[1] + t * (b[1] - a[1])
                    y = a[2] + t * (b[2] - a[2])
                elif up_axis == 1:  # Y-up: use X,Z as horizontal
                    x = a[0] + t * (b[0] - a[0])
                    y = a[2] + t * (b[2] - a[2])
                else:  # Z-up: use X,Y as horizontal
                    x = a[0] + t * (b[0] - a[0])
                    y = a[1] + t * (b[1] - a[1])
                    
                hits.append([x, y])
        
        if len(hits) >= 2:
            points.extend(hits[:2])
    
    return points


def rot_pt(p, angle):
    """Rotate point by angle in radians."""
    c = math.cos(angle)
    s = math.sin(angle)
    return [c * p[0] - s * p[1], s * p[0] + c * p[1]]


def histogram_typed(values, n_bins):
    """Create histogram with typed arrays for speed."""
    if len(values) == 0:
        return np.zeros(n_bins, dtype=np.int32)
    
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    
    bins = np.zeros(n_bins, dtype=np.int32)
    for val in values:
        bin_idx = min(n_bins - 1, int((val - min_val) / range_val * n_bins))
        bins[bin_idx] += 1
    
    return bins


def find_dominant_angle(points):
    """Find the dominant angle using histogram voting."""
    step = max(1, len(points) // 5000)
    sampled = points[::step]
    
    if len(sampled) == 0:
        return 0
    
    sampled_array = np.array(sampled)
    best_angle = 0
    best_score = 0
    
    # Coarse search: 0-180° in 2° steps
    for deg in range(0, 180, 2):
        rad = -deg * math.pi / 180
        c = math.cos(rad)
        s = math.sin(rad)
        
        x_buf = c * sampled_array[:, 0] - s * sampled_array[:, 1]
        z_buf = s * sampled_array[:, 0] + c * sampled_array[:, 1]
        
        x_hist = histogram_typed(x_buf, 80)
        z_hist = histogram_typed(z_buf, 80)
        
        score = np.sum(x_hist ** 2) + np.sum(z_hist ** 2)
        
        if score > best_score:
            best_score = score
            best_angle = deg
    
    # Fine search: best ±2° in 1° steps
    for deg in range(best_angle - 2, best_angle + 3):
        rad = -deg * math.pi / 180
        c = math.cos(rad)
        s = math.sin(rad)
        
        x_buf = c * sampled_array[:, 0] - s * sampled_array[:, 1]
        z_buf = s * sampled_array[:, 0] + c * sampled_array[:, 1]
        
        x_hist = histogram_typed(x_buf, 80)
        z_hist = histogram_typed(z_buf, 80)
        
        score = np.sum(x_hist ** 2) + np.sum(z_hist ** 2)
        
        if score > best_score:
            best_score = score
            best_angle = deg
    
    return ((best_angle % 180) + 180) % 180


def find_walls(rotated_points, axis, min_inliers=15, dist_thresh=0.04):
    """Find walls along an axis using histogram peak detection."""
    if len(rotated_points) == 0:
        return []
    
    rotated_array = np.array(rotated_points)
    coords = rotated_array[:, axis]
    other = rotated_array[:, 1 - axis]
    
    min_coord = np.min(coords)
    max_coord = np.max(coords)
    coord_range = max_coord - min_coord
    
    if coord_range <= 0:
        return []
    
    # Create histogram
    n_bins = max(60, int(coord_range / 0.02))
    bin_width = coord_range / n_bins
    
    hist = np.zeros(n_bins)
    for c in coords:
        bin_idx = min(n_bins - 1, int((c - min_coord) / coord_range * n_bins))
        hist[bin_idx] += 1
    
    # Find threshold - more strict than before
    sorted_hist = np.sort(hist)
    median = sorted_hist[n_bins // 2]
    threshold = max(median * 3.0, min_inliers)  # Increased from 2.5 to 3.0
    
    # Find peaks
    walls = []
    in_peak = False
    peak_weight = 0
    peak_sum = 0
    
    for i in range(n_bins + 1):
        bin_center = min_coord + (i + 0.5) * bin_width
        
        if i < n_bins and hist[i] > threshold:
            if not in_peak:
                in_peak = True
                peak_weight = 0
                peak_sum = 0
            peak_weight += hist[i]
            peak_sum += bin_center * hist[i]
        elif in_peak:
            # End of peak
            wall_pos = peak_sum / peak_weight
            
            # Collect nearby points
            near_pts = []
            for j, coord in enumerate(coords):
                if abs(coord - wall_pos) < dist_thresh * 2:
                    near_pts.append(other[j])
            
            if len(near_pts) >= min_inliers:
                near_pts.sort()
                # Use stricter percentiles for extent
                start = near_pts[int(len(near_pts) * 0.05)]  # 5th percentile
                end = near_pts[int(len(near_pts) * 0.95)]    # 95th percentile
                
                if end - start > 0.5:  # Increased minimum wall length
                    walls.append({
                        'axis': 'x' if axis == 0 else 'z',
                        'position': wall_pos,
                        'start': start,
                        'end': end,
                        'length': end - start,
                        'nPoints': len(near_pts)
                    })
            
            in_peak = False
    
    return walls


def select_boundary_walls_strict(walls):
    """Select boundary walls with allowance for significant interior walls."""
    if len(walls) == 0:
        return walls
    
    # Group walls by axis
    x_walls = [w for w in walls if w['axis'] == 'x']
    z_walls = [w for w in walls if w['axis'] == 'z']
    
    boundary_walls = []
    
    # For X walls: keep outermost plus any long interior walls
    if len(x_walls) >= 2:
        x_walls.sort(key=lambda w: w['position'])
        boundary_walls.append(x_walls[0])   # Leftmost
        boundary_walls.append(x_walls[-1])  # Rightmost
        
        # Add significant interior walls (long walls that might have openings)
        for wall in x_walls[1:-1]:
            if wall['length'] > 2.5 or wall['nPoints'] > 50:  # Significant wall
                boundary_walls.append(wall)
                
    elif len(x_walls) == 1:
        boundary_walls.extend(x_walls)
    
    # For Z walls: keep outermost plus any long interior walls
    if len(z_walls) >= 2:
        z_walls.sort(key=lambda w: w['position'])
        boundary_walls.append(z_walls[0])   # Bottommost
        boundary_walls.append(z_walls[-1])  # Topmost
        
        # Add significant interior walls (long walls that might have openings)
        for wall in z_walls[1:-1]:
            if wall['length'] > 2.5 or wall['nPoints'] > 50:  # Significant wall
                boundary_walls.append(wall)
                
    elif len(z_walls) == 1:
        boundary_walls.extend(z_walls)
    
    return boundary_walls


def merge_walls(walls, dist=0.2):
    """Merge parallel walls that are very close together."""
    if len(walls) <= 1:
        return walls
        
    merged = []
    used = set()
    
    for i, wall_i in enumerate(walls):
        if i in used:
            continue
        
        group = [wall_i]
        
        for j, wall_j in enumerate(walls[i+1:], i+1):
            if j in used or wall_i['axis'] != wall_j['axis']:
                continue
            
            if abs(wall_i['position'] - wall_j['position']) < dist:
                group.append(wall_j)
                used.add(j)
        
        # Compute weighted average position and merged extent
        total_points = sum(g['nPoints'] for g in group)
        avg_position = sum(g['position'] * g['nPoints'] for g in group) / total_points
        
        min_start = min(g['start'] for g in group)
        max_end = max(g['end'] for g in group)
        
        merged_wall = {
            **group[0],
            'position': avg_position,
            'start': min_start,
            'end': max_end,
            'length': max_end - min_start,
            'nPoints': total_points
        }
        
        merged.append(merged_wall)
        used.add(i)
    
    return merged


def build_rectilinear_room_polygon(walls, angle_rad):
    """Build a strict rectilinear room polygon from boundary walls."""
    if len(walls) < 4:
        return None
    
    # Group by axis and sort
    x_walls = sorted([w for w in walls if w['axis'] == 'x'], key=lambda w: w['position'])
    z_walls = sorted([w for w in walls if w['axis'] == 'z'], key=lambda w: w['position'])
    
    if len(x_walls) < 2 or len(z_walls) < 2:
        return None
    
    # Get boundary positions (outermost walls on each axis)
    left_x = x_walls[0]['position']      # Leftmost X wall
    right_x = x_walls[-1]['position']    # Rightmost X wall
    bottom_z = z_walls[0]['position']    # Bottommost Z wall  
    top_z = z_walls[-1]['position']      # Topmost Z wall
    
    # Create rectilinear room polygon (axis-aligned rectangle)
    # These are the 4 corners of the basic room rectangle
    rect_corners = [
        [left_x, bottom_z],    # Bottom-left
        [right_x, bottom_z],   # Bottom-right
        [right_x, top_z],      # Top-right
        [left_x, top_z]        # Top-left
    ]
    
    # Transform back to original coordinate system
    room_corners = [rot_pt(corner, angle_rad) for corner in rect_corners]
    
    # Ensure polygon is closed
    exterior = room_corners + [room_corners[0]]
    
    area = polygon_area(room_corners)
    perimeter = polygon_perimeter(room_corners)
    
    return {
        'exterior': exterior,
        'area': area,
        'perimeter': perimeter
    }


def detect_openings_strict(walls, rotated_points, angle_deg):
    """Strict opening detection - only on boundary walls, with high thresholds."""
    angle_rad = angle_deg * math.pi / 180
    gaps = []
    
    for w in walls:
        axis = 0 if w['axis'] == 'x' else 1
        near = []
        
        # Find points near this wall
        for p in rotated_points:
            if abs(p[axis] - w['position']) < 0.06:  # Slightly relaxed tolerance
                near.append(p[1 - axis])
        
        if len(near) < 4:  # Slightly reduced minimum
            continue
        
        near.sort()
        
        # Look for gaps with strict minimum threshold
        for i in range(len(near) - 1):
            gap = near[i + 1] - near[i]
            
            # STRICT: Minimum gap 0.4m to avoid noise (slightly lowered)
            if gap < 0.4:
                continue
                
            g_mid = (near[i] + near[i + 1]) / 2
            
            # Convert back to original coordinates
            if w['axis'] == 'x':
                p1 = rot_pt([w['position'], near[i]], angle_rad)
                p2 = rot_pt([w['position'], near[i + 1]], angle_rad)
                mid = rot_pt([w['position'], g_mid], angle_rad)
            else:
                p1 = rot_pt([near[i], w['position']], angle_rad)
                p2 = rot_pt([near[i + 1], w['position']], angle_rad)
                mid = rot_pt([g_mid, w['position']], angle_rad)
            
            # STRICT classification according to v9 spec:
            if 0.6 <= gap <= 1.2:     # Door range (strict)
                gap_type = 'door'
            elif 1.1 < gap <= 2.0:    # Window range (slightly relaxed upper bound)
                gap_type = 'window' 
            else:
                # Ignore everything else - too small or too large
                continue
            
            gaps.append({
                'type': gap_type,
                'width': gap,
                'start': p1,
                'end': p2,
                'mid': mid
            })
    
    # Sort by gap width (largest first) to prioritize main openings
    gaps.sort(key=lambda g: g['width'], reverse=True)
    
    return gaps


def polygon_area(pts):
    """Compute polygon area using shoelace formula."""
    area = 0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2


def polygon_perimeter(pts):
    """Compute polygon perimeter."""
    perimeter = 0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return perimeter


def analyze_mesh(mesh_file):
    """Main analysis function with strict boundary detection."""
    print(f"Loading mesh: {mesh_file}")
    
    mesh = trimesh.load(mesh_file)
    
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        raise ValueError("Invalid mesh file")
    
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Detect coordinate system
    up_axis_idx, up_axis_name = detect_up_axis(mesh)
    print(f"Detected coordinate system: {up_axis_name}-up")
    
    up_coords = mesh.vertices[:, up_axis_idx]
    up_min = np.min(up_coords)
    up_max = np.max(up_coords)
    up_range = up_max - up_min
    
    print(f"{up_axis_name} range: {up_min:.3f} to {up_max:.3f} (height: {up_range:.3f}m)")
    
    # Get triangles
    triangles = []
    for face in mesh.faces:
        triangle = [mesh.vertices[face[i]].tolist() for i in range(3)]
        triangles.append(triangle)
    
    # Step 1: Cross-section slicing
    print("Step 1: Cross-section slicing...")
    n_slices = 20  # Reduced slices - focus on quality
    all_horizontal = []
    slices = []
    
    for i in range(n_slices):
        height = up_min + up_range * (0.2 + 0.6 * i / (n_slices - 1))  # More focused height range
        pts = slice_at_height(triangles, height, up_axis_idx)
        
        if len(pts) < 4:
            continue
        
        all_horizontal.extend(pts)
        slices.append({
            'height': height,
            'points': pts,
            'n': len(pts)
        })
        print(f"  Slice {i+1}/{n_slices} at {up_axis_name}={height:.3f}: {len(pts)} points")
    
    if len(all_horizontal) < 50:
        print("Insufficient intersection points for analysis")
        return {
            'walls': [], 'room': None, 'gaps': [],
            'slices': slices, 'angle': 0,
            'allHorizontal': all_horizontal,
            'coordinate_system': f'{up_axis_name}-up'
        }
    
    print(f"Total intersection points: {len(all_horizontal)}")
    
    # Step 2: Find dominant angle
    print("Step 2: Finding dominant angle...")
    angle = find_dominant_angle(all_horizontal)
    angle_rad = angle * math.pi / 180
    rotated = [rot_pt(p, -angle_rad) for p in all_horizontal]
    print(f"  Dominant angle: {angle:.1f}°")
    
    # Step 3: Find walls
    print("Step 3: Finding walls...")
    x_walls = find_walls(rotated, 0)
    z_walls = find_walls(rotated, 1)
    all_raw_walls = x_walls + z_walls
    print(f"  Raw walls found: {len(x_walls)} X-walls, {len(z_walls)} Z-walls")
    
    # Step 4: STRICT boundary selection - only outermost walls
    print("Step 4: Selecting boundary walls (strict)...")
    boundary_walls = select_boundary_walls_strict(all_raw_walls)
    print(f"  Strict boundary selection: {len(all_raw_walls)} → {len(boundary_walls)} boundary walls")
    
    # Step 5: Merge very close parallel walls
    print("Step 5: Merging parallel walls...")
    merged_walls = merge_walls(boundary_walls, 0.15)
    print(f"  Merged: {len(boundary_walls)} → {len(merged_walls)} walls")
    
    # Convert coordinates back to original frame
    for w in merged_walls:
        if w['axis'] == 'x':
            w['startPt'] = rot_pt([w['position'], w['start']], angle_rad)
            w['endPt'] = rot_pt([w['position'], w['end']], angle_rad)
        else:
            w['startPt'] = rot_pt([w['start'], w['position']], angle_rad)
            w['endPt'] = rot_pt([w['end'], w['position']], angle_rad)
    
    # Step 6: Build RECTILINEAR room polygon (no convex hull)
    print("Step 6: Building rectilinear room polygon...")
    room = build_rectilinear_room_polygon(merged_walls, angle_rad)
    
    # Step 7: STRICT gap detection - boundary walls only, high thresholds
    print("Step 7: Detecting openings (strict)...")
    gaps = detect_openings_strict(merged_walls, rotated, angle)
    
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    
    print(f"  Found: {len(doors)} doors, {len(windows)} windows")
    
    results = {
        'walls': merged_walls,
        'room': room,
        'gaps': gaps,
        'slices': slices,
        'angle': angle,
        'allHorizontal': all_horizontal,
        'rotated': rotated,
        'coordinate_system': f'{up_axis_name}-up'
    }
    
    # Print summary
    print(f"\n=== Analysis Summary (FINAL) ===")
    print(f"Coordinate System: {up_axis_name}-up")
    print(f"Walls: {len(merged_walls)}")
    print(f"Doors: {len(doors)}")  
    print(f"Windows: {len(windows)}")
    print(f"Total openings: {len(gaps)}")
    if room:
        print(f"Room area: {room['area']:.1f}m²")
        print(f"Room perimeter: {room['perimeter']:.1f}m")
    
    return results


def visualize_results(results, output_path):
    """Create visualization with clean styling."""
    print(f"Creating visualization: {output_path}")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Draw room polygon
    if results['room'] and results['room']['exterior']:
        room_poly = results['room']['exterior']
        x_coords = [p[0] for p in room_poly]
        y_coords = [p[1] for p in room_poly]
        ax.fill(x_coords, y_coords, color='gray', alpha=0.3, label='Room')
        ax.plot(x_coords, y_coords, color='gray', linewidth=1, alpha=0.7)
    
    # Draw walls with labels
    for wall in results['walls']:
        start = wall['startPt']
        end = wall['endPt']
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                color='white', linewidth=4, solid_capstyle='round')
        
        # Add length label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        length_text = f"{wall['length']:.2f}m"
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle > 90 or angle < -90:
            angle += 180
        
        ax.text(mid_x, mid_y, length_text, 
                ha='center', va='bottom', color='yellow',
                fontsize=10, fontweight='bold',
                rotation=angle, rotation_mode='anchor')
    
    # Draw openings
    for gap in results['gaps']:
        start = gap['start']
        end = gap['end']
        mid = gap['mid']
        
        if gap['type'] == 'door':
            color = 'cyan'
        else:  # window
            color = 'lime'
        
        # Draw arc
        radius = gap['width'] / 4
        arc = patches.Arc((mid[0], mid[1]), radius*2, radius*2,
                        theta1=0, theta2=180, color=color, linewidth=2)
        ax.add_patch(arc)
        
        # Draw gap line
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                color=color, linewidth=2, linestyle='--', alpha=0.8)
        
        # Add width label
        ax.text(mid[0], mid[1], f"{gap['width']:.2f}m", 
                ha='center', va='center', color=color,
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Add title
    coord_sys = results.get('coordinate_system', 'Z-up')
    if results['room']:
        title = f"Floor Plan Final ({coord_sys}) - {results['room']['area']:.1f}m²"
        ax.text(0.02, 0.98, title, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', color='white',
                verticalalignment='top')
        
        # Detailed stats
        doors = len([g for g in results['gaps'] if g['type'] == 'door'])
        windows = len([g for g in results['gaps'] if g['type'] == 'window'])
        
        stats = f"Area: {results['room']['area']:.1f}m²\n"
        stats += f"Walls: {len(results['walls'])}\n"
        stats += f"Doors: {doors}\n"
        stats += f"Windows: {windows}"
        
        ax.text(0.98, 0.98, stats, transform=ax.transAxes,
                fontsize=12, color='white',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    # Clean legend
    legend_elements = [
        plt.Line2D([0], [0], color='white', linewidth=4, label='Walls'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='Doors'),
        plt.Line2D([0], [0], color='lime', linewidth=2, linestyle='--', label='Windows'),
        patches.Patch(color='gray', alpha=0.3, label='Room Area')
    ]
    ax.legend(handles=legend_elements, loc='lower right', facecolor='black', edgecolor='white')
    
    # Grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', color='white')
    ax.set_ylabel('Y (meters)', color='white')
    
    # Auto-scale
    all_x = []
    all_y = []
    for wall in results['walls']:
        all_x.extend([wall['startPt'][0], wall['endPt'][0]])
        all_y.extend([wall['startPt'][1], wall['endPt'][1]])
    
    if all_x and all_y:
        margin = 0.5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()


def save_results_json(results, output_path):
    """Save results JSON."""
    compact_results = {
        'summary': {
            'coordinate_system': results.get('coordinate_system', 'Z-up'),
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'total_openings': len(results['gaps']),
            'area_m2': results['room']['area'] if results['room'] else 0,
            'perimeter_m': results['room']['perimeter'] if results['room'] else 0,
            'dominant_angle_deg': results['angle']
        },
        'walls': [
            {
                'axis': w['axis'],
                'length_m': round(w['length'], 2),
                'start': [round(w['startPt'][0], 3), round(w['startPt'][1], 3)],
                'end': [round(w['endPt'][0], 3), round(w['endPt'][1], 3)]
            }
            for w in results['walls']
        ],
        'openings': [
            {
                'type': g['type'],
                'width_m': round(g['width'], 2),
                'position': [round(g['mid'][0], 3), round(g['mid'][1], 3)]
            }
            for g in results['gaps']
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(compact_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v9 Algorithm Port - FINAL')
    parser.add_argument('mesh_file', help='Path to mesh file (.obj)')
    parser.add_argument('--output-dir', default='results/v11_v9port_tests/', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get mesh name for output files
    mesh_name = Path(args.mesh_file).stem
    dataset_name = Path(args.mesh_file).parts[-3] if len(Path(args.mesh_file).parts) > 3 else 'data'
    output_prefix = f"{dataset_name}_{mesh_name}_final"
    
    try:
        # Run analysis
        results = analyze_mesh(args.mesh_file)
        
        # Save outputs
        viz_path = output_dir / f"{output_prefix}_floorplan.png"
        json_path = output_dir / f"{output_prefix}_results.json"
        
        visualize_results(results, viz_path)
        save_results_json(results, json_path)
        
        print(f"\n=== Outputs ===")
        print(f"Visualization: {viz_path}")
        print(f"Results JSON: {json_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
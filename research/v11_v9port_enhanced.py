#!/usr/bin/env python3
"""
mesh2plan v9 Algorithm Port to Python - ENHANCED

Enhanced version with ultra-sensitive gap detection to find all openings.
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


def find_walls(rotated_points, axis, min_inliers=8, dist_thresh=0.04):
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
    
    # Create histogram with higher resolution
    n_bins = max(60, int(coord_range / 0.015))  # Increased resolution
    bin_width = coord_range / n_bins
    
    hist = np.zeros(n_bins)
    for c in coords:
        bin_idx = min(n_bins - 1, int((c - min_coord) / coord_range * n_bins))
        hist[bin_idx] += 1
    
    # Find threshold (lower to catch more walls)
    sorted_hist = np.sort(hist)
    median = sorted_hist[n_bins // 2]
    threshold = max(median * 2.5, min_inliers)  # Reduced from 3 to 2.5
    
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
                # Use slightly wider percentiles for extent
                start = near_pts[int(len(near_pts) * 0.01)]  # 1st percentile
                end = near_pts[int(len(near_pts) * 0.99)]    # 99th percentile
                
                if end - start > 0.25:  # Reduced minimum wall length
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


def filter_boundary_walls_smart(walls, rotated_points):
    """Smart boundary filtering - keep outermost walls plus any interior walls with gaps."""
    if len(walls) == 0:
        return walls
    
    # Group walls by axis
    x_walls = [w for w in walls if w['axis'] == 'x']
    z_walls = [w for w in walls if w['axis'] == 'z']
    
    boundary_walls = []
    
    # For X walls, keep outermost and any with significant gaps
    if x_walls:
        x_walls.sort(key=lambda w: w['position'])
        boundary_walls.append(x_walls[0])  # Leftmost
        if len(x_walls) > 1:
            boundary_walls.append(x_walls[-1])  # Rightmost
        
        # Add interior walls if they have good coverage or gaps
        for wall in x_walls[1:-1]:
            if wall['length'] > 2.0 or wall['nPoints'] > 50:
                boundary_walls.append(wall)
    
    # For Z walls, keep outermost and any with significant gaps
    if z_walls:
        z_walls.sort(key=lambda w: w['position'])
        boundary_walls.append(z_walls[0])  # Bottommost
        if len(z_walls) > 1:
            boundary_walls.append(z_walls[-1])  # Topmost
        
        # Add interior walls if they have good coverage or gaps
        for wall in z_walls[1:-1]:
            if wall['length'] > 2.0 or wall['nPoints'] > 50:
                boundary_walls.append(wall)
    
    return boundary_walls


def merge_walls(walls, dist=0.25):
    """Merge parallel walls that are close together."""
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


def detect_gaps_ultra_sensitive(walls, rotated_points, angle_deg):
    """Ultra-sensitive gap detection to find all openings."""
    angle_rad = angle_deg * math.pi / 180
    gaps = []
    
    for w in walls:
        axis = 0 if w['axis'] == 'x' else 1
        near = []
        
        # Much tighter tolerance for finding points near the wall
        for p in rotated_points:
            if abs(p[axis] - w['position']) < 0.04:  # Very tight
                near.append(p[1 - axis])
        
        if len(near) < 3:  # Very low minimum
            continue
        
        near.sort()
        
        # Look for gaps with very low minimum threshold
        for i in range(len(near) - 1):
            gap = near[i + 1] - near[i]
            if gap > 0.15:  # Very low threshold to catch everything
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
                
                # Enhanced classification 
                if 0.4 <= gap <= 1.1:  # Door range
                    gap_type = 'door'
                elif 1.1 < gap <= 3.0:  # Window range (expanded)
                    gap_type = 'window'
                else:
                    gap_type = 'opening'
                
                gaps.append({
                    'type': gap_type,
                    'width': gap,
                    'start': p1,
                    'end': p2,
                    'mid': mid
                })
    
    # Sort gaps by width to prioritize larger ones
    gaps.sort(key=lambda g: g['width'], reverse=True)
    
    return gaps


def build_room_polygon_improved(walls, angle_rad):
    """Improved room polygon construction using all wall endpoints."""
    if len(walls) < 3:
        return None
    
    # Collect all wall endpoints
    all_endpoints = []
    for w in walls:
        all_endpoints.extend([w['startPt'], w['endPt']])
    
    if len(all_endpoints) < 6:
        return None
    
    # Compute convex hull of all endpoints
    hull = convex_hull_2d(all_endpoints)
    
    if len(hull) < 3:
        return None
    
    exterior = hull + [hull[0]]
    return {
        'exterior': exterior,
        'area': polygon_area(hull),
        'perimeter': polygon_perimeter(hull)
    }


def convex_hull_2d(points):
    """Compute 2D convex hull using Graham scan."""
    if len(points) < 3:
        return points
    
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]


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
    """Main analysis function with enhanced gap detection."""
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
    n_slices = 25  # More slices for better coverage
    all_horizontal = []
    slices = []
    
    for i in range(n_slices):
        height = up_min + up_range * (0.15 + 0.7 * i / (n_slices - 1))
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
    
    # Step 4: Smart boundary filtering
    print("Step 4: Filtering boundary walls...")
    boundary_walls = filter_boundary_walls_smart(all_raw_walls, rotated)
    print(f"  Smart filtering: {len(all_raw_walls)} → {len(boundary_walls)} boundary walls")
    
    # Step 5: Merge parallel walls
    print("Step 5: Merging parallel walls...")
    merged_walls = merge_walls(boundary_walls, 0.25)
    print(f"  Merged: {len(boundary_walls)} → {len(merged_walls)} walls")
    
    # Convert coordinates back to original frame
    for w in merged_walls:
        if w['axis'] == 'x':
            w['startPt'] = rot_pt([w['position'], w['start']], angle_rad)
            w['endPt'] = rot_pt([w['position'], w['end']], angle_rad)
        else:
            w['startPt'] = rot_pt([w['start'], w['position']], angle_rad)
            w['endPt'] = rot_pt([w['end'], w['position']], angle_rad)
    
    # Step 6: Build room polygon
    print("Step 6: Building room polygon...")
    room = build_room_polygon_improved(merged_walls, angle_rad)
    
    # Step 7: Ultra-sensitive gap detection
    print("Step 7: Detecting openings (ultra-sensitive)...")
    gaps = detect_gaps_ultra_sensitive(merged_walls, rotated, angle)
    
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    openings = [g for g in gaps if g['type'] == 'opening']
    
    print(f"  Found: {len(doors)} doors, {len(windows)} windows, {len(openings)} other openings")
    
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
    print(f"\n=== Analysis Summary ===")
    print(f"Coordinate System: {up_axis_name}-up")
    print(f"Walls: {len(merged_walls)}")
    print(f"Doors: {len(doors)}")  
    print(f"Windows: {len(windows)}")
    print(f"Other openings: {len(openings)}")
    if room:
        print(f"Room area: {room['area']:.1f}m²")
        print(f"Room perimeter: {room['perimeter']:.1f}m")
    
    return results


def visualize_results(results, output_path):
    """Create visualization with enhanced opening display."""
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
    
    # Draw openings with different colors and labels
    for gap in results['gaps']:
        start = gap['start']
        end = gap['end']
        mid = gap['mid']
        
        if gap['type'] == 'door':
            color = 'cyan'
        elif gap['type'] == 'window':
            color = 'lime'
        else:
            color = 'orange'
        
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
        title = f"Floor Plan ({coord_sys}) - {results['room']['area']:.1f}m²"
        ax.text(0.02, 0.98, title, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', color='white',
                verticalalignment='top')
        
        # Detailed stats
        doors = len([g for g in results['gaps'] if g['type'] == 'door'])
        windows = len([g for g in results['gaps'] if g['type'] == 'window'])
        openings = len([g for g in results['gaps'] if g['type'] == 'opening'])
        
        stats = f"Area: {results['room']['area']:.1f}m²\n"
        stats += f"Walls: {len(results['walls'])}\n"
        stats += f"Doors: {doors}\n"
        stats += f"Windows: {windows}"
        if openings > 0:
            stats += f"\nOther: {openings}"
        
        ax.text(0.98, 0.98, stats, transform=ax.transAxes,
                fontsize=12, color='white',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    # Enhanced legend
    legend_elements = [
        plt.Line2D([0], [0], color='white', linewidth=4, label='Walls'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='Doors'),
        plt.Line2D([0], [0], color='lime', linewidth=2, linestyle='--', label='Windows'),
        plt.Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Other Openings'),
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
    """Save enhanced results JSON."""
    compact_results = {
        'summary': {
            'coordinate_system': results.get('coordinate_system', 'Z-up'),
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'other_openings': len([g for g in results['gaps'] if g['type'] == 'opening']),
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
    parser = argparse.ArgumentParser(description='mesh2plan v9 Algorithm Port - ENHANCED')
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
    output_prefix = f"{dataset_name}_{mesh_name}_enhanced"
    
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
#!/usr/bin/env python3
"""
mesh2plan v20 - Hough Line Detection

Treats the 2D point density as an image, applies edge detection,
then uses Hough transform to find wall lines. Classic CV approach.

Advantages: No histogram binning assumptions, finds walls at any angle,
naturally handles non-rectilinear rooms.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage


def detect_up_axis(mesh):
    vertices = mesh.vertices
    ranges = [np.ptp(vertices[:, i]) for i in range(3)]
    axis_names = ['X', 'Y', 'Z']
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
    else:
        return np.argmin(ranges), axis_names[np.argmin(ranges)]


def project_vertices(mesh, up_axis_idx):
    verts = mesh.vertices
    if up_axis_idx == 1:
        return verts[:, 0], verts[:, 2]
    elif up_axis_idx == 2:
        return verts[:, 0], verts[:, 1]
    else:
        return verts[:, 1], verts[:, 2]


def rot_pt(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [p[0]*c - p[1]*s, p[0]*s + p[1]*c]


def build_density_image(rx, rz, cell_size=0.02):
    """Build a high-res density image from projected vertices."""
    x_min, x_max = rx.min() - 0.2, rx.max() + 0.2
    z_min, z_max = rz.min() - 0.2, rz.max() + 0.2
    
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    
    img = np.zeros((nz, nx), dtype=np.float32)
    
    xi = ((rx - x_min) / cell_size).astype(int)
    zi = ((rz - z_min) / cell_size).astype(int)
    
    # Clip to bounds
    xi = np.clip(xi, 0, nx - 1)
    zi = np.clip(zi, 0, nz - 1)
    
    # Accumulate
    np.add.at(img, (zi, xi), 1)
    
    return img, x_min, z_min, cell_size, nx, nz


def find_dominant_angle_from_image(img):
    """Find dominant wall angle using Hough on the density image."""
    # Normalize to 0-255
    img_norm = np.clip(img / max(np.percentile(img[img > 0], 95), 1) * 255, 0, 255).astype(np.uint8)
    
    # Edge detection
    edges = cv2.Canny(img_norm, 30, 100)
    
    # Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return 0.0
    
    # Collect angles, fold to 0-90
    angles = []
    for line in lines:
        theta = line[0][1] * 180 / np.pi
        angles.append(theta % 90)
    
    # Histogram peak
    hist, bins = np.histogram(angles, bins=90, range=(0, 90))
    peak = np.argmax(hist)
    return (bins[peak] + bins[peak + 1]) / 2


def detect_walls_hough(img, x_min, z_min, cell_size, angle_deg):
    """Use Hough line transform to find wall lines."""
    # Normalize
    img_norm = np.clip(img / max(np.percentile(img[img > 0], 90), 1) * 255, 0, 255).astype(np.uint8)
    
    # Blur slightly
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    
    # Edge detection with tight thresholds to get wall edges
    edges = cv2.Canny(img_blur, 50, 150)
    
    # Probabilistic Hough — gives line segments
    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=50,
                            minLineLength=int(0.5 / cell_size),  # min 0.5m
                            maxLineGap=int(0.3 / cell_size))     # bridge 0.3m gaps
    
    if lines is None:
        return [], edges
    
    angle_rad = angle_deg * math.pi / 180
    
    walls = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Convert pixel to world coords
        wx1 = x_min + x1 * cell_size
        wz1 = z_min + y1 * cell_size
        wx2 = x_min + x2 * cell_size
        wz2 = z_min + y2 * cell_size
        
        length = math.sqrt((wx2-wx1)**2 + (wz2-wz1)**2)
        
        if length < 0.3:
            continue
        
        # Determine angle of this line segment
        seg_angle = math.atan2(wz2-wz1, wx2-wx1) * 180 / math.pi
        
        # Check if it's roughly axis-aligned (within 15° of 0, 90, 180, -90)
        angle_mod = seg_angle % 90
        if angle_mod > 45:
            angle_mod = 90 - angle_mod
        
        if angle_mod > 15:
            continue  # Skip non-axis-aligned segments
        
        # Classify as X-wall (vertical) or Z-wall (horizontal)
        dx = abs(wx2 - wx1)
        dz = abs(wz2 - wz1)
        
        if dz > dx:
            # Vertical = X-wall
            axis = 'x'
            position = (wx1 + wx2) / 2
            start = min(wz1, wz2)
            end = max(wz1, wz2)
        else:
            # Horizontal = Z-wall
            axis = 'z'
            position = (wz1 + wz2) / 2
            start = min(wx1, wx2)
            end = max(wx1, wx2)
        
        walls.append({
            'axis': axis,
            'position': position,
            'start': start,
            'end': end,
            'length': length,
            'nPoints': 0,
            'startPt': [wx1, wz1],
            'endPt': [wx2, wz2],
        })
    
    return walls, edges


def cluster_walls(walls, axis, dist_threshold=0.15):
    """Cluster nearby parallel walls into single wall lines."""
    axis_walls = sorted([w for w in walls if w['axis'] == axis], 
                       key=lambda w: w['position'])
    
    if not axis_walls:
        return []
    
    clusters = []
    current_cluster = [axis_walls[0]]
    
    for w in axis_walls[1:]:
        if abs(w['position'] - current_cluster[-1]['position']) < dist_threshold:
            current_cluster.append(w)
        else:
            clusters.append(current_cluster)
            current_cluster = [w]
    clusters.append(current_cluster)
    
    # Merge each cluster
    merged = []
    for cluster in clusters:
        total_length = sum(w['length'] for w in cluster)
        avg_pos = sum(w['position'] * w['length'] for w in cluster) / total_length
        min_start = min(w['start'] for w in cluster)
        max_end = max(w['end'] for w in cluster)
        
        merged.append({
            'axis': axis,
            'position': avg_pos,
            'start': min_start,
            'end': max_end,
            'length': max_end - min_start,
            'nPoints': len(cluster),
            'total_segment_length': total_length,
        })
    
    return merged


def select_boundary_walls(x_clusters, z_clusters):
    """Select boundary walls: outermost per axis + partial step walls."""
    boundary = []
    
    # X walls: keep outermost only
    if len(x_clusters) >= 2:
        boundary.append(x_clusters[0])   # Left
        boundary.append(x_clusters[-1])  # Right
        for w in x_clusters[1:-1]:
            print(f"    Rejected interior X wall at {w['position']:.2f}")
    elif x_clusters:
        boundary.extend(x_clusters)
    
    full_width = (x_clusters[-1]['position'] - x_clusters[0]['position']) if len(x_clusters) >= 2 else 999
    
    # Z walls: keep bottom (longest near lowest), main top (widest), and partial step walls
    if len(z_clusters) >= 2:
        # Bottom: longest wall at lowest position
        bottom_pos = z_clusters[0]['position']
        bottom_candidates = [w for w in z_clusters if abs(w['position'] - bottom_pos) < 0.3]
        bottom = max(bottom_candidates, key=lambda w: w['length'])
        boundary.append(bottom)
        
        remaining = [w for w in z_clusters if w is not bottom and 
                    abs(w['position'] - bottom['position']) > 0.3]
        
        # Main top: full-width wall closest to bottom (but above it)
        main_top = None
        for w in remaining:
            span = w['length'] / full_width if full_width > 0 else 0
            if span > 0.6:
                if main_top is None or w['position'] < main_top['position']:
                    main_top = w
        
        if main_top:
            boundary.append(main_top)
        
        # Partial step walls (above main top)
        for w in remaining:
            if w is main_top:
                continue
            span = w['length'] / full_width if full_width > 0 else 0
            if 0.15 < span < 0.6 and main_top and w['position'] > main_top['position']:
                boundary.append(w)
                print(f"    Step wall at Z={w['position']:.2f} (span={span:.0%})")
    elif z_clusters:
        boundary.extend(z_clusters)
    
    return boundary


def build_room_polygon(boundary_walls, angle_rad):
    """Build L-shape or rectangle from boundary walls."""
    x_walls = sorted([w for w in boundary_walls if w['axis'] == 'x'], 
                    key=lambda w: w['position'])
    z_walls = sorted([w for w in boundary_walls if w['axis'] == 'z'], 
                    key=lambda w: w['position'])
    
    if len(x_walls) < 2 or len(z_walls) < 1:
        return None
    
    left_x = x_walls[0]['position']
    right_x = x_walls[-1]['position']
    bottom_z = z_walls[0]['position']
    
    # Find main top and extension
    full_width = right_x - left_x
    main_top_z = None
    ext_top_z = None
    step_x = None
    
    for w in z_walls[1:] if len(z_walls) > 1 else []:
        span = w['length'] / full_width
        if span > 0.6 and main_top_z is None:
            main_top_z = w['position']
        elif 0.15 < span < 0.6 and main_top_z and w['position'] > main_top_z:
            ext_top_z = w['position']
            step_x = w['start']
    
    if main_top_z is None:
        # Fallback: use the highest Z wall
        main_top_z = z_walls[-1]['position']
    
    if ext_top_z and step_x:
        # L-shape
        polygon = [
            [left_x, bottom_z],
            [right_x, bottom_z],
            [right_x, ext_top_z],
            [step_x, ext_top_z],
            [step_x, main_top_z],
            [left_x, main_top_z],
        ]
    else:
        # Rectangle
        polygon = [
            [left_x, bottom_z],
            [right_x, bottom_z],
            [right_x, main_top_z],
            [left_x, main_top_z],
        ]
    
    room_corners = [rot_pt(p, angle_rad) for p in polygon]
    exterior = room_corners + [room_corners[0]]
    
    n = len(room_corners)
    area = abs(sum(room_corners[i][0]*room_corners[(i+1)%n][1] - 
                   room_corners[(i+1)%n][0]*room_corners[i][1] for i in range(n))) / 2
    perimeter = sum(math.sqrt((room_corners[(i+1)%n][0]-room_corners[i][0])**2 + 
                              (room_corners[(i+1)%n][1]-room_corners[i][1])**2) 
                   for i in range(n))
    
    return {
        'exterior': exterior,
        'area': area,
        'perimeter': perimeter,
        'polygon_rotated': polygon,
    }


def detect_openings(boundary_walls, rotated_points, angle_rad):
    """Detect openings (gaps) in boundary walls."""
    gaps = []
    pts = np.array(rotated_points)
    
    for w in boundary_walls:
        axis_idx = 0 if w['axis'] == 'x' else 1
        other_idx = 1 - axis_idx
        
        # Points near this wall
        near_mask = np.abs(pts[:, axis_idx] - w['position']) < 0.12
        near = pts[near_mask][:, other_idx]
        
        if len(near) < 5:
            continue
        
        near = np.sort(near)
        
        # Find gaps
        for i in range(len(near) - 1):
            gap = near[i+1] - near[i]
            if 0.4 < gap < 3.0:
                mid_other = (near[i] + near[i+1]) / 2
                
                if w['axis'] == 'x':
                    mid_pt = rot_pt([w['position'], mid_other], angle_rad)
                    start_pt = rot_pt([w['position'], near[i]], angle_rad)
                    end_pt = rot_pt([w['position'], near[i+1]], angle_rad)
                else:
                    mid_pt = rot_pt([mid_other, w['position']], angle_rad)
                    start_pt = rot_pt([near[i], w['position']], angle_rad)
                    end_pt = rot_pt([near[i+1], w['position']], angle_rad)
                
                gap_type = 'door' if gap < 1.2 else 'window'
                
                gaps.append({
                    'type': gap_type,
                    'width': gap,
                    'mid': mid_pt,
                    'start': start_pt,
                    'end': end_pt,
                })
    
    return gaps


def analyze_mesh(mesh_file):
    """Main analysis using Hough line detection."""
    print(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    up_axis_idx, up_axis_name = detect_up_axis(mesh)
    print(f"Coordinate system: {up_axis_name}-up")
    
    up_coords = mesh.vertices[:, up_axis_idx]
    up_min, up_max = np.min(up_coords), np.max(up_coords)
    up_range = up_max - up_min
    
    # Project vertices (mid-height only)
    x_raw, z_raw = project_vertices(mesh, up_axis_idx)
    height_mask = (up_coords >= up_min + up_range * 0.15) & (up_coords <= up_min + up_range * 0.85)
    x_mid, z_mid = x_raw[height_mask], z_raw[height_mask]
    print(f"Vertices: {len(x_raw)} total, {len(x_mid)} mid-height")
    
    # Build density image
    print("Building density image...")
    cell_size = 0.02  # 2cm resolution
    img, x_min, z_min, _, nx, nz = build_density_image(x_mid, z_mid, cell_size)
    print(f"  Image: {nx}x{nz} ({cell_size*100:.0f}cm/pixel)")
    
    # Find dominant angle
    print("Finding dominant angle...")
    angle = find_dominant_angle_from_image(img)
    angle_rad = angle * math.pi / 180
    print(f"  Angle: {angle:.1f}°")
    
    # Rotate points
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    # Rebuild image in rotated frame
    img_rot, x_min_r, z_min_r, _, nx_r, nz_r = build_density_image(rx, rz, cell_size)
    
    # Detect walls via Hough
    print("Detecting walls (Hough)...")
    raw_walls, edges = detect_walls_hough(img_rot, x_min_r, z_min_r, cell_size, angle)
    print(f"  Raw Hough segments: {len(raw_walls)}")
    
    # Cluster parallel walls
    x_clusters = cluster_walls(raw_walls, 'x', 0.15)
    z_clusters = cluster_walls(raw_walls, 'z', 0.15)
    print(f"  Clustered: {len(x_clusters)} X-walls, {len(z_clusters)} Z-walls")
    
    for w in x_clusters:
        print(f"    X={w['position']:.2f}, len={w['length']:.2f}m, segments={w['nPoints']}")
    for w in z_clusters:
        print(f"    Z={w['position']:.2f}, len={w['length']:.2f}m, segments={w['nPoints']}")
    
    # Select boundary walls
    print("Selecting boundary walls...")
    boundary = select_boundary_walls(x_clusters, z_clusters)
    print(f"  Boundary walls: {len(boundary)}")
    
    # Build room polygon
    print("Building room polygon...")
    room = build_room_polygon(boundary, angle_rad)
    
    # Add startPt/endPt to boundary walls for visualization
    for w in boundary:
        if 'startPt' not in w:
            if w['axis'] == 'x':
                w['startPt'] = rot_pt([w['position'], w['start']], angle_rad)
                w['endPt'] = rot_pt([w['position'], w['end']], angle_rad)
            else:
                w['startPt'] = rot_pt([w['start'], w['position']], angle_rad)
                w['endPt'] = rot_pt([w['end'], w['position']], angle_rad)
    
    # Detect openings
    print("Detecting openings...")
    rotated_pts = list(zip(rx.tolist(), rz.tolist()))
    gaps = detect_openings(boundary, rotated_pts, angle_rad)
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    print(f"  Openings: {len(doors)} doors, {len(windows)} windows")
    
    results = {
        'walls': boundary,
        'room': room,
        'gaps': gaps,
        'angle': angle,
        'coordinate_system': f'{up_axis_name}-up',
        'edges_image': edges,
    }
    
    print(f"\n=== v20 Hough Lines Summary ===")
    print(f"Walls: {len(boundary)}")
    print(f"Doors: {len(doors)}, Windows: {len(windows)}")
    if room:
        print(f"Area: {room['area']:.1f}m², Perimeter: {room['perimeter']:.1f}m")
    
    return results


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Edge detection image
    if results.get('edges_image') is not None:
        axes[0].imshow(results['edges_image'], cmap='gray', origin='lower')
        axes[0].set_title('Canny Edges (rotated density)', color='white', fontsize=14)
        axes[0].set_xlabel('X pixels')
        axes[0].set_ylabel('Z pixels')
    
    # Right: Floor plan
    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    if results['room'] and results['room']['exterior']:
        room_poly = results['room']['exterior']
        ax.fill([p[0] for p in room_poly], [p[1] for p in room_poly], 
                color='gray', alpha=0.3)
        ax.plot([p[0] for p in room_poly], [p[1] for p in room_poly], 
                color='gray', linewidth=1, alpha=0.7)
    
    for wall in results['walls']:
        start, end = wall['startPt'], wall['endPt']
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                color='white', linewidth=4, solid_capstyle='round')
        mid_x, mid_y = (start[0]+end[0])/2, (start[1]+end[1])/2
        ax.text(mid_x, mid_y, f"{wall['length']:.2f}m", 
                ha='center', va='bottom', color='yellow', fontsize=9, fontweight='bold')
    
    for gap in results['gaps']:
        color = 'cyan' if gap['type'] == 'door' else 'lime'
        ax.plot([gap['start'][0], gap['end'][0]], [gap['start'][1], gap['end'][1]], 
                color=color, linewidth=2, linestyle='--', alpha=0.8)
        ax.text(gap['mid'][0], gap['mid'][1], f"{gap['width']:.2f}m",
                ha='center', va='center', color=color, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    if results['room']:
        ax.set_title(f"v20 Hough Lines — {results['room']['area']:.1f}m²", 
                    color='white', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v20_hough_lines',
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'area_m2': results['room']['area'] if results['room'] else 0,
        },
        'walls': [{'axis': w['axis'], 'length_m': round(w['length'], 2),
                   'start': [round(w['startPt'][0], 3), round(w['startPt'][1], 3)],
                   'end': [round(w['endPt'][0], 3), round(w['endPt'][1], 3)]}
                  for w in results['walls']],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v20 - Hough Lines')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v20_hough/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_name = Path(args.mesh_file).stem
    prefix = f"v20_{mesh_name}"
    
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

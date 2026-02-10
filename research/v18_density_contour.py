#!/usr/bin/env python3
"""
mesh2plan v18 - Density Contour Approach

Instead of detecting walls via histogram peaks, this approach:
1. Projects all mesh vertices to 2D (top-down)
2. Builds a fine density grid
3. Finds the room boundary as the contour of the occupied region
4. Simplifies the contour into a rectilinear polygon (Manhattan constraint)
5. Detects openings by finding gaps in wall-adjacent point density

No wall histogram detection needed — the room shape comes directly
from the point cloud footprint.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
from pathlib import Path
import math
from scipy import ndimage
import cv2


def detect_up_axis(mesh):
    """Detect the up-axis by finding which coordinate has reasonable room height."""
    vertices = mesh.vertices
    ranges = []
    axis_names = ['X', 'Y', 'Z']
    
    for i in range(3):
        coords = vertices[:, i]
        range_val = np.max(coords) - np.min(coords)
        ranges.append(range_val)
    
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
    else:
        min_idx = np.argmin(ranges)
        return min_idx, axis_names[min_idx]


def find_dominant_angle(points_2d):
    """Find dominant wall angle using gradient direction histogram."""
    pts = np.array(points_2d)
    
    # Create a density image
    cell = 0.02  # 2cm resolution
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    
    nx = int((x_max - x_min) / cell) + 1
    ny = int((y_max - y_min) / cell) + 1
    
    if nx < 10 or ny < 10:
        return 0.0
    
    img = np.zeros((ny, nx), dtype=np.float32)
    for p in pts:
        xi = int((p[0] - x_min) / cell)
        yi = int((p[1] - y_min) / cell)
        if 0 <= xi < nx and 0 <= yi < ny:
            img[yi, xi] += 1
    
    # Blur and find edges
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    
    # Use gradient directions
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitudes = np.sqrt(gx**2 + gy**2)
    angles = np.arctan2(gy, gx) * 180 / np.pi
    
    # Only consider strong gradients
    mask = magnitudes > np.percentile(magnitudes, 80)
    strong_angles = angles[mask]
    
    # Fold to 0-90 range (walls are perpendicular)
    folded = strong_angles % 90
    
    # Histogram to find dominant angle
    hist, bins = np.histogram(folded, bins=90, range=(0, 90))
    peak_bin = np.argmax(hist)
    dominant = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    
    # Convert back to rotation angle
    return dominant


def rot_pt(p, angle):
    """Rotate a 2D point."""
    c = math.cos(angle)
    s = math.sin(angle)
    return [p[0] * c - p[1] * s, p[0] * s + p[1] * c]


def project_vertices(mesh, up_axis_idx):
    """Project mesh vertices to 2D (top-down view)."""
    verts = mesh.vertices
    if up_axis_idx == 1:  # Y-up
        return verts[:, 0], verts[:, 2]
    elif up_axis_idx == 2:  # Z-up
        return verts[:, 0], verts[:, 1]
    else:  # X-up
        return verts[:, 1], verts[:, 2]


def build_density_grid(rx, rz, cell_size=0.05):
    """Build a 2D density grid from projected vertices."""
    x_min, x_max = rx.min() - 0.1, rx.max() + 0.1
    z_min, z_max = rz.min() - 0.1, rz.max() + 0.1
    
    x_bins = np.arange(x_min, x_max + cell_size, cell_size)
    z_bins = np.arange(z_min, z_max + cell_size, cell_size)
    
    hist, xedges, zedges = np.histogram2d(rx, rz, bins=[x_bins, z_bins])
    
    return hist, xedges, zedges, x_min, z_min, cell_size


def find_room_mask(hist, threshold_percentile=30):
    """Find the room region in the density grid using morphological operations."""
    # Threshold: cells above a density threshold are "room"
    threshold = np.percentile(hist[hist > 0], threshold_percentile) if np.any(hist > 0) else 1
    binary = (hist > threshold).astype(np.uint8)
    
    # Close small gaps (walls between rooms shouldn't break the region)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Fill holes
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    
    # Keep largest connected component
    labeled, n_features = ndimage.label(binary)
    if n_features > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
        largest = np.argmax(sizes) + 1
        binary = (labeled == largest).astype(np.uint8)
    
    # Erode slightly to remove thin extensions (hallway bleed through doors)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel_erode)
    
    # Re-dilate to restore room size
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel_erode)
    
    # Fill holes again after erosion
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    
    # Keep largest again
    labeled, n_features = ndimage.label(binary)
    if n_features > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
        largest = np.argmax(sizes) + 1
        binary = (labeled == largest).astype(np.uint8)
    
    return binary


def contour_to_polygon(binary, xedges, zedges, cell_size):
    """Extract the room contour and convert to world coordinates."""
    # Find contours using OpenCV
    contours, _ = cv2.findContours(binary.T.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Convert pixel coordinates to world coordinates
    # contour points are in (col, row) = (z_idx, x_idx) because we transposed
    world_pts = []
    for pt in contour:
        z_idx, x_idx = pt[0]  # After transpose: col=z, row=x
        x = xedges[0] + x_idx * cell_size
        z = zedges[0] + z_idx * cell_size
        world_pts.append([x, z])
    
    return world_pts


def simplify_to_rectilinear(polygon_pts, angle_tolerance=15):
    """Simplify a polygon to rectilinear (Manhattan) form.
    
    Snaps edges to be axis-aligned (horizontal or vertical) and
    removes small jogs/steps.
    """
    if len(polygon_pts) < 4:
        return polygon_pts
    
    pts = np.array(polygon_pts)
    
    # Douglas-Peucker simplification first
    epsilon = 0.1  # 10cm tolerance
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts_cv, epsilon, True)
    pts = simplified.reshape(-1, 2)
    
    if len(pts) < 4:
        return pts.tolist()
    
    # Now snap each edge to be axis-aligned
    rectilinear = [pts[0].tolist()]
    
    for i in range(1, len(pts)):
        prev = rectilinear[-1]
        curr = pts[i].tolist()
        
        dx = abs(curr[0] - prev[0])
        dz = abs(curr[1] - prev[1])
        
        if dx < 0.15:
            # Nearly vertical → snap X
            curr[0] = prev[0]
        elif dz < 0.15:
            # Nearly horizontal → snap Z
            curr[1] = prev[1]
        else:
            # Diagonal → insert a corner (go horizontal first, then vertical)
            rectilinear.append([curr[0], prev[1]])
        
        rectilinear.append(curr)
    
    # Remove very short segments (jogs < 20cm)
    cleaned = [rectilinear[0]]
    for i in range(1, len(rectilinear)):
        dx = abs(rectilinear[i][0] - cleaned[-1][0])
        dz = abs(rectilinear[i][1] - cleaned[-1][1])
        if dx > 0.2 or dz > 0.2:
            cleaned.append(rectilinear[i])
    
    # Final pass: merge collinear segments
    if len(cleaned) < 3:
        return cleaned
    
    final = [cleaned[0]]
    for i in range(1, len(cleaned) - 1):
        prev = final[-1]
        curr = cleaned[i]
        nxt = cleaned[i + 1]
        
        # Skip if collinear (all same X or all same Z)
        same_x = abs(prev[0] - curr[0]) < 0.05 and abs(curr[0] - nxt[0]) < 0.05
        same_z = abs(prev[1] - curr[1]) < 0.05 and abs(curr[1] - nxt[1]) < 0.05
        
        if not (same_x or same_z):
            final.append(curr)
    
    final.append(cleaned[-1])
    
    return final


def detect_openings_from_density(binary, xedges, zedges, cell_size, polygon_pts):
    """Detect doors and windows by finding gaps in the room boundary wall density."""
    # For each edge of the polygon, check point density along it
    # Gaps in density = openings
    gaps = []
    
    for i in range(len(polygon_pts)):
        j = (i + 1) % len(polygon_pts)
        p1 = polygon_pts[i]
        p2 = polygon_pts[j]
        
        edge_len = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if edge_len < 0.5:
            continue
        
        # Sample points along this edge, slightly outside the room
        n_samples = max(10, int(edge_len / 0.05))
        
        # Normal direction (pointing outward)
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        nx_dir = -dz / edge_len
        nz_dir = dx / edge_len
        
        # Check density along the wall (at the wall position)
        wall_density = []
        for k in range(n_samples):
            t = k / (n_samples - 1)
            x = p1[0] + t * dx
            z = p1[1] + t * dz
            
            # Check density at wall position (and just inside)
            total = 0
            for offset in [0, -0.05, -0.1]:
                sx = x + nx_dir * offset
                sz = z + nz_dir * offset
                
                xi = int((sx - xedges[0]) / cell_size)
                zi = int((sz - zedges[0]) / cell_size)
                
                if 0 <= xi < binary.shape[0] and 0 <= zi < binary.shape[1]:
                    total += binary[xi, zi]
            
            wall_density.append((t * edge_len, total))
        
        # Find gaps (runs of zero density)
        in_gap = False
        gap_start = 0
        
        for pos, density in wall_density:
            if density == 0 and not in_gap:
                in_gap = True
                gap_start = pos
            elif density > 0 and in_gap:
                gap_width = pos - gap_start
                if 0.4 < gap_width < 3.0:
                    gap_mid_t = (gap_start + pos) / 2 / edge_len
                    mid_x = p1[0] + gap_mid_t * dx
                    mid_z = p1[1] + gap_mid_t * dz
                    
                    gap_type = 'door' if gap_width < 1.2 else 'window'
                    
                    start_t = gap_start / edge_len
                    end_t = pos / edge_len
                    
                    gaps.append({
                        'type': gap_type,
                        'width': gap_width,
                        'start': [p1[0] + start_t * dx, p1[1] + start_t * dz],
                        'end': [p1[0] + end_t * dx, p1[1] + end_t * dz],
                        'mid': [mid_x, mid_z],
                    })
                in_gap = False
    
    return gaps


def polygon_area(pts):
    """Shoelface formula."""
    n = len(pts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2


def polygon_perimeter(pts):
    perimeter = 0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return perimeter


def analyze_mesh(mesh_file):
    """Main analysis using density contour approach."""
    print(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        raise ValueError("Invalid mesh file")
    
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    up_axis_idx, up_axis_name = detect_up_axis(mesh)
    print(f"Detected coordinate system: {up_axis_name}-up")
    
    up_coords = mesh.vertices[:, up_axis_idx]
    up_min, up_max = np.min(up_coords), np.max(up_coords)
    up_range = up_max - up_min
    print(f"{up_axis_name} range: {up_min:.3f} to {up_max:.3f} (height: {up_range:.3f}m)")
    
    # Step 1: Project vertices to 2D
    print("Step 1: Projecting vertices to 2D...")
    x_raw, z_raw = project_vertices(mesh, up_axis_idx)
    
    # Optional: filter to mid-height only (exclude floor/ceiling artifacts)
    height_mask = (up_coords >= up_min + up_range * 0.15) & (up_coords <= up_min + up_range * 0.85)
    x_mid = x_raw[height_mask]
    z_mid = z_raw[height_mask]
    print(f"  Total vertices: {len(x_raw)}, mid-height: {len(x_mid)}")
    
    # Step 2: Find dominant angle
    print("Step 2: Finding dominant angle...")
    pts_2d = list(zip(x_mid.tolist(), z_mid.tolist()))
    angle = find_dominant_angle(pts_2d)
    angle_rad = angle * math.pi / 180
    print(f"  Dominant angle: {angle:.1f}°")
    
    # Rotate all vertices
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    # Step 3: Build density grid
    print("Step 3: Building density grid...")
    cell_size = 0.05  # 5cm cells
    hist, xedges, zedges, x_min, z_min, _ = build_density_grid(rx, rz, cell_size)
    print(f"  Grid size: {hist.shape}")
    
    # Step 4: Find room mask
    print("Step 4: Finding room region...")
    room_mask = find_room_mask(hist)
    occupied_cells = np.sum(room_mask > 0)
    estimated_area = occupied_cells * cell_size * cell_size
    print(f"  Occupied cells: {occupied_cells}, estimated area: {estimated_area:.1f}m²")
    
    # Step 5: Extract contour and simplify
    print("Step 5: Extracting room contour...")
    raw_polygon = contour_to_polygon(room_mask, xedges, zedges, cell_size)
    
    if raw_polygon is None:
        print("  ERROR: No contour found")
        return {'walls': [], 'room': None, 'gaps': [], 'angle': angle,
                'coordinate_system': f'{up_axis_name}-up'}
    
    print(f"  Raw contour: {len(raw_polygon)} points")
    
    # Step 6: Simplify to rectilinear
    print("Step 6: Simplifying to rectilinear polygon...")
    rect_polygon = simplify_to_rectilinear(raw_polygon)
    print(f"  Simplified: {len(rect_polygon)} vertices")
    
    # Transform back to original coordinates
    room_corners = [rot_pt(p, angle_rad) for p in rect_polygon]
    exterior = room_corners + [room_corners[0]]
    
    area = polygon_area(room_corners)
    perimeter = polygon_perimeter(room_corners)
    
    room = {
        'exterior': exterior,
        'area': area,
        'perimeter': perimeter
    }
    
    print(f"  Room: {area:.1f}m², {perimeter:.1f}m perimeter")
    
    # Step 7: Detect openings
    print("Step 7: Detecting openings...")
    gaps = detect_openings_from_density(room_mask, xedges, zedges, cell_size, rect_polygon)
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    print(f"  Found: {len(doors)} doors, {len(windows)} windows")
    
    # Convert gaps to original coordinates
    for g in gaps:
        g['start'] = rot_pt(g['start'], angle_rad)
        g['end'] = rot_pt(g['end'], angle_rad)
        g['mid'] = rot_pt(g['mid'], angle_rad)
    
    # Build wall list from polygon edges
    walls = []
    for i in range(len(rect_polygon)):
        j = (i + 1) % len(rect_polygon)
        p1 = rect_polygon[i]
        p2 = rect_polygon[j]
        
        length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if length < 0.1:
            continue
        
        # Determine axis
        dx = abs(p2[0] - p1[0])
        dz = abs(p2[1] - p1[1])
        
        if dx > dz:
            axis = 'z'
            position = (p1[1] + p2[1]) / 2
        else:
            axis = 'x'
            position = (p1[0] + p2[0]) / 2
        
        walls.append({
            'axis': axis,
            'position': position,
            'start': min(p1[0] if axis == 'x' else p1[1], p2[0] if axis == 'x' else p2[1]),
            'end': max(p1[0] if axis == 'x' else p1[1], p2[0] if axis == 'x' else p2[1]),
            'length': length,
            'nPoints': 0,
            'startPt': rot_pt(p1, angle_rad),
            'endPt': rot_pt(p2, angle_rad),
        })
    
    results = {
        'walls': walls,
        'room': room,
        'gaps': gaps,
        'angle': angle,
        'coordinate_system': f'{up_axis_name}-up',
        'rect_polygon_rotated': rect_polygon,
        'raw_polygon_rotated': raw_polygon,
    }
    
    print(f"\n=== Analysis Summary (v18 Density Contour) ===")
    print(f"Coordinate System: {up_axis_name}-up")
    print(f"Walls: {len(walls)}")
    print(f"Doors: {len(doors)}")
    print(f"Windows: {len(windows)}")
    print(f"Room area: {area:.1f}m²")
    print(f"Room perimeter: {perimeter:.1f}m")
    
    return results


def visualize_results(results, output_path):
    """Create visualization."""
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
        
        color = 'cyan' if gap['type'] == 'door' else 'lime'
        
        radius = gap['width'] / 4
        arc = patches.Arc((mid[0], mid[1]), radius*2, radius*2,
                        theta1=0, theta2=180, color=color, linewidth=2)
        ax.add_patch(arc)
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                color=color, linewidth=2, linestyle='--', alpha=0.8)
        
        ax.text(mid[0], mid[1], f"{gap['width']:.2f}m", 
                ha='center', va='center', color=color,
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Title and stats
    coord_sys = results.get('coordinate_system', 'Z-up')
    if results['room']:
        title = f"v18 Density Contour ({coord_sys}) - {results['room']['area']:.1f}m²"
        ax.text(0.02, 0.98, title, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', color='white',
                verticalalignment='top')
        
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
    
    legend_elements = [
        plt.Line2D([0], [0], color='white', linewidth=4, label='Walls'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='Doors'),
        plt.Line2D([0], [0], color='lime', linewidth=2, linestyle='--', label='Windows'),
        patches.Patch(color='gray', alpha=0.3, label='Room Area')
    ]
    ax.legend(handles=legend_elements, loc='lower right', facecolor='black', edgecolor='white')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', color='white')
    ax.set_ylabel('Y (meters)', color='white')
    
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
    compact_results = {
        'summary': {
            'approach': 'v18_density_contour',
            'coordinate_system': results.get('coordinate_system', 'Z-up'),
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
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
    parser = argparse.ArgumentParser(description='mesh2plan v18 - Density Contour')
    parser.add_argument('mesh_file', help='Path to mesh file (.obj)')
    parser.add_argument('--output-dir', default='results/v18_density/', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_name = Path(args.mesh_file).stem
    dataset_name = Path(args.mesh_file).parts[-3] if len(Path(args.mesh_file).parts) > 3 else 'data'
    output_prefix = f"{dataset_name}_{mesh_name}_v18"
    
    try:
        results = analyze_mesh(args.mesh_file)
        
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

#!/usr/bin/env python3
"""
mesh2plan v21 - Raycasting Approach

Finds room boundaries by casting rays outward from the room centroid.
Each ray finds the first "wall hit" — a dense band of points.
The resulting hit positions trace the room boundary directly.

Advantages: No wall detection needed, naturally handles any room shape,
immune to interior furniture/artifacts (rays pass through sparse objects).
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


def find_dominant_angle(pts_2d, cell=0.02):
    """Find dominant angle from 2D point gradient directions."""
    pts = np.array(pts_2d)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    
    nx = int((x_max - x_min) / cell) + 1
    ny = int((y_max - y_min) / cell) + 1
    
    img = np.zeros((ny, nx), dtype=np.float32)
    xi = np.clip(((pts[:, 0] - x_min) / cell).astype(int), 0, nx-1)
    yi = np.clip(((pts[:, 1] - y_min) / cell).astype(int), 0, ny-1)
    np.add.at(img, (yi, xi), 1)
    
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    
    mask = mag > np.percentile(mag, 80)
    folded = ang[mask] % 90
    
    hist, bins = np.histogram(folded, bins=90, range=(0, 90))
    peak = np.argmax(hist)
    return (bins[peak] + bins[peak+1]) / 2


def cast_rays(density_img, x_min, z_min, cell_size, center_x, center_z, n_rays=360):
    """Cast rays from center outward through density image. 
    Returns hit points where rays hit dense regions (walls)."""
    nz, nx = density_img.shape
    
    # Threshold for "wall" density
    wall_threshold = np.percentile(density_img[density_img > 0], 75)
    
    hits = []
    
    for i in range(n_rays):
        angle = 2 * math.pi * i / n_rays
        dx = math.cos(angle)
        dz = math.sin(angle)
        
        # March along ray
        found_interior = False
        hit_pos = None
        
        for step in range(1, 500):
            dist = step * cell_size * 2  # Step size = 2 cells
            rx = center_x + dx * dist
            rz = center_z + dz * dist
            
            # Convert to pixel coords
            px = int((rx - x_min) / cell_size)
            pz = int((rz - z_min) / cell_size)
            
            if px < 0 or px >= nx or pz < 0 or pz >= nz:
                # Hit image boundary — use last known position
                if hit_pos is None:
                    hit_pos = (rx - dx * cell_size * 2, rz - dz * cell_size * 2)
                break
            
            density = density_img[pz, px]
            
            if density > wall_threshold and not found_interior:
                # First wall hit from center — this is likely an interior wall or furniture
                # Skip it and keep going
                found_interior = True
                continue
            
            if density > wall_threshold and found_interior:
                # We've passed through interior, now hitting boundary wall
                hit_pos = (rx, rz)
                # Keep going a bit to find the outer edge of the wall
                continue
            
            if density <= wall_threshold / 4 and found_interior and hit_pos:
                # We've exited through the wall — the last dense point was the boundary
                break
        
        if hit_pos:
            hits.append(hit_pos)
    
    return hits


def cast_rays_simple(density_img, x_min, z_min, cell_size, center_x, center_z, n_rays=720):
    """Simpler raycasting: find the LAST dense point before empty space in each direction."""
    nz, nx = density_img.shape
    
    # Build binary mask: where are there enough points to be "room"?
    threshold = max(np.percentile(density_img[density_img > 0], 20), 3)
    room_mask = density_img > threshold
    
    # Fill holes
    room_mask = ndimage.binary_fill_holes(room_mask)
    # Keep largest component
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = labeled == (np.argmax(sizes) + 1)
    
    # Erode to remove thin extensions (hallway bleed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    room_mask = cv2.morphologyEx(room_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    hits = []
    
    for i in range(n_rays):
        angle = 2 * math.pi * i / n_rays
        dx = math.cos(angle)
        dz = math.sin(angle)
        
        last_inside = None
        
        for step in range(1, 500):
            dist = step * cell_size
            rx = center_x + dx * dist
            rz = center_z + dz * dist
            
            px = int((rx - x_min) / cell_size)
            pz = int((rz - z_min) / cell_size)
            
            if px < 0 or px >= nx or pz < 0 or pz >= nz:
                break
            
            if room_mask[pz, px]:
                last_inside = (rx, rz)
            elif last_inside is not None:
                # Just exited the room
                break
        
        if last_inside:
            hits.append(last_inside)
    
    return hits, room_mask


def simplify_hits_to_rectilinear(hits, snap_tolerance=0.2):
    """Convert contour points to a clean rectilinear polygon."""
    if len(hits) < 10:
        return [list(h) for h in hits]
    
    pts = np.array(hits)
    
    # Step 1: Aggressive Douglas-Peucker
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts_cv, 0.2, True)
    pts = simplified.reshape(-1, 2)
    
    if len(pts) < 4:
        return pts.tolist()
    
    print(f"    After D-P: {len(pts)} points")
    
    # Step 2: Snap to axis-aligned
    result = [pts[0].tolist()]
    for i in range(1, len(pts)):
        prev = result[-1]
        curr = pts[i].tolist()
        
        dx = abs(curr[0] - prev[0])
        dz = abs(curr[1] - prev[1])
        
        if dx < snap_tolerance:
            curr[0] = prev[0]
        elif dz < snap_tolerance:
            curr[1] = prev[1]
        else:
            result.append([curr[0], prev[1]])
        
        result.append(curr)
    
    # Step 3: Iteratively remove tiny jogs (< 0.4m)
    for _ in range(5):
        cleaned = [result[0]]
        i = 1
        while i < len(result):
            d = math.sqrt((result[i][0]-cleaned[-1][0])**2 + (result[i][1]-cleaned[-1][1])**2)
            if d < 0.4 and i < len(result) - 1:
                i += 1  # skip tiny segment
            else:
                cleaned.append(result[i])
            i += 1
        if len(cleaned) == len(result):
            break
        result = cleaned
    
    print(f"    After jog removal: {len(result)} points")
    
    # Step 4: Remove collinear
    final = [result[0]]
    for i in range(1, len(result)-1):
        p, c, n = final[-1], result[i], result[i+1]
        same_x = abs(p[0]-c[0]) < 0.05 and abs(c[0]-n[0]) < 0.05
        same_z = abs(p[1]-c[1]) < 0.05 and abs(c[1]-n[1]) < 0.05
        if not (same_x or same_z):
            final.append(c)
    final.append(result[-1])
    
    print(f"    Final: {len(final)} points")
    return final


def analyze_mesh(mesh_file):
    print(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    up_idx, up_name = detect_up_axis(mesh)
    print(f"Coordinate system: {up_name}-up")
    
    up_coords = mesh.vertices[:, up_idx]
    up_min, up_max, up_range = up_coords.min(), up_coords.max(), np.ptp(up_coords)
    
    # Project mid-height vertices
    x_raw, z_raw = project_vertices(mesh, up_idx)
    mask = (up_coords >= up_min + up_range*0.15) & (up_coords <= up_min + up_range*0.85)
    x_mid, z_mid = x_raw[mask], z_raw[mask]
    
    # Find angle
    print("Finding dominant angle...")
    pts_2d = list(zip(x_mid.tolist(), z_mid.tolist()))
    angle = find_dominant_angle(pts_2d)
    angle_rad = angle * math.pi / 180
    print(f"  Angle: {angle:.1f}°")
    
    # Rotate
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    # Build density image — use coarser grid for room mask
    cell_size = 0.05  # 5cm for better interior coverage
    img = np.zeros((int((rz.max()-rz.min()+0.4)/cell_size)+1,
                    int((rx.max()-rx.min()+0.4)/cell_size)+1), dtype=np.float32)
    x_min = rx.min() - 0.2
    z_min = rz.min() - 0.2
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, img.shape[1]-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, img.shape[0]-1)
    np.add.at(img, (zi, xi), 1)
    
    # Find room centroid (median of all points)
    center_x = np.median(rx)
    center_z = np.median(rz)
    print(f"  Room center estimate: ({center_x:.2f}, {center_z:.2f})")
    
    # Build room mask (no raycasting needed — just use the mask directly)
    print("Building room mask...")
    # Low threshold — walls create a ring of dense points, interior is sparser
    room_mask = (img > 1).astype(np.uint8)
    
    # Close gaps in the wall ring so fill_holes can work
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Fill the interior
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)
    
    # Keep largest component
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    # Gentle opening to trim hallway bleed
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel_open)
    room_mask = ndimage.binary_fill_holes(room_mask).astype(np.uint8)
    
    # Keep largest again
    labeled, n = ndimage.label(room_mask)
    if n > 1:
        sizes = ndimage.sum(room_mask, labeled, range(1, n+1))
        room_mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    
    occupied = np.sum(room_mask > 0)
    print(f"  Room mask: {occupied} cells, ~{occupied * cell_size * cell_size:.1f}m²")
    
    # Extract contour directly from mask
    print("Extracting contour from room mask...")
    contours, _ = cv2.findContours(room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  ERROR: No contours found")
        return {'walls': [], 'room': None, 'gaps': [], 'angle': angle,
                'coordinate_system': f'{up_name}-up', 'room_mask': room_mask, 'hits': []}
    
    contour = max(contours, key=cv2.contourArea)
    
    # Convert contour pixels to world coordinates
    # Note: contour is in (col, row) = (x_pixel, z_pixel) for the img array
    hits = []
    for pt in contour:
        x_px, z_px = pt[0]
        wx = x_min + x_px * cell_size
        wz = z_min + z_px * cell_size
        hits.append((wx, wz))
    
    print(f"  Contour points: {len(hits)}")
    
    # Simplify to rectilinear with more conservative settings
    print("Simplifying to rectilinear polygon...")
    polygon_rot = simplify_hits_to_rectilinear(hits, snap_tolerance=0.2)
    print(f"  Polygon vertices: {len(polygon_rot)}")
    
    # Transform back
    room_corners = [rot_pt(p, angle_rad) for p in polygon_rot]
    exterior = room_corners + [room_corners[0]]
    
    n = len(room_corners)
    area = abs(sum(room_corners[i][0]*room_corners[(i+1)%n][1] - 
                   room_corners[(i+1)%n][0]*room_corners[i][1] for i in range(n))) / 2
    perimeter = sum(math.sqrt((room_corners[(i+1)%n][0]-room_corners[i][0])**2 + 
                              (room_corners[(i+1)%n][1]-room_corners[i][1])**2) 
                   for i in range(n))
    
    room = {'exterior': exterior, 'area': area, 'perimeter': perimeter}
    
    # Build wall list from polygon edges
    walls = []
    for i in range(len(polygon_rot)):
        j = (i + 1) % len(polygon_rot)
        p1, p2 = polygon_rot[i], polygon_rot[j]
        length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if length < 0.15:
            continue
        dx, dz = abs(p2[0]-p1[0]), abs(p2[1]-p1[1])
        axis = 'z' if dx > dz else 'x'
        walls.append({
            'axis': axis,
            'position': (p1[0]+p2[0])/2 if axis == 'x' else (p1[1]+p2[1])/2,
            'start': min(p1[0] if axis == 'x' else p1[1], p2[0] if axis == 'x' else p2[1]),
            'end': max(p1[0] if axis == 'x' else p1[1], p2[0] if axis == 'x' else p2[1]),
            'length': length,
            'nPoints': 0,
            'startPt': rot_pt(p1, angle_rad),
            'endPt': rot_pt(p2, angle_rad),
        })
    
    # Detect openings on boundary walls
    rotated_pts = list(zip(rx.tolist(), rz.tolist()))
    gaps = detect_openings_simple(walls, rotated_pts, angle_rad)
    
    results = {
        'walls': walls,
        'room': room,
        'gaps': gaps,
        'angle': angle,
        'coordinate_system': f'{up_name}-up',
        'room_mask': room_mask,
        'hits': hits,
    }
    
    print(f"\n=== v21 Raycasting Summary ===")
    print(f"Walls: {len(walls)}, Area: {area:.1f}m²")
    doors = [g for g in gaps if g['type'] == 'door']
    windows = [g for g in gaps if g['type'] == 'window']
    print(f"Doors: {len(doors)}, Windows: {len(windows)}")
    
    return results


def detect_openings_simple(walls, rotated_points, angle_rad):
    """Simple opening detection on polygon walls."""
    gaps = []
    pts = np.array(rotated_points)
    
    for w in walls:
        axis_idx = 0 if w['axis'] == 'x' else 1
        other_idx = 1 - axis_idx
        
        near_mask = np.abs(pts[:, axis_idx] - w['position']) < 0.10
        near = np.sort(pts[near_mask][:, other_idx])
        
        if len(near) < 5:
            continue
        
        for i in range(len(near)-1):
            gap = near[i+1] - near[i]
            if 0.4 < gap < 3.0:
                mid = (near[i] + near[i+1]) / 2
                if w['axis'] == 'x':
                    mid_pt = rot_pt([w['position'], mid], angle_rad)
                    s_pt = rot_pt([w['position'], near[i]], angle_rad)
                    e_pt = rot_pt([w['position'], near[i+1]], angle_rad)
                else:
                    mid_pt = rot_pt([mid, w['position']], angle_rad)
                    s_pt = rot_pt([near[i], w['position']], angle_rad)
                    e_pt = rot_pt([near[i+1], w['position']], angle_rad)
                
                gaps.append({
                    'type': 'door' if gap < 1.2 else 'window',
                    'width': gap,
                    'mid': mid_pt, 'start': s_pt, 'end': e_pt,
                })
    
    return gaps


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Room mask with ray hits
    ax0 = axes[0]
    if results.get('room_mask') is not None:
        ax0.imshow(results['room_mask'], cmap='gray', origin='lower')
    if results.get('hits'):
        # Would need coordinate transform — skip for now
        pass
    ax0.set_title('Room Mask (density threshold)', color='white', fontsize=14)
    
    # Right: Floor plan
    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    if results['room']:
        poly = results['room']['exterior']
        ax.fill([p[0] for p in poly], [p[1] for p in poly], color='gray', alpha=0.3)
        ax.plot([p[0] for p in poly], [p[1] for p in poly], color='gray', linewidth=1, alpha=0.7)
    
    for w in results['walls']:
        s, e = w['startPt'], w['endPt']
        ax.plot([s[0], e[0]], [s[1], e[1]], color='white', linewidth=4, solid_capstyle='round')
        mx, my = (s[0]+e[0])/2, (s[1]+e[1])/2
        ax.text(mx, my, f"{w['length']:.2f}m", ha='center', va='bottom', 
                color='yellow', fontsize=9, fontweight='bold')
    
    for g in results['gaps']:
        c = 'cyan' if g['type'] == 'door' else 'lime'
        ax.plot([g['start'][0], g['end'][0]], [g['start'][1], g['end'][1]], 
                color=c, linewidth=2, linestyle='--')
        ax.text(g['mid'][0], g['mid'][1], f"{g['width']:.2f}m", ha='center', va='center',
                color=c, fontsize=8, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    if results['room']:
        ax.set_title(f"v21 Raycasting — {results['room']['area']:.1f}m²", fontsize=14, color='white')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v21_raycasting',
            'walls': len(results['walls']),
            'doors': len([g for g in results['gaps'] if g['type'] == 'door']),
            'windows': len([g for g in results['gaps'] if g['type'] == 'window']),
            'area_m2': results['room']['area'] if results['room'] else 0,
        },
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v21 - Raycasting')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v21_raycast/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"v21_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()

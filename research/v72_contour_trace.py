#!/usr/bin/env python3
"""
mesh2plan v72 - Wall contour tracing from density map

Instead of fitting known rectangles, actually trace wall contours from the mesh data:
1. Build wall-face density at detected angle
2. Threshold + morphological ops to get wall mask
3. Find contours of wall regions
4. Extract wall segments (line fitting on contours)
5. Build room polygons from wall intersections
6. Render architectural floor plan from traced geometry
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Arc, Rectangle
import cv2
from scipy.signal import find_peaks
from pathlib import Path

RESOLUTION = 0.01  # 1cm resolution for better contour tracing
ANGLE = 60.0  # Correct rotation: wall normals at 150°, walls at 60°, rotate by -60° to axis-align

def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh

def rotate_points(pts, angle_deg, center=None):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if center is not None: pts = pts - center
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: rotated += center
    return rotated

def build_wall_density(mesh, angle_deg, resolution=RESOLUTION):
    """Build high-res wall density from wall-facing triangles."""
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]  # mirror fix
    center = pts_xz.mean(axis=0)
    
    # Wall faces only
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    
    # Use triangle centroids weighted by area for better density
    wall_areas = mesh.area_faces[wall_mask]
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    
    # Also get all vertices for extent
    all_rot = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = all_rot.min(axis=0) - 0.3
    xmax, zmax = all_rot.max(axis=0) + 0.3
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / resolution).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / resolution).astype(int), 0, h - 1)
    # Weight by triangle area
    np.add.at(density, (py, px), wall_areas)
    
    # Gentle blur
    density = cv2.GaussianBlur(density, (3, 3), 0.5)
    
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, 
                center=center, resolution=resolution)
    return density, grid, all_rot

def extract_wall_mask(density, grid):
    """Threshold density to get binary wall mask."""
    # Adaptive threshold based on density distribution
    nonzero = density[density > 0]
    if len(nonzero) == 0:
        return np.zeros_like(density, dtype=np.uint8)
    
    thresh = np.percentile(nonzero, 60)
    mask = (density > thresh).astype(np.uint8) * 255
    
    # Morphological ops: close small gaps, then thin
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    return mask

def extract_wall_segments(mask, grid, min_length_m=0.3):
    """Use Hough lines to find wall segments from the mask."""
    res = grid['resolution']
    min_length_px = int(min_length_m / res)
    
    # Edge detection
    edges = cv2.Canny(mask, 50, 150)
    
    # Hough line segments
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                             minLineLength=min_length_px, maxLineGap=int(0.1 / res))
    
    if lines is None:
        return []
    
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Convert to meters
        mx1 = x1 * res + grid['xmin']
        my1 = y1 * res + grid['zmin']
        mx2 = x2 * res + grid['xmin']
        my2 = y2 * res + grid['zmin']
        length = np.sqrt((mx2-mx1)**2 + (my2-my1)**2)
        angle = np.degrees(np.arctan2(my2-my1, mx2-mx1)) % 180
        segments.append({
            'p1': (mx1, my1), 'p2': (mx2, my2),
            'length': length, 'angle': angle,
            'px': (x1, y1, x2, y2)
        })
    
    return segments

def detect_dominant_angles(segments, n=2):
    """Find the two dominant wall angles from segments."""
    if not segments:
        return [0, 90]
    angles = np.array([s['angle'] for s in segments])
    lengths = np.array([s['length'] for s in segments])
    # Weighted histogram
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)
    hist = np.convolve(hist, np.ones(5)/5, mode='same')
    peaks, _ = find_peaks(hist, distance=30, height=hist.max()*0.1)
    if len(peaks) >= 2:
        top2 = sorted(peaks, key=lambda p: hist[p], reverse=True)[:2]
        return sorted(top2)
    elif len(peaks) == 1:
        return [peaks[0], (peaks[0] + 90) % 180]
    return [30, 120]

def classify_segments(segments, h_angle=None, v_angle=None, tolerance=10):
    """Classify segments by dominant angles."""
    if h_angle is None or v_angle is None:
        angles = detect_dominant_angles(segments)
        h_angle, v_angle = angles[0], angles[1]
        print(f"  Dominant angles: {h_angle}° and {v_angle}°")
    
    h_segs = []
    v_segs = []
    for s in segments:
        a = s['angle']
        dh = min(abs(a - h_angle), abs(a - h_angle - 180), abs(a - h_angle + 180))
        dv = min(abs(a - v_angle), abs(a - v_angle - 180), abs(a - v_angle + 180))
        if dh < tolerance:
            h_segs.append(s)
        elif dv < tolerance:
            v_segs.append(s)
    return h_segs, v_segs

def cluster_wall_lines(segments, axis='v', cluster_dist_m=0.15, wall_angle_deg=None):
    """Cluster parallel wall segments by perpendicular position."""
    if not segments:
        return []
    
    if wall_angle_deg is not None:
        # Project midpoints onto perpendicular axis
        perp_rad = np.radians(wall_angle_deg + 90)
        nx, ny = np.cos(perp_rad), np.sin(perp_rad)
        positions = [((s['p1'][0]+s['p2'][0])/2)*nx + ((s['p1'][1]+s['p2'][1])/2)*ny for s in segments]
    elif axis == 'v':
        positions = [(s['p1'][0] + s['p2'][0]) / 2 for s in segments]
    else:
        positions = [(s['p1'][1] + s['p2'][1]) / 2 for s in segments]
    
    # Sort
    indexed = sorted(enumerate(positions), key=lambda x: x[1])
    
    clusters = []
    current_cluster = [indexed[0]]
    for i in range(1, len(indexed)):
        if indexed[i][1] - indexed[i-1][1] < cluster_dist_m:
            current_cluster.append(indexed[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [indexed[i]]
    clusters.append(current_cluster)
    
    # For each cluster, compute weighted position and extent
    wall_lines = []
    for cluster in clusters:
        segs_in = [segments[idx] for idx, _ in cluster]
        total_length = sum(s['length'] for s in segs_in)
        if total_length < 0.3:
            continue
        
        if axis == 'v':
            weighted_pos = sum(((s['p1'][0]+s['p2'][0])/2) * s['length'] for s in segs_in) / total_length
            min_extent = min(min(s['p1'][1], s['p2'][1]) for s in segs_in)
            max_extent = max(max(s['p1'][1], s['p2'][1]) for s in segs_in)
        else:
            weighted_pos = sum(((s['p1'][1]+s['p2'][1])/2) * s['length'] for s in segs_in) / total_length
            min_extent = min(min(s['p1'][0], s['p2'][0]) for s in segs_in)
            max_extent = max(max(s['p1'][0], s['p2'][0]) for s in segs_in)
        
        wall_lines.append({
            'position': weighted_pos,
            'extent': (min_extent, max_extent),
            'total_length': total_length,
            'n_segments': len(segs_in),
            'segments': segs_in,
        })
    
    wall_lines.sort(key=lambda x: x['position'])
    return wall_lines

def find_room_polygons(v_walls, h_walls, density, grid):
    """Build room polygons from wall line intersections."""
    # For each pair of adjacent vertical walls and horizontal walls,
    # check if there's an enclosed room (density is low inside = floor)
    rooms = []
    res = grid['resolution']
    
    for i in range(len(v_walls) - 1):
        for j in range(len(h_walls) - 1):
            x0 = v_walls[i]['position']
            x1 = v_walls[i+1]['position']
            y0 = h_walls[j]['position']
            y1 = h_walls[j+1]['position']
            
            w = x1 - x0
            h_room = y1 - y0
            
            # Skip tiny cells
            if w < 0.5 or h_room < 0.5:
                continue
            
            # Check if this cell has walls on its boundaries
            # by looking at density along edges
            px0 = int((x0 - grid['xmin']) / res)
            px1 = int((x1 - grid['xmin']) / res)
            py0 = int((y0 - grid['zmin']) / res)
            py1 = int((y1 - grid['zmin']) / res)
            
            px0 = np.clip(px0, 0, grid['w']-1)
            px1 = np.clip(px1, 0, grid['w']-1)
            py0 = np.clip(py0, 0, grid['h']-1)
            py1 = np.clip(py1, 0, grid['h']-1)
            
            # Check density inside (should be low for a room, high = wall/furniture)
            if px1 > px0 + 5 and py1 > py0 + 5:
                interior = density[py0+3:py1-3, px0+3:px1-3]
                if interior.size > 0:
                    interior_mean = interior.mean()
                    # Check edges have density (walls present)
                    left_edge = density[py0:py1, max(0,px0-2):px0+2].mean()
                    right_edge = density[py0:py1, px1-2:min(grid['w'],px1+2)].mean()
                    bottom_edge = density[max(0,py0-2):py0+2, px0:px1].mean()
                    top_edge = density[py1-2:min(grid['h'],py1+2), px0:px1].mean()
                    
                    # Room = low interior, some wall edges present
                    edge_mean = (left_edge + right_edge + bottom_edge + top_edge) / 4
                    # Require: meaningful size AND edges stronger than interior
                    if w * h_room > 1.5 and (edge_mean > interior_mean * 1.5 or edge_mean > 0.01):
                        rooms.append({
                            'bounds': (x0, y0, x1, y1),
                            'width': w,
                            'height': h_room,
                            'area': w * h_room,
                            'interior_density': float(interior_mean),
                            'edge_density': float(edge_mean),
                        })
    
    return rooms

def render_traced_floorplan(v_walls, h_walls, rooms, wall_mask, density, grid, segments, output_path):
    """Render the traced floor plan."""
    res = grid['resolution']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Top-left: density map
    ax = axes[0, 0]
    ex = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    ax.set_title('Wall density (area-weighted)')
    
    # Top-right: wall mask + Hough segments
    ax = axes[0, 1]
    ax.imshow(wall_mask, origin='lower', cmap='gray', extent=ex, aspect='equal')
    for s in segments:
        x1, y1 = s['p1']
        x2, y2 = s['p2']
        ax.plot([x1, x2], [y1, y2], 'g-', lw=0.5, alpha=0.5)
    ax.set_title(f'Wall mask + {len(segments)} Hough segments')
    
    # Bottom-left: clustered wall lines
    ax = axes[1, 0]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal', alpha=0.5)
    for wl in v_walls:
        pos = wl['position']
        ext = wl['extent']
        ax.plot([pos, pos], [ext[0], ext[1]], 'lime', lw=2)
        ax.text(pos, ext[1] + 0.1, f'{pos:.2f}', color='lime', fontsize=6, ha='center')
    for wl in h_walls:
        pos = wl['position']
        ext = wl['extent']
        ax.plot([ext[0], ext[1]], [pos, pos], 'cyan', lw=2)
        ax.text(ext[1] + 0.1, pos, f'{pos:.2f}', color='cyan', fontsize=6, va='center')
    ax.set_title(f'{len(v_walls)} V walls, {len(h_walls)} H walls')
    
    # Bottom-right: detected rooms
    ax = axes[1, 1]
    ax.set_facecolor('#F5F5F5')
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 1)))
    for i, room in enumerate(rooms):
        x0, y0, x1, y1 = room['bounds']
        rect = Rectangle((x0, y0), x1-x0, y1-y0, 
                         facecolor=colors[i % len(colors)], alpha=0.5,
                         edgecolor='black', lw=2)
        ax.add_patch(rect)
        cx, cy = (x0+x1)/2, (y0+y1)/2
        ax.text(cx, cy, f"{room['area']:.1f}m²\n{room['width']:.2f}×{room['height']:.2f}",
                ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Also draw wall lines
    for wl in v_walls:
        pos = wl['position']
        ext = wl['extent']
        ax.plot([pos, pos], [ext[0], ext[1]], 'k-', lw=1.5)
    for wl in h_walls:
        pos = wl['position']
        ext = wl['extent']
        ax.plot([ext[0], ext[1]], [pos, pos], 'k-', lw=1.5)
    
    ax.set_xlim(grid['xmin'], grid['xmax'])
    ax.set_ylim(grid['zmin'], grid['zmax'])
    ax.set_aspect('equal')
    ax.set_title(f'{len(rooms)} rooms detected from traced walls')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Build density
    density, grid, all_rot = build_wall_density(mesh, ANGLE)
    print(f"Density grid: {grid['w']}×{grid['h']} ({grid['resolution']}m/px)")
    
    # Extract wall mask
    mask = extract_wall_mask(density, grid)
    print(f"Wall mask: {mask.sum() // 255} wall pixels")
    
    # Extract Hough segments - lower threshold for interior walls
    segments = extract_wall_segments(mask, grid, min_length_m=0.25)
    print(f"Hough segments: {len(segments)}")
    
    # With correct rotation (-60°), walls should be H/V in image space
    h_segs, v_segs = classify_segments(segments, h_angle=0, v_angle=90, tolerance=15)
    print(f"H segments: {len(h_segs)}, V segments: {len(v_segs)}")
    
    # Cluster into wall lines — require minimum total length
    v_walls = cluster_wall_lines(v_segs, axis='v', cluster_dist_m=0.2)
    h_walls = cluster_wall_lines(h_segs, axis='h', cluster_dist_m=0.2)
    
    # Filter: keep only wall lines with significant total length (>1.0m)
    min_wall_length = 1.0
    v_walls = [w for w in v_walls if w['total_length'] > min_wall_length]
    h_walls = [w for w in h_walls if w['total_length'] > min_wall_length]
    print(f"After filtering (>{min_wall_length}m): {len(v_walls)} V, {len(h_walls)} H")
    
    # Supplement with density profile analysis to find walls Hough misses
    v_profile = density.sum(axis=0)  # sum along Y → profile along X
    h_profile = density.sum(axis=1)  # sum along X → profile along Y
    
    # Smooth profiles
    kernel = np.ones(5) / 5
    v_smooth = np.convolve(v_profile, kernel, mode='same')
    h_smooth = np.convolve(h_profile, kernel, mode='same')
    
    # Find peaks in profiles
    v_thresh = np.percentile(v_smooth[v_smooth > 0], 30) if (v_smooth > 0).any() else 0
    h_thresh = np.percentile(h_smooth[h_smooth > 0], 30) if (h_smooth > 0).any() else 0
    
    v_peaks, _ = find_peaks(v_smooth, height=v_thresh, distance=int(0.3/grid['resolution']),
                             prominence=v_thresh * 0.3)
    h_peaks, _ = find_peaks(h_smooth, height=h_thresh, distance=int(0.3/grid['resolution']),
                             prominence=h_thresh * 0.3)
    
    v_positions = [p * grid['resolution'] + grid['xmin'] for p in v_peaks]
    h_positions = [p * grid['resolution'] + grid['zmin'] for p in h_peaks]
    
    print(f"\nDensity profile peaks:")
    print(f"  V peaks: {[f'{p:.2f}' for p in v_positions]}")
    print(f"  H peaks: {[f'{p:.2f}' for p in h_positions]}")
    
    # Add any profile peaks not already covered by Hough walls
    for vp in v_positions:
        if not any(abs(vp - w['position']) < 0.3 for w in v_walls):
            # Estimate extent from where profile is above threshold
            col = int((vp - grid['xmin']) / grid['resolution'])
            col_data = density[:, max(0,col-2):col+3].sum(axis=1)
            above = np.where(col_data > 0)[0]
            if len(above) > 0:
                ext_min = above.min() * grid['resolution'] + grid['zmin']
                ext_max = above.max() * grid['resolution'] + grid['zmin']
                if ext_max - ext_min > 0.5:
                    v_walls.append({'position': vp, 'extent': (ext_min, ext_max),
                                    'total_length': ext_max - ext_min, 'n_segments': 0,
                                    'segments': [], 'source': 'profile'})
                    print(f"  Added V wall from profile: x={vp:.2f}, extent={ext_min:.2f}→{ext_max:.2f}")
    
    for hp_pos in h_positions:
        if not any(abs(hp_pos - w['position']) < 0.3 for w in h_walls):
            row = int((hp_pos - grid['zmin']) / grid['resolution'])
            row_data = density[max(0,row-2):row+3, :].sum(axis=0)
            above = np.where(row_data > 0)[0]
            if len(above) > 0:
                ext_min = above.min() * grid['resolution'] + grid['xmin']
                ext_max = above.max() * grid['resolution'] + grid['xmin']
                if ext_max - ext_min > 0.5:
                    h_walls.append({'position': hp_pos, 'extent': (ext_min, ext_max),
                                    'total_length': ext_max - ext_min, 'n_segments': 0,
                                    'segments': [], 'source': 'profile'})
                    print(f"  Added H wall from profile: y={hp_pos:.2f}, extent={ext_min:.2f}→{ext_max:.2f}")
    
    v_walls.sort(key=lambda x: x['position'])
    h_walls.sort(key=lambda x: x['position'])
    print(f"\nDetected wall lines:")
    print(f"  V walls ({len(v_walls)}):")
    for wl in v_walls:
        print(f"    x={wl['position']:.2f}m, extent={wl['extent'][0]:.2f}→{wl['extent'][1]:.2f}m, "
              f"length={wl['total_length']:.2f}m ({wl['n_segments']} segs)")
    print(f"  H walls ({len(h_walls)}):")
    for wl in h_walls:
        print(f"    y={wl['position']:.2f}m, extent={wl['extent'][0]:.2f}→{wl['extent'][1]:.2f}m, "
              f"length={wl['total_length']:.2f}m ({wl['n_segments']} segs)")
    
    # Find rooms
    rooms = find_room_polygons(v_walls, h_walls, density, grid)
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nDetected rooms ({len(rooms)}):")
    total = 0
    for r in rooms:
        x0, y0, x1, y1 = r['bounds']
        print(f"  {r['width']:.2f}×{r['height']:.2f} = {r['area']:.1f}m² "
              f"(int_dens={r['interior_density']:.1f}, edge_dens={r['edge_density']:.1f})")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # Render
    output = '/tmp/v72_contour_trace.png'
    render_traced_floorplan(v_walls, h_walls, rooms, mask, density, grid, segments, output)
    
    # Copy to workspace
    import shutil
    shutil.copy(output, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))


if __name__ == '__main__':
    main()

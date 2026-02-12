#!/usr/bin/env python3
"""
mesh2plan v73 - Flood-fill room detection from wall mask

Approach:
1. Build wall density at correct rotation (-60°)
2. Create binary wall mask (thick walls = barriers)
3. Flood-fill from interior points to find rooms
4. Extract room contours → simplify to polygons
5. Render architectural floor plan
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Arc, Rectangle
import cv2
from pathlib import Path
from scipy import ndimage

RESOLUTION = 0.02  # 2cm — good balance of detail vs noise
ANGLE = 60.0

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

def build_wall_density(mesh, angle_deg=ANGLE, resolution=RESOLUTION):
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center = pts_xz.mean(axis=0)
    
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_areas = mesh.area_faces[wall_mask]
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    
    all_rot = rotate_points(pts_xz, -angle_deg, center)
    xmin, zmin = all_rot.min(axis=0) - 0.5
    xmax, zmax = all_rot.max(axis=0) + 0.5
    w = int((xmax - xmin) / resolution)
    h = int((zmax - zmin) / resolution)
    
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / resolution).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / resolution).astype(int), 0, h - 1)
    np.add.at(density, (py, px), wall_areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h,
                center=center, resolution=resolution)
    return density, grid, all_rot

def create_wall_mask(density, resolution=RESOLUTION):
    """Create thick binary wall mask."""
    nonzero = density[density > 0]
    if len(nonzero) == 0:
        return np.zeros_like(density, dtype=np.uint8)
    
    # Threshold - get wall pixels
    thresh = np.percentile(nonzero, 65)
    mask = (density > thresh).astype(np.uint8) * 255
    
    # Dilate to make walls solid barriers
    wall_thickness_px = max(3, int(0.12 / resolution))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wall_thickness_px, wall_thickness_px))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Close gaps along H and V directions separately (to bridge door openings in walls)
    # Use elongated kernels: thin in perpendicular direction, long in wall direction
    door_width_px = int(1.0 / resolution)  # close gaps up to ~1m (door width)
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (door_width_px, 1))
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, door_width_px))
    
    # Close horizontal walls (bridge vertical gaps)
    h_component = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_close)
    # Close vertical walls (bridge horizontal gaps)
    v_component = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, v_close)
    
    # Combine: a pixel is wall if either directional close says so
    mask = cv2.bitwise_or(h_component, v_component)
    
    return mask

def flood_fill_rooms(wall_mask, density, grid, min_area_m2=1.0):
    """Find rooms by flood-filling non-wall areas."""
    # Invert: rooms are where walls aren't
    floor = (wall_mask == 0).astype(np.uint8)
    
    # Label connected components
    labeled, n_labels = ndimage.label(floor)
    print(f"  Found {n_labels} connected regions")
    
    res = grid['resolution']
    rooms = []
    
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_px = region.sum()
        area_m2 = area_px * res * res
        
        if area_m2 < min_area_m2:
            continue
        
        # Get bounding box
        ys, xs = np.where(region)
        bbox_x0 = xs.min() * res + grid['xmin']
        bbox_x1 = xs.max() * res + grid['xmin']
        bbox_y0 = ys.min() * res + grid['zmin']
        bbox_y1 = ys.max() * res + grid['zmin']
        
        # Get contour
        region_u8 = region.astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour (approximate polygon)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert contour to meters
        contour_m = []
        for pt in approx[:, 0]:
            mx = pt[0] * res + grid['xmin']
            my = pt[1] * res + grid['zmin']
            contour_m.append((mx, my))
        
        rooms.append({
            'label': label_id,
            'area': area_m2,
            'bbox': (bbox_x0, bbox_y0, bbox_x1, bbox_y1),
            'width': bbox_x1 - bbox_x0,
            'height': bbox_y1 - bbox_y0,
            'contour_px': approx,
            'contour_m': contour_m,
            'n_vertices': len(approx),
            'centroid': ((bbox_x0+bbox_x1)/2, (bbox_y0+bbox_y1)/2),
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    return rooms, labeled

def snap_contour_to_hv(contour_m, angle_tolerance=15):
    """Snap near-H/V edges to exact H/V."""
    if len(contour_m) < 3:
        return contour_m
    
    snapped = [contour_m[0]]
    for i in range(1, len(contour_m)):
        x0, y0 = snapped[-1]
        x1, y1 = contour_m[i]
        dx, dy = x1 - x0, y1 - y0
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        
        if angle < angle_tolerance:
            # Near horizontal — snap y
            snapped.append((x1, y0))
        elif angle > 90 - angle_tolerance:
            # Near vertical — snap x
            snapped.append((x0, y1))
        else:
            snapped.append((x1, y1))
    
    return snapped

def render_architectural(rooms, wall_mask, density, grid, output_path):
    """Render architectural floor plan from detected rooms."""
    res = grid['resolution']
    ex = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Top-left: density
    ax = axes[0, 0]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    ax.set_title('Wall density')
    
    # Top-right: wall mask
    ax = axes[0, 1]
    ax.imshow(wall_mask, origin='lower', cmap='gray', extent=ex, aspect='equal')
    ax.set_title('Wall mask (thresholded + dilated + closed)')
    
    # Bottom-left: rooms on wall mask
    ax = axes[1, 0]
    ax.imshow(wall_mask, origin='lower', cmap='gray', extent=ex, aspect='equal', alpha=0.3)
    colors_list = ['#FFB3BA', '#BAE1FF', '#FFFFBA', '#BAFFC9', '#E8BAFF', '#FFE0BA', '#B3FFE0']
    for i, room in enumerate(rooms):
        contour_m = room['contour_m']
        if len(contour_m) >= 3:
            poly = MplPoly(contour_m, closed=True, 
                          facecolor=colors_list[i % len(colors_list)], alpha=0.6,
                          edgecolor='blue', lw=2)
            ax.add_patch(poly)
        cx, cy = room['centroid']
        ax.text(cx, cy, f"R{i+1}\n{room['area']:.1f}m²\n{room['n_vertices']}v",
                ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3])
    ax.set_aspect('equal')
    ax.set_title(f'{len(rooms)} rooms detected')
    
    # Bottom-right: architectural render
    ax = axes[1, 1]
    ax.set_facecolor('white')
    
    # Light grid
    for x in np.arange(int(ex[0])-1, int(ex[1])+2, 0.5):
        ax.axvline(x, color='#F0F0F0', lw=0.3, zorder=0)
    for y in np.arange(int(ex[2])-1, int(ex[3])+2, 0.5):
        ax.axhline(y, color='#F0F0F0', lw=0.3, zorder=0)
    
    # Room fills with snapped contours
    for i, room in enumerate(rooms):
        snapped = snap_contour_to_hv(room['contour_m'])
        if len(snapped) >= 3:
            # Fill
            fill_color = '#E8D5B7' if room['area'] > 10 else '#F5F5F5'
            poly = MplPoly(snapped, closed=True, facecolor=fill_color, edgecolor='none', zorder=1)
            ax.add_patch(poly)
            room['snapped'] = snapped
    
    # Draw walls from mask contours
    mask_contours, _ = cv2.findContours(wall_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in mask_contours:
        if cv2.contourArea(cnt) < 50:
            continue
        pts = [(p[0][0] * res + grid['xmin'], p[0][1] * res + grid['zmin']) for p in cnt]
        if len(pts) >= 3:
            poly = MplPoly(pts, closed=True, facecolor='#333', edgecolor='#333', lw=0.3, zorder=10)
            ax.add_patch(poly)
    
    # Room labels
    for i, room in enumerate(rooms):
        cx, cy = room['centroid']
        ax.text(cx, cy + 0.15, f'Room {i+1}', ha='center', va='center',
                fontsize=8, color='#888', fontweight='bold', zorder=20)
        ax.text(cx, cy - 0.15, f'({room["area"]:.1f} m²)', ha='center', va='center',
                fontsize=7, color='#AAA', zorder=20)
        # Dimension
        ax.text(cx, cy - 0.45, f'{room["width"]:.2f} × {room["height"]:.2f} m',
                ha='center', va='center', fontsize=6, color='#BBB', zorder=20)
    
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Architectural floor plan')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    density, grid, all_rot = build_wall_density(mesh)
    print(f"Density grid: {grid['w']}×{grid['h']}")
    
    wall_mask = create_wall_mask(density)
    print(f"Wall mask: {wall_mask.sum()//255} wall px")
    
    rooms, labeled = flood_fill_rooms(wall_mask, density, grid, min_area_m2=1.5)
    print(f"\nDetected {len(rooms)} rooms:")
    total = 0
    for i, r in enumerate(rooms):
        print(f"  Room {i+1}: {r['width']:.2f}×{r['height']:.2f} = {r['area']:.1f}m² ({r['n_vertices']} vertices)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    output = '/tmp/v73_floodfill.png'
    render_architectural(rooms, wall_mask, density, grid, output)
    
    import shutil
    shutil.copy(output, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))


if __name__ == '__main__':
    main()

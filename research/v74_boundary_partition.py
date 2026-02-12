#!/usr/bin/env python3
"""
mesh2plan v74 - Boundary extraction + wall partition

Strategy:
1. Build wall density at -60° rotation (walls become H/V)
2. Get apartment boundary from convex hull / alpha shape of all vertices
3. Draw boundary as thick wall on mask
4. Add interior walls from Hough line detection (only strong, long walls)
5. Close door gaps along each wall line individually
6. Flood fill to get rooms
7. Simplify contours and render
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from pathlib import Path

RESOLUTION = 0.015  # 1.5cm
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
    r = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: r += center
    return r

def m2px(x, y, grid):
    return (int((x - grid['xmin']) / grid['resolution']),
            int((y - grid['zmin']) / grid['resolution']))

def px2m(px, py, grid):
    return (px * grid['resolution'] + grid['xmin'],
            py * grid['resolution'] + grid['zmin'])

def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Rotate all vertices
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center = pts_xz.mean(axis=0)
    all_rot = rotate_points(pts_xz, -ANGLE, center)
    
    # Wall faces
    normals = mesh.face_normals
    wall_mask_faces = np.abs(normals[:, 1]) < 0.3
    wall_areas = mesh.area_faces[wall_mask_faces]
    wall_c = mesh.triangles_center[wall_mask_faces][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -ANGLE, center)
    
    # Grid
    xmin, zmin = all_rot.min(axis=0) - 0.5
    xmax, zmax = all_rot.max(axis=0) + 0.5
    res = RESOLUTION
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, resolution=res, center=center)
    
    # Wall density
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), wall_areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    print(f"Density: {w}×{h}")
    
    # ============================================================
    # STEP 1: Apartment boundary from vertex density
    # ============================================================
    # All-vertex density (not just walls)
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((all_rot[:, 0] - xmin) / res).astype(int), 0, w - 1)
    apy = np.clip(((all_rot[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(all_density, (apy, apx), 1)
    all_density = cv2.GaussianBlur(all_density, (11, 11), 3.0)
    
    # Threshold to get apartment footprint
    apt_thresh = np.percentile(all_density[all_density > 0], 10) if (all_density > 0).any() else 0
    apt_mask = (all_density > apt_thresh).astype(np.uint8) * 255
    
    # Clean up
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    
    # Find largest contour = apartment boundary
    contours, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("ERROR: No apartment boundary found")
        return
    apt_contour = max(contours, key=cv2.contourArea)
    
    # Simplify boundary
    epsilon = 0.01 * cv2.arcLength(apt_contour, True)
    apt_simple = cv2.approxPolyDP(apt_contour, epsilon, True)
    print(f"Apartment boundary: {len(apt_simple)} vertices")
    
    # ============================================================
    # STEP 2: Build partition mask — boundary + interior walls
    # ============================================================
    partition_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw apartment boundary as thick wall
    wall_px = max(4, int(0.15 / res))
    cv2.drawContours(partition_mask, [apt_contour], 0, 255, wall_px)
    
    # Fill outside apartment
    # Flood fill from corner (should be outside)
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition_mask, outside, (0, 0), 128)
    # Now 128 = outside, 255 = boundary, 0 = inside
    
    # ============================================================
    # STEP 3: Detect interior walls via Hough on wall density
    # ============================================================
    wall_thresh = np.percentile(density[density > 0], 55) if (density > 0).any() else 0
    wall_binary = (density > wall_thresh).astype(np.uint8) * 255
    wall_binary = cv2.dilate(wall_binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    edges = cv2.Canny(wall_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                             minLineLength=int(0.5 / res), maxLineGap=int(0.15 / res))
    
    if lines is None:
        lines = []
    else:
        lines = lines.tolist()
    
    print(f"Hough lines: {len(lines)}")
    
    # Filter to H/V lines only
    hv_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        if angle < 12 or angle > 78:  # Near H or V
            hv_lines.append(line[0])
    
    print(f"H/V lines: {len(hv_lines)}")
    
    # Draw interior walls onto partition mask
    for x1, y1, x2, y2 in hv_lines:
        # Only draw if inside apartment boundary
        mx, my = (x1+x2)//2, (y1+y2)//2
        if 0 <= my < h and 0 <= mx < w and partition_mask[my, mx] != 128:
            cv2.line(partition_mask, (x1, y1), (x2, y2), 255, max(2, wall_px//2))
    
    # ============================================================
    # STEP 4: Cluster Hough lines and bridge gaps along wall lines
    # ============================================================
    # Instead of global directional closing, bridge gaps along each detected wall line
    # Cluster lines into wall groups by position
    
    h_lines = []
    v_lines = []
    for x1, y1, x2, y2 in hv_lines:
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        if angle < 12:
            h_lines.append((x1, y1, x2, y2))
        else:
            v_lines.append((x1, y1, x2, y2))
    
    def cluster_lines(lines, axis_idx, cluster_dist_px=10):
        """Cluster lines by position on perpendicular axis."""
        if not lines:
            return []
        positions = [((l[axis_idx] + l[axis_idx+2]) / 2) for l in lines]
        indexed = sorted(zip(positions, lines), key=lambda x: x[0])
        clusters = [[indexed[0]]]
        for i in range(1, len(indexed)):
            if indexed[i][0] - indexed[i-1][0] < cluster_dist_px:
                clusters[-1].append(indexed[i])
            else:
                clusters.append([indexed[i]])
        return clusters
    
    # For each wall cluster, draw a continuous line bridging all segments
    wall_line_thickness = max(3, int(0.08 / res))
    
    for cluster in cluster_lines(h_lines, 1, int(0.3/res)):
        # All segments in this horizontal wall
        if len(cluster) < 2:
            continue
        # Find overall extent
        all_x = []
        y_avg = np.mean([pos for pos, _ in cluster])
        for _, (x1, y1, x2, y2) in cluster:
            all_x.extend([x1, x2])
        x_min, x_max = min(all_x), max(all_x)
        # Draw continuous wall line
        cv2.line(partition_mask, (x_min, int(y_avg)), (x_max, int(y_avg)), 255, wall_line_thickness)
    
    for cluster in cluster_lines(v_lines, 0, int(0.3/res)):
        if len(cluster) < 2:
            continue
        all_y = []
        x_avg = np.mean([pos for pos, _ in cluster])
        for _, (x1, y1, x2, y2) in cluster:
            all_y.extend([y1, y2])
        y_min, y_max = min(all_y), max(all_y)
        cv2.line(partition_mask, (int(x_avg), y_min), (int(x_avg), y_max), 255, wall_line_thickness)
    
    # Make sure outside stays marked
    partition_mask[outside[1:-1, 1:-1] == 1] = 128
    
    # ============================================================
    # STEP 5: Flood fill rooms
    # ============================================================
    # Interior = pixels that are 0 (not wall=255, not outside=128)
    interior = (partition_mask == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    print(f"Connected regions: {n_labels}")
    
    rooms = []
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_px = region.sum()
        area_m2 = area_px * res * res
        
        if area_m2 < 1.0:
            continue
        
        ys, xs = np.where(region)
        bbox = (xs.min()*res+xmin, ys.min()*res+zmin, xs.max()*res+xmin, ys.max()*res+zmin)
        
        # Contour
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        
        contour_m = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx]
        
        rooms.append({
            'area': area_m2,
            'bbox': bbox,
            'width': bbox[2]-bbox[0],
            'height': bbox[3]-bbox[1],
            'contour_m': contour_m,
            'n_verts': len(approx),
            'centroid': ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2),
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nDetected {len(rooms)} rooms:")
    total = 0
    for i, r in enumerate(rooms):
        print(f"  Room {i+1}: {r['width']:.2f}×{r['height']:.2f} = {r['area']:.1f}m² ({r['n_verts']}v)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # ============================================================
    # RENDER
    # ============================================================
    ex = [xmin, xmax, zmin, zmax]
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    axes[0,0].imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    axes[0,0].set_title('Wall density')
    
    axes[0,1].imshow(apt_mask, origin='lower', cmap='gray', extent=ex, aspect='equal')
    # Draw simplified boundary
    bpts = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in apt_simple]
    if len(bpts) >= 3:
        poly = MplPoly(bpts, closed=True, fill=False, edgecolor='lime', lw=2)
        axes[0,1].add_patch(poly)
    axes[0,1].set_title(f'Apartment boundary ({len(apt_simple)}v)')
    
    axes[0,2].imshow(wall_binary, origin='lower', cmap='gray', extent=ex, aspect='equal')
    for x1,y1,x2,y2 in hv_lines:
        mx1,my1 = px2m(x1,y1,grid)
        mx2,my2 = px2m(x2,y2,grid)
        axes[0,2].plot([mx1,mx2],[my1,my2],'g-',lw=1)
    axes[0,2].set_title(f'{len(hv_lines)} H/V wall lines')
    
    axes[1,0].imshow(partition_mask, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[1,0].set_title('Partition mask (boundary + walls + closed)')
    
    # Rooms overlay
    axes[1,1].imshow(partition_mask, origin='lower', cmap='gray', extent=ex, aspect='equal', alpha=0.3)
    colors = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9','#E8BAFF','#FFE0BA','#B3FFE0','#FFD5B3']
    for i, room in enumerate(rooms):
        if len(room['contour_m']) >= 3:
            poly = MplPoly(room['contour_m'], closed=True,
                          facecolor=colors[i%len(colors)], alpha=0.6, edgecolor='blue', lw=2)
            axes[1,1].add_patch(poly)
        cx,cy = room['centroid']
        axes[1,1].text(cx, cy, f"R{i+1}\n{room['area']:.1f}m²", ha='center', va='center',
                       fontsize=8, fontweight='bold')
    axes[1,1].set_xlim(ex[0],ex[1]); axes[1,1].set_ylim(ex[2],ex[3])
    axes[1,1].set_aspect('equal')
    axes[1,1].set_title(f'{len(rooms)} rooms')
    
    # Architectural render
    ax = axes[1,2]
    ax.set_facecolor('white')
    for x in np.arange(-5, 10, 0.5):
        ax.axvline(x, color='#F0F0F0', lw=0.3, zorder=0)
    for y in np.arange(-5, 10, 0.5):
        ax.axhline(y, color='#F0F0F0', lw=0.3, zorder=0)
    
    # Room fills
    for i, room in enumerate(rooms):
        if len(room['contour_m']) >= 3:
            fc = '#E8D5B7' if room['area'] > 10 else '#F5F5F5'
            poly = MplPoly(room['contour_m'], closed=True, fc=fc, ec='none', zorder=1)
            ax.add_patch(poly)
    
    # Walls from partition mask contours
    wall_only = ((partition_mask == 255)).astype(np.uint8) * 255
    wcnts, _ = cv2.findContours(wall_only, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in wcnts:
        if cv2.contourArea(cnt) < 20: continue
        pts = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in cnt]
        if len(pts) >= 3:
            poly = MplPoly(pts, closed=True, fc='#333', ec='#333', lw=0.2, zorder=10)
            ax.add_patch(poly)
    
    for i, room in enumerate(rooms):
        cx,cy = room['centroid']
        ax.text(cx, cy+0.1, f'Room {i+1}', ha='center', va='center', fontsize=8, color='#888', zorder=20)
        ax.text(cx, cy-0.2, f'({room["area"]:.1f} m²)', ha='center', va='center', fontsize=7, color='#AAA', zorder=20)
    
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3])
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Architectural')
    
    plt.tight_layout()
    out = '/tmp/v74_boundary_partition.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    
    import shutil
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))


if __name__ == '__main__':
    main()

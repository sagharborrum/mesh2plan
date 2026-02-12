#!/usr/bin/env python3
"""
mesh2plan v77 - Angled Hough lines (no rotation)

Instead of rotating to align H/V, detect lines at the actual wall angles.
Use standard Hough transform (not probabilistic) with theta restricted to 
wall angles ±5°. Then cluster parallel lines into walls, find intersections
to build room polygons.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from pathlib import Path
import shutil

RESOLUTION = 0.012

def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh

def detect_wall_angles(mesh):
    """Return the two dominant wall line directions in degrees."""
    normals = mesh.face_normals
    areas = mesh.area_faces
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wn = normals[wall_mask][:, [0, 2]]
    wa = areas[wall_mask]
    # Normal angles
    angles = np.degrees(np.arctan2(wn[:, 1], wn[:, 0])) % 180
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=wa)
    hist = ndimage.gaussian_filter1d(hist, sigma=2)
    peaks, props = find_peaks(hist, height=hist.max() * 0.2, distance=20)
    top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
    # IMPORTANT: apply mirror fix to normals before computing angles
    wn[:, 0] = -wn[:, 0]
    angles = np.degrees(np.arctan2(wn[:, 1], wn[:, 0])) % 180
    hist, _ = np.histogram(angles, bins=bins, weights=wa)
    hist = ndimage.gaussian_filter1d(hist, sigma=2)
    peaks, props = find_peaks(hist, height=hist.max() * 0.2, distance=20)
    top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
    # Normal angle + 90° = wall line angle
    wall_angles = sorted([(a + 90) % 180 for a in top2])
    print(f"Wall normal peaks: {sorted(top2)}° → wall line angles: {wall_angles}°")
    return wall_angles

def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    wall_angles = detect_wall_angles(mesh)
    
    # Project XZ, mirror X
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    
    # Wall faces
    normals = mesh.face_normals
    wall_mask_faces = np.abs(normals[:, 1]) < 0.3
    wall_areas = mesh.area_faces[wall_mask_faces]
    wall_c = mesh.triangles_center[wall_mask_faces][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    
    # Grid (no rotation)
    res = RESOLUTION
    xmin, zmin = pts_xz.min(axis=0) - 0.5
    xmax, zmax = pts_xz.max(axis=0) + 0.5
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    
    # Wall density
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_c[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((wall_c[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), wall_areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    print(f"Grid: {w}×{h}")
    
    # Apartment boundary
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((pts_xz[:, 0] - xmin) / res).astype(int), 0, w - 1)
    apy = np.clip(((pts_xz[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(all_density, (apy, apx), 1)
    all_density = cv2.GaussianBlur(all_density, (11, 11), 3.0)
    apt_thresh = np.percentile(all_density[all_density > 0], 10)
    apt_mask = (all_density > apt_thresh).astype(np.uint8) * 255
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    contours, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apt_contour = max(contours, key=cv2.contourArea)
    
    # ============================================================
    # Hough lines at wall angles
    # ============================================================
    wall_thresh = np.percentile(density[density > 0], 60)
    wall_binary = (density > wall_thresh).astype(np.uint8) * 255
    edges = cv2.Canny(wall_binary, 50, 150)
    
    # Standard Hough transform - get all lines
    # theta in Hough = angle of normal to line from origin
    # For a wall running at angle α, the Hough theta = α + 90° (or α - 90°)
    # But let's just use HoughLinesP and filter by angle
    
    lines_all = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=12,
                                 minLineLength=int(0.25 / res), maxLineGap=int(0.08 / res))
    if lines_all is None: lines_all = []
    else: lines_all = [l[0] for l in lines_all]
    print(f"Total Hough lines: {len(lines_all)}")
    
    # Filter lines to wall angles ±15°
    angle_tol = 15
    wall_lines = {i: [] for i in range(len(wall_angles))}
    
    for x1, y1, x2, y2 in lines_all:
        line_angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        for i, wa in enumerate(wall_angles):
            diff = min(abs(line_angle - wa), 180 - abs(line_angle - wa))
            if diff < angle_tol:
                wall_lines[i].append((x1, y1, x2, y2))
                break
    
    for i, wa in enumerate(wall_angles):
        print(f"  Wall angle {wa}°: {len(wall_lines[i])} lines")
    
    # ============================================================
    # Cluster parallel lines into wall groups
    # ============================================================
    def line_perp_distance(x1, y1, x2, y2, angle_deg):
        """Project line midpoint onto perpendicular axis."""
        rad = np.radians(angle_deg)
        # Perpendicular direction
        nx, ny = -np.sin(rad), np.cos(rad)
        mx, my = (x1+x2)/2, (y1+y2)/2
        return mx*nx + my*ny
    
    def line_para_extent(x1, y1, x2, y2, angle_deg):
        """Project line endpoints onto parallel axis, return (min, max)."""
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
        p1 = x1*dx + y1*dy
        p2 = x2*dx + y2*dy
        return min(p1, p2), max(p1, p2)
    
    all_walls = []
    cluster_dist = int(0.25 / res)
    
    for angle_idx, wa in enumerate(wall_angles):
        lines = wall_lines[angle_idx]
        if not lines: continue
        
        # Compute perpendicular distance for each line
        items = []
        for x1,y1,x2,y2 in lines:
            perp = line_perp_distance(x1,y1,x2,y2, wa)
            pmin, pmax = line_para_extent(x1,y1,x2,y2, wa)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            items.append((perp, pmin, pmax, length, (x1,y1,x2,y2)))
        
        items.sort(key=lambda x: x[0])
        
        # Cluster
        clusters = [[items[0]]]
        for i in range(1, len(items)):
            if items[i][0] - items[i-1][0] < cluster_dist:
                clusters[-1].append(items[i])
            else:
                clusters.append([items[i]])
        
        for cluster in clusters:
            perp_avg = np.mean([c[0] for c in cluster])
            para_min = min(c[1] for c in cluster)
            para_max = max(c[2] for c in cluster)
            total_length = sum(c[3] for c in cluster)
            
            if total_length < int(0.5 / res):
                continue
            
            # Convert back to pixel coords for endpoints
            rad = np.radians(wa)
            dx, dy = np.cos(rad), np.sin(rad)
            nx, ny = -np.sin(rad), np.cos(rad)
            
            # Start and end points
            sx = perp_avg * nx + para_min * dx
            sy = perp_avg * ny + para_min * dy
            ex = perp_avg * nx + para_max * dx
            ey = perp_avg * ny + para_max * dy
            
            all_walls.append({
                'angle': wa,
                'perp': perp_avg,
                'para_min': para_min,
                'para_max': para_max,
                'start_px': (int(sx), int(sy)),
                'end_px': (int(ex), int(ey)),
                'total_length': total_length,
                'n_segments': len(cluster),
            })
    
    print(f"\nClustered walls: {len(all_walls)}")
    for i, wall in enumerate(all_walls):
        length_m = wall['total_length'] * res
        span_m = (wall['para_max'] - wall['para_min']) * res
        print(f"  Wall {i}: {wall['angle']:.0f}° span={span_m:.1f}m ({wall['n_segments']} segs)")
    
    # ============================================================
    # Build partition mask
    # ============================================================
    wall_px = max(4, int(0.12 / res))
    partition = np.zeros((h, w), dtype=np.uint8)
    
    # Apartment boundary
    cv2.drawContours(partition, [apt_contour], 0, 255, wall_px + 2)
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition, outside, (0, 0), 128)
    
    # Interior walls
    for wall in all_walls:
        sx, sy = wall['start_px']
        ex, ey = wall['end_px']
        # Check midpoint is inside apartment
        mx, my = (sx+ex)//2, (sy+ey)//2
        if 0 <= my < h and 0 <= mx < w and partition[my, mx] != 128:
            cv2.line(partition, (sx, sy), (ex, ey), 255, wall_px)
    
    partition[outside[1:-1, 1:-1] == 1] = 128
    
    # ============================================================
    # Flood fill rooms
    # ============================================================
    interior = (partition == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    print(f"Connected regions: {n_labels}")
    
    rooms = []
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_m2 = region.sum() * res * res
        if area_m2 < 1.0: continue
        
        ys, xs = np.where(region)
        bbox = (xs.min()*res+xmin, ys.min()*res+zmin, xs.max()*res+xmin, ys.max()*res+zmin)
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        contour_m = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx]
        
        rooms.append({
            'area': area_m2, 'bbox': bbox,
            'width': bbox[2]-bbox[0], 'height': bbox[3]-bbox[1],
            'contour_m': contour_m, 'n_verts': len(approx),
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
    axes[0,0].set_title('Wall density (no rotation)')
    
    axes[0,1].imshow(wall_binary, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[0,1].set_title('Wall binary')
    
    # All Hough lines colored by angle group
    axes[0,2].imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal', alpha=0.5)
    line_colors = ['lime', 'cyan']
    for angle_idx, wa in enumerate(wall_angles):
        for x1,y1,x2,y2 in wall_lines[angle_idx]:
            mx1 = x1*res+xmin; my1 = y1*res+zmin
            mx2 = x2*res+xmin; my2 = y2*res+zmin
            axes[0,2].plot([mx1,mx2],[my1,my2], color=line_colors[angle_idx], lw=0.5)
    # Clustered walls
    for wall in all_walls:
        sx,sy = wall['start_px']
        exx,ey = wall['end_px']
        mx1 = sx*res+xmin; my1 = sy*res+zmin
        mx2 = exx*res+xmin; my2 = ey*res+zmin
        axes[0,2].plot([mx1,mx2],[my1,my2], 'r-', lw=3)
    axes[0,2].set_title(f'{len(all_walls)} walls (red=clustered)')
    
    axes[1,0].imshow(partition, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[1,0].set_title('Partition mask')
    
    # Rooms
    axes[1,1].imshow(partition, origin='lower', cmap='gray', extent=ex, aspect='equal', alpha=0.3)
    colors = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9','#E8BAFF','#FFE0BA']
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
    
    # Architectural
    ax = axes[1,2]
    ax.set_facecolor('#F8F8F8')
    for room in rooms:
        if len(room['contour_m']) >= 3:
            fc = '#E8D5B7' if room['area'] > 12 else '#FFFFFF'
            poly = MplPoly(room['contour_m'], closed=True, fc=fc, ec='none', zorder=1)
            ax.add_patch(poly)
    
    wall_region = (partition == 255).astype(np.uint8) * 255
    wcnts, _ = cv2.findContours(wall_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in wcnts:
        if cv2.contourArea(cnt) < 30: continue
        pts = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in cnt]
        if len(pts) >= 3:
            poly = MplPoly(pts, closed=True, fc='#333', ec='#222', lw=0.3, zorder=10)
            ax.add_patch(poly)
    
    for i, room in enumerate(rooms):
        cx, cy = room['centroid']
        ax.text(cx, cy+0.1, f'Room {i+1}', ha='center', fontsize=9, color='#666', zorder=20)
        ax.text(cx, cy-0.15, f'({room["area"]:.1f} m²)', ha='center', fontsize=7, color='#999', zorder=20)
    
    ax.set_xlim(xmin, xmax); ax.set_ylim(zmin, zmax)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Architectural')
    
    plt.tight_layout()
    out = '/tmp/v77_angled_hough.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))

if __name__ == '__main__':
    main()

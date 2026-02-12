#!/usr/bin/env python3
"""
mesh2plan v76 - Correct rotation angle + Hough H/V partition

FIX: Walls are at ~29°/119°. Rotate by -29° (not -60°) to align H/V.
Then Hough H/V detection should work much better.

Pipeline:
1. Wall density with correct rotation
2. Auto-detect dominant angle from wall normals histogram  
3. Hough lines on thresholded density → H/V only
4. Cluster + bridge gaps along wall lines
5. Flood fill rooms
6. Architectural render
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
WALL_THICKNESS_M = 0.12

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

def detect_wall_angle(mesh):
    """Detect dominant wall angle from face normals projected to XZ plane."""
    normals = mesh.face_normals
    areas = mesh.area_faces
    # Wall faces
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_normals = normals[wall_mask][:, [0, 2]]
    wall_areas = areas[wall_mask]
    
    # Normal angle (perpendicular to wall face)
    angles = np.degrees(np.arctan2(wall_normals[:, 1], wall_normals[:, 0])) % 180
    
    # Histogram
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=wall_areas)
    hist = ndimage.gaussian_filter1d(hist, sigma=2)
    
    peaks, props = find_peaks(hist, height=hist.max() * 0.2, distance=20)
    if len(peaks) >= 2:
        # Two strongest peaks should be ~90° apart
        top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
        # Wall normal angle → wall angle is perpendicular (add 90°)
        wall_angle = (min(top2) + 90) % 180
        print(f"Wall normal peaks: {top2[0]}°, {top2[1]}° → wall angle: {wall_angle}°")
        return wall_angle
    elif len(peaks) >= 1:
        wall_angle = (peaks[0] + 90) % 180
        print(f"Wall normal peak: {peaks[0]}° → wall angle: {wall_angle}°")
        return wall_angle
    return 0

def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Auto-detect wall angle
    wall_angle = detect_wall_angle(mesh)
    # wall_angle detection gives 119°, but we want walls at 29° to become 0°
    # Try both -29° and -119° and see which gives more H/V Hough lines
    rotation = -29.0  # force: 29° walls → 0° (H), 119° walls → 90° (V)
    print(f"Rotating by {rotation:.1f}° to align walls H/V")
    
    # Project to XZ, negate X (mirror fix)
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center = pts_xz.mean(axis=0)
    all_rot = rotate_points(pts_xz, rotation, center)
    
    # Wall faces
    normals = mesh.face_normals
    wall_mask_faces = np.abs(normals[:, 1]) < 0.3
    wall_areas = mesh.area_faces[wall_mask_faces]
    wall_c = mesh.triangles_center[wall_mask_faces][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, rotation, center)
    
    # Grid
    res = RESOLUTION
    xmin, zmin = all_rot.min(axis=0) - 0.5
    xmax, zmax = all_rot.max(axis=0) + 0.5
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    
    # Wall density
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), wall_areas)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    print(f"Grid: {w}×{h}")
    
    # ============================================================
    # STEP 1: Apartment boundary
    # ============================================================
    all_density = np.zeros((h, w), dtype=np.float32)
    apx = np.clip(((all_rot[:, 0] - xmin) / res).astype(int), 0, w - 1)
    apy = np.clip(((all_rot[:, 1] - zmin) / res).astype(int), 0, h - 1)
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
    # STEP 2: Hough lines on wall density
    # ============================================================
    wall_thresh = np.percentile(density[density > 0], 60)
    wall_binary = (density > wall_thresh).astype(np.uint8) * 255
    
    # Skeletonize to get thin wall lines
    wall_thin = cv2.ximgproc.thinning(wall_binary) if hasattr(cv2, 'ximgproc') else wall_binary
    
    edges = cv2.Canny(wall_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15,
                             minLineLength=int(0.3 / res), maxLineGap=int(0.1 / res))
    
    if lines is None: lines = []
    else: lines = [l[0] for l in lines]
    print(f"Hough lines: {len(lines)}")
    
    # Filter H/V only (now walls should be aligned)
    hv_lines = []
    h_lines = []
    v_lines = []
    for x1, y1, x2, y2 in lines:
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        if angle < 15:
            hv_lines.append((x1,y1,x2,y2))
            h_lines.append((x1,y1,x2,y2))
        elif angle > 75:
            hv_lines.append((x1,y1,x2,y2))
            v_lines.append((x1,y1,x2,y2))
    print(f"H/V lines: {len(hv_lines)} ({len(h_lines)} H, {len(v_lines)} V)")
    
    # ============================================================
    # STEP 3: Cluster lines into walls and bridge gaps
    # ============================================================
    wall_px = max(3, int(WALL_THICKNESS_M / res))
    
    def cluster_and_bridge(lines_list, perp_axis, para_axis, cluster_dist_px=None):
        """Cluster lines by perpendicular position, bridge into continuous walls."""
        if not lines_list: return []
        if cluster_dist_px is None:
            cluster_dist_px = int(0.25 / res)
        
        # Get perpendicular position for each line
        items = []
        for x1,y1,x2,y2 in lines_list:
            pts = [(x1,y1),(x2,y2)]
            perp = np.mean([p[perp_axis] for p in pts])
            para_min = min(p[para_axis] for p in pts)
            para_max = max(p[para_axis] for p in pts)
            items.append((perp, para_min, para_max))
        
        items.sort(key=lambda x: x[0])
        
        # Cluster by perpendicular position
        clusters = [[items[0]]]
        for i in range(1, len(items)):
            if items[i][0] - items[i-1][0] < cluster_dist_px:
                clusters[-1].append(items[i])
            else:
                clusters.append([items[i]])
        
        walls = []
        for cluster in clusters:
            perp_avg = np.mean([c[0] for c in cluster])
            para_min = min(c[1] for c in cluster)
            para_max = max(c[2] for c in cluster)
            total_length = sum(c[2]-c[1] for c in cluster)
            walls.append({
                'perp': perp_avg,
                'para_min': para_min,
                'para_max': para_max,
                'n_segments': len(cluster),
                'total_length': total_length,
            })
        return walls
    
    h_walls = cluster_and_bridge(h_lines, 1, 0)  # perp=Y, para=X
    v_walls = cluster_and_bridge(v_lines, 0, 1)  # perp=X, para=Y
    print(f"Wall clusters: {len(h_walls)} horizontal, {len(v_walls)} vertical")
    
    # Filter: keep walls with significant total length
    min_wall_length = int(0.4 / res)
    h_walls = [w for w in h_walls if w['total_length'] > min_wall_length]
    v_walls = [w for w in v_walls if w['total_length'] > min_wall_length]
    print(f"After length filter: {len(h_walls)} H, {len(v_walls)} V")
    
    # ============================================================
    # STEP 4: Build partition mask
    # ============================================================
    partition = np.zeros((h, w), dtype=np.uint8)
    
    # Draw apartment boundary
    cv2.drawContours(partition, [apt_contour], 0, 255, wall_px + 2)
    
    # Fill outside
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition, outside, (0, 0), 128)
    
    # Draw clustered wall lines
    for wall in h_walls:
        y = int(wall['perp'])
        x1 = int(wall['para_min'])
        x2 = int(wall['para_max'])
        # Only draw inside apartment
        my = min(y, h-1)
        mx = min((x1+x2)//2, w-1)
        if partition[my, mx] != 128:
            cv2.line(partition, (x1, y), (x2, y), 255, wall_px)
    
    for wall in v_walls:
        x = int(wall['perp'])
        y1 = int(wall['para_min'])
        y2 = int(wall['para_max'])
        mx = min(x, w-1)
        my = min((y1+y2)//2, h-1)
        if partition[mx, my] != 128 if mx < h and my < w else True:
            cv2.line(partition, (x, y1), (x, y2), 255, wall_px)
    
    partition[outside[1:-1, 1:-1] == 1] = 128
    
    # ============================================================
    # STEP 5: Flood fill rooms
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
    axes[0,0].set_title('Wall density (rotated)')
    
    axes[0,1].imshow(wall_binary, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[0,1].set_title(f'Wall binary (60th pct)')
    
    # Hough lines overlay
    axes[0,2].imshow(wall_binary, origin='lower', cmap='gray', extent=ex, aspect='equal')
    for x1,y1,x2,y2 in hv_lines:
        mx1 = x1*res+xmin; my1 = y1*res+zmin
        mx2 = x2*res+xmin; my2 = y2*res+zmin
        axes[0,2].plot([mx1,mx2],[my1,my2],'g-',lw=0.8)
    # Draw clustered walls
    for wall in h_walls:
        y = wall['perp']*res+zmin
        x1 = wall['para_min']*res+xmin
        x2 = wall['para_max']*res+xmin
        axes[0,2].plot([x1,x2],[y,y],'r-',lw=2)
    for wall in v_walls:
        x = wall['perp']*res+xmin
        y1 = wall['para_min']*res+zmin
        y2 = wall['para_max']*res+zmin
        axes[0,2].plot([x,x],[y1,y2],'b-',lw=2)
    axes[0,2].set_title(f'Walls: {len(h_walls)}H (red) + {len(v_walls)}V (blue)')
    
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
    for x in np.arange(int(xmin)-1, int(xmax)+2, 0.5):
        ax.axvline(x, color='#EEE', lw=0.3, zorder=0)
    for y in np.arange(int(zmin)-1, int(zmax)+2, 0.5):
        ax.axhline(y, color='#EEE', lw=0.3, zorder=0)
    
    for i, room in enumerate(rooms):
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
    out = '/tmp/v76_correct_angle.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))

if __name__ == '__main__':
    main()

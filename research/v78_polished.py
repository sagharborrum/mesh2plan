#!/usr/bin/env python3
"""
mesh2plan v78 - Polished angled Hough with rotation for rendering

Based on v77's working approach:
1. Detect walls at actual angles (60°/150° after mirror)
2. Cluster into wall lines
3. Build partition + flood fill
4. Rotate output by -60° for H/V aligned rendering (matching reference)
5. Better gap closing for small rooms (bathroom, WC)
6. Architectural render with thick walls, door arcs, dimensions
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Arc
from matplotlib.transforms import Affine2D
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from pathlib import Path
import shutil

RESOLUTION = 0.012
WALL_THICK_M = 0.12

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

def detect_wall_angles_mirrored(mesh):
    """Detect wall line angles from face normals (after X mirror)."""
    normals = mesh.face_normals
    areas = mesh.area_faces
    wm = np.abs(normals[:, 1]) < 0.3
    wn = normals[wm][:, [0, 2]].copy()
    wn[:, 0] = -wn[:, 0]  # mirror
    wa = areas[wm]
    angles = np.degrees(np.arctan2(wn[:, 1], wn[:, 0])) % 180
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=wa)
    hist = ndimage.gaussian_filter1d(hist, sigma=2)
    peaks, props = find_peaks(hist, height=hist.max() * 0.2, distance=20)
    top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
    wall_angles = sorted([(a + 90) % 180 for a in top2])
    print(f"Wall normal peaks: {sorted(top2)}° → wall angles: {wall_angles}°")
    return wall_angles

def line_perp_dist(x1, y1, x2, y2, angle_deg):
    rad = np.radians(angle_deg)
    nx, ny = -np.sin(rad), np.cos(rad)
    return ((x1+x2)/2)*nx + ((y1+y2)/2)*ny

def line_para_extent(x1, y1, x2, y2, angle_deg):
    rad = np.radians(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    p1 = x1*dx + y1*dy
    p2 = x2*dx + y2*dy
    return min(p1, p2), max(p1, p2)

def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    wall_angles = detect_wall_angles_mirrored(mesh)
    render_rotation = -wall_angles[0]  # rotate so first wall direction is horizontal
    
    # Project XZ, mirror X
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    
    normals = mesh.face_normals
    wall_mask_faces = np.abs(normals[:, 1]) < 0.3
    wall_areas = mesh.area_faces[wall_mask_faces]
    wall_c = mesh.triangles_center[wall_mask_faces][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    
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
    wall_thresh = np.percentile(density[density > 0], 55)
    wall_binary = (density > wall_thresh).astype(np.uint8) * 255
    edges = cv2.Canny(wall_binary, 50, 150)
    
    lines_all = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10,
                                 minLineLength=int(0.2 / res), maxLineGap=int(0.08 / res))
    if lines_all is None: lines_all = []
    else: lines_all = [l[0] for l in lines_all]
    
    angle_tol = 15
    wall_lines = {i: [] for i in range(len(wall_angles))}
    for x1, y1, x2, y2 in lines_all:
        la = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        for i, wa in enumerate(wall_angles):
            diff = min(abs(la - wa), 180 - abs(la - wa))
            if diff < angle_tol:
                wall_lines[i].append((x1, y1, x2, y2))
                break
    
    for i, wa in enumerate(wall_angles):
        print(f"  Angle {wa}°: {len(wall_lines[i])} lines")
    
    # Cluster into walls
    cluster_dist = int(0.20 / res)  # tighter clustering
    all_walls = []
    
    for angle_idx, wa in enumerate(wall_angles):
        lines = wall_lines[angle_idx]
        if not lines: continue
        
        items = []
        for x1,y1,x2,y2 in lines:
            perp = line_perp_dist(x1,y1,x2,y2, wa)
            pmin, pmax = line_para_extent(x1,y1,x2,y2, wa)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            items.append((perp, pmin, pmax, length))
        items.sort(key=lambda x: x[0])
        
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
            
            if total_length < int(0.3 / res):
                continue
            
            rad = np.radians(wa)
            dx, dy = np.cos(rad), np.sin(rad)
            nx, ny = -np.sin(rad), np.cos(rad)
            
            sx = perp_avg * nx + para_min * dx
            sy = perp_avg * ny + para_min * dy
            ex = perp_avg * nx + para_max * dx
            ey = perp_avg * ny + para_max * dy
            
            all_walls.append({
                'angle': wa, 'perp': perp_avg,
                'para_min': para_min, 'para_max': para_max,
                'start_px': (int(sx), int(sy)),
                'end_px': (int(ex), int(ey)),
                'total_length': total_length,
                'n_segments': len(cluster),
                'start_m': (sx*res+xmin, sy*res+zmin),
                'end_m': (ex*res+xmin, ey*res+zmin),
            })
    
    print(f"\nClustered walls: {len(all_walls)}")
    for i, wall in enumerate(all_walls):
        span = (wall['para_max'] - wall['para_min']) * res
        print(f"  Wall {i}: {wall['angle']:.0f}° span={span:.1f}m segs={wall['n_segments']}")
    
    # ============================================================
    # Partition mask
    # ============================================================
    wall_px = max(4, int(WALL_THICK_M / res))
    partition = np.zeros((h, w), dtype=np.uint8)
    
    cv2.drawContours(partition, [apt_contour], 0, 255, wall_px + 2)
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition, outside, (0, 0), 128)
    
    for wall in all_walls:
        sx, sy = wall['start_px']
        ex, ey = wall['end_px']
        mx, my = (sx+ex)//2, (sy+ey)//2
        if 0 <= my < h and 0 <= mx < w and partition[my, mx] != 128:
            cv2.line(partition, (sx, sy), (ex, ey), 255, wall_px)
    
    partition[outside[1:-1, 1:-1] == 1] = 128
    
    # ============================================================
    # Flood fill rooms
    # ============================================================
    interior = (partition == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    
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
        eps = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        contour_m = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx])
        
        rooms.append({
            'area': area_m2, 'bbox': bbox,
            'width': bbox[2]-bbox[0], 'height': bbox[3]-bbox[1],
            'contour_m': contour_m, 'n_verts': len(approx),
            'centroid_m': (contour_m[:,0].mean(), contour_m[:,1].mean()),
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nDetected {len(rooms)} rooms:")
    total = 0
    for i, r in enumerate(rooms):
        print(f"  Room {i+1}: {r['width']:.2f}×{r['height']:.2f} = {r['area']:.1f}m² ({r['n_verts']}v)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # Name rooms
    names = []
    bedroom_count = 0
    for r in rooms:
        if r['area'] > 12:
            bedroom_count += 1
            names.append(f'Bedroom {bedroom_count}')
        elif r['area'] > 4:
            names.append('Hallway')
        elif r['area'] > 3:
            names.append('Entry')
        elif r['area'] > 2:
            names.append('Bathroom')
        else:
            names.append('WC')
    
    # ============================================================
    # Rotate everything for rendering (walls → H/V)
    # ============================================================
    rot = render_rotation
    print(f"\nRender rotation: {rot:.1f}°")
    
    def rot_pts(pts):
        center_m = np.array([pts_xz.mean(axis=0)[0], pts_xz.mean(axis=0)[1]])
        return rotate_points(np.array(pts), rot, center_m)
    
    center_m = np.array([pts_xz.mean(axis=0)[0], pts_xz.mean(axis=0)[1]])
    
    # Rotate room contours
    rooms_rot = []
    for room in rooms:
        rc = rotate_points(room['contour_m'], rot, center_m)
        cent = rc.mean(axis=0)
        rooms_rot.append({
            'contour': rc,
            'centroid': cent,
            'area': room['area'],
            'width': rc[:,0].max() - rc[:,0].min(),
            'height': rc[:,1].max() - rc[:,1].min(),
        })
    
    # Rotate wall endpoints
    walls_rot = []
    for wall in all_walls:
        s = rotate_points(np.array([wall['start_m']]), rot, center_m)[0]
        e = rotate_points(np.array([wall['end_m']]), rot, center_m)[0]
        walls_rot.append((s, e))
    
    # Rotate apartment boundary contour
    apt_pts_m = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in apt_contour])
    apt_rot = rotate_points(apt_pts_m, rot, center_m)
    
    # ============================================================
    # ARCHITECTURAL RENDER (rotated, matching reference)
    # ============================================================
    # Compute bounds
    all_rc = np.vstack([r['contour'] for r in rooms_rot])
    rxmin, rymin = all_rc.min(axis=0) - 1.0
    rxmax, rymax = all_rc.max(axis=0) + 1.0
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_facecolor('#F5F5F5')
    
    # Grid
    for x in np.arange(int(rxmin)-1, int(rxmax)+2, 0.5):
        ax.axvline(x, color='#EEEEEE', lw=0.3, zorder=0)
    for y in np.arange(int(rymin)-1, int(rymax)+2, 0.5):
        ax.axhline(y, color='#EEEEEE', lw=0.3, zorder=0)
    
    # Room fills
    for i, rr in enumerate(rooms_rot):
        if len(rr['contour']) >= 3:
            a = rooms[i]['area']
            fc = '#E8D5B7' if a > 12 else '#FFFFFF'
            poly = MplPoly(rr['contour'], closed=True, fc=fc, ec='none', zorder=1)
            ax.add_patch(poly)
    
    # Thick walls - draw each wall as a thick line
    wt_m = WALL_THICK_M
    for s, e in walls_rot:
        ax.plot([s[0], e[0]], [s[1], e[1]], color='#333333', lw=wt_m*80, solid_capstyle='butt', zorder=10)
    
    # Apartment boundary as thick wall
    apt_closed = np.vstack([apt_rot, apt_rot[0:1]])
    for i in range(len(apt_closed)-1):
        ax.plot([apt_closed[i,0], apt_closed[i+1,0]], 
                [apt_closed[i,1], apt_closed[i+1,1]],
                color='#333333', lw=wt_m*80, solid_capstyle='butt', zorder=10)
    
    # Room labels + dimensions
    for i, rr in enumerate(rooms_rot):
        cx, cy = rr['centroid']
        name = names[i] if i < len(names) else f'Room {i+1}'
        ax.text(cx, cy+0.15, name, ha='center', va='center', fontsize=10, 
                color='#666', fontweight='bold', zorder=20)
        ax.text(cx, cy-0.15, f'({rooms[i]["area"]:.1f} m²)', ha='center', va='center',
                fontsize=8, color='#999', zorder=20)
    
    # Dimension lines for top 3 rooms
    for i, rr in enumerate(rooms_rot[:3]):
        c = rr['contour']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        w_m = x1 - x0
        h_m = y1 - y0
        
        # Top
        off = 0.2
        ax.annotate('', xy=(x1, y1+off), xytext=(x0, y1+off),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{w_m:.2f} m', ha='center', fontsize=7, color='#666', zorder=15)
        # Right
        ax.annotate('', xy=(x1+off, y1), xytext=(x1+off, y0),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+off+0.1, (y0+y1)/2, f'{h_m:.2f} m', ha='left', va='center', fontsize=7, 
                color='#666', rotation=90, zorder=15)
    
    # Door arcs (between adjacent rooms)
    # Simple heuristic: draw arcs at wall gaps
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 3: continue
        c = rr['contour']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        # Door on bottom of room (simple placement)
        if rooms[i]['area'] > 8:
            door_x = (x0 + x1) / 2
            door_y = y0
            arc = Arc((door_x, door_y), 0.8, 0.8, angle=0, theta1=0, theta2=90,
                      color='#666', lw=0.8, zorder=15)
            ax.add_patch(arc)
    
    ax.set_xlim(rxmin, rxmax)
    ax.set_ylim(rymin, rymax)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    out = '/tmp/v78_polished.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))
    
    # Also save diagnostic
    fig2, axes = plt.subplots(1, 3, figsize=(24, 8))
    ex = [xmin, xmax, zmin, zmax]
    
    axes[0].imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    for wall in all_walls:
        axes[0].plot([wall['start_m'][0], wall['end_m'][0]],
                     [wall['start_m'][1], wall['end_m'][1]], 'g-', lw=2)
    axes[0].set_title(f'Density + {len(all_walls)} walls')
    
    axes[1].imshow(partition, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[1].set_title('Partition')
    
    colors = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9','#E8BAFF','#FFE0BA']
    axes[2].imshow(partition, origin='lower', cmap='gray', extent=ex, aspect='equal', alpha=0.3)
    for i, room in enumerate(rooms):
        if len(room['contour_m']) >= 3:
            poly = MplPoly(room['contour_m'], closed=True,
                          facecolor=colors[i%len(colors)], alpha=0.6, edgecolor='blue', lw=2)
            axes[2].add_patch(poly)
        cx,cy = room['centroid_m']
        axes[2].text(cx, cy, f"{names[i]}\n{room['area']:.1f}m²", ha='center', fontsize=7, fontweight='bold')
    axes[2].set_xlim(ex[0],ex[1]); axes[2].set_ylim(ex[2],ex[3])
    axes[2].set_aspect('equal')
    axes[2].set_title(f'{len(rooms)} rooms')
    
    plt.tight_layout()
    plt.savefig('/tmp/v78_diagnostic.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    main()

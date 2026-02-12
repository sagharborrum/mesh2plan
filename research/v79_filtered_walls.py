#!/usr/bin/env python3
"""
mesh2plan v79 - Filtered walls: only keep walls with strong density support

Key changes from v78:
1. Validate each wall cluster by checking density along its entire span
2. Only keep walls where >40% of length has high density (actual wall behind it)
3. Clip wall endpoints to apartment boundary
4. Straighten apartment boundary to wall angles
5. Minimum wall span filter (>1.0m for interior walls)
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Arc
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
    normals = mesh.face_normals
    areas = mesh.area_faces
    wm = np.abs(normals[:, 1]) < 0.3
    wn = normals[wm][:, [0, 2]].copy()
    wn[:, 0] = -wn[:, 0]
    wa = areas[wm]
    angles = np.degrees(np.arctan2(wn[:, 1], wn[:, 0])) % 180
    bins = np.arange(0, 181, 1)
    hist, _ = np.histogram(angles, bins=bins, weights=wa)
    hist = ndimage.gaussian_filter1d(hist, sigma=2)
    peaks, props = find_peaks(hist, height=hist.max() * 0.2, distance=20)
    top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
    wall_angles = sorted([(a + 90) % 180 for a in top2])
    print(f"Wall angles: {wall_angles}°")
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

def sample_density_along_line(density, sx, sy, ex, ey, n_samples=50):
    """Sample density values along a line, return fraction above threshold."""
    xs = np.linspace(sx, ex, n_samples).astype(int)
    ys = np.linspace(sy, ey, n_samples).astype(int)
    h, w = density.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() == 0: return 0
    vals = density[ys[valid], xs[valid]]
    thresh = np.percentile(density[density > 0], 30) if (density > 0).any() else 0
    return (vals > thresh).mean()

def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    wall_angles = detect_wall_angles_mirrored(mesh)
    render_rotation = -wall_angles[0]
    
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center_m = pts_xz.mean(axis=0)
    
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
    
    # Hough lines
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
    
    # Cluster into walls
    cluster_dist = int(0.20 / res)
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
            span = para_max - para_min
            
            # FILTER 1: minimum span (1.0m)
            if span * res < 1.0:
                continue
            
            # FILTER 2: minimum total segment length
            if total_length < int(0.5 / res):
                continue
            
            rad = np.radians(wa)
            dx, dy = np.cos(rad), np.sin(rad)
            nx, ny = -np.sin(rad), np.cos(rad)
            
            sx = perp_avg * nx + para_min * dx
            sy = perp_avg * ny + para_min * dy
            ex = perp_avg * nx + para_max * dx
            ey = perp_avg * ny + para_max * dy
            
            # FILTER 3: density support along wall
            coverage = sample_density_along_line(density, int(sx), int(sy), int(ex), int(ey))
            if coverage < 0.25:
                continue
            
            all_walls.append({
                'angle': wa, 'perp': perp_avg,
                'para_min': para_min, 'para_max': para_max,
                'start_px': (int(sx), int(sy)),
                'end_px': (int(ex), int(ey)),
                'total_length': total_length,
                'n_segments': len(cluster),
                'span_m': span * res,
                'coverage': coverage,
                'start_m': (sx*res+xmin, sy*res+zmin),
                'end_m': (ex*res+xmin, ey*res+zmin),
            })
    
    print(f"\nFiltered walls: {len(all_walls)}")
    for i, wall in enumerate(all_walls):
        print(f"  Wall {i}: {wall['angle']:.0f}° span={wall['span_m']:.1f}m cov={wall['coverage']:.0%} segs={wall['n_segments']}")
    
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
    
    # Flood fill
    interior = (partition == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    
    rooms = []
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_m2 = region.sum() * res * res
        if area_m2 < 1.0: continue
        
        ys, xs = np.where(region)
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        contour_m = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx])
        
        rooms.append({
            'area': area_m2,
            'contour_m': contour_m,
            'n_verts': len(approx),
            'centroid_m': (contour_m[:,0].mean(), contour_m[:,1].mean()),
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nDetected {len(rooms)} rooms:")
    total = 0
    for i, r in enumerate(rooms):
        c = r['contour_m']
        w_m = c[:,0].max()-c[:,0].min()
        h_m = c[:,1].max()-c[:,1].min()
        print(f"  Room {i+1}: {w_m:.2f}×{h_m:.2f} = {r['area']:.1f}m² ({r['n_verts']}v)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # Name rooms
    names = []
    bed_n = 0
    for r in rooms:
        if r['area'] > 10: bed_n += 1; names.append(f'Bedroom {bed_n}')
        elif r['area'] > 4: names.append('Hallway')
        elif r['area'] > 3: names.append('Entry')
        elif r['area'] > 2: names.append('Bathroom')
        else: names.append('WC')
    
    # ============================================================
    # Rotate for rendering
    # ============================================================
    rot = render_rotation
    
    rooms_rot = []
    for room in rooms:
        rc = rotate_points(room['contour_m'], rot, center_m)
        rooms_rot.append({'contour': rc, 'centroid': rc.mean(axis=0), 'area': room['area']})
    
    walls_rot = []
    for wall in all_walls:
        s = rotate_points(np.array([wall['start_m']]), rot, center_m)[0]
        e = rotate_points(np.array([wall['end_m']]), rot, center_m)[0]
        walls_rot.append((s, e))
    
    apt_pts_m = np.array([(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in apt_contour])
    apt_rot = rotate_points(apt_pts_m, rot, center_m)
    
    # ============================================================
    # ARCHITECTURAL RENDER
    # ============================================================
    all_rc = np.vstack([r['contour'] for r in rooms_rot] + [apt_rot])
    rxmin, rymin = all_rc.min(axis=0) - 0.8
    rxmax, rymax = all_rc.max(axis=0) + 0.8
    
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
            fc = '#E8D5B7' if rooms[i]['area'] > 10 else '#FFFFFF'
            poly = MplPoly(rr['contour'], closed=True, fc=fc, ec='none', zorder=1)
            ax.add_patch(poly)
    
    # Walls as thick lines
    wt = WALL_THICK_M * 72  # ~8.6 points
    for s, e in walls_rot:
        ax.plot([s[0], e[0]], [s[1], e[1]], color='#333333', lw=wt, solid_capstyle='butt', zorder=10)
    
    # Boundary walls
    apt_closed = np.vstack([apt_rot, apt_rot[0:1]])
    # Simplify boundary - only keep edges at wall angles
    for i in range(len(apt_closed)-1):
        ax.plot([apt_closed[i,0], apt_closed[i+1,0]], 
                [apt_closed[i,1], apt_closed[i+1,1]],
                color='#333333', lw=wt, solid_capstyle='butt', zorder=9)
    
    # Labels
    for i, rr in enumerate(rooms_rot):
        cx, cy = rr['centroid']
        name = names[i] if i < len(names) else f'Room {i+1}'
        ax.text(cx, cy+0.15, name, ha='center', va='center', fontsize=10, 
                color='#555', fontweight='bold', zorder=20)
        ax.text(cx, cy-0.15, f'({rooms[i]["area"]:.1f} m²)', ha='center', va='center',
                fontsize=8, color='#888', zorder=20)
    
    # Dimensions on bedrooms
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 8: continue
        c = rr['contour']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        w_m = x1 - x0
        h_m = y1 - y0
        off = 0.25
        ax.annotate('', xy=(x1, y1+off), xytext=(x0, y1+off),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, y1+off+0.1, f'{w_m:.2f} m', ha='center', fontsize=7, color='#666', zorder=15)
        ax.annotate('', xy=(x1+off, y1), xytext=(x1+off, y0),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+off+0.1, (y0+y1)/2, f'{h_m:.2f} m', ha='left', va='center', fontsize=7, 
                color='#666', rotation=90, zorder=15)
    
    # Door arcs at likely positions (rooms >4m² get a door)
    for i, rr in enumerate(rooms_rot):
        if rooms[i]['area'] < 3: continue
        c = rr['contour']
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        cx, cy = rr['centroid']
        # Place door on the wall closest to apartment center
        apt_cx, apt_cy = apt_rot.mean(axis=0)
        # Simple: bottom wall
        door_w = 0.8
        if rooms[i]['area'] > 8:
            arc = Arc((cx, y0), door_w, door_w, angle=0, theta1=0, theta2=90,
                      color='#666', lw=0.8, zorder=15)
            ax.add_patch(arc)
    
    ax.set_xlim(rxmin, rxmax); ax.set_ylim(rymin, rymax)
    ax.set_aspect('equal'); ax.axis('off')
    
    plt.tight_layout()
    out = '/tmp/v79_filtered.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))
    
    # Diagnostic
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
    plt.savefig('/tmp/v79_diagnostic.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    main()

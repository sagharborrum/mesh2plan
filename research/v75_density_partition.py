#!/usr/bin/env python3
"""
mesh2plan v75 - Direct density wall extraction + morphological partition

Key improvements over v74:
- Use wall density directly (threshold + morphology) instead of Hough lines
- Directional closing (H then V) to bridge door gaps
- Lower area threshold (0.5m²) to capture small rooms (WC ~2m²)
- Better architectural rendering matching reference style
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Arc, FancyArrowPatch
from matplotlib.collections import PatchCollection
import cv2
from scipy import ndimage
from pathlib import Path
import shutil

RESOLUTION = 0.012  # 1.2cm for better detail
ANGLE = 60.0  # rotation to align walls H/V
WALL_THICKNESS_M = 0.12  # wall thickness in meters

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

def main():
    mesh_path = Path('../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Project to XZ, negate X (mirror fix), rotate to align walls
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center = pts_xz.mean(axis=0)
    all_rot = rotate_points(pts_xz, -ANGLE, center)
    
    # Wall faces (normal nearly horizontal)
    normals = mesh.face_normals
    wall_mask_faces = np.abs(normals[:, 1]) < 0.3
    wall_areas = mesh.area_faces[wall_mask_faces]
    wall_c = mesh.triangles_center[wall_mask_faces][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -ANGLE, center)
    
    # Grid setup
    res = RESOLUTION
    xmin, zmin = all_rot.min(axis=0) - 0.5
    xmax, zmax = all_rot.max(axis=0) + 0.5
    w = int((xmax - xmin) / res)
    h = int((zmax - zmin) / res)
    
    # Wall density (area-weighted)
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / res).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / res).astype(int), 0, h - 1)
    np.add.at(density, (py, px), wall_areas)
    density = cv2.GaussianBlur(density, (3, 3), 0.8)
    print(f"Grid: {w}×{h}, resolution: {res}m")
    
    # ============================================================
    # STEP 1: Wall mask from density
    # ============================================================
    # Adaptive threshold - walls are high density
    wall_pct = np.percentile(density[density > 0], 40)
    wall_mask = (density > wall_pct).astype(np.uint8) * 255
    
    # Thin walls to single-pixel ridges using skeletonization
    # Then thicken to consistent width
    wall_px = max(3, int(WALL_THICKNESS_M / res))
    
    # Directional closing to bridge gaps (doors are ~0.8m)
    door_gap_px = int(0.6 / res)  # slightly less than door width
    
    # Horizontal closing (bridges vertical door gaps in horizontal walls)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (door_gap_px, 1))
    wall_h = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, h_kernel)
    
    # Vertical closing
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, door_gap_px))
    wall_v = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, v_kernel)
    
    # Combine: use closed version where original wall exists nearby
    wall_closed = np.maximum(wall_h, wall_v)
    
    # Clean up small noise
    wall_closed = cv2.morphologyEx(wall_closed, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # ============================================================
    # STEP 2: Apartment boundary
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
    # STEP 3: Build partition mask
    # ============================================================
    partition = np.zeros((h, w), dtype=np.uint8)
    
    # Draw apartment boundary as thick wall
    cv2.drawContours(partition, [apt_contour], 0, 255, wall_px + 2)
    
    # Fill outside
    outside = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(partition, outside, (0, 0), 128)
    
    # Add interior walls from closed wall mask (only inside apartment)
    interior_wall = wall_closed.copy()
    interior_wall[outside[1:-1, 1:-1] == 1] = 0  # don't add outside
    
    # Dilate walls slightly to ensure they connect to boundary
    interior_wall = cv2.dilate(interior_wall, 
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    partition[interior_wall > 0] = 255
    partition[outside[1:-1, 1:-1] == 1] = 128  # restore outside
    
    # ============================================================
    # STEP 4: Flood fill rooms
    # ============================================================
    interior = (partition == 0).astype(np.uint8)
    labeled, n_labels = ndimage.label(interior)
    print(f"Connected regions: {n_labels}")
    
    rooms = []
    for label_id in range(1, n_labels + 1):
        region = (labeled == label_id)
        area_m2 = region.sum() * res * res
        if area_m2 < 0.5:
            continue
        
        ys, xs = np.where(region)
        bbox = (xs.min()*res+xmin, ys.min()*res+zmin, xs.max()*res+xmin, ys.max()*res+zmin)
        
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        
        contour_m = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in approx]
        
        rooms.append({
            'area': area_m2,
            'bbox': bbox,
            'width': bbox[2]-bbox[0],
            'height': bbox[3]-bbox[1],
            'contour_m': contour_m,
            'contour_px': approx,
            'n_verts': len(approx),
            'centroid': ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2),
            'label_id': label_id,
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    print(f"\nDetected {len(rooms)} rooms:")
    total = 0
    for i, r in enumerate(rooms):
        print(f"  Room {i+1}: {r['width']:.2f}×{r['height']:.2f} = {r['area']:.1f}m² ({r['n_verts']}v)")
        total += r['area']
    print(f"  Total: {total:.1f}m²")
    
    # ============================================================
    # STEP 5: Name rooms based on size/position
    # ============================================================
    room_names = []
    for i, r in enumerate(rooms):
        if r['area'] > 12:
            room_names.append('Bedroom')
        elif r['area'] > 4:
            room_names.append('Hallway')
        elif r['area'] > 3:
            room_names.append('Entry')
        elif r['area'] > 2:
            room_names.append('Bathroom')
        else:
            room_names.append('WC')
    
    # ============================================================
    # RENDER: 6-panel diagnostic + architectural
    # ============================================================
    ex = [xmin, xmax, zmin, zmax]
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Panel 1: Wall density
    axes[0,0].imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    axes[0,0].set_title('Wall density')
    
    # Panel 2: Wall mask (threshold)
    axes[0,1].imshow(wall_mask, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[0,1].set_title(f'Wall mask (threshold)')
    
    # Panel 3: Closed wall mask
    axes[0,2].imshow(wall_closed, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[0,2].set_title('Wall mask (directional close)')
    
    # Panel 4: Partition mask
    axes[1,0].imshow(partition, origin='lower', cmap='gray', extent=ex, aspect='equal')
    axes[1,0].set_title('Partition mask')
    
    # Panel 5: Rooms colored
    axes[1,1].imshow(partition, origin='lower', cmap='gray', extent=ex, aspect='equal', alpha=0.3)
    colors = ['#FFB3BA','#BAE1FF','#FFFFBA','#BAFFC9','#E8BAFF','#FFE0BA']
    for i, room in enumerate(rooms):
        if len(room['contour_m']) >= 3:
            poly = MplPoly(room['contour_m'], closed=True,
                          facecolor=colors[i%len(colors)], alpha=0.6, edgecolor='blue', lw=2)
            axes[1,1].add_patch(poly)
        cx,cy = room['centroid']
        name = room_names[i] if i < len(room_names) else f'Room {i+1}'
        axes[1,1].text(cx, cy, f"{name}\n{room['area']:.1f}m²", ha='center', va='center',
                       fontsize=8, fontweight='bold')
    axes[1,1].set_xlim(ex[0],ex[1]); axes[1,1].set_ylim(ex[2],ex[3])
    axes[1,1].set_aspect('equal')
    axes[1,1].set_title(f'{len(rooms)} rooms')
    
    # Panel 6: Architectural render
    ax = axes[1,2]
    ax.set_facecolor('#F8F8F8')
    
    # Grid lines
    for x in np.arange(int(xmin)-1, int(xmax)+2, 0.5):
        ax.axvline(x, color='#EEEEEE', lw=0.3, zorder=0)
    for y in np.arange(int(zmin)-1, int(zmax)+2, 0.5):
        ax.axhline(y, color='#EEEEEE', lw=0.3, zorder=0)
    
    # Room fills - wood for bedrooms, white for others
    for i, room in enumerate(rooms):
        if len(room['contour_m']) >= 3:
            fc = '#E8D5B7' if room['area'] > 12 else '#FFFFFF'
            poly = MplPoly(room['contour_m'], closed=True, fc=fc, ec='none', zorder=1)
            ax.add_patch(poly)
    
    # Draw walls from partition mask
    wall_region = (partition == 255).astype(np.uint8) * 255
    wcnts, _ = cv2.findContours(wall_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in wcnts:
        if cv2.contourArea(cnt) < 30: continue
        pts = [(p[0][0]*res+xmin, p[0][1]*res+zmin) for p in cnt]
        if len(pts) >= 3:
            poly = MplPoly(pts, closed=True, fc='#333333', ec='#222222', lw=0.3, zorder=10)
            ax.add_patch(poly)
    
    # Room labels
    for i, room in enumerate(rooms):
        cx, cy = room['centroid']
        name = room_names[i] if i < len(room_names) else f'Room {i+1}'
        ax.text(cx, cy+0.15, name, ha='center', va='center', fontsize=9, color='#666', zorder=20)
        ax.text(cx, cy-0.15, f'({room["area"]:.1f} m²)', ha='center', va='center', fontsize=7, color='#999', zorder=20)
    
    # Dimensions on largest rooms
    for i, room in enumerate(rooms[:3]):
        x0, z0, x1, z1 = room['bbox']
        w_m = room['width']
        h_m = room['height']
        # Top dimension
        ax.annotate('', xy=(x1, z1+0.15), xytext=(x0, z1+0.15),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text((x0+x1)/2, z1+0.25, f'{w_m:.2f} m', ha='center', va='bottom', fontsize=6, color='#666', zorder=15)
        # Right dimension
        ax.annotate('', xy=(x1+0.15, z1), xytext=(x1+0.15, z0),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8), zorder=15)
        ax.text(x1+0.25, (z0+z1)/2, f'{h_m:.2f} m', ha='left', va='center', fontsize=6, color='#666', rotation=90, zorder=15)
    
    ax.set_xlim(xmin, xmax); ax.set_ylim(zmin, zmax)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Architectural')
    
    plt.tight_layout()
    out = '/tmp/v75_density_partition.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    shutil.copy(out, str(Path.home() / '.openclaw/workspace/latest_floorplan.png'))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v31 - Hallway-First Approach

Strategy: Identify the hallway FIRST using skeleton + distance transform,
then use the hallway as a natural separator for rooms.

1. Build room mask
2. Skeletonize → find the "spine" of the apartment
3. The narrow part of the spine (low distance transform) = hallway
4. Grow hallway to wall boundaries
5. Subtract hallway from mask → remaining connected components = rooms
6. Clean polygons, doors, windows
7. Architectural rendering

This avoids the v29 problem of hallway being absorbed into rooms during merging.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndlabel, uniform_filter1d
from skimage.morphology import skeletonize


# ─── Utilities ───

def detect_up_axis(mesh):
    ranges = [np.ptp(mesh.vertices[:, i]) for i in range(3)]
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges): return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges): return 2, 'Z'
    return np.argmin(ranges), ['X','Y','Z'][np.argmin(ranges)]

def project_vertices(mesh, up_axis_idx):
    v = mesh.vertices
    if up_axis_idx == 1: return v[:, 0], v[:, 2]
    elif up_axis_idx == 2: return v[:, 0], v[:, 1]
    return v[:, 1], v[:, 2]

def find_dominant_angle(rx, rz, cell=0.02):
    x_min, x_max = rx.min(), rx.max()
    z_min, z_max = rz.min(), rz.max()
    nx = int((x_max - x_min) / cell) + 1
    nz = int((z_max - z_min) / cell) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell).astype(int), 0, nx - 1)
    zi = np.clip(((rz - z_min) / cell).astype(int), 0, nz - 1)
    np.add.at(img, (zi, xi), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    mask = mag > np.percentile(mag, 80)
    folded = ang[mask] % 90
    hist, bins = np.histogram(folded, bins=90, range=(0, 90))
    peak = np.argmax(hist)
    return (bins[peak] + bins[peak + 1]) / 2

def build_density_image(rx, rz, cell_size=0.02, margin=0.3):
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx - 1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz - 1)
    np.add.at(img, (zi, xi), 1)
    return img, x_min, z_min, cell_size

def build_room_mask(density_img, cell_size):
    occupied = (density_img > 0).astype(np.uint8)
    k_size = max(3, int(0.15 / cell_size)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    lbl, n = ndlabel(filled)
    if n > 1:
        sizes = ndimage.sum(filled, lbl, range(1, n+1))
        filled = (lbl == np.argmax(sizes)+1).astype(np.uint8)
    return filled

def hough_wall_positions(density_img, x_min, z_min, cell_size, nms_dist=0.3):
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)
    proj_x = smoothed.sum(axis=0)
    proj_z = smoothed.sum(axis=1)
    def find_peaks_nms(profile, origin, cs, min_dist_cells):
        prof = uniform_filter1d(profile.astype(float), size=5)
        local_max = maximum_filter(prof, size=max(3, min_dist_cells)) == prof
        threshold = prof.mean() + 0.15 * prof.std()
        peaks = np.where(local_max & (prof > threshold))[0]
        positions = origin + peaks * cs
        if len(positions) == 0: return np.array([])
        strengths = prof[peaks]
        order = np.argsort(-strengths)
        kept = []
        for i in order:
            pos = positions[i]
            if any(abs(pos - k) < nms_dist for k in kept): continue
            kept.append(pos)
        return np.sort(kept)
    min_dist = int(nms_dist / cell_size)
    return find_peaks_nms(proj_x, x_min, cell_size, min_dist), \
           find_peaks_nms(proj_z, z_min, cell_size, min_dist)


# ─── Hallway-First Detection ───

def find_hallway(room_mask, density_img, cell_size, 
                  hallway_max_width_m=1.4, min_hallway_length_m=2.0):
    """
    Find the hallway by:
    1. Distance transform → find narrow regions
    2. Skeleton → find the spine
    3. Narrow spine pixels = hallway center
    4. Grow hallway to fill the narrow region between walls
    """
    # Distance transform
    dist = cv2.distanceTransform(room_mask, cv2.DIST_L2, 5)
    
    # Skeleton
    skel = skeletonize(room_mask > 0).astype(np.uint8)
    
    # Narrow skeleton: skeleton pixels where distance < hallway half-width
    hallway_half_width_px = hallway_max_width_m / (2 * cell_size)
    narrow_skel = (skel > 0) & (dist < hallway_half_width_px) & (dist > 0)
    narrow_skel = narrow_skel.astype(np.uint8)
    
    # Connect nearby narrow skeleton pieces
    k_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    narrow_connected = cv2.morphologyEx(narrow_skel, cv2.MORPH_CLOSE, k_connect)
    
    # Label connected narrow regions
    lbl, n = ndlabel(narrow_connected)
    
    # Find significant hallway segments (long enough)
    min_length_px = min_hallway_length_m / cell_size
    hallway_mask = np.zeros_like(room_mask)
    
    for i in range(1, n + 1):
        component = (lbl == i)
        rows, cols = np.where(component)
        if len(rows) == 0: continue
        
        # Check length (max dimension)
        h = rows.max() - rows.min()
        w = cols.max() - cols.min()
        length = max(h, w)
        
        if length >= min_length_px:
            # Grow this hallway segment to fill the narrow region
            # Use the distance transform: hallway = all pixels close to the skeleton
            # within the narrow region
            grown = cv2.dilate(component.astype(np.uint8), 
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            
            # Grow iteratively: add pixels that are in room_mask AND within hallway width
            for _ in range(int(hallway_half_width_px)):
                expanded = cv2.dilate(grown, np.ones((3,3), np.uint8))
                # Only expand within room mask and narrow regions
                expanded = expanded & room_mask & (dist < hallway_half_width_px * 1.2).astype(np.uint8)
                if np.array_equal(expanded, grown):
                    break
                grown = expanded
            
            hallway_mask = hallway_mask | grown
    
    # Clean up: open to remove noise, close to fill gaps
    hallway_mask = cv2.morphologyEx(hallway_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    hallway_mask = cv2.morphologyEx(hallway_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    
    # Ensure hallway is within room mask
    hallway_mask = hallway_mask & room_mask
    
    return hallway_mask, dist, skel


def extract_rooms_after_hallway(room_mask, hallway_mask, density_img, cell_size, 
                                 x_min, z_min, x_walls, z_walls, min_area=2.0):
    """Extract rooms by subtracting hallway from room mask."""
    # Room mask minus hallway
    room_only = room_mask.copy()
    room_only[hallway_mask > 0] = 0
    
    # Slight erosion to separate rooms that touch at corners
    room_only = cv2.erode(room_only, np.ones((3,3), np.uint8))
    room_only = cv2.dilate(room_only, np.ones((3,3), np.uint8))
    
    # Label connected components
    lbl, n = ndlabel(room_only)
    
    rooms = []
    for i in range(1, n + 1):
        mask = (lbl == i).astype(np.uint8)
        area = np.sum(mask) * cell_size * cell_size
        if area < min_area:
            continue
        rooms.append({'mask': mask, 'area_m2': area})
    
    rooms.sort(key=lambda r: -r['area_m2'])
    return rooms


def mask_to_polygon(mask, density_img, x_min, z_min, cell_size, x_walls, z_walls):
    """Convert mask to polygon with wall snapping."""
    rows, cols = np.where(mask)
    if len(rows) == 0: return []
    
    masked_density = density_img * mask
    dr, dc = np.where(masked_density > 0)
    if len(dr) > 0:
        wx_min = x_min + np.percentile(dc, 1) * cell_size
        wx_max = x_min + np.percentile(dc, 99) * cell_size
        wz_min = z_min + np.percentile(dr, 1) * cell_size
        wz_max = z_min + np.percentile(dr, 99) * cell_size
    else:
        wx_min = x_min + cols.min() * cell_size
        wx_max = x_min + cols.max() * cell_size
        wz_min = z_min + rows.min() * cell_size
        wz_max = z_min + rows.max() * cell_size
    
    def snap(val, positions, max_snap=0.25):
        if len(positions) == 0: return val
        dists = np.abs(np.array(positions) - val)
        idx = np.argmin(dists)
        return float(positions[idx]) if dists[idx] < max_snap else val
    
    bbox_area = (cols.max()-cols.min()+1) * (rows.max()-rows.min()+1)
    fill_ratio = len(rows) / max(1, bbox_area)
    
    x0 = snap(wx_min, x_walls)
    x1 = snap(wx_max, x_walls)
    z0 = snap(wz_min, z_walls)
    z1 = snap(wz_max, z_walls)
    
    if fill_ratio > 0.75:
        return [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
    
    # Contour for non-rectangular
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
    
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(3, int(0.2 / cell_size))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    poly = []
    for pt in approx.reshape(-1, 2):
        wx = snap(x_min + pt[0] * cell_size, x_walls)
        wz = snap(z_min + pt[1] * cell_size, z_walls)
        poly.append([wx, wz])
    
    # Make rectilinear
    rect = [poly[0]]
    for i in range(1, len(poly)):
        prev = rect[-1]; curr = poly[i]
        dx, dz = abs(curr[0]-prev[0]), abs(curr[1]-prev[1])
        if dx > 0.05 and dz > 0.05:
            if dx < dz: rect.append([curr[0], prev[1]])
            else: rect.append([prev[0], curr[1]])
        rect.append(curr)
    
    # Remove collinear
    cleaned = [rect[0]]
    for p in rect[1:]:
        if abs(p[0]-cleaned[-1][0]) > 0.01 or abs(p[1]-cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    if len(cleaned) < 3: return [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]
    
    result = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i-1)%n]; curr = cleaned[i]; nxt = cleaned[(i+1)%n]
        cross = (curr[0]-prev[0])*(nxt[1]-curr[1]) - (curr[1]-prev[1])*(nxt[0]-curr[0])
        if abs(cross) > 0.001: result.append(curr)
    
    return result if len(result) >= 3 else [[x0,z0], [x1,z0], [x1,z1], [x0,z1]]


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    return abs(sum(poly[i][0]*poly[(i+1)%n][1] - poly[(i+1)%n][0]*poly[i][1] for i in range(n))) / 2

def polygon_centroid(poly):
    n = len(poly)
    if n == 0: return 0, 0
    return sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n


# ─── Rendering ───

ROOM_COLORS = ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#F3E5F5', '#FFFDE7', '#E0F7FA']
HALLWAY_COLOR = '#F5F5F5'
CLOSET_COLOR = '#EFEBE9'

def render_debug(density_img, room_mask, dist, skel, hallway_mask, rooms_data,
                  x_min, z_min, cs, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    extent = [x_min, x_min + density_img.shape[1]*cs, z_min, z_min + density_img.shape[0]*cs]
    
    axes[0,0].imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal')
    axes[0,0].set_title('1. Density')
    
    axes[0,1].imshow(dist * room_mask, origin='lower', extent=extent, cmap='hot', aspect='equal')
    axes[0,1].set_title('2. Distance Transform')
    
    # Skeleton overlay on dist
    axes[0,2].imshow(dist * room_mask, origin='lower', extent=extent, cmap='gray', aspect='equal', alpha=0.5)
    skel_overlay = np.zeros((*skel.shape, 4))
    skel_overlay[skel > 0] = [0, 1, 0, 1]
    axes[0,2].imshow(skel_overlay, origin='lower', extent=extent, aspect='equal')
    axes[0,2].set_title('3. Skeleton on Distance')
    
    # Hallway mask
    axes[1,0].imshow(room_mask, origin='lower', extent=extent, cmap='gray', aspect='equal', alpha=0.3)
    hallway_overlay = np.zeros((*hallway_mask.shape, 4))
    hallway_overlay[hallway_mask > 0] = [1, 0, 0, 0.7]
    axes[1,0].imshow(hallway_overlay, origin='lower', extent=extent, aspect='equal')
    axes[1,0].set_title('4. Hallway (red) on Mask')
    
    # Rooms
    ax = axes[1,1]
    ax.set_facecolor('white')
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        if rd['type'] == 'hallway': color = HALLWAY_COLOR
        elif rd['type'] == 'closet': color = CLOSET_COLOR
        else:
            color = ROOM_COLORS[ci % len(ROOM_COLORS)]
            ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.8)
        n = len(poly)
        for k in range(n):
            p1, p2 = poly[k], poly[(k+1)%n]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=2)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²", ha='center', va='center',
                fontsize=7, fontweight='bold', color='#333')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('5. Room Polygons')
    
    # Overlay
    ax = axes[1,2]
    ax.imshow(np.log1p(density_img), origin='lower', extent=extent, cmap='hot', aspect='equal', alpha=0.4)
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        if rd['type'] == 'hallway': color = HALLWAY_COLOR
        elif rd['type'] == 'closet': color = CLOSET_COLOR
        else:
            color = ROOM_COLORS[ci % len(ROOM_COLORS)]
            ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.4)
        n = len(poly)
        for k in range(n):
            p1, p2 = poly[k], poly[(k+1)%n]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'w-', lw=2)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title('6. Overlay')
    
    fig.suptitle('v31 Hallway-First — Debug', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def render_architectural(rooms_data, doors, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    
    all_x, all_z = [], []
    for rd in rooms_data:
        for p in rd['polygon']:
            all_x.append(p[0]); all_z.append(p[1])
    if not all_x:
        plt.savefig(output_path, dpi=200); plt.close(); return
    
    margin = 0.8; wall_w = 0.08
    
    ci = 0
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        if rd['type'] == 'hallway': color = HALLWAY_COLOR
        elif rd['type'] == 'closet': color = CLOSET_COLOR
        else:
            color = ROOM_COLORS[ci % len(ROOM_COLORS)]
            ci += 1
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        ax.fill(xs, zs, color=color, alpha=0.9, zorder=1)
    
    for rd in rooms_data:
        poly = rd['polygon']
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i+1)%n]
            if abs(p1[0]-p2[0]) < 0.05:
                x = p1[0]; zl, zh = min(p1[1],p2[1]), max(p1[1],p2[1])
                ax.add_patch(patches.Rectangle((x-wall_w/2, zl), wall_w, zh-zl,
                             facecolor='black', edgecolor='none', zorder=2))
            elif abs(p1[1]-p2[1]) < 0.05:
                z = p1[1]; xl, xh = min(p1[0],p2[0]), max(p1[0],p2[0])
                ax.add_patch(patches.Rectangle((xl, z-wall_w/2), xh-xl, wall_w,
                             facecolor='black', edgecolor='none', zorder=2))
            else:
                ax.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-', lw=3, zorder=2)
    
    for door in doors:
        x, z, w = door['x'], door['z'], door.get('width', 0.8)
        if door['orientation'] == 'vertical':
            ax.add_patch(patches.Rectangle((x-wall_w, z-w/2), wall_w*2, w,
                         facecolor='white', edgecolor='none', zorder=3))
            ax.add_patch(Arc((x, z-w/2), w, w, angle=0, theta1=0, theta2=90,
                         color='#666', lw=1.5, zorder=4))
        else:
            ax.add_patch(patches.Rectangle((x-w/2, z-wall_w), w, wall_w*2,
                         facecolor='white', edgecolor='none', zorder=3))
            ax.add_patch(Arc((x-w/2, z), w, w, angle=0, theta1=0, theta2=90,
                         color='#666', lw=1.5, zorder=4))
    
    for rd in rooms_data:
        poly = rd['polygon']
        if len(poly) < 3: continue
        cx, cz = polygon_centroid(poly)
        fs = 9 if rd['area'] > 5 else 7
        ax.text(cx, cz+0.15, rd['name'], ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#333', zorder=5)
        ax.text(cx, cz-0.15, f"{rd['area']:.1f} m²", ha='center', va='center',
                fontsize=fs-1, color='#666', zorder=5)
    
    # Scale bar
    sx, sy = min(all_x)+0.3, min(all_z)-0.4
    ax.plot([sx, sx+1], [sy, sy], 'k-', lw=2)
    ax.plot([sx, sx], [sy-0.05, sy+0.05], 'k-', lw=2)
    ax.plot([sx+1, sx+1], [sy-0.05, sy+0.05], 'k-', lw=2)
    ax.text(sx+0.5, sy-0.15, '1 m', ha='center', va='top', fontsize=8)
    
    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if rd['type'] == 'room')
    nh = sum(1 for rd in rooms_data if rd['type'] == 'hallway')
    nc = sum(1 for rd in rooms_data if rd['type'] == 'closet')
    parts = []
    if nr: parts.append(f"{nr} room{'s' if nr!=1 else ''}")
    if nh: parts.append(f"{nh} hallway")
    if nc: parts.append(f"{nc} closet")
    
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_z)-margin, max(all_z)+margin)
    ax.set_aspect('equal')
    ax.set_title(f"Floor Plan\n{' · '.join(parts)} · {total:.1f} m²",
                 fontsize=13, fontweight='bold', pad=15, color='#333')
    ax.axis('off')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--hallway-width', type=float, default=1.4, help='Max hallway width (m)')
    parser.add_argument('--min-room-area', type=float, default=2.0)
    parser.add_argument('-o', '--output', default='results/v31_hallway_first')
    args = parser.parse_args()
    
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = Path(args.mesh).stem
    
    print(f"Loading: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh')
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    up_idx, _ = detect_up_axis(mesh)
    rx, rz = project_vertices(mesh, up_idx)
    angle = find_dominant_angle(rx, rz, cell=args.cell)
    print(f"  Rotation: {angle:.1f}°")
    
    cos_a, sin_a = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a
    
    density_img, x_min, z_min, cs = build_density_image(rx2, rz2, cell_size=args.cell)
    room_mask = build_room_mask(density_img, cs)
    x_walls, z_walls = hough_wall_positions(density_img, x_min, z_min, cs)
    
    mask_area = np.sum(room_mask) * cs * cs
    print(f"  Room mask: {mask_area:.1f}m²")
    print(f"  Hough walls: {len(x_walls)}X, {len(z_walls)}Z")
    
    # Step 1: Find hallway
    print("Finding hallway...")
    hallway_mask, dist, skel = find_hallway(room_mask, density_img, cs, 
                                             hallway_max_width_m=args.hallway_width)
    hallway_area = np.sum(hallway_mask) * cs * cs
    print(f"  Hallway area: {hallway_area:.1f}m²")
    
    # Step 2: Extract rooms
    print("Extracting rooms...")
    rooms = extract_rooms_after_hallway(room_mask, hallway_mask, density_img, cs,
                                        x_min, z_min, x_walls, z_walls,
                                        min_area=args.min_room_area)
    print(f"  Rooms found: {len(rooms)}")
    
    # Step 3: Build room data
    rooms_data = []
    
    # Hallway polygon
    if hallway_area > 1.0:
        hall_poly = mask_to_polygon(hallway_mask, density_img, x_min, z_min, cs, x_walls, z_walls)
        hall_area = compute_polygon_area(hall_poly) if len(hall_poly) >= 3 else hallway_area
        rooms_data.append({
            'name': 'Hallway', 'polygon': hall_poly, 'area': hall_area,
            'type': 'hallway', 'mask': hallway_mask
        })
    
    # Room polygons
    rn = 1
    for r in rooms:
        poly = mask_to_polygon(r['mask'], density_img, x_min, z_min, cs, x_walls, z_walls)
        area = compute_polygon_area(poly) if len(poly) >= 3 else r['area_m2']
        
        if area < 3.0:
            name = 'Closet'
            rtype = 'closet'
        else:
            name = f'Room {rn}'
            rtype = 'room'
            rn += 1
        
        rooms_data.append({
            'name': name, 'polygon': poly, 'area': area,
            'type': rtype, 'mask': r['mask']
        })
    
    for rd in rooms_data:
        print(f"  {rd['name']}: {rd['area']:.1f}m² ({rd['type']}) poly={len(rd['polygon'])}v")
    
    # Doors (simplified)
    doors = []
    
    total = sum(rd['area'] for rd in rooms_data)
    nr = sum(1 for rd in rooms_data if rd['type'] == 'room')
    nh = sum(1 for rd in rooms_data if rd['type'] == 'hallway')
    nc = sum(1 for rd in rooms_data if rd['type'] == 'closet')
    
    print(f"\n=== v31 RESULTS ===")
    print(f"  {nr} rooms, {nh} hallways, {nc} closets — {total:.1f}m²")
    
    # Render
    render_debug(density_img, room_mask, dist, skel, hallway_mask, rooms_data,
                  x_min, z_min, cs, out_dir / f"v31_{mesh_name}_debug.png")
    render_architectural(rooms_data, doors, out_dir / f"v31_{mesh_name}_plan.png")
    
    # JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    
    results = {
        'approach': 'v31_hallway_first',
        'rooms': [{
            'name': rd['name'], 'area_m2': round(rd['area'],1), 'type': rd['type'],
            'polygon': [[round(p[0],3), round(p[1],3)] for p in rd['polygon']],
        } for rd in rooms_data],
        'doors': doors,
    }
    with open(out_dir / f"v31_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    
    print("Done!")

if __name__ == '__main__':
    main()

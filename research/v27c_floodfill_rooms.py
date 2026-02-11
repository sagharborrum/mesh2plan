#!/usr/bin/env python3
"""
mesh2plan v27c - Flood-Fill Room Detection

Strategy: Instead of wall-based splitting, find rooms by eroding the density mask
to separate at narrow doorways, then use connected components.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage


# ─── Base functions (same as v27) ───

def detect_up_axis(mesh):
    ranges = [np.ptp(mesh.vertices[:, i]) for i in range(3)]
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges):
        return 1, 'Y'
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges):
        return 2, 'Z'
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
    xi = np.clip(((rx - x_min) / cell).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell).astype(int), 0, nz-1)
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
    return (bins[peak] + bins[peak+1]) / 2

def build_density_image(rx, rz, cell_size=0.01, margin=0.3):
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz-1)
    np.add.at(img, (zi, xi), 1)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img, x_min, z_min, cell_size

def multi_edge_detection(img):
    if img.max() > 0:
        img_norm = (img / np.percentile(img[img > 0], 95) * 255).clip(0, 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian = np.abs(cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(img_blur, cv2.MORPH_GRADIENT, kernel).astype(np.float64)
    def normalize(x):
        mx = x.max()
        return x / mx if mx > 0 else x
    combined = 0.5 * normalize(sobel_mag) + 0.25 * normalize(laplacian) + 0.25 * normalize(morph_grad)
    sobel_dir = np.arctan2(sobel_y, sobel_x)
    return combined, sobel_dir

def non_max_suppression(edge_mag, edge_dir):
    nz, nx = edge_mag.shape
    suppressed = np.zeros_like(edge_mag)
    angle = edge_dir * 180 / np.pi
    angle[angle < 0] += 180
    for i in range(1, nz-1):
        for j in range(1, nx-1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = edge_mag[i, j-1], edge_mag[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = edge_mag[i-1, j+1], edge_mag[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = edge_mag[i-1, j], edge_mag[i+1, j]
            else:
                n1, n2 = edge_mag[i-1, j-1], edge_mag[i+1, j+1]
            if edge_mag[i, j] >= n1 and edge_mag[i, j] >= n2:
                suppressed[i, j] = edge_mag[i, j]
    return suppressed

def extract_wall_segments(nms_edges, x_min, z_min, cell_size, min_length=0.3):
    thresh = max(np.percentile(nms_edges[nms_edges > 0], 70) if np.any(nms_edges > 0) else 0.1, 0.05)
    binary = (nms_edges > thresh).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=int(min_length / cell_size),
                            maxLineGap=int(0.2 / cell_size))
    if lines is None:
        return [], []
    x_positions, z_positions = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        wx1, wz1 = x_min + x1 * cell_size, z_min + y1 * cell_size
        wx2, wz2 = x_min + x2 * cell_size, z_min + y2 * cell_size
        length = math.sqrt((wx2-wx1)**2 + (wz2-wz1)**2)
        if length < min_length: continue
        dx, dz = abs(wx2-wx1), abs(wz2-wz1)
        if dx + dz < 0.01: continue
        angle_mod = math.atan2(min(dx, dz), max(dx, dz)) * 180 / math.pi
        if angle_mod > 15: continue
        if dz > dx:
            x_positions.append(((wx1+wx2)/2, length))
        else:
            z_positions.append(((wz1+wz2)/2, length))
    x_walls = cluster_positions(x_positions, 0.15, min_total_length=0.8)
    z_walls = cluster_positions(z_positions, 0.15, min_total_length=3.0)
    return x_walls, z_walls

def cluster_positions(positions, dist_threshold=0.15, min_total_length=0.8):
    if not positions: return []
    sorted_pos = sorted(positions, key=lambda p: p[0])
    clusters, current = [], [sorted_pos[0]]
    for p in sorted_pos[1:]:
        if abs(p[0] - current[-1][0]) < dist_threshold:
            current.append(p)
        else:
            clusters.append(current); current = [p]
    clusters.append(current)
    result = []
    for cluster in clusters:
        total_len = sum(p[1] for p in cluster)
        if total_len < min_total_length: continue
        avg_pos = sum(p[0]*p[1] for p in cluster) / total_len
        result.append(avg_pos)
    return sorted(result)

def build_room_mask(density_img):
    mask = (density_img > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.dilate(mask, kernel2)
    mask = cv2.erode(mask, kernel2)
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, labeled, range(1, n+1))
        largest = np.argmax(sizes) + 1
        filled = (labeled == largest).astype(np.uint8)
    return filled


# ─── v27c: Flood-fill room detection ───

def find_rooms_by_erosion(room_mask, cell_size, nms_edges, x_min, z_min):
    """Find rooms by progressive erosion to separate at doorways."""
    nz_img, nx_img = room_mask.shape
    
    # Strategy: erode with increasing kernel sizes until we get reasonable room count
    # Doorways are typically 0.7-1.0m wide, walls are ~0.1-0.2m thick
    # At 1cm cell size, doorway = 70-100 pixels wide
    # We need erosion radius of ~half doorway width to pinch them off
    
    # Also use edge information: draw strong edges as barriers before flood fill
    # This helps separate rooms even without full erosion
    
    # Build edge barrier: threshold NMS edges and use as walls
    if np.any(nms_edges > 0):
        edge_thresh = np.percentile(nms_edges[nms_edges > 0], 80)
    else:
        edge_thresh = 0.1
    edge_barrier = (nms_edges > edge_thresh).astype(np.uint8)
    # Thicken edge barrier slightly
    edge_barrier = cv2.dilate(edge_barrier, np.ones((3, 3), np.uint8))
    
    best_labeled = None
    best_n = 0
    best_erosion = 0
    
    # Try different erosion sizes
    for erosion_px in range(15, 55, 5):  # 0.15m to 0.55m radius
        eroded = room_mask.copy()
        
        # Apply edge barrier: remove edge pixels from mask
        eroded = eroded & (~edge_barrier).astype(np.uint8)
        
        # Erode
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px*2+1, erosion_px*2+1))
        eroded = cv2.erode(eroded, kernel)
        
        # Connected components
        labeled, n_comp = ndimage.label(eroded)
        
        # Filter tiny components
        min_pixels = int(1.0 / (cell_size * cell_size))  # 1m² minimum
        valid = 0
        for i in range(1, n_comp + 1):
            if np.sum(labeled == i) >= min_pixels:
                valid += 1
        
        erosion_m = erosion_px * cell_size
        print(f"    Erosion {erosion_m:.2f}m: {n_comp} components, {valid} valid (>1m²)")
        
        # We want 2-8 rooms for a multiroom case, or 1 for single room
        if 2 <= valid <= 8 and valid > best_n:
            best_n = valid
            best_erosion = erosion_px
            best_labeled = labeled.copy()
        elif valid == 1 and best_n == 0:
            best_n = 1
            best_erosion = erosion_px
            best_labeled = labeled.copy()
    
    # If no good erosion found, try without edge barrier
    if best_n <= 1:
        print("  Trying without edge barrier...")
        for erosion_px in range(20, 60, 5):
            eroded = room_mask.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px*2+1, erosion_px*2+1))
            eroded = cv2.erode(eroded, kernel)
            labeled, n_comp = ndimage.label(eroded)
            min_pixels = int(1.0 / (cell_size * cell_size))
            valid = 0
            for i in range(1, n_comp + 1):
                if np.sum(labeled == i) >= min_pixels:
                    valid += 1
            erosion_m = erosion_px * cell_size
            print(f"    Erosion {erosion_m:.2f}m (no barrier): {valid} valid")
            if 2 <= valid <= 8 and valid > best_n:
                best_n = valid
                best_erosion = erosion_px
                best_labeled = labeled.copy()
    
    if best_labeled is None:
        print("  No good segmentation found, using whole mask as 1 room")
        best_labeled = room_mask.astype(np.int32)
        best_labeled[best_labeled > 0] = 1
        best_n = 1
        best_erosion = 0
    
    print(f"  Best: erosion={best_erosion * cell_size:.2f}m → {best_n} rooms")
    
    # Now expand each eroded component back to fill the original mask
    # Use watershed-style expansion
    labeled_full = expand_rooms_to_mask(best_labeled, room_mask, cell_size)
    
    # Get valid room labels
    room_labels = []
    min_pixels = int(1.0 / (cell_size * cell_size))
    for i in range(1, labeled_full.max() + 1):
        if np.sum(labeled_full == i) >= min_pixels:
            room_labels.append(i)
    
    return labeled_full, room_labels


def expand_rooms_to_mask(seed_labeled, room_mask, cell_size):
    """Expand seed labels to fill the entire room mask (watershed-like)."""
    # Use distance-based assignment: for each unlabeled mask pixel,
    # assign to nearest labeled component
    
    labeled = seed_labeled.copy()
    unlabeled = (room_mask > 0) & (labeled == 0)
    
    # Iterative dilation to fill
    max_iters = 500
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max_iters):
        if not np.any(unlabeled):
            break
        # Dilate each label by 1 pixel
        new_labeled = labeled.copy()
        for label_val in range(1, labeled.max() + 1):
            mask = (labeled == label_val).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel)
            # Only assign to unlabeled pixels within room mask
            assign = (dilated > 0) & unlabeled
            new_labeled[assign] = label_val
        
        newly_labeled = (new_labeled > 0) & (labeled == 0) & (room_mask > 0)
        if not np.any(newly_labeled):
            break
        labeled = new_labeled
        unlabeled = (room_mask > 0) & (labeled == 0)
    
    return labeled


def snap_rooms_to_walls(labeled, room_labels, x_walls, z_walls, x_min, z_min, cell_size):
    """Snap room boundaries to nearest Hough walls."""
    # For each room, find its bounding box and snap edges to walls
    # This is cosmetic - the main room finding is done by flood fill
    pass  # Snapping happens during polygon extraction


# ─── Polygon extraction ───

def extract_room_polygon(room_component, x_walls, z_walls, x_min, z_min, cell_size):
    mask_u8 = (room_component > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    contour = max(contours, key=cv2.contourArea)
    world_pts = [[x_min + pt[0][0] * cell_size, z_min + pt[0][1] * cell_size] for pt in contour]
    if len(world_pts) < 3: return world_pts
    pts_arr = np.array(world_pts, dtype=np.float32)
    simplified = cv2.approxPolyDP(pts_arr, 0.15, True)
    poly = simplified[:, 0].tolist()
    all_x, all_z = sorted(x_walls), sorted(z_walls)
    snapped = []
    for p in poly:
        sx = snap_to_nearest(p[0], all_x, 0.25)
        sz = snap_to_nearest(p[1], all_z, 0.25)
        snapped.append([sx, sz])
    snapped = axis_snap_polygon(snapped)
    cleaned = [snapped[0]]
    for p in snapped[1:]:
        if abs(p[0] - cleaned[-1][0]) > 0.01 or abs(p[1] - cleaned[-1][1]) > 0.01:
            cleaned.append(p)
    return cleaned

def snap_to_nearest(val, positions, tolerance=0.25):
    best, best_d = val, tolerance
    for p in positions:
        d = abs(p - val)
        if d < best_d: best, best_d = p, d
    return best

def axis_snap_polygon(poly):
    if len(poly) < 3: return poly
    result = [poly[0]]
    for i in range(1, len(poly)):
        prev, cur = result[-1], poly[i]
        dx, dz = abs(cur[0] - prev[0]), abs(cur[1] - prev[1])
        if dx < dz: result.append([prev[0], cur[1]])
        else: result.append([cur[0], prev[1]])
    return result

def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2


# ─── Main ───

ROOM_COLORS = [
    '#4A90D9', '#E8834A', '#67B868', '#C75B8F', '#8B6CC1',
    '#D4A843', '#4ABFBF', '#D96060', '#7B8FD4', '#A0C75B',
]

def analyze_mesh(mesh_file):
    print(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    up_idx, up_name = detect_up_axis(mesh)
    up_coords = mesh.vertices[:, up_idx]
    up_min, up_range = up_coords.min(), np.ptp(up_coords)
    x_raw, z_raw = project_vertices(mesh, up_idx)
    hmask = (up_coords >= up_min + up_range*0.15) & (up_coords <= up_min + up_range*0.85)
    x_mid, z_mid = x_raw[hmask], z_raw[hmask]
    
    angle = find_dominant_angle(x_mid, z_mid)
    angle_rad = angle * math.pi / 180
    rx = x_mid * math.cos(-angle_rad) - z_mid * math.sin(-angle_rad)
    rz = x_mid * math.sin(-angle_rad) + z_mid * math.cos(-angle_rad)
    
    print("Building density image...")
    density_img, img_x_min, img_z_min, cell_size = build_density_image(rx, rz, cell_size=0.01)
    
    print("Edge detection + NMS...")
    combined_edges, edge_dir = multi_edge_detection(density_img)
    nms = non_max_suppression(combined_edges, edge_dir)
    
    print("Extracting wall segments (for snapping)...")
    x_walls, z_walls = extract_wall_segments(nms, img_x_min, img_z_min, cell_size)
    print(f"  X-walls: {[f'{w:.2f}' for w in x_walls]}")
    print(f"  Z-walls: {[f'{w:.2f}' for w in z_walls]}")
    
    print("Building room mask...")
    room_mask = build_room_mask(density_img)
    mask_area = np.sum(room_mask) * cell_size * cell_size
    print(f"  Mask area: {mask_area:.1f} m²")
    
    print("Finding rooms by erosion + flood fill...")
    labeled, room_labels = find_rooms_by_erosion(room_mask, cell_size, nms,
                                                   img_x_min, img_z_min)
    
    print("Extracting room polygons...")
    rooms = []
    total_area = 0
    for idx, label in enumerate(room_labels):
        component = (labeled == label).astype(np.uint8)
        pixel_area = np.sum(component) * cell_size * cell_size
        poly = extract_room_polygon(component, x_walls, z_walls,
                                     img_x_min, img_z_min, cell_size)
        poly_area = compute_polygon_area(poly)
        area = poly_area if poly_area > 0.5 else pixel_area
        rooms.append({
            'label': label, 'polygon_rot': poly, 'area': area,
            'pixel_area': pixel_area, 'name': f"Room {idx+1}",
        })
        total_area += area
        print(f"  {rooms[-1]['name']}: {area:.1f} m²")
    
    print(f"\n=== v27c Summary ===")
    print(f"Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    
    return {
        'rooms': rooms, 'total_area': total_area,
        'x_walls': x_walls, 'z_walls': z_walls,
        'interior_x': [], 'interior_z': [],
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'combined_edges': combined_edges, 'nms': nms,
        'room_mask': room_mask, 'labeled': labeled, 'room_labels': room_labels,
        'img_origin': (img_x_min, img_z_min, cell_size),
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(36, 9))
    ix_min, iz_min, cs = results['img_origin']
    
    axes[0].imshow(results['combined_edges'], cmap='inferno', origin='lower')
    axes[0].set_title('Combined Edge Response', color='white', fontsize=14)
    axes[0].axis('off')
    
    ax1 = axes[1]
    nms = results['nms']
    ax1.imshow(nms, cmap='hot', origin='lower', alpha=0.8)
    for xw in results['x_walls']:
        ax1.axvline(x=(xw - ix_min) / cs, color='lime', linewidth=1, alpha=0.7)
    for zw in results['z_walls']:
        ax1.axhline(y=(zw - iz_min) / cs, color='lime', linewidth=1, alpha=0.7)
    ax1.set_title('NMS + Hough Walls', color='white', fontsize=14)
    ax1.axis('off')
    
    ax2 = axes[2]
    labeled = results['labeled']
    seg_img = np.zeros((*labeled.shape, 3), dtype=np.float32)
    for i, label in enumerate(results['room_labels']):
        ch = ROOM_COLORS[i % len(ROOM_COLORS)]
        r, g, b = int(ch[1:3], 16)/255, int(ch[3:5], 16)/255, int(ch[5:7], 16)/255
        seg_img[labeled == label] = [r, g, b]
    ax2.imshow(seg_img, origin='lower')
    for room in results['rooms']:
        rows, cols = np.where(labeled == room['label'])
        if len(rows) > 0:
            ax2.text(np.mean(cols), np.mean(rows), f"{room['name']}\n{room['area']:.1f}m²",
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ax2.set_title(f'v27c Flood-Fill Rooms ({len(results["rooms"])} rooms)', color='white', fontsize=14)
    ax2.axis('off')
    
    ax3 = axes[3]
    ax3.set_aspect('equal'); ax3.set_facecolor('#1a1a2e')
    all_x, all_z = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3: continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs, zs = [p[0] for p in pc], [p[1] for p in pc]
        ax3.fill(xs, zs, color=color, alpha=0.3)
        all_x.extend(xs); all_z.extend(zs)
        for j in range(len(poly)):
            k = (j+1) % len(poly)
            ax3.plot([poly[j][0], poly[k][0]], [poly[j][1], poly[k][1]],
                    color='white', linewidth=3, solid_capstyle='round')
        cx = sum(p[0] for p in poly) / len(poly)
        cz = sum(p[1] for p in poly) / len(poly)
        ax3.text(cx, cz, f"{room['name']}\n{room['area']:.1f}m²",
                ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.5))
    ax3.set_title(f'v27c Floor Plan — {len(results["rooms"])} rooms, {results["total_area"]:.1f}m²',
                  color='white', fontsize=14)
    ax3.grid(True, alpha=0.2, color='gray')
    if all_x:
        m = 0.5
        ax3.set_xlim(min(all_x)-m, max(all_x)+m)
        ax3.set_ylim(min(all_z)-m, max(all_z)+m)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_json(results, output_path):
    data = {
        'summary': {
            'approach': 'v27c_floodfill',
            'num_rooms': len(results['rooms']),
            'total_area_m2': round(results['total_area'], 1),
        },
        'rooms': [{'name': r['name'], 'area_m2': round(r['area'], 1),
                    'polygon': [[round(p[0], 3), round(p[1], 3)] for p in r['polygon_rot']]}
                  for r in results['rooms']],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v27c - Flood-Fill Room Detection')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27c/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27c_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

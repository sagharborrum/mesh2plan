#!/usr/bin/env python3
"""
mesh2plan v27e - Density-Based Room Detection

Strategy: Instead of edge→Hough→split, use the density image directly.
Walls have HIGH vertex density, room interiors have LOW density.
Threshold high to get wall pixels, invert to find rooms as connected components.
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


# ─── Base functions (from v27d) ───

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

def build_density_image(rx, rz, cell_size=0.02, margin=0.3):
    x_min, z_min = rx.min() - margin, rz.min() - margin
    x_max, z_max = rx.max() + margin, rz.max() + margin
    nx = int((x_max - x_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx - x_min) / cell_size).astype(int), 0, nx-1)
    zi = np.clip(((rz - z_min) / cell_size).astype(int), 0, nz-1)
    np.add.at(img, (zi, xi), 1)
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


# ─── v27e: Density-based room detection ───

def density_room_detection(density_img, x_min, z_min, cell_size, min_room_area=1.5):
    """Detect rooms as low-density gaps between high-density walls."""
    nz_img, nx_img = density_img.shape
    
    # Build occupied region from raw density (any nonzero pixel)
    has_data = (density_img > 0).astype(np.uint8)
    # Close small gaps to make a solid occupied region
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    occupied = cv2.morphologyEx(has_data, cv2.MORPH_CLOSE, k_close)
    occupied = ndimage.binary_fill_holes(occupied).astype(np.uint8)
    # Keep largest component
    lbl_occ, n_occ = ndimage.label(occupied)
    if n_occ > 1:
        sizes = ndimage.sum(occupied, lbl_occ, range(1, n_occ+1))
        largest = np.argmax(sizes) + 1
        occupied = (lbl_occ == largest).astype(np.uint8)
    
    occupied_area = np.sum(occupied) * cell_size * cell_size
    print(f"  Occupied area: {occupied_area:.1f} m²")
    
    # Strategy: walls are high-density lines. We want to:
    # 1. Threshold density to get wall seed pixels
    # 2. Use directional morphological closing to connect wall segments into continuous lines
    # 3. The connected low-density regions between these wall lines = rooms
    
    nonzero = density_img[density_img > 0]
    if len(nonzero) == 0:
        return np.zeros_like(density_img, dtype=np.int32), []
    
    best_labeled = None
    best_labels = []
    best_score = -1
    
    # Key insight: walls have density >> floor. At 2cm, wall pixels might have
    # 20-180 vertices, floor pixels have 1-5. Use absolute density thresholds.
    p50 = np.percentile(nonzero, 50)
    p75 = np.percentile(nonzero, 75)
    p90 = np.percentile(nonzero, 90)
    print(f"  Density p50={p50:.0f} p75={p75:.0f} p90={p90:.0f}")
    
    for thresh_val in [p50 * 0.5, p50, p50 * 1.5, p75, p75 * 1.5, p90]:
        wall_seed = (density_img >= thresh_val).astype(np.uint8)
        
        # Connect wall fragments with directional closing
        for close_len in [7, 13, 21, 31]:
            k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (close_len, 1))
            walls_h = cv2.morphologyEx(wall_seed, cv2.MORPH_CLOSE, k_h)
            k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_len))
            walls_v = cv2.morphologyEx(wall_seed, cv2.MORPH_CLOSE, k_v)
            walls = walls_h | walls_v
            
            # Thin dilation to fill 1px gaps
            k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            walls = cv2.dilate(walls, k_d)
            
            # Rooms = occupied minus walls
            rooms = occupied & (~walls).astype(np.uint8)
            
            # Open to remove tiny noise
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            rooms = cv2.morphologyEx(rooms, cv2.MORPH_OPEN, k_open)
            
            labeled, n_rooms = ndimage.label(rooms)
            
            valid_labels = []
            for i in range(1, n_rooms + 1):
                area = np.sum(labeled == i) * cell_size * cell_size
                if area >= min_room_area:
                    valid_labels.append(i)
            
            n_valid = len(valid_labels)
            if n_valid < 1:
                continue
            
            total_room_area = sum(np.sum(labeled == l) * cell_size * cell_size for l in valid_labels)
            coverage = total_room_area / max(occupied_area, 1)
            
            if coverage < 0.15:
                continue
            
            # Score: strongly prefer more rooms, with coverage penalty
            room_score = n_valid ** 1.5 if n_valid <= 12 else 12 ** 1.5 - (n_valid - 12)
            score = room_score * coverage
            
            if score > best_score:
                best_score = score
                best_labeled = labeled.copy()
                best_labels = valid_labels[:]
                print(f"    t={thresh_val:.0f} cl={close_len}: {n_valid} rooms, "
                      f"coverage={coverage:.2f}, score={score:.2f}")
    
    if best_labeled is None:
        return np.zeros_like(density_img, dtype=np.int32), []
    
    print(f"  Best: {len(best_labels)} rooms (score={best_score:.2f})")
    return best_labeled, best_labels


def extract_room_polygon(room_component, x_min, z_min, cell_size):
    """Extract bounding rectangle from room component, snapped to density peaks."""
    rows, cols = np.where(room_component > 0)
    if len(rows) == 0:
        return []
    
    # Get bounding box in world coords
    left = x_min + cols.min() * cell_size
    right = x_min + (cols.max() + 1) * cell_size
    bottom = z_min + rows.min() * cell_size
    top = z_min + (rows.max() + 1) * cell_size
    
    # Simple rectangle
    return [[left, bottom], [right, bottom], [right, top], [left, top]]


def compute_polygon_area(poly):
    n = len(poly)
    if n < 3: return 0
    area = sum(poly[i][0] * poly[(i+1)%n][1] - poly[(i+1)%n][0] * poly[i][1] for i in range(n))
    return abs(area) / 2


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
    
    cell_size = 0.02  # 2cm resolution
    print(f"Building density image (cell={cell_size}m)...")
    density_img, img_x_min, img_z_min, cs = build_density_image(rx, rz, cell_size=cell_size)
    
    print("Edge detection (for overlay)...")
    # Build a finer density for edge detection
    density_fine, fx_min, fz_min, fcs = build_density_image(rx, rz, cell_size=0.01)
    combined_edges, edge_dir = multi_edge_detection(density_fine)
    nms = non_max_suppression(combined_edges, edge_dir)
    
    print("Density-based room detection...")
    labeled, room_labels = density_room_detection(density_img, img_x_min, img_z_min, cs)
    
    print("Extracting room polygons...")
    rooms = []
    total_area = 0
    for idx, label in enumerate(room_labels):
        component = (labeled == label).astype(np.uint8)
        pixel_area = np.sum(component) * cs * cs
        poly = extract_room_polygon(component, img_x_min, img_z_min, cs)
        poly_area = compute_polygon_area(poly)
        area = pixel_area  # Use pixel area as it's more accurate for irregular shapes
        rooms.append({
            'label': label, 'polygon_rot': poly, 'area': area,
            'pixel_area': pixel_area, 'poly_area': poly_area,
            'name': f"Room {idx+1}",
        })
        total_area += area
        print(f"  {rooms[-1]['name']}: {area:.1f} m² (bbox: {poly_area:.1f} m²)")
    
    print(f"\n=== v27e Summary ===")
    print(f"Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    
    return {
        'rooms': rooms, 'total_area': total_area,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'density_img': density_img,
        'combined_edges': combined_edges, 'nms': nms,
        'labeled': labeled, 'room_labels': room_labels,
        'img_origin': (img_x_min, img_z_min, cs),
        'fine_origin': (fx_min, fz_min, fcs),
    }


def visualize_results(results, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(45, 9))
    ix_min, iz_min, cs = results['img_origin']
    fx_min, fz_min, fcs = results['fine_origin']
    
    # Panel 1: Density image
    density = results['density_img']
    d_display = density.copy()
    if d_display.max() > 0:
        d_display = (d_display / np.percentile(d_display[d_display > 0], 95)).clip(0, 1)
    axes[0].imshow(d_display, cmap='hot', origin='lower')
    axes[0].set_title('Density Image (2cm)', color='white', fontsize=14)
    axes[0].axis('off')
    
    # Panel 2: Wall mask (high density threshold)
    smoothed = cv2.GaussianBlur(density, (5, 5), 1.0)
    nonzero = smoothed[smoothed > 0]
    if len(nonzero) > 0:
        thresh = np.percentile(nonzero, 60)
        wall_mask = (smoothed >= thresh).astype(np.uint8)
    else:
        wall_mask = np.zeros_like(density, dtype=np.uint8)
    ax1 = axes[1]
    # Show wall mask with dilation
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_dilated = cv2.dilate(wall_mask, k)
    display = np.zeros((*wall_mask.shape, 3), dtype=np.float32)
    display[wall_mask > 0] = [1, 0.3, 0.3]  # walls red
    display[wall_dilated > 0] = np.maximum(display[wall_dilated > 0], [0.5, 0.15, 0.15])
    ax1.imshow(display, origin='lower')
    ax1.set_title('Wall Mask (high density)', color='white', fontsize=14)
    ax1.axis('off')
    
    # Panel 3: Room segmentation colored
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
    ax2.set_title(f'v27e Density Rooms ({len(results["rooms"])} rooms)', color='white', fontsize=14)
    ax2.axis('off')
    
    # Panel 4: Edge + Polygon Overlay
    ax_overlay = axes[3]
    ax_overlay.set_aspect('equal'); ax_overlay.set_facecolor('black')
    nms_data = results['nms']
    edge_rows, edge_cols = np.where(nms_data > 0.05)
    if len(edge_rows) > 0:
        edge_x = fx_min + edge_cols * fcs
        edge_z = fz_min + edge_rows * fcs
        intensities = nms_data[edge_rows, edge_cols]
        ax_overlay.scatter(edge_x, edge_z, c=intensities, cmap='hot', s=0.3, alpha=0.5)
    
    overlay_xs, overlay_zs = [], []
    for i, room in enumerate(results['rooms']):
        poly = room['polygon_rot']
        if len(poly) < 3: continue
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        pc = poly + [poly[0]]
        xs, zs = [p[0] for p in pc], [p[1] for p in pc]
        ax_overlay.plot(xs, zs, color=color, linewidth=2.5, alpha=0.9)
        overlay_xs.extend(xs); overlay_zs.extend(zs)
    
    ax_overlay.set_title('Edge + Floor Plan Overlay', color='white', fontsize=14)
    ax_overlay.grid(True, alpha=0.2)
    ax_overlay.set_xlabel('X (meters)')
    ax_overlay.set_ylabel('Z (meters)')
    if overlay_xs:
        m = 0.5
        ax_overlay.set_xlim(min(overlay_xs)-m, max(overlay_xs)+m)
        ax_overlay.set_ylim(min(overlay_zs)-m, max(overlay_zs)+m)
    
    # Panel 5: Final floor plan
    ax3 = axes[4]
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
    ax3.set_title(f'v27e Floor Plan — {len(results["rooms"])} rooms, {results["total_area"]:.1f}m²',
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
            'approach': 'v27e_density_rooms',
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
    parser = argparse.ArgumentParser(description='mesh2plan v27e - Density-Based Rooms')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27e/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27e_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

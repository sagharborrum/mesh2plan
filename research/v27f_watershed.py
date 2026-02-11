#!/usr/bin/env python3
"""
mesh2plan v27f - Watershed Room Segmentation

Strategy: Use OpenCV watershed with density-derived edge ridges as barriers
and low-density room interiors as seeds.
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


# ─── v27f: Watershed room segmentation ───

def watershed_room_detection(density_img, x_min, z_min, cell_size, min_room_area=1.5):
    """Use watershed segmentation on density image to find rooms."""
    nz_img, nx_img = density_img.shape
    
    # Smooth density
    smoothed = cv2.GaussianBlur(density_img, (5, 5), 1.0)
    
    # Normalize to 8-bit
    if smoothed.max() > 0:
        img8 = (smoothed / np.percentile(smoothed[smoothed > 0], 95) * 255).clip(0, 255).astype(np.uint8)
    else:
        return np.zeros_like(density_img, dtype=np.int32), []
    
    # Edge map (ridges for watershed)
    sobel_x = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Occupied region
    occupied = (smoothed > 0).astype(np.uint8)
    k_occ = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, k_occ)
    occupied = ndimage.binary_fill_holes(occupied).astype(np.uint8)
    lbl_occ, n_occ = ndimage.label(occupied)
    if n_occ > 1:
        sizes = ndimage.sum(occupied, lbl_occ, range(1, n_occ+1))
        largest = np.argmax(sizes) + 1
        occupied = (lbl_occ == largest).astype(np.uint8)
    
    # Sure background: high density (walls) + outside
    nonzero = smoothed[smoothed > 0]
    wall_thresh = np.percentile(nonzero, 65)
    sure_bg = ((smoothed >= wall_thresh) | (occupied == 0)).astype(np.uint8)
    
    # Sure foreground: low density regions far from walls (room interiors)
    # Use distance transform on inverted wall mask within occupied region
    wall_mask = (smoothed >= wall_thresh).astype(np.uint8)
    
    # Dilate walls
    k_w = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_dilated = cv2.dilate(wall_mask, k_w)
    
    # Room interior = occupied AND NOT wall
    interior = occupied & (~wall_dilated).astype(np.uint8)
    
    # Distance transform to find room centers
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    
    # Find peaks in distance transform (room centers)
    # Threshold at fraction of max distance
    if dist.max() > 0:
        # Try multiple thresholds
        best_markers = None
        best_n = 0
        best_score = -1
        
        for dist_frac in [0.2, 0.3, 0.4, 0.5, 0.6]:
            _, fg = cv2.threshold(dist, dist_frac * dist.max(), 255, 0)
            fg = fg.astype(np.uint8)
            
            # Clean up
            k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_clean)
            
            # Label markers
            markers_labeled, n_markers = ndimage.label(fg)
            
            # Filter small markers
            valid = []
            for i in range(1, n_markers + 1):
                area = np.sum(markers_labeled == i) * cell_size * cell_size
                if area >= 0.3:  # At least 0.3 m²
                    valid.append(i)
            
            n_valid = len(valid)
            if n_valid >= 1:
                total_area = sum(np.sum(markers_labeled == v) for v in valid) * cell_size * cell_size
                score = n_valid * (total_area / max(np.sum(occupied) * cell_size * cell_size, 1))
                
                if score > best_score:
                    best_score = score
                    best_n = n_valid
                    best_markers = markers_labeled.copy()
                    # Remap to contiguous labels
                    remap = np.zeros(n_markers + 1, dtype=np.int32)
                    for new_idx, old_idx in enumerate(valid, 1):
                        remap[old_idx] = new_idx
                    best_markers = remap[best_markers]
                    print(f"    dist_frac={dist_frac}: {n_valid} markers, score={score:.2f}")
        
        if best_markers is None:
            return np.zeros_like(density_img, dtype=np.int32), []
        
        markers = best_markers.copy()
        n_rooms = best_n
    else:
        return np.zeros_like(density_img, dtype=np.int32), []
    
    # Mark background as separate label
    markers[occupied == 0] = n_rooms + 1
    
    # Prepare image for watershed (needs 3-channel uint8)
    img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    
    # Watershed needs markers as int32
    markers_ws = markers.astype(np.int32)
    cv2.watershed(img_color, markers_ws)
    
    # Watershed sets boundaries to -1
    # Extract room labels (exclude background label and boundaries)
    result = markers_ws.copy()
    result[result == -1] = 0  # boundaries
    result[result == n_rooms + 1] = 0  # background
    
    # Filter by area
    valid_labels = []
    for i in range(1, n_rooms + 1):
        area = np.sum(result == i) * cell_size * cell_size
        if area >= min_room_area:
            valid_labels.append(i)
        print(f"    Room {i}: {area:.1f} m²")
    
    print(f"  Watershed: {len(valid_labels)} rooms (from {n_rooms} markers)")
    return result, valid_labels


def extract_room_polygon(room_component, x_min, z_min, cell_size):
    rows, cols = np.where(room_component > 0)
    if len(rows) == 0:
        return []
    left = x_min + cols.min() * cell_size
    right = x_min + (cols.max() + 1) * cell_size
    bottom = z_min + rows.min() * cell_size
    top = z_min + (rows.max() + 1) * cell_size
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
    
    cell_size = 0.02
    print(f"Building density image (cell={cell_size}m)...")
    density_img, img_x_min, img_z_min, cs = build_density_image(rx, rz, cell_size=cell_size)
    
    print("Edge detection (for overlay)...")
    density_fine, fx_min, fz_min, fcs = build_density_image(rx, rz, cell_size=0.01)
    combined_edges, edge_dir = multi_edge_detection(density_fine)
    nms = non_max_suppression(combined_edges, edge_dir)
    
    print("Watershed room segmentation...")
    labeled, room_labels = watershed_room_detection(density_img, img_x_min, img_z_min, cs)
    
    # Also compute edge map for visualization
    img8 = density_img.copy()
    if img8.max() > 0:
        img8 = (img8 / np.percentile(img8[img8 > 0], 95) * 255).clip(0, 255).astype(np.uint8)
    sobel_x = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
    
    print("Extracting room polygons...")
    rooms = []
    total_area = 0
    for idx, label in enumerate(room_labels):
        component = (labeled == label).astype(np.uint8)
        pixel_area = np.sum(component) * cs * cs
        poly = extract_room_polygon(component, img_x_min, img_z_min, cs)
        poly_area = compute_polygon_area(poly)
        area = pixel_area
        rooms.append({
            'label': label, 'polygon_rot': poly, 'area': area,
            'pixel_area': pixel_area, 'poly_area': poly_area,
            'name': f"Room {idx+1}",
        })
        total_area += area
        print(f"  {rooms[-1]['name']}: {area:.1f} m² (bbox: {poly_area:.1f} m²)")
    
    print(f"\n=== v27f Summary ===")
    print(f"Rooms: {len(rooms)}, Total area: {total_area:.1f} m²")
    
    return {
        'rooms': rooms, 'total_area': total_area,
        'angle': angle, 'coordinate_system': f'{up_name}-up',
        'density_img': density_img, 'edge_map': edge_map,
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
    
    # Panel 2: Edge map (Sobel magnitude)
    edge_map = results['edge_map']
    if edge_map.max() > 0:
        edge_display = (edge_map / edge_map.max()).clip(0, 1)
    else:
        edge_display = edge_map
    axes[1].imshow(edge_display, cmap='inferno', origin='lower')
    axes[1].set_title('Edge Map (Sobel)', color='white', fontsize=14)
    axes[1].axis('off')
    
    # Panel 3: Room segmentation colored
    ax2 = axes[2]
    labeled = results['labeled']
    seg_img = np.zeros((*labeled.shape, 3), dtype=np.float32)
    for i, label in enumerate(results['room_labels']):
        ch = ROOM_COLORS[i % len(ROOM_COLORS)]
        r, g, b = int(ch[1:3], 16)/255, int(ch[3:5], 16)/255, int(ch[5:7], 16)/255
        seg_img[labeled == label] = [r, g, b]
    # Mark watershed boundaries
    seg_img[labeled == -1] = [1, 1, 1]
    ax2.imshow(seg_img, origin='lower')
    for room in results['rooms']:
        rows, cols = np.where(labeled == room['label'])
        if len(rows) > 0:
            ax2.text(np.mean(cols), np.mean(rows), f"{room['name']}\n{room['area']:.1f}m²",
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ax2.set_title(f'v27f Watershed ({len(results["rooms"])} rooms)', color='white', fontsize=14)
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
    ax3.set_title(f'v27f Floor Plan — {len(results["rooms"])} rooms, {results["total_area"]:.1f}m²',
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
            'approach': 'v27f_watershed',
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
    parser = argparse.ArgumentParser(description='mesh2plan v27f - Watershed Rooms')
    parser.add_argument('mesh_file')
    parser.add_argument('--output-dir', default='results/v27f/')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"v27f_{Path(args.mesh_file).stem}"
    results = analyze_mesh(args.mesh_file)
    visualize_results(results, output_dir / f"{prefix}_floorplan.png")
    save_results_json(results, output_dir / f"{prefix}_results.json")


if __name__ == '__main__':
    main()

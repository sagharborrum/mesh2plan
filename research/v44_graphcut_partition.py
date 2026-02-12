#!/usr/bin/env python3
"""
mesh2plan v44 - Graph-Cut Room Partition

FUNDAMENTALLY NEW approach: Model the floor plan as a graph and use spectral
clustering / normalized cuts to partition into rooms.

Key insight: Walls are high-density ridges between low-density room interiors.
A graph where edge weights are INVERSELY proportional to wall density will
naturally have weak connections through walls → graph cut finds room boundaries.

Pipeline:
1. Normal-filtered wall density → clean wall signal
2. Build pixel adjacency graph on apartment interior
3. Edge weights = exp(-wall_density) → low weight across walls
4. Spectral clustering (normalized cuts) → room partition
5. Per-room contour + Hough angle-snap polygon extraction
6. Hallway detection: elongated rooms with high perimeter/area ratio

Why this should work:
- Graph cuts are OPTIMAL for finding boundaries along high-cost edges
- No manual threshold tuning for wall barriers (continuous weights)
- Naturally handles complex room shapes (L-shapes, alcoves)
- Spectral clustering finds the "natural" number of clusters
- Well-studied in image segmentation (Shi & Malik 2000)
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage, sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def create_wall_density(mesh, resolution=0.02, normal_thresh=0.5):
    """Create wall-only density image using face normals (from v41b)."""
    normals = mesh.face_normals
    centroids = mesh.triangles_center
    y_comp = np.abs(normals[:, 1])
    wall_mask = y_comp < normal_thresh
    wall_weights = 1.0 - y_comp
    wall_centroids = centroids[wall_mask]
    ww = wall_weights[wall_mask]
    
    x, z = wall_centroids[:, 0], wall_centroids[:, 2]
    pad = 0.3
    x_min, x_max = x.min() - pad, x.max() + pad
    z_min, z_max = z.min() - pad, z.max() + pad
    w = int((x_max - x_min) / resolution) + 1
    h = int((z_max - z_min) / resolution) + 1
    
    wall_density = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(wall_density, (zi, xi), ww)
    
    # All-vertex density for apartment mask
    all_x, all_z = mesh.vertices[:, 0], mesh.vertices[:, 2]
    all_density = np.zeros((h, w), dtype=np.float32)
    axi = np.clip(((all_x - x_min) / resolution).astype(int), 0, w - 1)
    azi = np.clip(((all_z - z_min) / resolution).astype(int), 0, h - 1)
    np.add.at(all_density, (azi, axi), 1)
    
    print(f"  Wall faces: {wall_mask.sum()} / {len(normals)} ({wall_mask.mean()*100:.1f}%)")
    print(f"  Image size: {w}x{h}")
    return wall_density, all_density, (x_min, z_min, resolution)


def get_apartment_mask(density, threshold=1):
    mask = (density >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 1, -1)
    # Fill holes
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask


def detect_wall_angles(wall_density, apt_mask):
    """Detect dominant wall angles using Hough lines."""
    wall_norm = wall_density / (wall_density.max() + 1e-6)
    wall_img = (wall_norm * 255 * apt_mask).astype(np.uint8)
    thresh = np.percentile(wall_img[apt_mask > 0], 80)
    edges = (wall_img > thresh).astype(np.uint8) * 255
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    if lines is None:
        return [0, 90]
    
    angles_deg = []
    for line in lines:
        theta = line[0][1]
        angle = np.degrees(theta) % 180
        angles_deg.append(angle)
    
    # Cluster angles
    angles_arr = np.array(angles_deg).reshape(-1, 1)
    # Use histogram to find peaks
    hist, bin_edges = np.histogram(angles_arr, bins=180, range=(0, 180))
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
    
    from scipy.signal import find_peaks
    peaks, props = find_peaks(hist_smooth, height=max(hist_smooth)*0.15, distance=15)
    peak_angles = [(bin_edges[p] + bin_edges[p+1])/2 for p in peaks]
    
    print(f"  Hough lines: {len(lines)}, dominant angles: {[f'{a:.0f}°' for a in peak_angles]}")
    return peak_angles if peak_angles else [0, 90]


def build_graph_and_partition(wall_density, apt_mask, n_rooms=5, sigma=None):
    """Build pixel graph weighted by wall density, spectral clustering."""
    h, w = apt_mask.shape
    
    # Downsample for tractability (spectral clustering is O(n²) memory)
    scale = 4  # Process at 1/4 resolution
    small_mask = cv2.resize(apt_mask, (w//scale, h//scale), interpolation=cv2.INTER_NEAREST)
    small_wall = cv2.resize(wall_density, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
    sh, sw = small_mask.shape
    
    # Smooth wall density for better gradients
    small_wall = cv2.GaussianBlur(small_wall, (5, 5), 1.0)
    
    # Normalize wall density
    wall_max = small_wall.max()
    if wall_max > 0:
        wall_norm = small_wall / wall_max
    else:
        wall_norm = small_wall
    
    # Get interior pixel indices
    interior = np.where(small_mask > 0)
    n_pixels = len(interior[0])
    print(f"  Graph: {n_pixels} interior pixels at {scale}x downsample ({sh}x{sw})")
    
    if n_pixels > 50000:
        # Further downsample
        scale2 = 2
        small_mask = cv2.resize(small_mask, (sw//scale2, sh//scale2), interpolation=cv2.INTER_NEAREST)
        small_wall = cv2.resize(wall_norm, (sw//scale2, sh//scale2), interpolation=cv2.INTER_LINEAR)
        wall_norm = small_wall
        sh, sw = small_mask.shape
        interior = np.where(small_mask > 0)
        n_pixels = len(interior[0])
        scale *= scale2
        print(f"  Further downsampled: {n_pixels} pixels at {scale}x ({sh}x{sw})")
    
    # Create pixel index map
    pixel_idx = np.full((sh, sw), -1, dtype=np.int32)
    pixel_idx[interior] = np.arange(n_pixels)
    
    # Build sparse adjacency matrix (4-connected)
    rows, cols, weights = [], [], []
    
    # Auto-tune sigma based on wall density distribution
    wall_vals = wall_norm[interior]
    if sigma is None:
        sigma = max(np.median(wall_vals[wall_vals > 0]) * 2, 0.1)
    print(f"  Sigma: {sigma:.3f}")
    
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        ny = interior[0] + dy
        nx = interior[1] + dx
        valid = (ny >= 0) & (ny < sh) & (nx >= 0) & (nx < sw)
        valid &= small_mask[ny[valid].clip(0, sh-1), nx[valid].clip(0, sw-1)] > 0 if valid.any() else valid
        
        # Recompute with proper bounds
        vy, vx = interior[0][valid], interior[1][valid]
        nvy, nvx = vy + dy, vx + dx
        bounds = (nvy >= 0) & (nvy < sh) & (nvx >= 0) & (nvx < sw)
        vy, vx = vy[bounds], vx[bounds]
        nvy, nvx = nvy[bounds], nvx[bounds]
        in_mask = small_mask[nvy, nvx] > 0
        vy, vx = vy[in_mask], vx[in_mask]
        nvy, nvx = nvy[in_mask], nvx[in_mask]
        
        src_idx = pixel_idx[vy, vx]
        dst_idx = pixel_idx[nvy, nvx]
        valid_edges = (src_idx >= 0) & (dst_idx >= 0)
        src_idx = src_idx[valid_edges]
        dst_idx = dst_idx[valid_edges]
        
        # Weight = exp(-wall_cost / sigma)
        # wall_cost = max wall density along edge
        wall_cost = np.maximum(wall_norm[vy[valid_edges], vx[valid_edges]], 
                               wall_norm[nvy[valid_edges], nvx[valid_edges]])
        w_vals = np.exp(-wall_cost / sigma)
        
        rows.extend(src_idx.tolist())
        cols.extend(dst_idx.tolist())
        weights.extend(w_vals.tolist())
    
    print(f"  Edges: {len(rows)}")
    
    # Build sparse symmetric matrix
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_pixels, n_pixels))
    W = W + W.T  # Symmetrize
    
    # Degree matrix
    D = sparse.diags(np.array(W.sum(axis=1)).flatten())
    
    # Normalized Laplacian: D^{-1/2} (D - W) D^{-1/2}
    d_inv_sqrt = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d_inv_sqrt > 0, 1.0 / np.sqrt(d_inv_sqrt), 0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    
    L_norm = sparse.eye(n_pixels) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Find smallest eigenvectors (Fiedler vectors)
    n_eig = min(n_rooms + 1, n_pixels - 1)
    print(f"  Computing {n_eig} eigenvectors...")
    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_eig, which='SM', tol=1e-4, maxiter=1000)
    except Exception as e:
        print(f"  Eigsh failed: {e}, trying with more iterations")
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_eig, which='SM', tol=1e-3, maxiter=3000)
    
    print(f"  Eigenvalues: {eigenvalues}")
    
    # Determine number of clusters from eigengap
    eigengaps = np.diff(eigenvalues)
    if len(eigengaps) > 1:
        # Best k = argmax eigengap (after first near-zero eigenvalue)
        best_k = np.argmax(eigengaps[1:]) + 2  # +2 because we skip gap[0] and 1-index
        best_k = max(2, min(best_k, n_rooms + 2))
        print(f"  Eigengap suggests {best_k} clusters (gaps: {eigengaps})")
    else:
        best_k = n_rooms
    
    # Try requested n_rooms and eigengap suggestion
    results = {}
    for k in set([n_rooms, best_k]):
        # K-means on eigenvectors
        features = eigenvectors[:, 1:k+1]  # Skip first (constant) eigenvector
        # Normalize rows
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        features = features / norms
        
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(features)
        
        # Map back to image
        label_img = np.full((sh, sw), -1, dtype=np.int32)
        label_img[interior] = labels
        
        # Upscale back to original resolution
        label_full = cv2.resize(label_img.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        label_full[apt_mask == 0] = -1
        
        results[k] = label_full
        print(f"  k={k}: labels {np.unique(labels)}")
    
    return results, scale


def extract_room_polygons(label_img, wall_angles, transform, min_area_m2=1.5):
    """Extract polygons for each room label with angle-snapped simplification."""
    x_min, z_min, resolution = transform
    rooms = []
    
    unique_labels = [l for l in np.unique(label_img) if l >= 0]
    
    for label in unique_labels:
        room_mask = (label_img == label).astype(np.uint8)
        area_px = room_mask.sum()
        area_m2 = area_px * resolution * resolution
        
        if area_m2 < min_area_m2:
            continue
        
        # Find contour
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        cnt = max(contours, key=cv2.contourArea)
        
        # Simplify with Douglas-Peucker
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.015 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Snap edges to detected wall angles
        pts = approx.reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        
        snapped = snap_to_angles(pts, wall_angles, angle_tolerance=12)
        
        # Convert to world coordinates
        world_pts = []
        for px, py in snapped:
            wx = px * resolution + x_min
            wz = py * resolution + z_min
            world_pts.append([wx, wz])
        
        # Classify room
        bbox = cv2.boundingRect(cnt)
        aspect = max(bbox[2], bbox[3]) / (min(bbox[2], bbox[3]) + 1)
        perimeter_m = perimeter * resolution
        compactness = (perimeter_m ** 2) / (4 * math.pi * area_m2) if area_m2 > 0 else 99
        
        if area_m2 < 3:
            name = "Closet"
        elif area_m2 < 5:
            if compactness > 2.5 or aspect > 3:
                name = "Hallway"
            else:
                name = "Bathroom"
        elif compactness > 2.5 or aspect > 3:
            name = "Hallway"
        else:
            name = "Room"
        
        rooms.append({
            'label': int(label),
            'name': name,
            'area_m2': round(area_m2, 1),
            'vertices': len(snapped),
            'polygon_world': world_pts,
            'polygon_px': snapped.tolist(),
            'centroid_px': [int(np.mean(snapped[:, 0])), int(np.mean(snapped[:, 1]))],
            'compactness': round(compactness, 1),
            'aspect': round(aspect, 1),
        })
    
    # Number rooms
    room_idx = 1
    for r in sorted(rooms, key=lambda x: -x['area_m2']):
        if r['name'] == 'Room':
            r['name'] = f"Room {room_idx}"
            room_idx += 1
    bath_idx = 1
    for r in sorted(rooms, key=lambda x: -x['area_m2']):
        if r['name'] == 'Bathroom':
            if bath_idx > 1:
                r['name'] = f"Bathroom {bath_idx}"
            bath_idx += 1
    hall_idx = 1
    for r in sorted(rooms, key=lambda x: -x['area_m2']):
        if r['name'] == 'Hallway':
            if hall_idx > 1:
                r['name'] = f"Hallway {hall_idx}"
            hall_idx += 1
    closet_idx = 1
    for r in sorted(rooms, key=lambda x: -x['area_m2']):
        if r['name'] == 'Closet':
            if closet_idx > 1:
                r['name'] = f"Closet {closet_idx}"
            closet_idx += 1
    
    return rooms


def snap_to_angles(pts, angles_deg, angle_tolerance=12):
    """Snap polygon edges to nearest detected wall angle."""
    n = len(pts)
    if n < 3:
        return pts
    
    snapped = pts.copy()
    for i in range(n):
        j = (i + 1) % n
        dx = pts[j, 0] - pts[i, 0]
        dy = pts[j, 1] - pts[i, 1]
        edge_len = math.sqrt(dx*dx + dy*dy)
        if edge_len < 3:
            continue
        
        edge_angle = math.degrees(math.atan2(dy, dx)) % 180
        
        # Find nearest wall angle
        best_angle = edge_angle
        best_diff = angle_tolerance
        for wa in angles_deg:
            for a in [wa, (wa + 90) % 180]:
                diff = abs(edge_angle - a)
                diff = min(diff, 180 - diff)
                if diff < best_diff:
                    best_diff = diff
                    best_angle = a
        
        if best_diff < angle_tolerance:
            # Rotate edge to snap angle, keeping midpoint
            mid = (pts[i] + pts[j]) / 2
            rad = math.radians(best_angle)
            half = edge_len / 2
            snapped[i] = mid - np.array([math.cos(rad), math.sin(rad)]) * half
            snapped[j] = mid + np.array([math.cos(rad), math.sin(rad)]) * half
    
    # Recompute vertices as edge intersections for clean corners
    result = []
    for i in range(n):
        prev = (i - 1) % n
        # Intersect edge prev→i with edge i→next
        p1, p2 = snapped[prev], snapped[i]
        p3, p4 = snapped[i], snapped[(i+1) % n]
        pt = line_intersection(p1, p2, p3, p4)
        if pt is not None and not np.any(np.isnan(pt)):
            result.append(pt)
        else:
            result.append(snapped[i])
    
    return np.array(result)


def line_intersection(p1, p2, p3, p4):
    """Find intersection of two lines (p1-p2) and (p3-p4)."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def detect_doors(label_img, wall_density, transform, rooms):
    """Detect doors as thin connections between rooms."""
    resolution = transform[2]
    doors = []
    
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            mask1 = (label_img == r1['label']).astype(np.uint8)
            mask2 = (label_img == r2['label']).astype(np.uint8)
            
            # Dilate both and find overlap
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            d1 = cv2.dilate(mask1, kernel)
            d2 = cv2.dilate(mask2, kernel)
            overlap = d1 & d2
            
            if overlap.sum() > 5:
                # Find centroid of overlap
                ys, xs = np.where(overlap > 0)
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                wx = cx * resolution + transform[0]
                wz = cy * resolution + transform[1]
                doors.append({
                    'rooms': [r1['name'], r2['name']],
                    'pos_world': [round(wx, 2), round(wz, 2)],
                    'pos_px': [cx, cy],
                    'width_px': int(overlap.sum() ** 0.5),
                })
    
    return doors


def plot_results(label_img, wall_density, apt_mask, rooms, doors, transform, 
                 wall_angles, out_path, mesh_name, k):
    """Plot floor plan with room polygons."""
    resolution = transform[2]
    x_min, z_min = transform[0], transform[1]
    h, w = label_img.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Color rooms
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 1)))
    for idx, room in enumerate(rooms):
        mask = (label_img == room['label']).astype(float)
        color_rgba = list(colors[idx % len(colors)])
        color_rgba[3] = 0.3
        colored = np.zeros((h, w, 4))
        colored[mask > 0] = color_rgba
        
        extent = [x_min, x_min + w * resolution, z_min, z_min + h * resolution]
        ax.imshow(colored, origin='lower', extent=extent)
    
    # Draw room polygons
    for idx, room in enumerate(rooms):
        poly = np.array(room['polygon_world'])
        if len(poly) >= 3:
            poly_closed = np.vstack([poly, poly[0]])
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', linewidth=2)
        
        # Label
        cx = np.mean(poly[:, 0]) if len(poly) > 0 else 0
        cy = np.mean(poly[:, 1]) if len(poly) > 0 else 0
        ax.text(cx, cy, f"{room['name']}\n{room['area_m2']}m²\n({room['vertices']}v)",
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    
    # Door markers
    for door in doors:
        ax.plot(door['pos_world'][0], door['pos_world'][1], 's', 
                color='darkred', markersize=6)
    
    # Scale bar
    ax.plot([x_min + 0.3, x_min + 1.3], [z_min + 0.3, z_min + 0.3], 'k-', linewidth=3)
    ax.text(x_min + 0.8, z_min + 0.5, '1m', ha='center', fontsize=9)
    
    # Wall angles legend
    angle_str = ', '.join([f'{a:.0f}°' for a in wall_angles])
    ax.text(0.02, 0.98, f'Wall angles: {angle_str}', transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.set_title(f'v44 Graph-Cut Partition (k={k}) — {mesh_name}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_debug(wall_density, apt_mask, label_imgs, eigenvalues, transform, out_path):
    """Debug panels: wall density, apartment mask, eigenvalues, partitions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    resolution = transform[2]
    x_min, z_min = transform[0], transform[1]
    h, w = apt_mask.shape
    extent = [x_min, x_min + w * resolution, z_min, z_min + h * resolution]
    
    # Wall density
    ax = axes[0, 0]
    ax.imshow(wall_density, origin='lower', extent=extent, cmap='hot')
    ax.set_title('Wall-Only Density')
    
    # Apartment mask
    ax = axes[0, 1]
    ax.imshow(apt_mask, origin='lower', extent=extent, cmap='gray')
    ax.set_title('Apartment Mask')
    
    # Eigenvalues
    ax = axes[0, 2]
    if eigenvalues is not None:
        ax.bar(range(len(eigenvalues)), eigenvalues)
        ax.set_title('Eigenvalues (gaps = cluster boundaries)')
        ax.set_xlabel('Index')
    
    # Partitions for different k
    for idx, (k, label_img) in enumerate(label_imgs.items()):
        if idx >= 3:
            break
        ax = axes[1, idx]
        display = label_img.astype(float)
        display[label_img < 0] = np.nan
        ax.imshow(display, origin='lower', extent=extent, cmap='tab20', interpolation='nearest')
        n_rooms = len([l for l in np.unique(label_img) if l >= 0])
        ax.set_title(f'Partition k={k} ({n_rooms} rooms)')
    
    for ax in axes.flat:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved debug: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='mesh2plan v44 - Graph-Cut Partition')
    parser.add_argument('mesh_path', help='Path to mesh file')
    parser.add_argument('--output', '-o', default='results/v44_graphcut_partition')
    parser.add_argument('--n-rooms', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=None)
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_path)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_name = mesh_path.stem
    
    print(f"\n=== v44 Graph-Cut Partition ===")
    print(f"Mesh: {mesh_path}")
    print(f"Target rooms: {args.n_rooms}")
    
    # 1. Load mesh
    print("\n1. Loading mesh...")
    mesh = trimesh.load(mesh_path, process=False)
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    # 2. Create wall density
    print("\n2. Creating wall-only density...")
    wall_density, all_density, transform = create_wall_density(mesh)
    
    # 3. Apartment mask
    print("\n3. Apartment mask...")
    apt_mask = get_apartment_mask(all_density)
    apt_area = apt_mask.sum() * transform[2] ** 2
    print(f"  Apartment area: {apt_area:.1f} m²")
    
    # 4. Detect wall angles
    print("\n4. Detecting wall angles...")
    wall_angles = detect_wall_angles(wall_density, apt_mask)
    
    # 5. Build graph and spectral partition
    print("\n5. Building graph and computing spectral partition...")
    label_imgs, scale = build_graph_and_partition(
        wall_density, apt_mask, n_rooms=args.n_rooms, sigma=args.sigma
    )
    
    # 6. Extract room polygons for each k
    best_rooms = None
    best_k = None
    best_doors = None
    
    for k, label_img in label_imgs.items():
        print(f"\n6. Extracting rooms for k={k}...")
        rooms = extract_room_polygons(label_img, wall_angles, transform)
        doors = detect_doors(label_img, wall_density, transform, rooms)
        
        total_area = sum(r['area_m2'] for r in rooms)
        print(f"  Rooms: {len(rooms)}, Total area: {total_area:.1f}m², Doors: {len(doors)}")
        for r in sorted(rooms, key=lambda x: -x['area_m2']):
            print(f"    {r['name']}: {r['area_m2']}m² ({r['vertices']}v) [compact={r['compactness']}]")
        
        # Pick best k (closest to target room count)
        if best_rooms is None or abs(len(rooms) - args.n_rooms) < abs(len(best_rooms) - args.n_rooms):
            best_rooms = rooms
            best_k = k
            best_doors = doors
            best_label = label_img
    
    # 7. Plot
    print(f"\n7. Plotting (best k={best_k})...")
    plot_results(best_label, wall_density, apt_mask, best_rooms, best_doors,
                 transform, wall_angles, out_dir / f'floorplan_k{best_k}.png',
                 mesh_name, best_k)
    
    # Debug panels
    plot_debug(wall_density, apt_mask, label_imgs, None, transform, out_dir / 'debug.png')
    
    # Also save alternate k plots
    for k, label_img in label_imgs.items():
        if k != best_k:
            rooms_k = extract_room_polygons(label_img, wall_angles, transform)
            doors_k = detect_doors(label_img, wall_density, transform, rooms_k)
            plot_results(label_img, wall_density, apt_mask, rooms_k, doors_k,
                         transform, wall_angles, out_dir / f'floorplan_k{k}.png',
                         mesh_name, k)
    
    # 8. Save JSON
    result = {
        'version': 'v44_graphcut_partition',
        'mesh': str(mesh_path),
        'best_k': best_k,
        'wall_angles': [float(a) for a in wall_angles],
        'rooms': best_rooms,
        'doors': best_doors,
        'total_area_m2': round(sum(r['area_m2'] for r in best_rooms), 1),
    }
    
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Best: k={best_k}, {len(best_rooms)} rooms, {result['total_area_m2']}m²")


if __name__ == '__main__':
    main()

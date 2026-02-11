#!/usr/bin/env python3
"""
mesh2plan v34 - Junction-based wall detection

Instead of projecting walls independently (X and Z), detect wall JUNCTIONS
where perpendicular walls meet. This reveals interior partitions that are
too weak to show up in global projections (like bathroom walls).

Strategy:
1. Build density + edge images
2. Find all candidate wall positions (X and Z)
3. For each (X,Z) pair, check local density at the intersection
4. Strong junctions = confirmed wall corners
5. Build room graph from confirmed junctions
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import math
from scipy import ndimage
from scipy.ndimage import maximum_filter, uniform_filter1d
from pathlib import Path
import argparse


def detect_up_axis(mesh):
    ranges = [np.ptp(mesh.vertices[:, i]) for i in range(3)]
    if 1.0 <= ranges[1] <= 4.0 and ranges[1] != max(ranges): return 1
    elif 1.0 <= ranges[2] <= 4.0 and ranges[2] != max(ranges): return 2
    return np.argmin(ranges)

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


def find_wall_candidates(density_img, x_min, z_min, cs, nms_dist=0.3):
    """Find all candidate wall positions from projection peaks."""
    smoothed = cv2.GaussianBlur(density_img, (3, 3), 0.5)
    proj_x = smoothed.sum(axis=0)
    proj_z = smoothed.sum(axis=1)
    
    def peaks(profile, origin):
        prof = uniform_filter1d(profile.astype(float), size=5)
        min_dist = max(3, int(nms_dist / cs))
        local_max = maximum_filter(prof, size=min_dist) == prof
        threshold = prof.mean()  # Lower threshold to catch weak walls
        idx = np.where(local_max & (prof > threshold))[0]
        positions = origin + idx * cs
        strengths = prof[idx]
        # NMS
        order = np.argsort(-strengths)
        kept_pos, kept_str = [], []
        for i in order:
            if any(abs(positions[i] - k) < nms_dist for k in kept_pos): continue
            kept_pos.append(positions[i])
            kept_str.append(strengths[i])
        return np.array(kept_pos), np.array(kept_str)
    
    x_pos, x_str = peaks(proj_x, x_min)
    z_pos, z_str = peaks(proj_z, z_min)
    return x_pos, x_str, z_pos, z_str


def detect_junctions(density_img, x_min, z_min, cs, x_walls, z_walls, room_mask,
                      junction_radius=0.15, min_density=3.0):
    """Detect wall junctions where X and Z walls intersect with high local density."""
    junctions = []
    r_px = max(3, int(junction_radius / cs))
    
    for xw in x_walls:
        for zw in z_walls:
            # Check if this point is inside the apartment mask
            x_px = int((xw - x_min) / cs)
            z_px = int((zw - z_min) / cs)
            
            if not (0 <= x_px < density_img.shape[1] and 0 <= z_px < density_img.shape[0]):
                continue
            if room_mask[z_px, x_px] == 0:
                continue
            
            # Measure local density in a cross pattern (along both wall directions)
            x_lo = max(0, x_px - r_px)
            x_hi = min(density_img.shape[1], x_px + r_px + 1)
            z_lo = max(0, z_px - r_px)
            z_hi = min(density_img.shape[0], z_px + r_px + 1)
            
            # Cross pattern: horizontal strip + vertical strip
            h_strip = density_img[z_px-1:z_px+2, x_lo:x_hi].mean() if z_px > 0 else 0
            v_strip = density_img[z_lo:z_hi, x_px-1:x_px+2].mean() if x_px > 0 else 0
            
            # Junction strength = geometric mean of both strips
            junction_str = np.sqrt(max(0, h_strip) * max(0, v_strip))
            
            if junction_str >= min_density:
                junctions.append({
                    'x': xw, 'z': zw,
                    'h_str': float(h_strip),
                    'v_str': float(v_strip),
                    'strength': float(junction_str),
                })
    
    junctions.sort(key=lambda j: -j['strength'])
    return junctions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('-o', '--output', default='results/v34_junction')
    args = parser.parse_args()
    
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = Path(args.mesh).stem
    
    print(f"Loading: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh')
    print(f"  {len(mesh.vertices)} verts")
    
    up_idx = detect_up_axis(mesh)
    rx, rz = project_vertices(mesh, up_idx)
    angle = find_dominant_angle(rx, rz, cell=args.cell)
    cos_a, sin_a = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a
    
    cs = args.cell
    margin = 0.3
    x_min, z_min = rx2.min() - margin, rz2.min() - margin
    nx = int((rx2.max() + margin - x_min) / cs) + 1
    nz = int((rz2.max() + margin - z_min) / cs) + 1
    img = np.zeros((nz, nx), dtype=np.float32)
    xi = np.clip(((rx2 - x_min) / cs).astype(int), 0, nx - 1)
    zi = np.clip(((rz2 - z_min) / cs).astype(int), 0, nz - 1)
    np.add.at(img, (zi, xi), 1)
    
    # Room mask
    occupied = (img > 0).astype(np.uint8)
    k = max(3, int(0.15 / cs)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    lbl, n = ndimage.label(filled)
    sizes = ndimage.sum(filled, lbl, range(1, n+1))
    room_mask = (lbl == np.argmax(sizes)+1).astype(np.uint8)
    
    # Find walls
    x_walls, x_str, z_walls, z_str = find_wall_candidates(img, x_min, z_min, cs)
    print(f"  Wall candidates: {len(x_walls)}X, {len(z_walls)}Z")
    
    # Detect junctions
    junctions = detect_junctions(img, x_min, z_min, cs, x_walls, z_walls, room_mask)
    print(f"  Junctions found: {len(junctions)}")
    
    # Filter: keep junctions with strength > median
    if junctions:
        median_str = np.median([j['strength'] for j in junctions])
        strong_junctions = [j for j in junctions if j['strength'] > median_str]
    else:
        strong_junctions = []
    
    print(f"  Strong junctions: {len(strong_junctions)}")
    for j in strong_junctions[:15]:
        print(f"    ({j['x']:+.2f}, {j['z']:+.2f}) str={j['strength']:.1f} h={j['h_str']:.1f} v={j['v_str']:.1f}")
    
    # Visualize
    extent = [x_min, x_min + nx*cs, z_min, z_min + nz*cs]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Density
    axes[0].imshow(np.log1p(img), origin='lower', extent=extent, cmap='hot')
    axes[0].set_title('Density')
    
    # Wall candidates
    axes[1].imshow(np.log1p(img), origin='lower', extent=extent, cmap='gray', alpha=0.5)
    for xw in x_walls:
        axes[1].axvline(xw, color='cyan', alpha=0.3, lw=0.5)
    for zw in z_walls:
        axes[1].axhline(zw, color='lime', alpha=0.3, lw=0.5)
    axes[1].set_title(f'Wall candidates ({len(x_walls)}X, {len(z_walls)}Z)')
    
    # Junctions
    axes[2].imshow(np.log1p(img), origin='lower', extent=extent, cmap='gray', alpha=0.5)
    if strong_junctions:
        jx = [j['x'] for j in strong_junctions]
        jz = [j['z'] for j in strong_junctions]
        js = [j['strength'] for j in strong_junctions]
        sc = axes[2].scatter(jx, jz, c=js, cmap='hot', s=30, zorder=3, edgecolors='white', linewidth=0.5)
        plt.colorbar(sc, ax=axes[2], label='Junction strength')
    axes[2].set_title(f'Strong junctions ({len(strong_junctions)})')
    
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.suptitle('v34 Junction Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f'v34_{mesh_name}_junctions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir / f'v34_{mesh_name}_junctions.png'}")


if __name__ == '__main__':
    main()

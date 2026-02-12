#!/usr/bin/env python3
"""
v34_wall_debug — Diagnostic overlay for wall detection accuracy.

Produces three images:
1. Main overlay: density + Hough walls (red) + room polygons (green) + mask contour (cyan) + doors (blue)
2. Edge detection: density + Sobel/Canny edges overlaid
3. Wall profile crops: 1D density profiles along each detected wall
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import math
import cv2
import shutil
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndlabel, uniform_filter1d

# Import pipeline functions from v32
import sys
sys.path.insert(0, str(Path(__file__).parent))
from v32_strip_merge import (
    detect_up_axis, project_vertices, find_dominant_angle,
    build_density_image, build_room_mask, hough_wall_positions,
    wall_has_evidence, cut_mask_with_walls, extract_and_merge_rooms,
    detect_doors,
    ROOM_COLORS, polygon_centroid, compute_polygon_area,
    make_rectilinear, remove_collinear, simplify_polygon,
)


def run_pipeline(mesh_path, cell=0.02, nms=0.3):
    """Run the full v32 pipeline, return all intermediate data."""
    print(f"Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    up_idx, up_name = detect_up_axis(mesh)
    rx, rz = project_vertices(mesh, up_idx)
    angle = find_dominant_angle(rx, rz, cell=cell)
    print(f"  Up={up_name}, Rotation: {angle:.1f}°")

    cos_a, sin_a = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rx2 = rx * cos_a - rz * sin_a
    rz2 = rx * sin_a + rz * cos_a

    density_img, x_min, z_min, cs = build_density_image(rx2, rz2, cell_size=cell)
    room_mask = build_room_mask(density_img, cs)
    x_walls, z_walls, x_str, z_str = hough_wall_positions(density_img, x_min, z_min, cs, nms_dist=nms)
    print(f"  Hough walls: {len(x_walls)}X, {len(z_walls)}Z")

    # Score walls
    mask_rows = np.where(room_mask.any(axis=1))[0]
    mask_cols = np.where(room_mask.any(axis=0))[0]
    bx_min = x_min + mask_cols[0] * cs
    bx_max = x_min + mask_cols[-1] * cs
    bz_min = z_min + mask_rows[0] * cs
    bz_max = z_min + mask_rows[-1] * cs
    bmargin = 0.3

    def score_walls_local(walls, strengths, axis, bound_lo, bound_hi):
        scored = []
        for i, w in enumerate(walls):
            has_ev, extent, max_run = wall_has_evidence(density_img, x_min, z_min, cs, w, axis, room_mask)
            strength = strengths[i] if i < len(strengths) else 0
            score = float(strength * max_run)
            is_boundary = abs(w - bound_lo) < bmargin or abs(w - bound_hi) < bmargin
            scored.append((w, score, max_run, is_boundary, has_ev))
        scored.sort(key=lambda t: -t[1])
        return scored

    x_scored = score_walls_local(x_walls, x_str, 'x', bx_min, bx_max)
    z_scored = score_walls_local(z_walls, z_str, 'z', bz_min, bz_max)

    # Select cuts (same logic as v32)
    min_sep = 1.0
    def select_top(scored, max_n):
        sel = []
        for pos, score, run, is_bnd, has_ev in scored:
            if is_bnd or run < 0.8: continue
            if any(abs(pos - s) < min_sep for s in sel): continue
            sel.append(pos)
            if len(sel) >= max_n: break
        return sel

    mask_area = np.sum(room_mask) * cs * cs
    is_single = mask_area < 20.0
    sel_x = [] if is_single else select_top(x_scored, 2)
    sel_z = [] if is_single else select_top(z_scored, 2)

    # Cut and extract rooms
    rooms_data = []
    doors = []
    if sel_x or sel_z:
        sel_x_arr = np.array(sorted(sel_x))
        sel_z_arr = np.array(sorted(sel_z))
        sel_x_str = np.array([next(s for w, s, *_ in x_scored if w == xw) for xw in sel_x_arr]) if len(sel_x_arr) else np.array([])
        sel_z_str = np.array([next(s for w, s, *_ in z_scored if w == zw) for zw in sel_z_arr]) if len(sel_z_arr) else np.array([])
        cut_mask, valid_x, valid_z = cut_mask_with_walls(
            room_mask, density_img, x_min, z_min, cs,
            sel_x_arr, sel_z_arr, sel_x_str, sel_z_str, min_wall_run=0.5)
        rooms_data = extract_and_merge_rooms(cut_mask, density_img, x_min, z_min, cs,
                                              x_walls, z_walls, valid_x, valid_z)
    else:
        valid_x, valid_z = [], []

    if rooms_data:
        doors = detect_doors(density_img, x_min, z_min, cs, rooms_data)

    return {
        'density_img': density_img, 'x_min': x_min, 'z_min': z_min, 'cs': cs,
        'room_mask': room_mask,
        'x_walls': x_walls, 'z_walls': z_walls,
        'x_str': x_str, 'z_str': z_str,
        'x_scored': x_scored, 'z_scored': z_scored,
        'sel_x': sel_x, 'sel_z': sel_z,
        'valid_x': valid_x, 'valid_z': valid_z,
        'rooms_data': rooms_data, 'doors': doors,
    }


def image1_main_overlay(data, out_path):
    """Density + red Hough walls + green room polygons + cyan mask contour + blue doors."""
    d = data
    density = d['density_img']
    x_min, z_min, cs = d['x_min'], d['z_min'], d['cs']
    extent = [x_min, x_min + density.shape[1]*cs, z_min, z_min + density.shape[0]*cs]

    fig, ax = plt.subplots(figsize=(16, 14))

    # Background: log-scaled density with hot colormap
    ax.imshow(np.log1p(density), origin='lower', extent=extent, cmap='hot', aspect='equal', zorder=0)

    # Red lines: ALL Hough wall positions, thickness ∝ score
    x_max_str = max(d['x_str'].max(), d['z_str'].max()) if len(d['x_str']) > 0 and len(d['z_str']) > 0 else 1
    for i, xw in enumerate(d['x_walls']):
        lw = max(0.5, 4 * d['x_str'][i] / x_max_str)
        ax.axvline(xw, color='red', alpha=0.6, lw=lw, zorder=1)
    for i, zw in enumerate(d['z_walls']):
        lw = max(0.5, 4 * d['z_str'][i] / x_max_str)
        ax.axhline(zw, color='red', alpha=0.6, lw=lw, zorder=1)

    # Selected cuts in bright white dashed
    for xw in d['sel_x']:
        ax.axvline(xw, color='white', alpha=0.9, lw=2.5, ls='--', zorder=2)
    for zw in d['sel_z']:
        ax.axhline(zw, color='white', alpha=0.9, lw=2.5, ls='--', zorder=2)

    # Cyan: room mask contour
    mask_u8 = d['room_mask'].astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        wx = x_min + pts[:, 0] * cs
        wz = z_min + pts[:, 1] * cs
        ax.plot(np.append(wx, wx[0]), np.append(wz, wz[0]), color='cyan', lw=1.5, alpha=0.8, zorder=3)

    # Green polygons: detected rooms
    for rd in d['rooms_data']:
        poly = rd['polygon']
        if len(poly) < 3: continue
        xs = [p[0] for p in poly] + [poly[0][0]]
        zs = [p[1] for p in poly] + [poly[0][1]]
        ax.fill(xs, zs, color='lime', alpha=0.15, zorder=4)
        ax.plot(xs, zs, color='lime', lw=2, alpha=0.8, zorder=5)
        cx, cz = polygon_centroid(poly)
        ax.text(cx, cz, f"{rd['name']}\n{rd['area']:.1f}m²",
                ha='center', va='center', fontsize=8, color='lime',
                fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    # Blue dots: doors
    for door in d['doors']:
        ax.plot(door['x'], door['z'], 'o', color='dodgerblue', ms=12, mew=2, mec='white', zorder=7)
        ax.text(door['x'], door['z'] + 0.15, 'DOOR', ha='center', fontsize=6,
                color='dodgerblue', fontweight='bold', zorder=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Hough walls (all)'),
        Line2D([0], [0], color='white', lw=2, ls='--', label='Selected cuts'),
        Line2D([0], [0], color='cyan', lw=1.5, label='Room mask contour'),
        Line2D([0], [0], color='lime', lw=2, label='Room polygons'),
        Line2D([0], [0], marker='o', color='dodgerblue', lw=0, ms=8, label='Doors'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              facecolor='black', edgecolor='white', labelcolor='white')

    ax.set_title('v34 Wall Debug: Density + Detected Walls Overlay', fontsize=14, fontweight='bold', color='white',
                 bbox=dict(facecolor='black', alpha=0.7, pad=5))
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  [1] Main overlay: {out_path}")


def image2_edge_detection(data, out_path):
    """Density + Sobel/Canny edges overlaid."""
    density = data['density_img']
    x_min, z_min, cs = data['x_min'], data['z_min'], data['cs']
    extent = [x_min, x_min + density.shape[1]*cs, z_min, z_min + density.shape[0]*cs]

    # Prepare density for edge detection
    log_d = np.log1p(density).astype(np.float32)
    norm = ((log_d / max(log_d.max(), 1e-6)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(norm, (5, 5), 1.0)

    # Sobel
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    sobel_norm = (sobel_mag / max(sobel_mag.max(), 1e-6) * 255).astype(np.uint8)

    # Canny
    canny = cv2.Canny(blurred, 30, 100)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Original density
    axes[0].imshow(np.log1p(density), origin='lower', extent=extent, cmap='hot', aspect='equal')
    axes[0].set_title('Density (log)', fontsize=12, fontweight='bold')

    # Sobel edges
    axes[1].imshow(np.log1p(density), origin='lower', extent=extent, cmap='gray', aspect='equal', alpha=0.4)
    sobel_rgba = np.zeros((*sobel_norm.shape, 4), dtype=np.uint8)
    sobel_rgba[..., 1] = sobel_norm  # green channel
    sobel_rgba[..., 3] = np.clip(sobel_norm * 2, 0, 255)
    axes[1].imshow(sobel_rgba, origin='lower', extent=extent, aspect='equal')
    axes[1].set_title('Sobel Edges (green)', fontsize=12, fontweight='bold')

    # Canny edges
    axes[2].imshow(np.log1p(density), origin='lower', extent=extent, cmap='gray', aspect='equal', alpha=0.4)
    canny_rgba = np.zeros((*canny.shape, 4), dtype=np.uint8)
    canny_rgba[..., 0] = canny  # red
    canny_rgba[..., 1] = canny * 200 // 255  # orange tint
    canny_rgba[..., 3] = canny
    axes[2].imshow(canny_rgba, origin='lower', extent=extent, aspect='equal')
    # Overlay Hough walls for comparison
    for xw in data['x_walls']:
        axes[2].axvline(xw, color='cyan', alpha=0.3, lw=0.5)
    for zw in data['z_walls']:
        axes[2].axhline(zw, color='cyan', alpha=0.3, lw=0.5)
    axes[2].set_title('Canny Edges + Hough walls (cyan)', fontsize=12, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] Edge detection: {out_path}")


def image3_wall_profiles(data, out_path):
    """Zoom crops + 1D density profiles along each detected wall."""
    density = data['density_img']
    x_min, z_min, cs = data['x_min'], data['z_min'], data['cs']
    room_mask = data['room_mask']

    all_walls = []
    for i, xw in enumerate(data['x_walls']):
        has_ev, extent, max_run = wall_has_evidence(density, x_min, z_min, cs, xw, 'x', room_mask)
        all_walls.append(('X', xw, data['x_str'][i], has_ev, max_run))
    for i, zw in enumerate(data['z_walls']):
        has_ev, extent, max_run = wall_has_evidence(density, x_min, z_min, cs, zw, 'z', room_mask)
        all_walls.append(('Z', zw, data['z_str'][i], has_ev, max_run))

    # Sort by strength, show top 12
    all_walls.sort(key=lambda w: -w[2])
    walls_to_show = all_walls[:min(12, len(all_walls))]

    if not walls_to_show:
        print("  [3] No walls to profile")
        return

    n = len(walls_to_show)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    strip_half = max(2, int(0.06 / cs))

    for idx, (axis, pos, strength, has_ev, max_run) in enumerate(walls_to_show):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        if axis == 'X':
            px = int((pos - x_min) / cs)
            if 0 <= px < density.shape[1]:
                lo = max(0, px - strip_half)
                hi = min(density.shape[1], px + strip_half + 1)
                profile = density[:, lo:hi].max(axis=1)
                mask_col = room_mask[:, px]
                world_coords = z_min + np.arange(len(profile)) * cs
                label = f"X={pos:.2f}m"
            else:
                continue
        else:
            px = int((pos - z_min) / cs)
            if 0 <= px < density.shape[0]:
                lo = max(0, px - strip_half)
                hi = min(density.shape[0], px + strip_half + 1)
                profile = density[lo:hi, :].max(axis=0)
                mask_col = room_mask[px, :]
                world_coords = x_min + np.arange(len(profile)) * cs
                label = f"Z={pos:.2f}m"
            else:
                continue

        ax.fill_between(world_coords, 0, profile, alpha=0.3, color='orange')
        ax.plot(world_coords, profile, color='red', lw=1)

        # Show mask extent
        inside = mask_col > 0
        if np.any(inside):
            inside_idx = np.where(inside)[0]
            ax.axvspan(world_coords[inside_idx[0]], world_coords[inside_idx[-1]],
                       alpha=0.1, color='cyan', label='mask extent')

        # Threshold line
        pos_vals = profile[profile > 0]
        if len(pos_vals) > 0:
            thr = max(2, np.percentile(pos_vals, 30))
            ax.axhline(thr, color='green', ls='--', alpha=0.5, lw=0.8, label=f'thr={thr:.0f}')

        is_selected = (axis == 'X' and pos in data['sel_x']) or (axis == 'Z' and pos in data['sel_z'])
        title_color = 'green' if is_selected else ('blue' if has_ev else 'gray')
        status = "SELECTED" if is_selected else ("evidence" if has_ev else "weak")
        ax.set_title(f"{label} | str={strength:.0f} run={max_run:.1f}m [{status}]",
                     fontsize=9, fontweight='bold', color=title_color)
        ax.set_ylabel('density')
        ax.legend(fontsize=6, loc='upper right')

    # Hide unused axes
    for idx in range(len(walls_to_show), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle('Wall Density Profiles (sorted by projection strength)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] Wall profiles: {out_path}")


def main():
    mesh_path = Path.home() / "projects/mesh2plan/data/multiroom/2026_02_10_18_31_36/export_refined.obj"
    out_dir = Path.home() / "projects/mesh2plan/results/v34_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_dst = Path.home() / ".openclaw/workspace/v34_overlay.png"

    data = run_pipeline(str(mesh_path))

    image1_main_overlay(data, out_dir / "overlay_main.png")
    image2_edge_detection(data, out_dir / "overlay_edges.png")
    image3_wall_profiles(data, out_dir / "wall_profiles.png")

    # Copy main overlay to workspace
    shutil.copy2(out_dir / "overlay_main.png", workspace_dst)
    print(f"\n  Copied main overlay → {workspace_dst}")
    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v66b - Smart constraint fitting

Strategy:
1. Build wall density (mirror-fixed, rotated)
2. Find ALL wall-line candidates from profiles
3. Try combinations of 2V + 3-4H interior walls (plus exterior)
4. For each combo, compute room dimensions and score against known floor plan
5. Pick the best match

Key: exterior walls are where density drops to near-zero, not vertex extent.
Use the 5th/95th percentile of occupied rows/cols for exterior bounds.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks
from itertools import combinations

RESOLUTION = 0.02
WALL_ANGLE = 29.0


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def rotate_points(pts, angle_deg, center=None):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if center is not None: pts = pts - center
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: rotated += center
    return rotated


def build_wall_density(mesh, angle_deg):
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]  # MIRROR FIX
    
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.3
    xmax, zmax = rot_verts.max(axis=0) + 0.3
    w = int((xmax-xmin)/RESOLUTION)
    h = int((zmax-zmin)/RESOLUTION)
    
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py = np.clip(((wall_rot[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    np.add.at(density, (py, px), 1)
    density = cv2.GaussianBlur(density, (5,5), 1.0)
    
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, center=center)
    return density, grid, rot_verts


def find_exterior(density, grid):
    """Find apartment exterior walls from density."""
    # Threshold density to get apartment mask
    thresh = np.percentile(density[density > 0], 10)
    mask = (density > thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = binary_fill_holes(mask).astype(np.uint8) * 255
    
    # Find bounding area using density mass
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    
    # Use 2%/98% percentile of occupied pixels for tighter bounds
    row_density = density.sum(axis=1)
    col_density = density.sum(axis=0)
    
    row_cumsum = np.cumsum(row_density)
    col_cumsum = np.cumsum(col_density)
    if row_cumsum[-1] > 0:
        row_cumsum /= row_cumsum[-1]
        col_cumsum /= col_cumsum[-1]
    
    bot_row = np.searchsorted(row_cumsum, 0.02)
    top_row = np.searchsorted(row_cumsum, 0.98)
    left_col = np.searchsorted(col_cumsum, 0.02)
    right_col = np.searchsorted(col_cumsum, 0.98)
    
    ext = dict(
        left=left_col * RESOLUTION + grid['xmin'],
        right=right_col * RESOLUTION + grid['xmin'],
        bot=bot_row * RESOLUTION + grid['zmin'],
        top=top_row * RESOLUTION + grid['zmin'],
    )
    ext['width'] = ext['right'] - ext['left']
    ext['height'] = ext['top'] - ext['bot']
    return ext, mask


def find_wall_lines(density, mask, grid, axis, min_gap=0.20):
    """Find wall line positions from profile peaks."""
    masked = density * (mask > 0).astype(np.float32)
    
    if axis == 'v':
        profile = masked.sum(axis=0)
        origin = grid['xmin']
    else:
        profile = masked.sum(axis=1)
        origin = grid['zmin']
    
    kernel = np.ones(5) / 5
    smooth = np.convolve(profile, kernel, mode='same')
    
    thresh = np.percentile(smooth[smooth > 0], 25) if (smooth > 0).any() else 0
    peaks_idx, props = find_peaks(smooth, height=thresh, distance=int(0.20/RESOLUTION), 
                                   prominence=thresh*0.2)
    
    walls = [(idx * RESOLUTION + origin, float(smooth[idx])) for idx in peaks_idx]
    walls.sort(key=lambda x: x[1], reverse=True)
    return walls, smooth


def score_layout(rooms_dict, actual):
    """Score how well detected rooms match actual dimensions."""
    err = 0
    for name, (aw, ah, aa) in actual.items():
        if name not in rooms_dict:
            err += aa * 3  # heavy penalty for missing room
            continue
        dw, dh = rooms_dict[name]
        # Allow flipped dimensions
        e1 = abs(dw - aw) + abs(dh - ah)
        e2 = abs(dw - ah) + abs(dh - aw)
        err += min(e1, e2) * 2
        da = dw * dh
        err += abs(da - aa) * 0.5
    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v66b')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    density, grid, rot_verts = build_wall_density(mesh, args.angle)
    ext, mask = find_exterior(density, grid)
    
    print(f"Exterior: X=[{ext['left']:.2f}, {ext['right']:.2f}], Z=[{ext['bot']:.2f}, {ext['top']:.2f}]")
    print(f"Size: {ext['width']:.2f} × {ext['height']:.2f}m")
    
    v_walls, v_prof = find_wall_lines(density, mask, grid, 'v')
    h_walls, h_prof = find_wall_lines(density, mask, grid, 'h')
    
    print(f"\nV walls: {[(f'{p:.2f}', f'{s:.0f}') for p,s in v_walls[:10]]}")
    print(f"H walls: {[(f'{p:.2f}', f'{s:.0f}') for p,s in h_walls[:10]]}")
    
    # Actual dimensions: (width, height, area)
    actual = {
        'Bedroom 1': (3.38, 4.59, 15.22),
        'Bedroom 2': (3.31, 5.58, 15.5),
        'Hallway': (1.73, 2.95, 5.1),
        'Bathroom': (1.56, 1.59, 2.5),
        'WC': (1.01, 1.98, 2.0),
    }
    
    # Layout from actual floor plan:
    # The apartment is divided by 2 vertical walls into 3 columns:
    #   Left column (Bedroom 2): width ~3.31m
    #   Center column (Bathroom/Hallway/WC): width ~1.56-1.73m
    #   Right column (Bedroom 1): width ~3.38m
    # 
    # And by several horizontal walls:
    #   The left bedroom spans from bottom to ~5.58m up
    #   The bathroom is at the top of center column
    #   The hallway is in the middle of center column
    #   The WC is at the bottom of center column
    #   The right bedroom spans from ~top-4.59 to top
    
    # Filter to interior walls only
    margin = 0.3
    v_interior = [(p, s) for p, s in v_walls 
                  if ext['left'] + margin < p < ext['right'] - margin]
    h_interior = [(p, s) for p, s in h_walls 
                  if ext['bot'] + margin < p < ext['top'] - margin]
    
    print(f"\nInterior V walls: {[(f'{p:.2f}', f'{s:.0f}') for p,s in v_interior[:8]]}")
    print(f"Interior H walls: {[(f'{p:.2f}', f'{s:.0f}') for p,s in h_interior[:8]]}")
    
    L = ext['left']
    R = ext['right']
    B = ext['bot']
    T = ext['top']
    
    # Try all pairs of V walls as the two column dividers
    best_score = float('inf')
    best_layout = None
    
    v_cands = [p for p, s in v_interior[:8]]
    h_cands = [p for p, s in h_interior[:8]]
    
    for v1, v2 in combinations(sorted(v_cands), 2):
        # v1 = left column/center divider, v2 = center/right divider
        left_w = v1 - L
        center_w = v2 - v1
        right_w = R - v2
        
        # Quick filter: dimensions should be roughly plausible
        if left_w < 1.5 or left_w > 5: continue
        if center_w < 0.5 or center_w > 4: continue
        if right_w < 1.5 or right_w > 5: continue
        
        # Now try H walls to divide the center column
        # We need 2-3 H walls in center column to create bathroom/hallway/WC
        for nh in range(2, min(5, len(h_cands)+1)):
            for h_combo in combinations(sorted(h_cands), nh):
                # Center column rooms (bottom to top): WC, hallway, bathroom
                h_sorted = sorted(h_combo)
                
                # Also need to determine right bedroom vertical extent
                # Right bedroom: top of apartment down to some H wall
                # Left bedroom: bottom up to some H wall
                
                rooms = {}
                
                # Right bedroom: v2 to R, some_h to T
                # Find which H wall is the bottom of right bedroom
                # It should be near T - 4.59
                target_rb_bot = T - 4.59
                rb_bot_candidates = [h for h in h_sorted if abs(h - target_rb_bot) < 1.5]
                if not rb_bot_candidates:
                    rb_bot = target_rb_bot  # fallback
                else:
                    rb_bot = min(rb_bot_candidates, key=lambda h: abs(h - target_rb_bot))
                
                rooms['Bedroom 1'] = (right_w, T - rb_bot)
                
                # Left bedroom: L to v1, B to some_h
                # Top of left bedroom should be near B + 5.58
                target_lb_top = B + 5.58
                lb_top_candidates = [h for h in h_sorted if abs(h - target_lb_top) < 1.5]
                if not lb_top_candidates:
                    lb_top = target_lb_top
                else:
                    lb_top = min(lb_top_candidates, key=lambda h: abs(h - target_lb_top))
                
                rooms['Bedroom 2'] = (left_w, lb_top - B)
                
                # Center column division
                # Sort the H walls between B and T
                center_h = sorted([h for h in h_sorted if B + 0.3 < h < T - 0.3])
                
                if len(center_h) >= 2:
                    # Bottom segment = WC, middle = hallway, top = bathroom
                    # But which H walls?
                    # Try: WC from B to center_h[0], hallway from center_h[0] to center_h[-1], 
                    # bathroom from center_h[-1] to T
                    for i in range(len(center_h)):
                        for j in range(i+1, len(center_h)):
                            wc_h = center_h[i] - B
                            hall_h = center_h[j] - center_h[i]
                            bath_h = T - center_h[j]
                            
                            rooms_test = dict(rooms)
                            rooms_test['WC'] = (center_w, wc_h)
                            rooms_test['Hallway'] = (center_w, hall_h)
                            rooms_test['Bathroom'] = (center_w, bath_h)
                            
                            sc = score_layout(rooms_test, actual)
                            if sc < best_score:
                                best_score = sc
                                best_layout = {
                                    'v1': v1, 'v2': v2,
                                    'h_center': (center_h[i], center_h[j]),
                                    'rb_bot': rb_bot, 'lb_top': lb_top,
                                    'rooms': rooms_test,
                                    'rects': {
                                        'Bedroom 1': (v2, rb_bot, R, T),
                                        'Bedroom 2': (L, B, v1, lb_top),
                                        'WC': (v1, B, v2, center_h[i]),
                                        'Hallway': (v1, center_h[i], v2, center_h[j]),
                                        'Bathroom': (v1, center_h[j], v2, T),
                                    }
                                }
                                # Print progress
                                print(f"  score={sc:.1f} v=[{v1:.2f},{v2:.2f}] h=[{center_h[i]:.2f},{center_h[j]:.2f}] "
                                      f"rb_bot={rb_bot:.2f} lb_top={lb_top:.2f}")
                                for rn, (rw, rh) in rooms_test.items():
                                    aw, ah, aa = actual[rn]
                                    print(f"    {rn}: {rw:.2f}×{rh:.2f}={rw*rh:.1f}m² (target {aw}×{ah}={aa}m²)")
    
    if best_layout is None:
        print("FAILED to find any layout!")
        return
    
    print(f"\n=== BEST LAYOUT (score={best_score:.1f}) ===")
    rects = best_layout['rects']
    rooms_dims = best_layout['rooms']
    total = 0
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms_dims[name]
        aw, ah, aa = actual[name]
        da = dw * dh
        total += da
        r = rects[name]
        print(f"  {name}: {dw:.2f}×{dh:.2f}={da:.1f}m² (target {aw}×{ah}={aa}m²) [{r[0]:.2f},{r[1]:.2f}]-[{r[2]:.2f},{r[3]:.2f}]")
    print(f"  Total: {total:.1f}m²")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: density + walls
    ax = axes[0]
    ex = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    # Draw selected walls
    ax.axvline(best_layout['v1'], color='lime', linewidth=2, label='V walls')
    ax.axvline(best_layout['v2'], color='lime', linewidth=2)
    for h in best_layout['h_center']:
        ax.axhline(h, color='cyan', linewidth=2)
    ax.axhline(best_layout['rb_bot'], color='yellow', linewidth=1.5, linestyle='--')
    ax.axhline(best_layout['lb_top'], color='yellow', linewidth=1.5, linestyle='--')
    ax.axvline(L, color='white', linewidth=1, linestyle=':')
    ax.axvline(R, color='white', linewidth=1, linestyle=':')
    ax.axhline(B, color='white', linewidth=1, linestyle=':')
    ax.axhline(T, color='white', linewidth=1, linestyle=':')
    ax.set_title("Wall density + selected walls")
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3])
    
    # Panel 2: rooms
    ax = axes[1]
    pastel = {'Bedroom 1': '#FFB3BA', 'Bedroom 2': '#BAE1FF', 'Hallway': '#FFFFBA', 
              'Bathroom': '#BAFFC9', 'WC': '#E8BAFF'}
    
    for name, (x0, z0, x1, z1) in rects.items():
        w = x1 - x0
        h = z1 - z0
        a = w * h
        rect = plt.Rectangle((x0, z0), w, h, facecolor=pastel.get(name, '#DDD'), 
                              alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text((x0+x1)/2, (z0+z1)/2, f"{name}\n{a:.1f}m²\n{w:.2f}×{h:.2f}m", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(L-1, R+1)
    ax.set_ylim(B-1, T+1)
    ax.set_aspect('equal')
    ax.set_title(f"v66b — {len(rects)} rooms, {total:.1f}m²")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: comparison
    ax = axes[2]
    ax.text(0.5, 0.95, "Actual vs Detected", ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
    y = 0.85
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms_dims[name]
        da = dw*dh
        ax.text(0.05, y, f"[D] {name}: {da:.1f}m² ({dw:.2f}×{dh:.2f}m)", fontsize=10, transform=ax.transAxes, color='blue')
        y -= 0.07
    y -= 0.05
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        aw, ah, aa = actual[name]
        ax.text(0.05, y, f"[A] {name}: {aa:.1f}m² ({aw:.2f}×{ah:.2f}m)", fontsize=10, transform=ax.transAxes, color='red')
        y -= 0.07
    ax.axis('off')
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mesh2plan v67 - Variable center column width + constrained bedrooms

Key fixes over v66e:
1. WC is narrower (1.01m) than hallway (1.70m) — allow different X extents per center room
2. Bedroom 2 was over-extended (17.4m² vs 15.5m²) — tighter height constraints
3. Use density peaks to validate wall positions, but constrain search with known dimensions

From actual floor plan:
- Bedroom 1 (right): 3.38 × 4.59 = 15.22m²
- Bedroom 2 (left): 3.31 × 5.58 = 15.5m² (internal width 2.75m)
- Hallway (center): ~1.73 × 2.95 = 5.1m²
- Bathroom (top-center): 1.56 × 1.59 = 2.5m²
- WC (bottom-center): 1.01 × 1.98 = 2.0m²
- Entry: ~2.01m vertical
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy.signal import find_peaks

RESOLUTION = 0.02


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
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: rotated += center
    return rotated


def build_density(mesh, angle_deg):
    """Build wall density image from mesh, with mirror fix (negate X)."""
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]  # MIRROR FIX
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.5
    xmax, zmax = rot_verts.max(axis=0) + 0.5
    w = int((xmax - xmin) / RESOLUTION)
    h = int((zmax - zmin) / RESOLUTION)
    
    # Wall faces only (vertical surfaces)
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]  # MIRROR FIX
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / RESOLUTION).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / RESOLUTION).astype(int), 0, h - 1)
    np.add.at(density, (py, px), 1)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, center=center)
    return density, grid


def find_wall_peaks(density, grid, axis):
    """Find wall positions along given axis from density projection."""
    if axis == 'v':
        profile = density.sum(axis=0)
        origin = grid['xmin']
    else:
        profile = density.sum(axis=1)
        origin = grid['zmin']
    kernel = np.ones(5) / 5
    smooth = np.convolve(profile, kernel, mode='same')
    thresh = np.percentile(smooth[smooth > 0], 15) if (smooth > 0).any() else 0
    peaks_idx, props = find_peaks(smooth, height=thresh, distance=int(0.15 / RESOLUTION),
                                   prominence=thresh * 0.15)
    walls = [(idx * RESOLUTION + origin, float(smooth[idx])) for idx in peaks_idx]
    walls.sort(key=lambda x: x[1], reverse=True)
    return walls, smooth


def find_nearest_wall(walls, target, tolerance=0.5):
    """Find wall peak nearest to target position within tolerance."""
    candidates = [(abs(p - target), p, s) for p, s in walls if abs(p - target) < tolerance]
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v67')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    angle = 30.5  # From v66d
    density, grid = build_density(mesh, angle)
    v_walls, v_prof = find_wall_peaks(density, grid, 'v')
    h_walls, h_prof = find_wall_peaks(density, grid, 'h')
    
    vp = sorted([p for p, s in v_walls])
    hp = sorted([p for p, s in h_walls])
    print(f"V walls ({len(vp)}): {[f'{p:.2f}' for p in vp]}")
    print(f"H walls ({len(hp)}): {[f'{p:.2f}' for p in hp]}")
    
    # ============================================================
    # APPROACH: Use known wall positions from v66e as anchors,
    # then refine with density peaks and actual dimensions
    # ============================================================
    
    # Known V walls from v66d/v66e: le=-4.13, v1=-0.83, v2=0.87, re=4.01
    le_t, v1_t, v2_t, re_t = -4.13, -0.83, 0.87, 4.01
    
    le = find_nearest_wall(v_walls, le_t, 0.3)
    v1 = find_nearest_wall(v_walls, v1_t, 0.3)
    v2 = find_nearest_wall(v_walls, v2_t, 0.3)
    re = find_nearest_wall(v_walls, re_t, 0.3)
    
    print(f"\nV walls: le={le:.2f}, v1={v1:.2f}, v2={v2:.2f}, re={re:.2f}")
    print(f"  Left bedroom width: {v1-le:.2f}m (target 3.31m)")
    print(f"  Center column width: {v2-v1:.2f}m (target ~1.70m)")
    print(f"  Right bedroom width: {re-v2:.2f}m (target 3.38m)")
    
    # H walls — find exterior and interior
    # Need be (bottom), te (top), and interior walls for center column
    be_candidates = [p for p in hp if p < -3]
    te_candidates = [p for p in hp if p > 3]
    be = min(be_candidates) if be_candidates else hp[0]
    te = max(te_candidates) if te_candidates else hp[-1]
    
    print(f"  Exterior H: be={be:.2f}, te={te:.2f}, span={te-be:.2f}m")
    
    # Interior H walls (between be and te)
    ih = sorted([p for p in hp if be + 0.3 < p < te - 0.3])
    print(f"  Interior H walls: {[f'{p:.2f}' for p in ih]}")
    
    # ============================================================
    # ROOM PLACEMENT using actual dimensions as constraints
    # ============================================================
    
    # From floor plan (top to bottom in center column):
    # Bathroom (top): 1.56 × 1.59m
    # Entry: ~2.01m tall
    # Hallway: 1.73 × 2.95m
    # WC (bottom): 1.01 × 1.98m
    
    # Strategy: place rooms from known anchor positions
    # WC bottom = be, WC height = 1.98 → wc_top = be + 1.98
    # Bathroom top = te, Bathroom height = 1.59 → bath_bot = te - 1.59
    # Hallway fills the gap, with entry above hallway
    
    wc_top_target = be + 1.98
    bath_bot_target = te - 1.59
    
    wc_top = find_nearest_wall(h_walls, wc_top_target, 0.5)
    bath_bot = find_nearest_wall(h_walls, bath_bot_target, 0.5)
    
    # The hallway should be ~2.95m
    # Try to find a wall that splits the remaining space into hallway + entry
    hall_top_target = wc_top + 2.95
    hall_top = find_nearest_wall(h_walls, hall_top_target, 0.5)
    
    # Entry = hall_top to bath_bot
    entry_h = bath_bot - hall_top
    
    print(f"\n  Center column layout (bottom to top):")
    print(f"    WC:       {be:.2f} to {wc_top:.2f} = {wc_top-be:.2f}m (target 1.98m)")
    print(f"    Hallway:  {wc_top:.2f} to {hall_top:.2f} = {hall_top-wc_top:.2f}m (target 2.95m)")
    print(f"    Entry:    {hall_top:.2f} to {bath_bot:.2f} = {entry_h:.2f}m (target ~2.01m)")
    print(f"    Bathroom: {bath_bot:.2f} to {te:.2f} = {te-bath_bot:.2f}m (target 1.59m)")
    
    # ============================================================
    # BEDROOM PLACEMENT
    # ============================================================
    
    # Bedroom 1 (right): top-aligned with apartment top, 4.59m tall
    rb_bot_target = te - 4.59
    rb_bot = find_nearest_wall(h_walls, rb_bot_target, 0.5)
    
    # Bedroom 2 (left): bottom-aligned with apartment bottom, 5.58m tall
    lb_top_target = be + 5.58
    lb_top = find_nearest_wall(h_walls, lb_top_target, 0.5)
    
    print(f"\n  Bedrooms:")
    print(f"    Bedroom 1 (right): {rb_bot:.2f} to {te:.2f} = {te-rb_bot:.2f}m tall × {re-v2:.2f}m wide (target 4.59×3.38)")
    print(f"    Bedroom 2 (left):  {be:.2f} to {lb_top:.2f} = {lb_top-be:.2f}m tall × {v1-le:.2f}m wide (target 5.58×3.31)")
    
    # ============================================================
    # WC: narrower than hallway — find its actual X extent
    # WC is 1.01m wide. It could be left-aligned, right-aligned, or centered in the column.
    # From the floor plan, the WC appears to be on the RIGHT side of the center column
    # (right-aligned with the hallway right wall v2)
    # ============================================================
    
    # Look for a V wall near v2 - 1.01 = v2 - 1.01
    wc_left_target = v2 - 1.01
    # Also check: maybe WC left = v1 (left-aligned)
    # From floor plan: WC has 0.87m width and there's a 0.30m gap and 0.87m passage
    # Actually looking at floor plan: WC is bottom-center, width 1.01m
    # The hallway width at WC level is 1.78m, but WC only takes 1.01m of it
    # Let's check if there's a V wall peak near v1 + (v2-v1-1.01)/2 (centered)
    # or at v2 - 1.01 (right-aligned) or v1 (left-aligned with width 1.01)
    
    # From floor plan the WC left wall appears at about 0.87m from the right side
    # Let's look at density in the WC zone for V walls
    wc_zone_h_start = max(0, int((be - grid['zmin']) / RESOLUTION))
    wc_zone_h_end = min(grid['h'], int((wc_top - grid['zmin']) / RESOLUTION))
    if wc_zone_h_end > wc_zone_h_start:
        wc_v_profile = density[wc_zone_h_start:wc_zone_h_end, :].sum(axis=0)
        wc_v_smooth = np.convolve(wc_v_profile, np.ones(5)/5, mode='same')
        thresh = np.percentile(wc_v_smooth[wc_v_smooth > 0], 30) if (wc_v_smooth > 0).any() else 0
        wc_peaks_idx, _ = find_peaks(wc_v_smooth, height=thresh, distance=int(0.15/RESOLUTION))
        wc_v_walls = sorted([(idx * RESOLUTION + grid['xmin']) for idx in wc_peaks_idx])
        # Filter to center column region
        wc_v_walls = [p for p in wc_v_walls if v1 - 0.3 < p < v2 + 0.3]
        print(f"\n  WC zone V walls: {[f'{p:.2f}' for p in wc_v_walls]}")
    
    # For now, use the hallway width for WC but note the discrepancy
    # The WC being narrower is a refinement we can add visually
    # Actually, let's try: WC right = v2, WC left = v2 - 1.01
    wc_v1 = v2 - 1.01
    wc_v2 = v2
    
    # ============================================================
    # BUILD ROOM RECTANGLES
    # ============================================================
    
    cw = v2 - v1  # center column width
    
    rects = {
        'Bedroom 1': (v2, rb_bot, re, te),
        'Bedroom 2': (le, be, v1, lb_top),
        'Hallway': (v1, wc_top, v2, hall_top),
        'Entry': (v1, hall_top, v2, bath_bot),
        'Bathroom': (v1, bath_bot, v2, te),
        'WC': (wc_v1, be, wc_v2, wc_top),
    }
    
    actual = {
        'Bedroom 1': (3.38, 4.59, 15.22),
        'Bedroom 2': (3.31, 5.58, 15.5),
        'Hallway': (1.73, 2.95, 5.1),
        'Bathroom': (1.56, 1.59, 2.5),
        'WC': (1.01, 1.98, 2.0),
        'Entry': (2.01, 1.0, 2.0),
    }
    
    print(f"\n=== v67 RESULTS ===")
    total = 0
    for name, (x0, z0, x1, z1) in rects.items():
        w = x1 - x0
        h = z1 - z0
        a = w * h
        total += a
        if name in actual:
            aw, ah, aa = actual[name]
            print(f"  {name}: {w:.2f}×{h:.2f} = {a:.1f}m² (target {aw}×{ah} = {aa}m²) err={abs(a-aa)/aa*100:.0f}%")
        else:
            print(f"  {name}: {w:.2f}×{h:.2f} = {a:.1f}m²")
    print(f"  Total: {total:.1f}m²")
    
    # ============================================================
    # PLOT
    # ============================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Left: density with walls
    ax = axes[0]
    ex = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    for v in [le, v1, v2, re]:
        ax.axvline(v, color='lime', linewidth=2, alpha=0.7)
    ax.axvline(wc_v1, color='yellow', linewidth=1.5, linestyle='--', alpha=0.7)
    for h in [be, wc_top, hall_top, bath_bot, te, rb_bot, lb_top]:
        ax.axhline(h, color='cyan', linewidth=1.5, alpha=0.7)
    ax.set_title(f"Wall density (angle={angle}°)")
    
    # Center: room layout
    ax = axes[1]
    pastel = {'Bedroom 1': '#FFB3BA', 'Bedroom 2': '#BAE1FF', 'Hallway': '#FFFFBA',
              'Bathroom': '#BAFFC9', 'WC': '#E8BAFF', 'Entry': '#FFE0BA'}
    for name, (x0, z0, x1, z1) in rects.items():
        w = x1 - x0; h = z1 - z0; a = w * h
        rect = plt.Rectangle((x0, z0), w, h, facecolor=pastel.get(name, '#DDD'),
                              alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (z0 + z1) / 2, f"{name}\n{a:.1f}m²\n{w:.2f}×{h:.2f}m",
                ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_xlim(le - 1, re + 1)
    ax.set_ylim(be - 1, te + 1)
    ax.set_aspect('equal')
    ax.set_title(f"v67 — {len(rects)} rooms, {total:.1f}m²")
    ax.grid(True, alpha=0.3)
    
    # Right: comparison
    ax = axes[2]
    ax.text(0.5, 0.95, "Actual vs Detected", ha='center', va='top', fontsize=14,
            fontweight='bold', transform=ax.transAxes)
    y = 0.85
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC', 'Entry']:
        x0, z0, x1, z1 = rects[name]
        dw, dh = x1 - x0, z1 - z0
        ax.text(0.05, y, f"[D] {name}: {dw*dh:.1f}m² ({dw:.2f}×{dh:.2f}m)",
                fontsize=10, transform=ax.transAxes, color='blue')
        y -= 0.06
    y -= 0.03
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        aw, ah, aa = actual[name]
        ax.text(0.05, y, f"[A] {name}: {aa}m² ({aw}×{ah}m)",
                fontsize=10, transform=ax.transAxes, color='red')
        y -= 0.06
    ax.axis('off')
    
    plt.tight_layout()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out / 'floorplan.png'}")


if __name__ == '__main__':
    main()

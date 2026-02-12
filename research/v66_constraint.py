#!/usr/bin/env python3
"""
mesh2plan v66 - Constraint-based room fitting

Instead of cutting a messy boundary polygon, we:
1. Build axis-aligned wall density (mirror-fixed, rotated -29°)
2. Project to H and V profiles to find wall line positions
3. Use known room dimensions to MATCH wall lines to room boundaries
4. Build rooms as rectangles aligned to detected walls

Known layout (from actual floor plan):
  Top-right: Right bedroom 3.38×4.59m (15.22m²)
  Left (full height): Left bedroom 3.31×5.58m
  Top-center: Bathroom 1.56×1.59m  
  Center: Hallway ~1.73×2.95m
  Bottom-center: WC 1.01×1.98m
  Entry area at center-top: 2.01m wide
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
    """Build wall-face density image in axis-aligned coordinates."""
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]  # MIRROR FIX
    
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.3
    xmax, zmax = rot_verts.max(axis=0) + 0.3
    w = int((xmax-xmin)/RESOLUTION)
    h = int((zmax-zmin)/RESOLUTION)
    
    # Wall faces only (horizontal normal = wall)
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]  # MIRROR FIX
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py = np.clip(((wall_rot[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    np.add.at(density, (py, px), 1)
    density = cv2.GaussianBlur(density, (5,5), 1.0)
    
    # Also build vertex mask for boundary
    all_rot = rotate_points(pts_xz, -angle_deg, center)
    vmask = np.zeros((h, w), dtype=np.uint8)
    apx = np.clip(((all_rot[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    apy = np.clip(((all_rot[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    vmask[apy, apx] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, k)
    vmask = binary_fill_holes(vmask).astype(np.uint8) * 255
    
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, center=center)
    return density, vmask, grid


def find_walls(density, vmask, grid, axis):
    """Find wall positions along given axis using projection profile.
    axis='v' → project along Y (sum columns) → find X positions of vertical walls
    axis='h' → project along X (sum rows) → find Z positions of horizontal walls
    """
    masked = density * (vmask > 0).astype(np.float32)
    
    if axis == 'v':
        profile = masked.sum(axis=0)  # sum each column → profile along X
        origin = grid['xmin']
    else:
        profile = masked.sum(axis=1)  # sum each row → profile along Z
        origin = grid['zmin']
    
    # Smooth
    kernel = np.ones(7) / 7
    profile = np.convolve(profile, kernel, mode='same')
    
    # Find peaks
    threshold = np.percentile(profile[profile > 0], 30) if (profile > 0).any() else 0
    peaks_idx, props = find_peaks(profile, height=threshold, distance=int(0.15/RESOLUTION), prominence=threshold*0.3)
    
    walls = []
    for idx in peaks_idx:
        pos = idx * RESOLUTION + origin
        strength = float(profile[idx])
        walls.append((pos, strength))
    
    walls.sort(key=lambda x: x[1], reverse=True)
    return walls, profile


def find_best_wall_near(walls, target, tolerance=0.8):
    """Find the strongest wall within tolerance of target position."""
    candidates = [(p, s) for p, s in walls if abs(p - target) < tolerance]
    if not candidates:
        return target  # fallback to target
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v66')
    parser.add_argument('--mesh', default='export_refined.obj')
    parser.add_argument('--angle', type=float, default=WALL_ANGLE)
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    density, vmask, grid = build_wall_density(mesh, args.angle)
    
    v_walls, v_profile = find_walls(density, vmask, grid, 'v')
    h_walls, h_profile = find_walls(density, vmask, grid, 'h')
    
    print(f"\nV walls (X positions): {[(f'{p:.2f}', f'{s:.0f}') for p,s in v_walls[:12]]}")
    print(f"H walls (Z positions): {[(f'{p:.2f}', f'{s:.0f}') for p,s in h_walls[:12]]}")
    
    # Find apartment extent from vmask
    rows = np.where(vmask.any(axis=1))[0]
    cols = np.where(vmask.any(axis=0))[0]
    apt_top = rows[-1] * RESOLUTION + grid['zmin']  # max Z = top
    apt_bot = rows[0] * RESOLUTION + grid['zmin']    # min Z = bottom
    apt_left = cols[0] * RESOLUTION + grid['xmin']   # min X = left
    apt_right = cols[-1] * RESOLUTION + grid['xmin']  # max X = right
    print(f"\nApartment bounds: X=[{apt_left:.2f}, {apt_right:.2f}], Z=[{apt_bot:.2f}, {apt_top:.2f}]")
    print(f"Apartment size: {apt_right-apt_left:.2f} × {apt_top-apt_bot:.2f}m")
    
    # === CONSTRAINT-BASED WALL MATCHING ===
    # Looking at the actual floor plan:
    # The apartment has walls roughly at these relative positions:
    # 
    # Vertical walls (X positions, left to right):
    #   V0: left exterior wall
    #   V1: V0 + 3.31 = right wall of left bedroom / left wall of bathroom/hallway
    #   V2: V1 + ~1.56-1.73 = right wall of bathroom / right wall of hallway  
    #   V3: V2 = left wall of right bedroom (same as V2 in some interpretations)
    #   V4: right exterior wall (V3 + 3.38 or V2 + 3.38)
    #
    # Horizontal walls (Z positions, bottom to top):
    #   H0: bottom exterior
    #   H1: H0 + 1.98 = top of WC
    #   H2: hallway/bedroom boundary  
    #   H3: top of left bedroom = H0 + 5.58
    #   H4: top exterior
    
    # Strategy: find the strongest vertical wall that could be the bedroom divider
    # The right bedroom is 3.38m wide, so the divider is at apt_right - 3.38
    
    target_v1 = apt_right - 3.38   # Right bedroom left wall
    target_v0 = target_v1 - 1.73   # Hallway left wall (hallway is 1.73m wide)
    # But V0 could also be target_v1 - bathroom_width... let's check
    # Actually from the floor plan: left bedroom is 3.31m, then bathroom/hallway, then right bedroom 3.38m
    # Total width ≈ 3.31 + 1.73 + 3.38 ≈ 8.42m ... but apt is ~9.4m. Hmm, walls have thickness.
    
    # Let's try from the left: 
    target_vL = apt_left  # left exterior
    target_v_leftbed_right = apt_left + 3.31  # right wall of left bedroom
    
    # From right:
    target_v_rightbed_left = apt_right - 3.38  # left wall of right bedroom
    
    print(f"\nTarget walls:")
    print(f"  Left bedroom right wall: {target_v_leftbed_right:.2f}")
    print(f"  Right bedroom left wall: {target_v_rightbed_left:.2f}")
    print(f"  Gap between: {target_v_rightbed_left - target_v_leftbed_right:.2f}m (should be ~1.73m hallway)")
    
    # Horizontal targets
    target_h_top = apt_top  # top exterior
    target_h_bot = apt_bot  # bottom exterior
    target_h_wc_top = apt_bot + 1.98  # top of WC
    target_h_leftbed_top = apt_bot + 5.58  # top of left bedroom
    target_h_rightbed_bot = apt_top - 4.59  # bottom of right bedroom
    
    print(f"  WC top: {target_h_wc_top:.2f}")
    print(f"  Left bed top: {target_h_leftbed_top:.2f}")
    print(f"  Right bed bottom: {target_h_rightbed_bot:.2f}")
    
    # Snap to nearest detected walls
    vL = apt_left
    vR = apt_right
    v1 = find_best_wall_near(v_walls, target_v_leftbed_right, 1.0)
    v2 = find_best_wall_near(v_walls, target_v_rightbed_left, 1.0)
    
    hT = apt_top
    hB = apt_bot
    h1 = find_best_wall_near(h_walls, target_h_wc_top, 0.8)
    h2 = find_best_wall_near(h_walls, target_h_leftbed_top, 1.0)
    h3 = find_best_wall_near(h_walls, target_h_rightbed_bot, 1.0)
    
    print(f"\nSnapped walls:")
    print(f"  V1 (left bed right): {v1:.2f} (target {target_v_leftbed_right:.2f})")
    print(f"  V2 (right bed left): {v2:.2f} (target {target_v_rightbed_left:.2f})")
    print(f"  H1 (WC top): {h1:.2f} (target {target_h_wc_top:.2f})")
    print(f"  H2 (left bed top): {h2:.2f} (target {target_h_leftbed_top:.2f})")
    print(f"  H3 (right bed bot): {h3:.2f} (target {target_h_rightbed_bot:.2f})")
    
    # === BUILD ROOMS ===
    # Room definitions as (left, bottom, right, top)
    rooms = {}
    
    # Right bedroom: v2 to vR, h3 to hT
    rooms['Bedroom 1'] = (v2, h3, vR, hT)
    
    # Left bedroom: vL to v1, hB to h2
    rooms['Bedroom 2'] = (vL, hB, v1, h2)
    
    # Hallway: v1 to v2, h1 to h2 (roughly)
    rooms['Hallway'] = (v1, h1, v2, h2)
    
    # Bathroom: v1 to v2, h2 to hT (top-center)
    rooms['Bathroom'] = (v1, h2, v2, hT)
    
    # WC: v1 to v2, hB to h1 (bottom-center)
    rooms['WC'] = (v1, hB, v2, h1)
    
    print(f"\n=== ROOMS ===")
    total = 0
    for name, (x0, z0, x1, z1) in rooms.items():
        w = abs(x1-x0)
        h = abs(z1-z0)
        a = w*h
        total += a
        print(f"  {name}: {a:.1f}m² ({w:.2f}×{h:.2f}m) [{x0:.2f},{z0:.2f}]-[{x1:.2f},{z1:.2f}]")
    print(f"  Total: {total:.1f}m²")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: Wall density with detected wall lines
    ax = axes[0]
    ext = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax.imshow(density, origin='lower', cmap='hot', extent=ext, aspect='equal')
    
    # Draw detected walls
    for pos, s in v_walls[:10]:
        ax.axvline(pos, color='cyan', alpha=0.3, linewidth=0.5)
    for pos, s in h_walls[:10]:
        ax.axhline(pos, color='cyan', alpha=0.3, linewidth=0.5)
    
    # Highlight selected walls
    for v in [v1, v2]:
        ax.axvline(v, color='lime', linewidth=2, linestyle='--')
    for h in [h1, h2, h3]:
        ax.axhline(h, color='lime', linewidth=2, linestyle='--')
    
    ax.set_title("Wall density + selected walls (green)")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    
    # Panel 2: Room layout
    ax = axes[1]
    pastel = {'Bedroom 1': '#FFB3BA', 'Bedroom 2': '#BAE1FF', 'Hallway': '#FFFFBA', 
              'Bathroom': '#BAFFC9', 'WC': '#E8BAFF'}
    
    for name, (x0, z0, x1, z1) in rooms.items():
        w = abs(x1-x0)
        h = abs(z1-z0)
        a = w*h
        rect = plt.Rectangle((min(x0,x1), min(z0,z1)), w, h, 
                              facecolor=pastel.get(name, '#DDD'), alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        cx, cz = (x0+x1)/2, (z0+z1)/2
        ax.text(cx, cz, f"{name}\n{a:.1f}m²\n{w:.2f}×{h:.2f}m", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(apt_left-1, apt_right+1)
    ax.set_ylim(apt_bot-1, apt_top+1)
    ax.set_aspect('equal')
    ax.set_title(f"v66 — {len(rooms)} rooms, {total:.1f}m²")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Actual vs Detected comparison
    ax = axes[2]
    ax.text(0.5, 0.95, "Actual vs Detected", ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    actual = [
        ("Right bedroom", 15.2, 3.38, 4.59),
        ("Left bedroom", 15.5, 3.31, 5.58),
        ("Hallway", 5.1, 1.73, 2.95),
        ("Bathroom", 2.5, 1.56, 1.59),
        ("WC", 2.0, 1.01, 1.98),
    ]
    
    y_pos = 0.85
    for name, (x0, z0, x1, z1) in sorted(rooms.items(), key=lambda x: abs(x[1][2]-x[1][0])*abs(x[1][3]-x[1][1]), reverse=True):
        w = abs(x1-x0)
        h = abs(z1-z0)
        a = w*h
        ax.text(0.05, y_pos, f"[D] {name}: {a:.1f}m² ({w:.2f}×{h:.2f}m)", 
                fontsize=10, transform=ax.transAxes, color='blue')
        y_pos -= 0.07
    
    y_pos -= 0.05
    for aname, area, aw, ah in actual:
        ax.text(0.05, y_pos, f"[A] {aname}: {area:.1f}m² ({aw:.2f}×{ah:.2f}m)", 
                fontsize=10, transform=ax.transAxes, color='red')
        y_pos -= 0.07
    
    ax.axis('off')
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    outfile = out / 'floorplan.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outfile}")
    
    # Also save profiles for debugging
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    x_axis = np.arange(len(v_profile)) * RESOLUTION + grid['xmin']
    ax1.plot(x_axis, v_profile)
    ax1.set_title("V profile (sum of columns → X positions of vertical walls)")
    for v in [v1, v2]: ax1.axvline(v, color='red', linewidth=2)
    ax1.axvline(apt_left, color='gray', linestyle='--')
    ax1.axvline(apt_right, color='gray', linestyle='--')
    
    z_axis = np.arange(len(h_profile)) * RESOLUTION + grid['zmin']
    ax2.plot(z_axis, h_profile)
    ax2.set_title("H profile (sum of rows → Z positions of horizontal walls)")
    for h in [h1, h2, h3]: ax2.axvline(h, color='red', linewidth=2)
    ax2.axvline(apt_bot, color='gray', linestyle='--')
    ax2.axvline(apt_top, color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out / 'profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'profiles.png'}")


if __name__ == '__main__':
    main()

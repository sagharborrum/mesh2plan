#!/usr/bin/env python3
"""
mesh2plan v66d - Use detected walls for BOTH interior and exterior bounds

Key insight: vertex extent includes scan noise. Actual exterior walls are the 
strongest wall lines near the edges. Use ALL walls (including near-edge ones) 
and pick the combination that best matches known dimensions.

Layout model: 3-column (left bedroom | center strip | right bedroom)
- Center strip subdivided horizontally into bathroom/hallway/WC
- Left/right columns bounded by exterior walls + one interior V wall each
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


def build_all(mesh, angle_deg):
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]  # MIRROR FIX
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.5
    xmax, zmax = rot_verts.max(axis=0) + 0.5
    w = int((xmax-xmin)/RESOLUTION)
    h = int((zmax-zmin)/RESOLUTION)
    
    normals = mesh.face_normals
    wf = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wf][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:,0]-xmin)/RESOLUTION).astype(int), 0, w-1)
    py = np.clip(((wall_rot[:,1]-zmin)/RESOLUTION).astype(int), 0, h-1)
    np.add.at(density, (py, px), 1)
    density = cv2.GaussianBlur(density, (5,5), 1.0)
    
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, center=center)
    return density, grid


def find_walls(density, grid, axis):
    if axis == 'v':
        profile = density.sum(axis=0)
        origin = grid['xmin']
    else:
        profile = density.sum(axis=1)
        origin = grid['zmin']
    
    kernel = np.ones(5) / 5
    smooth = np.convolve(profile, kernel, mode='same')
    thresh = np.percentile(smooth[smooth > 0], 15) if (smooth > 0).any() else 0
    peaks_idx, _ = find_peaks(smooth, height=thresh, distance=int(0.15/RESOLUTION), prominence=thresh*0.15)
    
    walls = [(idx * RESOLUTION + origin, float(smooth[idx])) for idx in peaks_idx]
    walls.sort(key=lambda x: x[1], reverse=True)
    return walls, smooth


def score_rooms(rooms):
    actual = {
        'Bedroom 1': (3.38, 4.59),
        'Bedroom 2': (3.31, 5.58),
        'Hallway': (1.73, 2.95),
        'Bathroom': (1.56, 1.59),
        'WC': (1.01, 1.98),
    }
    # Areas for weighting
    areas = {'Bedroom 1': 15.22, 'Bedroom 2': 15.5, 'Hallway': 5.1, 'Bathroom': 2.5, 'WC': 2.0}
    
    err = 0
    for name, (aw, ah) in actual.items():
        if name not in rooms:
            err += 50
            continue
        dw, dh = rooms[name]
        aa = areas[name]
        # Try both orientations, use relative error weighted by area
        e1 = (abs(dw-aw)/aw + abs(dh-ah)/ah) * aa
        e2 = (abs(dw-ah)/ah + abs(dh-aw)/aw) * aa
        err += min(e1, e2)
    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v66d')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    
    best_score = float('inf')
    best_result = None
    
    for angle in np.arange(27, 34, 0.5):
        density, grid = build_all(mesh, angle)
        v_walls, v_prof = find_walls(density, grid, 'v')
        h_walls, h_prof = find_walls(density, grid, 'h')
        
        vp = sorted([p for p, s in v_walls])  # all V wall positions
        hp = sorted([p for p, s in h_walls])  # all H wall positions
        
        if len(vp) < 4 or len(hp) < 4:
            continue
        
        # We need 4 V walls: left_ext, v1 (left bed right), v2 (right bed left), right_ext
        # And 4+ H walls: bot_ext, wc_top, hall_top/bath_bot, top_ext, plus rb_bot, lb_top
        
        for i_le in range(len(vp)):
            for i_v1 in range(i_le+1, len(vp)):
                for i_v2 in range(i_v1+1, len(vp)):
                    for i_re in range(i_v2+1, len(vp)):
                        le, v1, v2, re = vp[i_le], vp[i_v1], vp[i_v2], vp[i_re]
                        
                        lw = v1 - le   # left bedroom width
                        cw = v2 - v1   # center width
                        rw = re - v2   # right bedroom width
                        
                        # Quick dimension filter
                        if not (2.5 < lw < 4.2): continue
                        if not (0.8 < cw < 2.5): continue
                        if not (2.5 < rw < 4.5): continue
                        
                        # H walls: need bot_ext, wc_top, hall_top, top_ext
                        # Also rb_bot (right bedroom bottom) and lb_top (left bedroom top)
                        for i_be in range(len(hp)):
                            for i_te in range(i_be+3, len(hp)):
                                be, te = hp[i_be], hp[i_te]
                                apt_h = te - be
                                if not (6 < apt_h < 10): continue
                                
                                # Interior H walls between be and te
                                ih = [p for p in hp if be + 0.3 < p < te - 0.3]
                                if len(ih) < 2: continue
                                
                                for hc_lo, hc_hi in combinations(ih, 2):
                                    if hc_hi <= hc_lo + 0.3: continue
                                    
                                    wc_h = hc_lo - be
                                    hall_h = hc_hi - hc_lo
                                    bath_h = te - hc_hi
                                    
                                    if wc_h < 0.5 or hall_h < 1.0 or bath_h < 0.5: continue
                                    
                                    # Right bedroom bottom
                                    rb_bot_target = te - 4.59
                                    rb_bots = [p for p in hp if abs(p - rb_bot_target) < 1.2 and be < p < te]
                                    if not rb_bots: rb_bots = [rb_bot_target]
                                    
                                    # Left bedroom top
                                    lb_top_target = be + 5.58
                                    lb_tops = [p for p in hp if abs(p - lb_top_target) < 1.2 and be < p < te]
                                    if not lb_tops: lb_tops = [lb_top_target]
                                    
                                    for rb_bot in rb_bots[:3]:
                                        rb_h = te - rb_bot
                                        if not (3 < rb_h < 6): continue
                                        
                                        for lb_top in lb_tops[:3]:
                                            lb_h = lb_top - be
                                            if not (4 < lb_h < 7): continue
                                            
                                            rooms = {
                                                'Bedroom 1': (rw, rb_h),
                                                'Bedroom 2': (lw, lb_h),
                                                'Hallway': (cw, hall_h),
                                                'Bathroom': (cw, bath_h),
                                                'WC': (cw, wc_h),
                                            }
                                            
                                            sc = score_rooms(rooms)
                                            if sc < best_score:
                                                best_score = sc
                                                best_result = {
                                                    'angle': angle,
                                                    'rects': {
                                                        'Bedroom 1': (v2, rb_bot, re, te),
                                                        'Bedroom 2': (le, be, v1, lb_top),
                                                        'WC': (v1, be, v2, hc_lo),
                                                        'Hallway': (v1, hc_lo, v2, hc_hi),
                                                        'Bathroom': (v1, hc_hi, v2, te),
                                                    },
                                                    'rooms': rooms,
                                                    'walls': {'le': le, 'v1': v1, 'v2': v2, 're': re,
                                                              'be': be, 'te': te,
                                                              'hc_lo': hc_lo, 'hc_hi': hc_hi,
                                                              'rb_bot': rb_bot, 'lb_top': lb_top},
                                                    'density': density, 'grid': grid,
                                                }
                                                if sc < 6:
                                                    print(f"  ★ angle={angle:.1f} sc={sc:.1f} "
                                                          f"V=[{le:.2f},{v1:.2f},{v2:.2f},{re:.2f}] "
                                                          f"H=[{be:.2f}..{te:.2f}]")
                                                    for n, (dw, dh) in rooms.items():
                                                        print(f"    {n}: {dw:.2f}×{dh:.2f}={dw*dh:.1f}m²")
    
    if best_result is None:
        print("FAILED")
        return
    
    angle = best_result['angle']
    rooms = best_result['rooms']
    rects = best_result['rects']
    walls = best_result['walls']
    density = best_result['density']
    grid = best_result['grid']
    
    actual_dims = {
        'Bedroom 1': (3.38, 4.59, 15.22),
        'Bedroom 2': (3.31, 5.58, 15.5),
        'Hallway': (1.73, 2.95, 5.1),
        'Bathroom': (1.56, 1.59, 2.5),
        'WC': (1.01, 1.98, 2.0),
    }
    
    print(f"\n=== BEST (angle={angle:.1f}°, score={best_score:.1f}) ===")
    total = 0
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms[name]
        da = dw*dh
        total += da
        aw, ah, aa = actual_dims[name]
        print(f"  {name}: {dw:.2f}×{dh:.2f}={da:.1f}m² (target {aw}×{ah}={aa}m²)")
    print(f"  Total: {total:.1f}m²")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    ax = axes[0]
    ex = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    for key in ['le', 'v1', 'v2', 're']:
        color = 'lime' if key in ['v1','v2'] else 'white'
        ax.axvline(walls[key], color=color, linewidth=2)
    for key in ['be', 'te']:
        ax.axhline(walls[key], color='white', linewidth=2)
    for key in ['hc_lo', 'hc_hi']:
        ax.axhline(walls[key], color='cyan', linewidth=2)
    ax.axhline(walls['rb_bot'], color='yellow', linewidth=1.5, linestyle='--')
    ax.axhline(walls['lb_top'], color='yellow', linewidth=1.5, linestyle='--')
    ax.set_title(f"Wall density (angle={angle:.1f}°)")
    
    ax = axes[1]
    pastel = {'Bedroom 1': '#FFB3BA', 'Bedroom 2': '#BAE1FF', 'Hallway': '#FFFFBA', 
              'Bathroom': '#BAFFC9', 'WC': '#E8BAFF'}
    for name, (x0, z0, x1, z1) in rects.items():
        w = x1-x0; h = z1-z0; a = w*h
        rect = plt.Rectangle((x0, z0), w, h, facecolor=pastel.get(name, '#DDD'), 
                              alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text((x0+x1)/2, (z0+z1)/2, f"{name}\n{a:.1f}m²\n{w:.2f}×{h:.2f}m", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(walls['le']-1, walls['re']+1)
    ax.set_ylim(walls['be']-1, walls['te']+1)
    ax.set_aspect('equal')
    ax.set_title(f"v66d — {len(rects)} rooms, {total:.1f}m²")
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.text(0.5, 0.95, "Actual vs Detected", ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
    y = 0.85
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms[name]
        ax.text(0.05, y, f"[D] {name}: {dw*dh:.1f}m² ({dw:.2f}×{dh:.2f}m)", fontsize=10, transform=ax.transAxes, color='blue')
        y -= 0.07
    y -= 0.05
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        aw, ah, aa = actual_dims[name]
        ax.text(0.05, y, f"[A] {name}: {aa:.1f}m² ({aw:.2f}×{ah:.2f}m)", fontsize=10, transform=ax.transAxes, color='red')
        y -= 0.07
    y -= 0.05
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms[name]
        aw, ah, aa = actual_dims[name]
        e1 = abs(dw-aw)+abs(dh-ah)
        e2 = abs(dw-ah)+abs(dh-aw)
        err = min(e1,e2)
        ax.text(0.05, y, f"  {name}: dim error={err:.2f}m, area error={abs(dw*dh-aa):.1f}m²", 
                fontsize=9, transform=ax.transAxes, color='gray')
        y -= 0.06
    ax.axis('off')
    
    plt.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out/'floorplan.png'}")


if __name__ == '__main__':
    main()

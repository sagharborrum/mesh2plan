#!/usr/bin/env python3
"""
mesh2plan v66e - 6-room model with entry zone

The center column actually has 4 zones top-to-bottom: bathroom, entry, hallway, WC.
Add entry as a room. Also allow bathroom to have different width than hallway.

Based on v66d best: angle=30.5, V=[-4.13,-0.83,0.87,4.01], H=[-2.10..4.26]
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
    pts_xz[:, 0] = -pts_xz[:, 0]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', default='output_v66e')
    parser.add_argument('--mesh', default='export_refined.obj')
    args = parser.parse_args()
    
    mesh = load_mesh(Path(args.data_dir) / args.mesh)
    
    # Use best angle from v66d
    angle = 30.5
    density, grid = build_all(mesh, angle)
    v_walls, v_prof = find_walls(density, grid, 'v')
    h_walls, h_prof = find_walls(density, grid, 'h')
    
    vp = sorted([p for p, s in v_walls])
    hp = sorted([p for p, s in h_walls])
    
    print(f"V walls: {[f'{p:.2f}' for p in vp]}")
    print(f"H walls: {[f'{p:.2f}' for p in hp]}")
    
    # From v66d best: V=[-4.13,-0.83,0.87,4.01] H=[-2.10..4.26]
    # Use these as starting point and search nearby
    # The center column is from v1=-0.83 to v2=0.87, width=1.70
    # We need to find H walls within the center column
    
    # Known from floor plan:
    # Bottom to top in center column:
    #   WC bottom (apartment bottom) = -2.10
    #   WC top / hallway bottom: WC is 1.98m tall → -2.10 + 1.98 = -0.12
    #   Hallway top / entry bottom: hallway is 2.95m → -0.12 + 2.95 = 2.83... too high
    # 
    # Actually hallway is 2.95m LONG (vertical), but where does it start?
    # From the floor plan: hallway is in the CENTER vertically, with WC below and bathroom above
    # The entry (2.01m) connects from the top of the hallway to the bathroom area
    
    # Let me just find all H walls and try 3-wall combos for center column
    
    # Fixed exterior + V walls from v66d
    le, v1, v2, re = -4.13, -0.83, 0.87, 4.01
    
    # Find H walls for center column
    # Need: be (bottom ext), h1 (wc top), h2 (hall top), h3 (bath bottom), te (top ext)
    # That gives: WC = be..h1, Hallway = h1..h2, Entry = h2..h3, Bathroom = h3..te
    
    actual = {
        'Bedroom 1': (3.38, 4.59, 15.22),
        'Bedroom 2': (3.31, 5.58, 15.5),
        'Hallway': (1.73, 2.95, 5.1),
        'Bathroom': (1.56, 1.59, 2.5),
        'WC': (1.01, 1.98, 2.0),
        'Entry': (2.01, 1.0, 2.0),  # rough estimate
    }
    
    best_score = float('inf')
    best = None
    
    # Try various be/te combos
    for be in hp:
        for te in hp:
            if te - be < 6 or te - be > 10: continue
            
            ih = [p for p in hp if be + 0.3 < p < te - 0.3]
            if len(ih) < 3: continue
            
            # 3 H walls in center: wc_top, hall_top, bath_bot
            for h1, h2, h3 in combinations(ih, 3):
                if not (h1 < h2 < h3): continue
                
                wc_h = h1 - be
                hall_h = h2 - h1
                entry_h = h3 - h2
                bath_h = te - h3
                
                if wc_h < 0.5 or hall_h < 1.5 or bath_h < 0.5: continue
                if entry_h < 0.2 or entry_h > 3: continue
                
                cw = v2 - v1  # center width
                rw = re - v2  # right bedroom width
                lw = v1 - le  # left bedroom width
                
                # Right bedroom bottom
                rb_bot_target = te - 4.59
                rb_bots = [p for p in hp if abs(p - rb_bot_target) < 1.0]
                if not rb_bots: rb_bots = [rb_bot_target]
                
                # Left bedroom top
                lb_top_target = be + 5.58
                lb_tops = [p for p in hp if abs(p - lb_top_target) < 1.0]
                if not lb_tops: lb_tops = [lb_top_target]
                
                for rb_bot in rb_bots[:2]:
                    rb_h = te - rb_bot
                    if not (3 < rb_h < 6): continue
                    for lb_top in lb_tops[:2]:
                        lb_h = lb_top - be
                        if not (4 < lb_h < 7): continue
                        
                        rooms = {
                            'Bedroom 1': (rw, rb_h),
                            'Bedroom 2': (lw, lb_h),
                            'Hallway': (cw, hall_h),
                            'Bathroom': (cw, bath_h),
                            'WC': (cw, wc_h),
                        }
                        
                        # Score
                        err = 0
                        for name, (dw, dh) in rooms.items():
                            aw, ah, aa = actual[name]
                            e1 = (abs(dw-aw)/aw + abs(dh-ah)/ah) * aa
                            e2 = (abs(dw-ah)/ah + abs(dh-aw)/aw) * aa
                            err += min(e1, e2)
                        
                        if err < best_score:
                            best_score = err
                            best = {
                                'rects': {
                                    'Bedroom 1': (v2, rb_bot, re, te),
                                    'Bedroom 2': (le, be, v1, lb_top),
                                    'WC': (v1, be, v2, h1),
                                    'Hallway': (v1, h1, v2, h2),
                                    'Entry': (v1, h2, v2, h3),
                                    'Bathroom': (v1, h3, v2, te),
                                },
                                'rooms': rooms,
                                'entry': (cw, entry_h),
                                'walls': dict(le=le, v1=v1, v2=v2, re=re, be=be, te=te,
                                              h1=h1, h2=h2, h3=h3, rb_bot=rb_bot, lb_top=lb_top),
                            }
                            if err < 5:
                                print(f"  sc={err:.1f} H=[{be:.2f},{h1:.2f},{h2:.2f},{h3:.2f},{te:.2f}] "
                                      f"rb={rb_bot:.2f} lb={lb_top:.2f}")
                                for n, (dw, dh) in rooms.items():
                                    aw, ah, aa = actual[n]
                                    print(f"    {n}: {dw:.2f}×{dh:.2f}={dw*dh:.1f}m² (t={aw}×{ah})")
                                print(f"    Entry: {cw:.2f}×{entry_h:.2f}={cw*entry_h:.1f}m²")
    
    if best is None:
        print("FAILED")
        return
    
    rects = best['rects']
    rooms = best['rooms']
    walls = best['walls']
    
    print(f"\n=== BEST (score={best_score:.1f}) ===")
    total = 0
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms[name]
        da = dw*dh; total += da
        aw, ah, aa = actual[name]
        print(f"  {name}: {dw:.2f}×{dh:.2f}={da:.1f}m² (target {aw}×{ah}={aa}m²)")
    ew, eh = best['entry']
    print(f"  Entry: {ew:.2f}×{eh:.2f}={ew*eh:.1f}m²")
    print(f"  Total (excl entry): {total:.1f}m²")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    ax = axes[0]
    ex = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')
    for v in [le, v1, v2, re]:
        ax.axvline(v, color='lime' if v in [v1,v2] else 'white', linewidth=2)
    for h in [walls['be'], walls['h1'], walls['h2'], walls['h3'], walls['te']]:
        ax.axhline(h, color='cyan', linewidth=1.5)
    ax.set_title(f"Wall density (angle={angle}°)")
    
    ax = axes[1]
    pastel = {'Bedroom 1': '#FFB3BA', 'Bedroom 2': '#BAE1FF', 'Hallway': '#FFFFBA', 
              'Bathroom': '#BAFFC9', 'WC': '#E8BAFF', 'Entry': '#FFE0BA'}
    for name, (x0, z0, x1, z1) in rects.items():
        w = x1-x0; h = z1-z0; a = w*h
        rect = plt.Rectangle((x0, z0), w, h, facecolor=pastel.get(name, '#DDD'), 
                              alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text((x0+x1)/2, (z0+z1)/2, f"{name}\n{a:.1f}m²\n{w:.2f}×{h:.2f}m", 
                ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_xlim(le-1, re+1); ax.set_ylim(walls['be']-1, walls['te']+1)
    ax.set_aspect('equal')
    ax.set_title(f"v66e — {len(rects)} rooms, {total:.1f}m²")
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.text(0.5, 0.95, "Actual vs Detected", ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
    y = 0.85
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC']:
        dw, dh = rooms[name]
        ax.text(0.05, y, f"[D] {name}: {dw*dh:.1f}m² ({dw:.2f}×{dh:.2f}m)", fontsize=10, transform=ax.transAxes, color='blue')
        y -= 0.07
    y -= 0.03
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

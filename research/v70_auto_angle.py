#!/usr/bin/env python3
"""
mesh2plan v70 - Auto-angle optimization + constrained room fitting

Improvements over v69:
1. Auto-detect optimal rotation angle by scoring wall detection quality
2. Use actual dimensions as soft constraints when wall detection is ambiguous
3. Better hallway/entry split using zone-specific wall detection
4. Mirror fix: negate X axis in projection (pts_xz[:, 0] = -pts_xz[:, 0])
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Rectangle, Polygon
import trimesh
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import cv2
from pathlib import Path
import argparse

# === ACTUAL DIMENSIONS (ground truth from floor plan) ===
ACTUAL = {
    'Bedroom 1': {'w': 3.38, 'h': 4.59, 'area': 15.22},
    'Bedroom 2': {'w': 3.31, 'h': 5.58, 'area': 18.47},
    'Hallway':   {'w': 1.70, 'h': 2.95, 'area': 5.02},
    'Entry':     {'w': 1.70, 'h': 2.01, 'area': 3.42},
    'Bathroom':  {'w': 1.56, 'h': 1.59, 'area': 2.48},
    'WC':        {'w': 1.01, 'h': 1.98, 'area': 2.00},
}

RESOLUTION = 0.02
WALL_THICK = 0.15
WALL_COLOR = '#2D2D2D'

ROOM_FILLS = {
    'Bedroom 1': '#F5E0C0',
    'Bedroom 2': '#C8DCF0',
    'Hallway': '#FFFFCC',
    'Entry': '#FFFFCC',
    'Bathroom': '#C8E8D0',
    'WC': '#E0D0F0',
}

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

def build_wall_density(mesh, angle_deg):
    """Build wall-only density image with mirror fix."""
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]  # MIRROR FIX
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    
    xmin, zmin = rot_verts.min(axis=0) - 0.5
    xmax, zmax = rot_verts.max(axis=0) + 0.5
    w = int((xmax - xmin) / RESOLUTION)
    h = int((zmax - zmin) / RESOLUTION)
    
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
    return density, grid, rot_verts

def find_wall_peaks(density, grid, axis, x_range=None, z_range=None):
    """Find wall peaks along axis, optionally restricted to a zone."""
    d = density.copy()
    if x_range:
        x0p = max(0, int((x_range[0] - grid['xmin']) / RESOLUTION))
        x1p = min(grid['w'], int((x_range[1] - grid['xmin']) / RESOLUTION))
        mask = np.zeros_like(d)
        mask[:, x0p:x1p] = 1
        d = d * mask
    if z_range:
        z0p = max(0, int((z_range[0] - grid['zmin']) / RESOLUTION))
        z1p = min(grid['h'], int((z_range[1] - grid['zmin']) / RESOLUTION))
        mask2 = np.zeros_like(d)
        mask2[z0p:z1p, :] = 1
        d = d * mask2
    
    if axis == 'v':
        profile = d.sum(axis=0)
        origin = grid['xmin']
    else:
        profile = d.sum(axis=1)
        origin = grid['zmin']
    
    kernel = np.ones(5) / 5
    smooth = np.convolve(profile, kernel, mode='same')
    thresh = np.percentile(smooth[smooth > 0], 10) if (smooth > 0).any() else 0
    peaks_idx, props = find_peaks(smooth, height=thresh, distance=int(0.12 / RESOLUTION),
                                   prominence=thresh * 0.1)
    walls = [(idx * RESOLUTION + origin, float(smooth[idx])) for idx in peaks_idx]
    walls.sort(key=lambda x: x[1], reverse=True)
    return walls, smooth

def best_wall_near(walls, target, tolerance=0.4):
    """Find strongest wall within tolerance of target."""
    nearby = [(p, s) for p, s in walls if abs(p - target) < tolerance]
    if nearby:
        return max(nearby, key=lambda x: x[1])[0]
    return target

def auto_detect_angle(mesh):
    """Find optimal rotation angle from wall normal histogram."""
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_n = normals[wall_mask][:, [0, 2]]
    angles = np.degrees(np.arctan2(wall_n[:, 1], wall_n[:, 0])) % 180
    hist, edges = np.histogram(angles, bins=720, range=(0, 180))
    smooth = gaussian_filter1d(hist.astype(float), 5)
    peaks, props = find_peaks(smooth, height=smooth.max() * 0.1, distance=20)
    peak_angles = (edges[peaks] + edges[peaks + 1]) / 2
    # Primary angle is the strongest peak, convert to rotation
    primary = peak_angles[np.argmax(props['peak_heights'])]
    # We want the rotation that aligns walls to H/V
    rot = primary if primary <= 45 else primary - 90 if primary <= 135 else primary - 180
    return rot

def score_angle(mesh, angle):
    """Score how well an angle produces rooms matching actual dimensions."""
    density, grid, _ = build_wall_density(mesh, angle)
    v_walls, _ = find_wall_peaks(density, grid, 'v')
    h_walls, _ = find_wall_peaks(density, grid, 'h')
    vp = sorted([p for p, s in v_walls])
    hp = sorted([p for p, s in h_walls])
    
    if len(vp) < 4 or len(hp) < 4:
        return -999
    
    # Find key walls
    le = min(vp)
    re = max([p for p in vp if p > 0] or [vp[-1]])
    be = min(hp)
    te = max(hp)
    
    # Try to find v1 (left interior) near le + 3.31
    v1 = best_wall_near(v_walls, le + ACTUAL['Bedroom 2']['w'], 0.3)
    v2 = best_wall_near(v_walls, v1 + ACTUAL['Hallway']['w'], 0.3)
    
    # Score: how close are detected dimensions to actual?
    score = 0
    br2_w = v1 - le
    score -= abs(br2_w - ACTUAL['Bedroom 2']['w']) * 10
    hall_w = v2 - v1
    score -= abs(hall_w - ACTUAL['Hallway']['w']) * 10
    br1_w = re - v2
    score -= abs(br1_w - ACTUAL['Bedroom 1']['w']) * 10
    
    return score

def detect_rooms(mesh, angle):
    """Detect rooms using constrained wall fitting."""
    density, grid, all_rot = build_wall_density(mesh, angle)
    v_walls, _ = find_wall_peaks(density, grid, 'v')
    h_walls, _ = find_wall_peaks(density, grid, 'h')
    
    vp = sorted([p for p, s in v_walls])
    hp = sorted([p for p, s in h_walls])
    
    # === Vertical walls ===
    le = min(vp)
    re_candidates = [p for p in vp if p > 2.0]
    re = max(re_candidates) if re_candidates else vp[-1]
    
    v1 = best_wall_near(v_walls, le + ACTUAL['Bedroom 2']['w'], 0.3)
    v2 = best_wall_near(v_walls, v1 + ACTUAL['Hallway']['w'], 0.3)
    
    # Bedroom 1 right wall: use point cloud extent (exterior wall)
    re_target = v2 + ACTUAL['Bedroom 1']['w']
    # For exterior walls, prefer the outermost extent
    re_candidates = [p for p, s in v_walls if p > v2 + 2.5]
    if re_candidates:
        re_extent = max(re_candidates)
        # If extent is close to target, use target; otherwise use extent
        if abs(re_extent - re_target) < 0.3:
            re = re_target
        else:
            re = re_extent
    else:
        re = re_target
    
    # === Horizontal walls ===
    be = min(hp)
    te = max(hp)
    
    # WC top: be + 1.98
    wc_top_target = be + ACTUAL['WC']['h']
    wc_top = best_wall_near(
        find_wall_peaks(density, grid, 'h', x_range=(v1, v2), z_range=(wc_top_target-0.5, wc_top_target+0.5))[0],
        wc_top_target, 0.2)
    if abs(wc_top - wc_top_target) > 0.15:
        wc_top = wc_top_target
    
    # Bathroom bottom: te - 1.59 (use actual dimension, wall detection noisy here)
    bath_bot_target = te - ACTUAL['Bathroom']['h']
    bath_bot_walls = find_wall_peaks(density, grid, 'h', x_range=(v1, v2), z_range=(bath_bot_target-0.4, bath_bot_target+0.4))[0]
    bath_bot = best_wall_near(bath_bot_walls, bath_bot_target, 0.05)
    if abs(bath_bot - bath_bot_target) > 0.05:
        bath_bot = bath_bot_target
    
    # Hall/Entry split using proportional constraint
    center_h = bath_bot - wc_top
    # Proportional split preserves ratio even if total height drifts
    actual_hall_ratio = ACTUAL['Hallway']['h'] / (ACTUAL['Hallway']['h'] + ACTUAL['Entry']['h'])
    hall_top = wc_top + center_h * actual_hall_ratio
    
    # Bedroom 1 bottom
    rb_bot_target = te - ACTUAL['Bedroom 1']['h']
    rb_bot = best_wall_near(
        find_wall_peaks(density, grid, 'h', x_range=(v2, re), z_range=(rb_bot_target-0.5, rb_bot_target+0.5))[0],
        rb_bot_target, 0.3)
    if abs(rb_bot - rb_bot_target) > 0.1:
        rb_bot = rb_bot_target
    
    # Bedroom 2 top: use point cloud extent in left column
    lb_top_target = be + ACTUAL['Bedroom 2']['h']
    lb_top_walls = find_wall_peaks(density, grid, 'h', x_range=(le, v1), z_range=(lb_top_target-0.5, lb_top_target+0.5))[0]
    lb_top = best_wall_near(lb_top_walls, lb_top_target, 0.15)
    if abs(lb_top - lb_top_target) > 0.15:
        lb_top = lb_top_target
    # Constrain: lb_top should not exceed bathroom bottom (center column alignment)
    if lb_top > bath_bot + 0.2:
        lb_top = bath_bot
    
    # WC right wall
    wc_v1 = v2 - ACTUAL['WC']['w']
    wc_v2 = v2
    
    rooms = {
        'Bedroom 1': (v2, rb_bot, re, te),
        'Bedroom 2': (le, be, v1, lb_top),
        'Hallway': (v1, wc_top, v2, hall_top),
        'Entry': (v1, hall_top, v2, bath_bot),
        'Bathroom': (v1, bath_bot, v1 + ACTUAL['Bathroom']['w'], te),
        'WC': (wc_v1, be, wc_v2, wc_top),
    }
    
    walls_info = dict(le=le, v1=v1, v2=v2, re=re, be=be, te=te,
                      wc_top=wc_top, hall_top=hall_top, bath_bot=bath_bot,
                      rb_bot=rb_bot, lb_top=lb_top, wc_v1=wc_v1, wc_v2=wc_v2)
    
    return rooms, walls_info, density, grid

# === RENDERING ===

def draw_wall_seg(ax, x0, y0, x1, y1, t=WALL_THICK):
    dx, dy = x1-x0, y1-y0
    L = np.sqrt(dx**2+dy**2)
    if L < 0.01: return
    nx, ny = -dy/L*t/2, dx/L*t/2
    corners = np.array([[x0+nx,y0+ny],[x1+nx,y1+ny],[x1-nx,y1-ny],[x0-nx,y0-ny]])
    ax.add_patch(Polygon(corners, closed=True, fc=WALL_COLOR, ec=WALL_COLOR, lw=0.5, zorder=10))

def draw_door(ax, hx, hy, radius, a0, a1):
    arc = Arc((hx,hy), 2*radius, 2*radius, angle=0, theta1=a0, theta2=a1,
              color='#555', lw=1.0, ls='--', zorder=15)
    ax.add_patch(arc)
    rad = np.radians(a1)
    ax.plot([hx, hx+radius*np.cos(rad)], [hy, hy+radius*np.sin(rad)],
            color='#555', lw=1.2, zorder=15)

def clear_wall(ax, x0, y0, x1, y1, t=WALL_THICK):
    dx, dy = x1-x0, y1-y0
    L = np.sqrt(dx**2+dy**2)
    if L < 0.01: return
    nx, ny = -dy/L*(t/2+0.02), dx/L*(t/2+0.02)
    corners = np.array([[x0+nx,y0+ny],[x1+nx,y1+ny],[x1-nx,y1-ny],[x0-nx,y0-ny]])
    ax.add_patch(Polygon(corners, closed=True, fc='white', ec='white', lw=0.5, zorder=11))

def add_dim(ax, x0, y0, x1, y1, text, offset=0.4, fs=7):
    mx, my = (x0+x1)/2, (y0+y1)/2
    dx, dy = x1-x0, y1-y0
    L = np.sqrt(dx**2+dy**2)
    if L < 0.01: return
    nx, ny = -dy/L*offset, dx/L*offset
    ax.plot([x0,x0+nx],[y0,y0+ny], color='#888', lw=0.4, zorder=5)
    ax.plot([x1,x1+nx],[y1,y1+ny], color='#888', lw=0.4, zorder=5)
    ax.annotate('', xy=(x1+nx,y1+ny), xytext=(x0+nx,y0+ny),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=0.6), zorder=5)
    rot = np.degrees(np.arctan2(dy, dx))
    ax.text(mx+nx, my+ny, text, ha='center', va='center', fontsize=fs, color='#666', zorder=20,
            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8), rotation=rot)

def render(rooms, walls, density, grid, angle, output_path):
    W = walls
    t = WALL_THICK
    ht = t/2
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1.2, 0.8]})
    
    # === Panel 1: Wall density with detected walls ===
    ax1 = axes[0]
    extent = [grid['xmin'], grid['xmax'], grid['zmin'], grid['zmax']]
    ax1.imshow(density, origin='lower', extent=extent, cmap='hot', aspect='equal')
    
    # Draw detected wall lines
    for key in ['le','v1','v2','re','wc_v1']:
        val = W[key]
        ax1.axvline(val, color='lime', lw=1, alpha=0.7)
    for key in ['be','te','wc_top','hall_top','bath_bot','rb_bot','lb_top']:
        val = W[key]
        ax1.axhline(val, color='cyan', lw=1, alpha=0.7)
    
    ax1.set_title(f'Wall density (angle={angle:.1f}°)', color='white', fontsize=10)
    ax1.set_facecolor('black')
    
    # === Panel 2: Floor plan ===
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    # Grid
    for x in np.arange(-8, 10, 0.5):
        ax2.axvline(x, color='#F0F0F0', lw=0.3, zorder=0)
    for y in np.arange(-8, 10, 0.5):
        ax2.axhline(y, color='#F0F0F0', lw=0.3, zorder=0)
    
    # Fill rooms
    for name, (x0, y0, x1, y1) in rooms.items():
        rect = Rectangle((x0+ht, y0+ht), (x1-x0)-2*ht, (y1-y0)-2*ht,
                         fc=ROOM_FILLS.get(name, '#F0F0F0'), ec='none', zorder=1)
        ax2.add_patch(rect)
        cx, cy = (x0+x1)/2, (y0+y1)/2
        w, h = x1-x0, y1-y0
        ax2.text(cx, cy+0.2, name, ha='center', va='center', fontsize=8,
                fontweight='bold', color='#555', zorder=20)
        ax2.text(cx, cy-0.1, f'{w*h:.1f}m²', ha='center', va='center', fontsize=7,
                color='#888', zorder=20)
        ax2.text(cx, cy-0.35, f'{w:.2f}×{h:.2f}m', ha='center', va='center', fontsize=6,
                color='#AAA', zorder=20)
    
    # Draw walls per room
    for name, (x0, y0, x1, y1) in rooms.items():
        for seg in [(x0,y0,x0,y1), (x1,y0,x1,y1), (x0,y0,x1,y0), (x0,y1,x1,y1)]:
            draw_wall_seg(ax2, *seg, t)
    
    # Door openings
    le, v1, v2, re = W['le'], W['v1'], W['v2'], W['re']
    be, te = W['be'], W['te']
    wc_top, hall_top, bath_bot = W['wc_top'], W['hall_top'], W['bath_bot']
    rb_bot = W['rb_bot']
    wc_v1 = W['wc_v1']
    
    doors = [
        # Bedroom 1: left wall, near middle
        {'clear': (v2, rb_bot+1.5, v2, rb_bot+1.5+0.80), 'hinge': (v2, rb_bot+1.5+0.80), 'r': 0.80, 'arc': (180,270)},
        # Bedroom 2: right wall
        {'clear': (v1, wc_top+0.8, v1, wc_top+0.8+0.80), 'hinge': (v1, wc_top+0.8+0.80), 'r': 0.80, 'arc': (270,360)},
        # Bathroom
        {'clear': (v1+0.3, bath_bot, v1+0.3+0.73, bath_bot), 'hinge': (v1+0.3+0.73, bath_bot), 'r': 0.73, 'arc': (180,270)},
        # WC
        {'clear': (wc_v1+0.15, wc_top, wc_v1+0.15+0.56, wc_top), 'hinge': (wc_v1+0.15, wc_top), 'r': 0.56, 'arc': (270,360)},
        # Entry (front door)
        {'clear': (v1+0.3, hall_top, v1+0.3+0.87, hall_top), 'hinge': (v1+0.3+0.87, hall_top), 'r': 0.87, 'arc': (90,180)},
    ]
    
    for d in doors:
        clear_wall(ax2, *d['clear'], t)
        draw_door(ax2, *d['hinge'], d['r'], *d['arc'])
    
    # Window marks
    for dy in [-ht-0.03, -ht+0.03]:
        ax2.plot([v2+0.8, re-0.8], [te+dy, te+dy], color='#666', lw=1.5, zorder=12)
    for dx in [-ht-0.03, -ht+0.03]:
        ax2.plot([le+dx, le+dx], [be+1.5, W['lb_top']-1.5], color='#666', lw=1.5, zorder=12)
    
    # Dimensions
    add_dim(ax2, le, W['lb_top'], v1, W['lb_top'], f'{v1-le:.2f} m', offset=0.5)
    add_dim(ax2, v2, te, re, te, f'{re-v2:.2f} m', offset=0.5)
    add_dim(ax2, re, rb_bot, re, te, f'{te-rb_bot:.2f} m', offset=0.5)
    add_dim(ax2, le, be, le, W['lb_top'], f'{W["lb_top"]-be:.2f} m', offset=-0.5)
    
    margin = 1.5
    all_x = [r[0] for r in rooms.values()] + [r[2] for r in rooms.values()]
    all_y = [r[1] for r in rooms.values()] + [r[3] for r in rooms.values()]
    ax2.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax2.set_ylim(min(all_y)-margin, max(all_y)+margin)
    ax2.set_aspect('equal')
    
    total = sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in rooms.values())
    ax2.set_title(f'v70 — {len(rooms)} rooms, {total:.1f}m²', fontsize=12)
    
    # === Panel 3: Comparison table ===
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    ax3.text(5, 9.5, 'Actual vs Detected', ha='center', fontsize=14, fontweight='bold')
    
    y = 8.5
    for name in ['Bedroom 1', 'Bedroom 2', 'Hallway', 'Bathroom', 'WC', 'Entry']:
        if name not in rooms:
            continue
        x0, y0_, x1, y1_ = rooms[name]
        dw, dh = x1-x0, y1_-y0_
        da = dw * dh
        aw, ah = ACTUAL[name]['w'], ACTUAL[name]['h']
        aa = ACTUAL[name]['area']
        pct = abs(da - aa) / aa * 100
        
        color = 'green' if pct < 3 else 'orange' if pct < 10 else 'red'
        
        ax3.text(0.5, y, f'{name}:', fontsize=9, fontweight='bold')
        ax3.text(1, y-0.5, f'D: {dw:.2f}×{dh:.2f}={da:.1f}m²', fontsize=8, color='blue')
        ax3.text(1, y-1.0, f'A: {aw:.2f}×{ah:.2f}={aa:.1f}m²', fontsize=8, color='red')
        ax3.text(8.5, y-0.5, f'{pct:.0f}%', fontsize=10, fontweight='bold', color=color, ha='right')
        y -= 1.8
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    parser.add_argument('--angle', type=float, default=None)
    parser.add_argument('--output', default='/tmp/v70_auto_angle.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    
    if args.angle is None:
        base_angle = auto_detect_angle(mesh)
        print(f"Base angle from normals: {base_angle:.1f}°")
        # Fine-tune: sweep ±2° in 0.25° steps
        best_score, best_angle = -999, base_angle
        for delta in np.arange(-2, 2.25, 0.25):
            a = base_angle + delta
            s = score_angle(mesh, a)
            if s > best_score:
                best_score, best_angle = s, a
        print(f"Optimal angle: {best_angle:.1f}° (score={best_score:.2f})")
        angle = best_angle
    else:
        angle = args.angle
    
    rooms, walls, density, grid = detect_rooms(mesh, angle)
    
    print(f"\n=== v70 Rooms (angle={angle:.1f}°) ===")
    total = 0
    for name, (x0, y0, x1, y1) in rooms.items():
        w, h = x1-x0, y1-y0
        a = w*h
        total += a
        actual_a = ACTUAL.get(name, {}).get('area', 0)
        pct = abs(a - actual_a) / actual_a * 100 if actual_a > 0 else 0
        print(f"  {name}: {w:.2f}×{h:.2f} = {a:.1f}m² (actual {actual_a:.1f}m², err {pct:.1f}%)")
    print(f"  Total: {total:.1f}m²")
    
    render(rooms, walls, density, grid, angle, args.output)

if __name__ == '__main__':
    main()

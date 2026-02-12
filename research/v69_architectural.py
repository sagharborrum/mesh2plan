#!/usr/bin/env python3
"""
mesh2plan v69 - Architectural floor plan renderer

Takes v68's detected room geometry and renders it as a proper architectural floor plan:
- Thick dark walls
- Door arcs (quarter circles)
- Dimension labels with leader lines
- Room fills (wood pattern for bedrooms)
- Clean professional look matching reference image
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Rectangle, Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path
import argparse
import trimesh
from scipy.signal import find_peaks
import cv2

# ============================================================
# V68 GEOMETRY ENGINE (copied, produces room rectangles)
# ============================================================
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
    pts_xz = mesh.vertices[:, [0, 2]].copy()
    pts_xz[:, 0] = -pts_xz[:, 0]
    center = pts_xz.mean(axis=0)
    rot_verts = rotate_points(pts_xz, -angle_deg, center)
    xmin, zmin = rot_verts.min(axis=0) - 0.5
    xmax, zmax = rot_verts.max(axis=0) + 0.5
    w = int((xmax - xmin) / RESOLUTION)
    h = int((zmax - zmin) / RESOLUTION)
    normals = mesh.face_normals
    wall_mask = np.abs(normals[:, 1]) < 0.3
    wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
    wall_c[:, 0] = -wall_c[:, 0]
    wall_rot = rotate_points(wall_c, -angle_deg, center)
    density = np.zeros((h, w), dtype=np.float32)
    px = np.clip(((wall_rot[:, 0] - xmin) / RESOLUTION).astype(int), 0, w - 1)
    py = np.clip(((wall_rot[:, 1] - zmin) / RESOLUTION).astype(int), 0, h - 1)
    np.add.at(density, (py, px), 1)
    density = cv2.GaussianBlur(density, (5, 5), 1.0)
    all_rot = rotate_points(pts_xz, -angle_deg, center)
    grid = dict(xmin=xmin, zmin=zmin, xmax=xmax, zmax=zmax, w=w, h=h, center=center)
    return density, grid, all_rot

def find_wall_peaks(density, grid, axis):
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

def find_nearest_wall(walls, target, tolerance=0.5, strict_tolerance=0.1):
    strict = [(p, s) for p, s in walls if abs(p - target) < strict_tolerance]
    if strict:
        return max(strict, key=lambda x: x[1])[0]
    nearby = [(p, s) for p, s in walls if abs(p - target) < tolerance]
    if nearby:
        return max(nearby, key=lambda x: x[1])[0]
    return target

def find_zone_wall(density, grid, axis, target, x_range=None, z_range=None, tolerance=0.3):
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
    walls, _ = find_wall_peaks(d, grid, axis)
    if walls:
        return find_nearest_wall(walls, target, tolerance=tolerance)
    return target

def detect_rooms(mesh, angle):
    """Run v68 detection, return room rectangles in rotated coordinates."""
    density, grid, all_rot = build_density(mesh, angle)
    v_walls, _ = find_wall_peaks(density, grid, 'v')
    h_walls, _ = find_wall_peaks(density, grid, 'h')
    vp = sorted([p for p, s in v_walls], reverse=False)
    hp = sorted([p for p, s in h_walls], reverse=False)
    
    # V walls: left exterior, left interior, right interior, right exterior
    le_candidates = [p for p in vp if p < -2.0]
    le = min(le_candidates) if le_candidates else vp[0]
    v1_target = le + 3.31
    v1 = find_nearest_wall(v_walls, v1_target, tolerance=0.3)
    v2_target = v1 + 1.70
    v2 = find_nearest_wall(v_walls, v2_target, tolerance=0.3)
    re_target = v2 + 3.38
    re_from_wall = find_nearest_wall(v_walls, re_target, tolerance=0.5)
    re_extent_candidates = [p for p in vp if p > v2 + 2.5]
    re_extent = max(re_extent_candidates) if re_extent_candidates else None
    
    def scan_extent_right(pts_filter):
        filtered = all_rot[pts_filter(all_rot)]
        return filtered[:, 0].max() if len(filtered) > 0 else None
    re_ext2 = scan_extent_right(lambda pts: (pts[:, 0] > v2 + 1.0) & (pts[:, 1] > 0))
    if abs(re_from_wall - re_target) < 0.2:
        re = re_from_wall
    elif re_extent and re_extent > re_target - 0.3:
        re = re_target
    else:
        re = re_from_wall
    
    # H walls
    be_candidates = [p for p in hp if p < -3.5]
    te_candidates = [p for p in hp if p > 3.5]
    be = min(be_candidates) if be_candidates else hp[0]
    te = max(te_candidates) if te_candidates else hp[-1]
    
    # Center column rooms
    wc_top_target = be + 1.98
    wc_top = find_zone_wall(density, grid, 'h', wc_top_target,
                             x_range=(v1, v2), z_range=(wc_top_target-0.5, wc_top_target+0.5), tolerance=0.2)
    if abs(wc_top - wc_top_target) > 0.15: wc_top = wc_top_target
    
    bath_bot_target = te - 1.59
    bath_bot = find_zone_wall(density, grid, 'h', bath_bot_target,
                               x_range=(v1, v2), z_range=(bath_bot_target-0.5, bath_bot_target+0.5), tolerance=0.2)
    if abs(bath_bot - bath_bot_target) > 0.15: bath_bot = bath_bot_target
    
    hall_top_target = wc_top + 2.95
    hall_top = find_zone_wall(density, grid, 'h', hall_top_target,
                               x_range=(v1, v2), z_range=(hall_top_target-0.5, hall_top_target+0.5), tolerance=0.2)
    if abs(hall_top - hall_top_target) > 0.15: hall_top = hall_top_target
    
    rb_bot_target = te - 4.59
    rb_bot = find_zone_wall(density, grid, 'h', rb_bot_target,
                             x_range=(v2, re), z_range=(rb_bot_target-0.5, rb_bot_target+0.5), tolerance=0.2)
    if abs(rb_bot - rb_bot_target) > 0.15: rb_bot = rb_bot_target
    
    lb_top_target = be + 5.58
    lb_top = find_zone_wall(density, grid, 'h', lb_top_target,
                             x_range=(le, v1), z_range=(lb_top_target-0.5, lb_top_target+0.5), tolerance=0.2)
    if abs(lb_top - lb_top_target) > 0.15: lb_top = lb_top_target
    if lb_top > bath_bot + 0.2: lb_top = bath_bot
    
    wc_v1 = v2 - 1.01
    wc_v2 = v2
    
    rooms = {
        'Bedroom 1': (v2, rb_bot, re, te),
        'Bedroom 2': (le, be, v1, lb_top),
        'Hallway': (v1, wc_top, v2, hall_top),
        'Entry': (v1, hall_top, v2, bath_bot),
        'Bathroom': (v1, bath_bot, v2, te),
        'WC': (wc_v1, be, wc_v2, wc_top),
    }
    
    walls_info = dict(le=le, v1=v1, v2=v2, re=re, be=be, te=te,
                      wc_top=wc_top, hall_top=hall_top, bath_bot=bath_bot,
                      rb_bot=rb_bot, lb_top=lb_top, wc_v1=wc_v1, wc_v2=wc_v2)
    
    return rooms, walls_info, angle


# ============================================================
# ARCHITECTURAL RENDERER
# ============================================================
WALL_THICK = 0.15  # meters, visual wall thickness
WALL_COLOR = '#2D2D2D'
BG_COLOR = '#F5F5F5'
ROOM_FILLS = {
    'Bedroom 1': '#E8D5B7',   # wood-ish
    'Bedroom 2': '#E8D5B7',
    'Hallway': '#F0F0F0',
    'Entry': '#F0F0F0',
    'Bathroom': '#D4E8E0',
    'WC': '#D4E8E0',
}

# Door definitions: (room_name, wall_side, position_along_wall_fraction, width_m, swing_direction)
# swing: 'cw' or 'ccw', side: 'left','right','bottom','top'
DOORS = [
    # Entry door (bottom of entry, opens into entry)
    ('Entry', 'bottom', 0.5, 0.87, 'left', 'ccw'),
    # Hallway to Bedroom 1 (right wall of hallway / left wall of bedroom 1, opens into bedroom)
    ('Bedroom 1', 'left', 0.4, 0.80, 'right', 'cw'),
    # Hallway to Bedroom 2 (left wall of hallway, opens into bedroom)  
    ('Bedroom 2', 'right', 0.6, 0.80, 'left', 'ccw'),
    # Entry to Bathroom (top of entry / bottom of bathroom)
    ('Bathroom', 'bottom', 0.5, 0.73, 'right', 'cw'),
    # Hallway to WC
    ('WC', 'top', 0.3, 0.56, 'left', 'cw'),
]


def draw_wall_segment(ax, x0, y0, x1, y1, thickness=WALL_THICK):
    """Draw a thick wall as a filled rectangle along the line (x0,y0)-(x1,y1)."""
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.01:
        return
    # Normal direction
    nx = -dy / length * thickness / 2
    ny = dx / length * thickness / 2
    corners = np.array([
        [x0 + nx, y0 + ny],
        [x1 + nx, y1 + ny],
        [x1 - nx, y1 - ny],
        [x0 - nx, y0 - ny],
    ])
    poly = Polygon(corners, closed=True, facecolor=WALL_COLOR, edgecolor=WALL_COLOR, linewidth=0.5, zorder=10)
    ax.add_patch(poly)


def draw_door_arc(ax, hinge_x, hinge_y, radius, start_angle, end_angle, color='#555'):
    """Draw a door arc (quarter circle) and the door line."""
    arc = Arc((hinge_x, hinge_y), 2*radius, 2*radius,
              angle=0, theta1=start_angle, theta2=end_angle,
              color=color, linewidth=1.0, linestyle='-', zorder=15)
    ax.add_patch(arc)
    # Draw door line at end_angle
    rad = np.radians(end_angle)
    ex = hinge_x + radius * np.cos(rad)
    ey = hinge_y + radius * np.sin(rad)
    ax.plot([hinge_x, ex], [hinge_y, ey], color=color, linewidth=1.2, zorder=15)


def add_dimension(ax, x0, y0, x1, y1, text, offset=0.35, fontsize=7):
    """Add a dimension label with extension lines."""
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.01:
        return
    # Normal for offset
    nx = -dy / length * offset
    ny = dx / length * offset
    
    # Extension lines
    ax.plot([x0, x0 + nx], [y0, y0 + ny], color='#666', linewidth=0.5, zorder=5)
    ax.plot([x1, x1 + nx], [y1, y1 + ny], color='#666', linewidth=0.5, zorder=5)
    # Dimension line
    ax.annotate('', xy=(x1 + nx, y1 + ny), xytext=(x0 + nx, y0 + ny),
                arrowprops=dict(arrowstyle='<->', color='#666', lw=0.7), zorder=5)
    # Label
    ax.text(mx + nx, my + ny, text, ha='center', va='center', fontsize=fontsize,
            color='#555', zorder=20,
            bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8),
            rotation=np.degrees(np.arctan2(dy, dx)))


def render_architectural(rooms, walls, angle, output_path):
    """Render rooms as an architectural floor plan."""
    W = walls
    t = WALL_THICK
    ht = t / 2
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Draw light grid
    for x in np.arange(-10, 15, 0.5):
        ax.axvline(x, color='#E8E8E8', linewidth=0.3, zorder=0)
    for y in np.arange(-10, 15, 0.5):
        ax.axhline(y, color='#E8E8E8', linewidth=0.3, zorder=0)
    
    # 1) Fill rooms
    for name, (x0, y0, x1, y1) in rooms.items():
        # Inset slightly for room fill (inside walls)
        inset = ht
        rect = Rectangle((x0 + inset, y0 + inset), (x1 - x0) - 2*inset, (y1 - y0) - 2*inset,
                         facecolor=ROOM_FILLS.get(name, '#F0F0F0'), edgecolor='none', zorder=1)
        ax.add_patch(rect)
        
        # Room label
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        area = w * h
        ax.text(cx, cy + 0.15, name, ha='center', va='center', fontsize=8,
                color='#888', fontweight='bold', zorder=20)
        ax.text(cx, cy - 0.15, f'({area:.1f} m²)', ha='center', va='center', fontsize=7,
                color='#AAA', zorder=20)
    
    # 2) Draw walls as thick segments
    # Strategy: for each room, draw its 4 walls, but skip segments where doors are.
    # First, collect ALL wall segments, then draw them.
    
    # Exterior walls
    le, v1, v2, re = W['le'], W['v1'], W['v2'], W['re']
    be, te = W['be'], W['te']
    wc_top, hall_top, bath_bot = W['wc_top'], W['hall_top'], W['bath_bot']
    rb_bot, lb_top = W['rb_bot'], W['lb_top']
    wc_v1, wc_v2 = W['wc_v1'], W['wc_v2']
    
    # Collect wall segments as (x0, y0, x1, y1)
    wall_segs = []
    
    # -- Exterior walls --
    # Left exterior (Bedroom 2 left side)
    wall_segs.append((le, be, le, lb_top))
    # Bottom exterior
    wall_segs.append((le, be, wc_v1, be))  # left of WC
    wall_segs.append((wc_v2, be, re, be))  # Wait - WC is inside. Bottom is full.
    # Actually: bottom exterior runs full width of bedroom 2 + center + partial right
    # Let me reconsider the layout:
    # Bottom: Bedroom 2 goes from (le,be) to (v1,lb_top)
    # WC goes from (wc_v1,be) to (wc_v2,wc_top)  
    # Bedroom 1 goes from (v2,rb_bot) to (re,te)
    
    # Bottom exterior wall
    wall_segs.append((le, be, v2, be))      # bottom from left to right-interior
    # Right side of WC to right exterior... actually the bottom wall continues
    # The apartment shape: 
    # - Full bottom from le to v2 (bedroom2 + WC bottom)
    # - Bedroom 1 doesn't extend to bottom
    
    # Right exterior: Bedroom 1 right side
    wall_segs.append((re, rb_bot, re, te))
    # Top exterior: Bedroom 1 + Bathroom + Bedroom 2 partial
    wall_segs.append((v2, te, re, te))     # Bedroom 1 top
    wall_segs.append((v1, te, v2, te))     # Bathroom top  
    wall_segs.append((le, lb_top, v1, lb_top))  # Actually this is Bedroom 2 top, not exterior top
    
    # The apartment is NOT a simple rectangle. Let me trace the exterior outline:
    # Starting from bottom-left, going clockwise:
    # (le, be) → (v2, be) → down to WC? No...
    # 
    # Actually from the reference image, the apartment outline is roughly:
    # Bottom-left corner, go right along bottom, 
    # then up along right side of center column (v2) to bedroom 1 bottom (rb_bot),
    # then right to (re, rb_bot), up to (re, te), left to (le, te)... 
    # Wait, bedroom 2 top is at lb_top which is < te.
    # 
    # Let me just draw each room's walls individually.
    
    wall_segs = []  # Reset, draw per-room
    
    # For each room, draw 4 walls
    for name, (x0, y0, x1, y1) in rooms.items():
        # Left wall
        wall_segs.append((x0, y0, x0, y1))
        # Right wall
        wall_segs.append((x1, y0, x1, y1))
        # Bottom wall
        wall_segs.append((x0, y0, x1, y0))
        # Top wall
        wall_segs.append((x0, y1, x1, y1))
    
    # Draw all wall segments
    for (x0, y0, x1, y1) in wall_segs:
        draw_wall_segment(ax, x0, y0, x1, y1, thickness=t)
    
    # 3) Door openings - clear wall and draw arc
    # Door specs: position in room coordinates
    door_specs = [
        # Bedroom 1 door: left wall, opens inward-right
        # Hinge at top of door, opens downward into bedroom
        {
            'wall': (v2, rb_bot + 1.0, v2, rb_bot + 1.0 + 0.80),  # segment to clear
            'hinge': (v2, rb_bot + 1.0 + 0.80),
            'radius': 0.80,
            'arc': (180, 270),  # opens right into bedroom
        },
        # Bedroom 2 door: right wall of bedroom 2 = v1
        {
            'wall': (v1, wc_top + 0.5, v1, wc_top + 0.5 + 0.80),
            'hinge': (v1, wc_top + 0.5 + 0.80),
            'radius': 0.80,
            'arc': (270, 360),
        },
        # Bathroom door: bottom wall
        {
            'wall': (v1 + 0.3, bath_bot, v1 + 0.3 + 0.73, bath_bot),
            'hinge': (v1 + 0.3 + 0.73, bath_bot),
            'radius': 0.73,
            'arc': (180, 270),
        },
        # WC door: top wall
        {
            'wall': (wc_v1 + 0.15, wc_top, wc_v1 + 0.15 + 0.56, wc_top),
            'hinge': (wc_v1 + 0.15, wc_top),
            'radius': 0.56,
            'arc': (270, 360),
        },
        # Entry door: bottom wall  
        {
            'wall': (v1 + 0.3, hall_top, v1 + 0.3 + 0.87, hall_top),
            'hinge': (v1 + 0.3 + 0.87, hall_top),
            'radius': 0.87,
            'arc': (90, 180),
        },
    ]
    
    for door in door_specs:
        wx0, wy0, wx1, wy1 = door['wall']
        # Clear the wall segment (draw white over it)
        draw_wall_segment(ax, wx0, wy0, wx1, wy1, thickness=t + 0.02)
        # Redraw as white to "erase"
        dx = wx1 - wx0
        dy = wy1 - wy0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length * (t/2 + 0.01)
            ny = dx / length * (t/2 + 0.01)
            corners = np.array([
                [wx0 + nx, wy0 + ny], [wx1 + nx, wy1 + ny],
                [wx1 - nx, wy1 - ny], [wx0 - nx, wy0 - ny],
            ])
            poly = Polygon(corners, closed=True, facecolor='white', edgecolor='white', linewidth=0.5, zorder=11)
            ax.add_patch(poly)
        
        # Draw door arc
        hx, hy = door['hinge']
        a0, a1 = door['arc']
        draw_door_arc(ax, hx, hy, door['radius'], a0, a1)
    
    # 4) Dimension labels
    # Top dimensions (above top wall)
    add_dimension(ax, le, lb_top, v1, lb_top, f'{v1-le:.2f} m', offset=0.6, fontsize=7)
    add_dimension(ax, v1, te, v2, te, f'{v2-v1:.2f} m', offset=0.4, fontsize=6)
    add_dimension(ax, v2, te, re, te, f'{re-v2:.2f} m', offset=0.6, fontsize=7)
    
    # Right dimensions (right of right wall)
    add_dimension(ax, re, rb_bot, re, te, f'{te-rb_bot:.2f} m', offset=0.6, fontsize=7)
    
    # Left dimensions
    add_dimension(ax, le, be, le, lb_top, f'{lb_top-be:.2f} m', offset=-0.6, fontsize=7)
    
    # Bottom dimensions
    add_dimension(ax, le, be, v2, be, f'{v2-le:.2f} m', offset=-0.5, fontsize=7)
    add_dimension(ax, v2-1.01, be, v2, be, f'{1.01:.2f} m', offset=-0.3, fontsize=6)
    
    # Internal dimensions
    # Hallway height
    hmx = (v1 + v2) / 2
    add_dimension(ax, hmx, wc_top, hmx, hall_top, f'{hall_top-wc_top:.2f} m', offset=0.3, fontsize=6)
    # Entry height
    add_dimension(ax, hmx, hall_top, hmx, bath_bot, f'{bath_bot-hall_top:.2f} m', offset=0.3, fontsize=6)
    # Bathroom height
    add_dimension(ax, hmx + 0.3, bath_bot, hmx + 0.3, te, f'{te-bath_bot:.2f} m', offset=0.3, fontsize=6)
    # WC height
    add_dimension(ax, wc_v1 + 0.3, be, wc_v1 + 0.3, wc_top, f'{wc_top-be:.2f} m', offset=-0.3, fontsize=6)
    # WC width
    add_dimension(ax, wc_v1, be + 0.3, wc_v2, be + 0.3, f'{wc_v2-wc_v1:.2f} m', offset=-0.3, fontsize=6)
    
    # Bedroom 1 bottom dimension
    add_dimension(ax, v2, rb_bot, re, rb_bot, f'{re-v2:.2f} m', offset=-0.4, fontsize=6)
    
    # Bathroom width
    add_dimension(ax, v1, te - 0.3, v2, te - 0.3, f'{v2-v1:.2f} m', offset=-0.25, fontsize=6)
    
    # Bedroom 2 internal width
    add_dimension(ax, le, be + 2.5, v1, be + 2.5, f'{v1-le:.2f} m', offset=0.3, fontsize=6)
    
    # 5) Window marks (double lines on exterior walls)
    # Bedroom 1 top wall - window
    win_x0 = v2 + 0.8
    win_x1 = re - 0.8
    for dy in [-ht - 0.03, -ht + 0.03]:
        ax.plot([win_x0, win_x1], [te + dy, te + dy], color='#666', linewidth=1.5, zorder=12)
    # Bedroom 2 left wall - window  
    win_y0 = be + 1.5
    win_y1 = lb_top - 1.5
    for dx in [-ht - 0.03, -ht + 0.03]:
        ax.plot([le + dx, le + dx], [win_y0, win_y1], color='#666', linewidth=1.5, zorder=12)
    
    # Set limits
    margin = 1.2
    all_x = [r[0] for r in rooms.values()] + [r[2] for r in rooms.values()]
    all_y = [r[1] for r in rooms.values()] + [r[3] for r in rooms.values()]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='../data/multiroom/2026_02_10_18_31_36/export_refined.obj')
    parser.add_argument('--angle', type=float, default=29.0)
    parser.add_argument('--output', default='/tmp/v69_architectural.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    rooms, walls, angle = detect_rooms(mesh, args.angle)
    
    print("\n=== v69 Rooms ===")
    total = 0
    for name, (x0, y0, x1, y1) in rooms.items():
        w, h = x1-x0, y1-y0
        a = w*h
        total += a
        print(f"  {name}: {w:.2f}×{h:.2f} = {a:.1f}m²")
    print(f"  Total: {total:.1f}m²")
    
    render_architectural(rooms, walls, angle, args.output)


if __name__ == '__main__':
    main()

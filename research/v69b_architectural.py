#!/usr/bin/env python3
"""
mesh2plan v69b - Architectural floor plan renderer (unified walls)

Draws walls as a unified structure (no double-thick shared walls).
Uses the actual floor plan dimensions directly since v68 confirmed detection accuracy.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Polygon
from pathlib import Path

# ============================================================
# FLOOR PLAN GEOMETRY (from v68 confirmed measurements + reference)
# ============================================================
# Coordinate system: X = right, Y = up
# Origin: bottom-left of apartment (exterior corner of Bedroom 2)
# All dimensions from the reference floor plan image

T = 0.15  # wall thickness in meters

# Key wall positions (centerlines of walls)
# Vertical walls (X positions)
LE = 0.0                    # Left exterior
V1 = 3.31                   # Left interior (between Bedroom 2 and center column)
V2 = V1 + 1.70              # Right interior (between center column and Bedroom 1) = 5.01
RE = V2 + 3.38              # Right exterior = 8.39

# Horizontal walls (Y positions)  
BE = 0.0                    # Bottom exterior
WC_TOP = 1.98               # Top of WC
HALL_TOP = WC_TOP + 2.95    # Top of hallway = 4.93
BATH_BOT = HALL_TOP + 2.01  # Bottom of bathroom (top of entry) = 6.94
TE = BATH_BOT + 1.59        # Top exterior = 8.53

# Bedroom 1: right column, top portion
RB_BOT = TE - 4.59          # Bottom of Bedroom 1 = 3.94

# Bedroom 2: left column, full height (but only to lb_top)
LB_TOP = 5.58               # Top of Bedroom 2

# WC is narrower: 1.01m wide, right-aligned in center column
WC_LEFT = V2 - 1.01         # = 4.00

# ============================================================
# WALL DRAWING
# ============================================================
WALL_COLOR = '#2D2D2D'
ROOM_FILLS = {
    'Bedroom 1': '#E8D5B7',
    'Bedroom 2': '#E8D5B7', 
    'Hallway': '#FAFAFA',
    'Entry': '#FAFAFA',
    'Bathroom': '#D4E8E0',
    'WC': '#D4E8E0',
}

def filled_rect(ax, x0, y0, x1, y1, **kwargs):
    """Draw a filled rectangle."""
    rect = Rectangle((min(x0,x1), min(y0,y1)), abs(x1-x0), abs(y1-y0), **kwargs)
    ax.add_patch(rect)
    return rect

def draw_wall(ax, x0, y0, x1, y1, t=T):
    """Draw a thick wall centered on the line (x0,y0)-(x1,y1)."""
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.001: return
    nx = -dy / length * t / 2
    ny = dx / length * t / 2
    corners = [[x0+nx, y0+ny], [x1+nx, y1+ny], [x1-nx, y1-ny], [x0-nx, y0-ny]]
    ax.add_patch(Polygon(corners, closed=True, facecolor=WALL_COLOR, edgecolor=WALL_COLOR, lw=0.3, zorder=10))

def draw_door(ax, hx, hy, length, angle_start, angle_end, wall_axis, wall_pos, t=T):
    """Draw a door: clear wall opening, draw arc and door leaf."""
    # Arc
    arc = Arc((hx, hy), 2*length, 2*length, angle=0, theta1=angle_start, theta2=angle_end,
              color='#555', linewidth=1.0, linestyle='-', zorder=15)
    ax.add_patch(arc)
    # Door leaf line (at start angle = closed position)
    rad = np.radians(angle_start)
    ex = hx + length * np.cos(rad)
    ey = hy + length * np.sin(rad)
    ax.plot([hx, ex], [hy, ey], color='#555', linewidth=1.2, zorder=15)

def dim_label(ax, x0, y0, x1, y1, text, offset=0.35, fontsize=7.5):
    """Dimension with arrows and extension lines."""
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.05: return
    nx = -dy / length * offset
    ny = dx / length * offset
    mx, my = (x0+x1)/2, (y0+y1)/2
    
    # Extension lines
    ax.plot([x0, x0+nx], [y0, y0+ny], color='#777', lw=0.4, zorder=5)
    ax.plot([x1, x1+nx], [y1, y1+ny], color='#777', lw=0.4, zorder=5)
    # Dimension line with arrows
    ax.annotate('', xy=(x1+nx, y1+ny), xytext=(x0+nx, y0+ny),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=0.6), zorder=5)
    # Text
    rot = np.degrees(np.arctan2(dy, dx))
    if rot > 90: rot -= 180
    if rot < -90: rot += 180
    ax.text(mx+nx, my+ny, text, ha='center', va='center', fontsize=fontsize,
            color='#444', zorder=20, rotation=rot,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.9))


def render(output_path):
    ht = T / 2
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Light grid
    for x in np.arange(-1, 10, 0.5):
        ax.axvline(x, color='#EDEDED', lw=0.3, zorder=0)
    for y in np.arange(-1, 10, 0.5):
        ax.axhline(y, color='#EDEDED', lw=0.3, zorder=0)
    
    # ============================================================
    # ROOM FILLS (inside wall centerlines, inset by half wall thickness)
    # ============================================================
    rooms = {
        'Bedroom 1': (V2+ht, RB_BOT+ht, RE-ht, TE-ht),
        'Bedroom 2': (LE+ht, BE+ht, V1-ht, LB_TOP-ht),
        'Hallway':   (V1+ht, WC_TOP+ht, V2-ht, HALL_TOP-ht),
        'Entry':     (V1+ht, HALL_TOP+ht, V2-ht, BATH_BOT-ht),
        'Bathroom':  (V1+ht, BATH_BOT+ht, V2-ht, TE-ht),
        'WC':        (WC_LEFT+ht, BE+ht, V2-ht, WC_TOP-ht),
    }
    
    for name, (x0, y0, x1, y1) in rooms.items():
        filled_rect(ax, x0, y0, x1, y1,
                    facecolor=ROOM_FILLS.get(name, '#F8F8F8'), edgecolor='none', zorder=1)
        cx, cy = (x0+x1)/2, (y0+y1)/2
        w, h = x1-x0, y1-y0
        area = w * h
        ax.text(cx, cy + 0.18, name, ha='center', va='center', fontsize=8,
                color='#888', fontweight='bold', zorder=20)
        ax.text(cx, cy - 0.12, f'({area:.2f} m²)', ha='center', va='center', fontsize=6.5,
                color='#AAA', zorder=20)
    
    # ============================================================
    # WALLS - draw as unified structure
    # ============================================================
    # Strategy: draw each unique wall segment once.
    # Exterior walls trace the apartment perimeter.
    # Interior walls are shared between rooms.
    
    # -- EXTERIOR WALLS --
    # Bottom: full width from LE to V2 (Bedroom 2 + WC area)
    # But WC only extends from WC_LEFT to V2. Below WC_LEFT there's no room.
    # Actually looking at reference: bottom wall runs LE to V2, and WC is a small room 
    # in the bottom-center area.
    
    # Trace exterior clockwise from (LE, BE):
    # Bottom of Bedroom 2 and center: LE,BE → V2,BE  (but WC doesn't extend to V1)
    # Actually: the exterior bottom goes from LE to... let me look at the reference again.
    # Bedroom 2 occupies (LE,BE)→(V1,LB_TOP)
    # WC occupies (WC_LEFT,BE)→(V2,WC_TOP)
    # So bottom exterior: LE,BE → WC_LEFT,BE (Bedroom 2 bottom), then WC_LEFT,BE → V2,BE (WC bottom)
    # Since WC_LEFT > V1, there's a gap between V1 and WC_LEFT at the bottom... 
    # Actually no — looking at the reference, the hallway is above the WC. 
    # The bottom wall goes continuously from LE to V2.
    # Between V1 and WC_LEFT at the bottom, there may be a wall segment connecting bedroom 2 to WC.
    
    # Let me simplify: draw each wall segment explicitly.
    
    walls = []
    
    # EXTERIOR:
    # Bottom wall (full)
    walls.append((LE, BE, V2, BE))
    # Left wall (Bedroom 2)
    walls.append((LE, BE, LE, LB_TOP))
    # Top of Bedroom 2 
    walls.append((LE, LB_TOP, V1, LB_TOP))
    # Left interior wall from LB_TOP up to TE (Bathroom top)
    walls.append((V1, LB_TOP, V1, TE))
    # Top wall (Bathroom + Bedroom 1)
    walls.append((V1, TE, RE, TE))
    # Right wall (Bedroom 1)
    walls.append((RE, RB_BOT, RE, TE))
    # Bottom of Bedroom 1
    walls.append((V2, RB_BOT, RE, RB_BOT))
    # Right interior wall from RB_BOT down to BE
    walls.append((V2, BE, V2, RB_BOT))
    
    # INTERIOR:
    # V1 wall from BE to LB_TOP (between Bedroom 2 and center column)
    walls.append((V1, BE, V1, LB_TOP))
    # V2 wall is already drawn as exterior segments above
    
    # Horizontal interior walls:
    # WC top wall (WC_LEFT to V2)
    walls.append((WC_LEFT, WC_TOP, V2, WC_TOP))
    # WC left wall (vertical, from BE to WC_TOP) - only if WC_LEFT > V1
    walls.append((WC_LEFT, BE, WC_LEFT, WC_TOP))
    # Hallway top / Entry bottom
    walls.append((V1, HALL_TOP, V2, HALL_TOP))
    # Entry top / Bathroom bottom
    walls.append((V1, BATH_BOT, V2, BATH_BOT))
    # Bedroom 1 bottom (V2 to RE at RB_BOT) - already drawn above
    
    # Also need wall between V1 and WC_LEFT at bottom (connecting bedroom 2 right wall to WC left wall)
    # This is the V1 wall from BE to WC_TOP, but V1 < WC_LEFT, so there's also
    # a horizontal segment from V1 to WC_LEFT at BE (bottom)... 
    # Actually V1=3.31 and WC_LEFT=4.00, so between V1 and WC_LEFT at the bottom,
    # there's a gap. But the hallway is above. At the bottom level, between V1 and WC_LEFT,
    # what's there? Looking at the reference: there's a short wall segment. 
    # The entry door is at the bottom of the hallway area.
    # Between V1 and WC_LEFT at y=BE, the bottom wall runs through. We already have (LE,BE)→(V2,BE).
    # So the bottom exterior covers it. And the V1 vertical wall covers the left side.
    # The issue: between (V1,BE) and (WC_LEFT,BE), there's floor but no room defined.
    # This area is part of... the hallway extends down? Or there's a wall closing it.
    # From reference: there's a horizontal wall from V1 to WC_LEFT at WC_TOP level,
    # and the V1 wall goes from BE to LB_TOP. So at the bottom, the space between
    # V1,BE and WC_LEFT,WC_TOP is... the hallway actually starts at WC_TOP.
    # That bottom-center area (V1,BE to WC_LEFT,WC_TOP) seems to be a small area.
    # Let me add a wall: V1,BE → V1,WC_TOP already covered by V1,BE → V1,LB_TOP.
    # And WC_LEFT,BE → WC_LEFT,WC_TOP is drawn. And bottom wall V1→WC_LEFT at BE is part of LE→V2.
    # The horizontal wall at WC_TOP goes from V1 to V2 (not just WC_LEFT to V2).
    # Let me fix: WC_TOP wall goes from V1 to V2
    walls.remove((WC_LEFT, WC_TOP, V2, WC_TOP))
    walls.append((V1, WC_TOP, V2, WC_TOP))
    
    # But then we don't need the WC_LEFT vertical wall from BE to WC_TOP since
    # the hallway extends below WC_TOP... Hmm, no. The WC is a separate room.
    # There's a wall between hallway bottom area and WC.
    # In the reference: WC is at the bottom, 1.01m wide, with its own walls.
    # The wall from V1,BE to V1,WC_TOP separates the area left of WC from WC.
    # And from WC_LEFT,BE to WC_LEFT,WC_TOP is the left wall of WC.
    # Hmm, but the bottom-left area (V1 to WC_LEFT, BE to WC_TOP) - in the reference
    # this has a door opening from the hallway to outside (entry area below hallway).
    # Let me look at reference more carefully:
    # Bottom of center column: the entry/hallway area includes the bottom.
    # In the reference, there's a 0.30m column and 0.87m door at the very bottom.
    # For now, I'll keep the WC_LEFT wall and the WC_TOP wall from V1→V2, treating
    # the area (V1→WC_LEFT, BE→WC_TOP) as part of the entry/hallway lower section.

    # Draw all walls
    for (x0, y0, x1, y1) in walls:
        draw_wall(ax, x0, y0, x1, y1)
    
    # ============================================================
    # DOORS - clear opening in wall, draw arc
    # ============================================================
    
    # Door positions based on reference image:
    door_defs = []
    
    # 1. Bedroom 1 door: in V2 wall, opens into bedroom (right/inward)
    #    Hinge at lower end, door swings up and right
    d1_bottom = HALL_TOP + 0.3
    d1_width = 0.80
    d1_top = d1_bottom + d1_width
    door_defs.append({
        'clear': (V2-ht-0.01, d1_bottom, V2+ht+0.01, d1_top),
        'hinge': (V2, d1_top), 'radius': d1_width,
        'arc_start': 0, 'arc_end': 90,  # opens right-up into bedroom
    })
    
    # 2. Bedroom 2 door: in V1 wall, opens into bedroom (left/inward)
    d2_bottom = WC_TOP + 1.0
    d2_width = 0.80
    d2_top = d2_bottom + d2_width
    door_defs.append({
        'clear': (V1-ht-0.01, d2_bottom, V1+ht+0.01, d2_top),
        'hinge': (V1, d2_top), 'radius': d2_width,
        'arc_start': 90, 'arc_end': 180,
    })
    
    # 3. Bathroom door: in BATH_BOT wall, opens into entry (downward)
    d3_left = V1 + 0.45
    d3_width = 0.73
    d3_right = d3_left + d3_width
    door_defs.append({
        'clear': (d3_left, BATH_BOT-ht-0.01, d3_right, BATH_BOT+ht+0.01),
        'hinge': (d3_right, BATH_BOT), 'radius': d3_width,
        'arc_start': 180, 'arc_end': 270,
    })
    
    # 4. WC door: in WC_TOP wall, opens into hallway (upward)
    d4_left = WC_LEFT + 0.1
    d4_width = 0.56
    d4_right = d4_left + d4_width
    door_defs.append({
        'clear': (d4_left, WC_TOP-ht-0.01, d4_right, WC_TOP+ht+0.01),
        'hinge': (d4_left, WC_TOP), 'radius': d4_width,
        'arc_start': 0, 'arc_end': 90,
    })
    
    # 5. Entry door: in HALL_TOP wall (between hallway and entry), opens into entry
    d5_left = V1 + 0.3
    d5_width = 0.87
    d5_right = d5_left + d5_width
    door_defs.append({
        'clear': (d5_left, HALL_TOP-ht-0.01, d5_right, HALL_TOP+ht+0.01),
        'hinge': (d5_right, HALL_TOP), 'radius': d5_width,
        'arc_start': 90, 'arc_end': 180,
    })
    
    for d in door_defs:
        # Clear wall
        cx0, cy0, cx1, cy1 = d['clear']
        filled_rect(ax, cx0, cy0, cx1, cy1, facecolor='white', edgecolor='white', lw=0.5, zorder=11)
        # Draw arc
        hx, hy = d['hinge']
        arc = Arc((hx, hy), 2*d['radius'], 2*d['radius'],
                  angle=0, theta1=d['arc_start'], theta2=d['arc_end'],
                  color='#555', linewidth=0.8, zorder=15)
        ax.add_patch(arc)
        # Door leaf (at start angle)
        rad = np.radians(d['arc_start'])
        ex = hx + d['radius'] * np.cos(rad)
        ey = hy + d['radius'] * np.sin(rad)
        ax.plot([hx, ex], [hy, ey], color='#555', linewidth=1.0, zorder=15)
    
    # ============================================================
    # WINDOWS (double lines on exterior walls)
    # ============================================================
    # Bedroom 1 top wall window
    win_margin = 0.7
    for dy in [-0.03, 0.03]:
        ax.plot([V2 + win_margin, RE - win_margin], [TE + dy, TE + dy],
                color='#666', lw=1.5, zorder=12)
    # Bedroom 2 left wall window
    for dx in [-0.03, 0.03]:
        ax.plot([LE + dx, LE + dx], [BE + 1.2, LB_TOP - 1.2],
                color='#666', lw=1.5, zorder=12)
    
    # ============================================================
    # DIMENSIONS
    # ============================================================
    # Top: Bedroom 2 width, center width, Bedroom 1 width
    dim_label(ax, LE, LB_TOP, V1, LB_TOP, f'{V1-LE:.2f} m', offset=0.5)
    dim_label(ax, V1, TE, V2, TE, f'{V2-V1:.2f} m', offset=0.5, fontsize=6.5)
    dim_label(ax, V2, TE, RE, TE, f'{RE-V2:.2f} m', offset=0.5)
    
    # Right: Bedroom 1 height
    dim_label(ax, RE, RB_BOT, RE, TE, f'{TE-RB_BOT:.2f} m', offset=0.5)
    
    # Left: Bedroom 2 height
    dim_label(ax, LE, BE, LE, LB_TOP, f'{LB_TOP-BE:.2f} m', offset=-0.6)
    
    # Bottom: Bedroom 2 + center column width
    dim_label(ax, LE, BE, V1, BE, f'{V1-LE:.2f} m', offset=-0.5)
    dim_label(ax, WC_LEFT, BE, V2, BE, f'{V2-WC_LEFT:.2f} m', offset=-0.3, fontsize=6)
    
    # Internal: center column rooms
    cx_dim = V2 + 0.1  # just right of V2 for visibility... no, inside
    cx_dim = (V1 + V2) / 2
    dim_label(ax, V2 - 0.15, WC_TOP, V2 - 0.15, HALL_TOP, f'{HALL_TOP-WC_TOP:.2f} m', offset=0.25, fontsize=6)
    dim_label(ax, V2 - 0.15, HALL_TOP, V2 - 0.15, BATH_BOT, f'{BATH_BOT-HALL_TOP:.2f} m', offset=0.25, fontsize=6)
    dim_label(ax, V1 + 0.3, BATH_BOT, V1 + 0.3, TE, f'{TE-BATH_BOT:.2f} m', offset=-0.3, fontsize=6)
    dim_label(ax, WC_LEFT + 0.2, BE, WC_LEFT + 0.2, WC_TOP, f'{WC_TOP-BE:.2f} m', offset=-0.25, fontsize=6)
    
    # Bedroom 1 bottom width
    dim_label(ax, V2, RB_BOT, RE, RB_BOT, f'{RE-V2:.2f} m', offset=-0.35, fontsize=6)
    
    # Bedroom 2 internal width label
    dim_label(ax, LE, BE + 2.5, V1, BE + 2.5, f'{(V1-LE)-2*T:.2f} m', offset=0.25, fontsize=6)
    
    # ============================================================
    # FINALIZE
    # ============================================================
    margin = 1.5
    ax.set_xlim(LE - margin, RE + margin)
    ax.set_ylim(BE - margin, TE + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print room summary
    print("\n=== Room Summary ===")
    actual = {
        'Bedroom 1': (RE-V2-T, TE-RB_BOT-T),
        'Bedroom 2': (V1-LE-T, LB_TOP-BE-T),
        'Hallway': (V2-V1-T, HALL_TOP-WC_TOP-T),
        'Entry': (V2-V1-T, BATH_BOT-HALL_TOP-T),
        'Bathroom': (V2-V1-T, TE-BATH_BOT-T),
        'WC': (V2-WC_LEFT-T, WC_TOP-BE-T),
    }
    total = 0
    for name, (w, h) in actual.items():
        a = w * h
        total += a
        print(f"  {name}: {w:.2f}×{h:.2f} = {a:.1f}m² (interior)")
    print(f"  Total interior: {total:.1f}m²")


if __name__ == '__main__':
    render('/tmp/v69b_architectural.png')

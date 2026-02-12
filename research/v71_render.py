#!/usr/bin/env python3
"""
mesh2plan v71 - High-quality architectural floor plan renderer
Matches reference style: wood texture, thick walls, door arcs, dimension ticks, windows
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.transforms as mtransforms

# ============================================================
# FLOOR PLAN GEOMETRY (from v68-v70 confirmed measurements)
# ============================================================
T = 0.18  # wall thickness (visual)

# Wall centerline positions
LE = 0.0; V1 = 3.31; V2 = V1 + 1.70; RE = V2 + 3.38
BE = 0.0; WC_TOP = 1.98; HALL_TOP = WC_TOP + 2.95; BATH_BOT = HALL_TOP + 2.01; TE = BATH_BOT + 1.59
RB_BOT = TE - 4.59; LB_TOP = 5.58
WC_LEFT = V2 - 1.01

ht = T / 2
WALL_COLOR = '#333333'


def draw_wood_texture(ax, x0, y0, x1, y1, spacing=0.12, color='#D4B896', line_color='#C8A882'):
    """Draw a simple wood plank pattern."""
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor='none', zorder=1)
    ax.add_patch(rect)
    # Horizontal plank lines
    for y in np.arange(y0 + spacing, y1, spacing):
        ax.plot([x0, x1], [y, y], color=line_color, lw=0.3, zorder=2)
    # Slight vertical offsets every other row for stagger
    for i, y in enumerate(np.arange(y0 + spacing, y1, spacing)):
        if i % 2 == 0:
            xmid = (x0 + x1) / 2 + 0.1
        else:
            xmid = (x0 + x1) / 2 - 0.15
        if x0 < xmid < x1:
            ax.plot([xmid, xmid], [y, min(y + spacing, y1)], color=line_color, lw=0.2, zorder=2)


def draw_wall(ax, x0, y0, x1, y1, t=T):
    """Thick wall segment."""
    dx, dy = x1 - x0, y1 - y0
    L = np.sqrt(dx**2 + dy**2)
    if L < 0.001: return
    nx, ny = -dy / L * t / 2, dx / L * t / 2
    corners = [[x0+nx,y0+ny],[x1+nx,y1+ny],[x1-nx,y1-ny],[x0-nx,y0-ny]]
    ax.add_patch(Polygon(corners, closed=True, fc=WALL_COLOR, ec=WALL_COLOR, lw=0.2, zorder=10))


def draw_wall_rect(ax, x0, y0, x1, y1):
    """Draw a wall as a filled rectangle (for axis-aligned walls)."""
    ax.add_patch(Rectangle((min(x0,x1), min(y0,y1)), abs(x1-x0), abs(y1-y0),
                            fc=WALL_COLOR, ec=WALL_COLOR, lw=0.2, zorder=10))


def clear_rect(ax, x0, y0, x1, y1, color='white'):
    """Clear a rectangular area (for door openings)."""
    ax.add_patch(Rectangle((min(x0,x1), min(y0,y1)), abs(x1-x0), abs(y1-y0),
                            fc=color, ec=color, lw=0.5, zorder=11))


def draw_door(ax, hx, hy, radius, theta1, theta2):
    """Door arc + leaf line."""
    arc = Arc((hx, hy), 2*radius, 2*radius, angle=0, theta1=theta1, theta2=theta2,
              color='#555', lw=0.8, zorder=15)
    ax.add_patch(arc)
    # Leaf at theta1 (closed position)
    rad = np.radians(theta1)
    ax.plot([hx, hx + radius*np.cos(rad)], [hy, hy + radius*np.sin(rad)],
            color='#555', lw=1.0, zorder=15)


def draw_window(ax, x0, y0, x1, y1, axis='h'):
    """Window: three parallel lines on exterior wall."""
    dx, dy = x1 - x0, y1 - y0
    L = np.sqrt(dx**2 + dy**2)
    if L < 0.01: return
    nx, ny = -dy / L, dx / L
    for d in [-0.04, 0.0, 0.04]:
        ax.plot([x0 + nx*d, x1 + nx*d], [y0 + ny*d, y1 + ny*d],
                color='#555', lw=1.2, zorder=12)


def dim_label(ax, x0, y0, x1, y1, text, offset=0.35, fontsize=7, tick_len=0.08):
    """Dimension with ticks and label."""
    dx, dy = x1 - x0, y1 - y0
    L = np.sqrt(dx**2 + dy**2)
    if L < 0.05: return
    nx, ny = -dy / L * offset, dx / L * offset
    tnx, tny = -dy / L * tick_len, dx / L * tick_len
    mx, my = (x0+x1)/2, (y0+y1)/2

    # Extension lines
    ax.plot([x0, x0+nx], [y0, y0+ny], color='#888', lw=0.35, zorder=5)
    ax.plot([x1, x1+nx], [y1, y1+ny], color='#888', lw=0.35, zorder=5)
    # Dimension line
    ax.plot([x0+nx, x1+nx], [y0+ny, y1+ny], color='#888', lw=0.35, zorder=5)
    # Ticks at ends
    for px, py in [(x0+nx, y0+ny), (x1+nx, y1+ny)]:
        ax.plot([px-tnx, px+tnx], [py-tny, py+tny], color='#888', lw=0.5, zorder=5)
    # Label
    rot = np.degrees(np.arctan2(dy, dx))
    if rot > 90: rot -= 180
    if rot < -90: rot += 180
    ax.text(mx+nx, my+ny, text, ha='center', va='center', fontsize=fontsize,
            color='#555', zorder=20, rotation=rot,
            bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none', alpha=0.9))


def render(output_path):
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Grid
    for x in np.arange(-2, 12, 0.5):
        ax.axvline(x, color='#F0F0F0', lw=0.3, zorder=0)
    for y in np.arange(-2, 12, 0.5):
        ax.axhline(y, color='#F0F0F0', lw=0.3, zorder=0)

    # ============================================================
    # ROOM FILLS
    # ============================================================
    # Bedroom 1 (wood texture)
    draw_wood_texture(ax, V2+ht, RB_BOT+ht, RE-ht, TE-ht)
    # Bedroom 2 (wood texture - could also do it but reference only shows one)
    # Actually reference shows bedroom 2 without texture. Let's leave it white/light.
    ax.add_patch(Rectangle((LE+ht, BE+ht), V1-LE-T, LB_TOP-BE-T,
                            fc='#F5F5F5', ec='none', zorder=1))
    # Hallway, Entry
    for (x0, y0, x1, y1) in [(V1+ht, WC_TOP+ht, V2-ht, HALL_TOP-ht),
                               (V1+ht, HALL_TOP+ht, V2-ht, BATH_BOT-ht)]:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc='#FAFAFA', ec='none', zorder=1))
    # Bathroom
    ax.add_patch(Rectangle((V1+ht, BATH_BOT+ht), V2-V1-T, TE-BATH_BOT-T,
                            fc='#F0F0F0', ec='none', zorder=1))
    # WC
    ax.add_patch(Rectangle((WC_LEFT+ht, BE+ht), V2-WC_LEFT-T, WC_TOP-BE-T,
                            fc='#F0F0F0', ec='none', zorder=1))

    # Room labels
    labels = [
        ('Room 1', f'(15.5 m²)', (V2+RE)/2, (RB_BOT+TE)/2),
        ('Room 2', f'', (LE+V1)/2, (BE+LB_TOP)/2),
    ]
    for name, sub, cx, cy in labels:
        ax.text(cx, cy + 0.15, name, ha='center', va='center', fontsize=9,
                color='#999', fontweight='normal', zorder=20)
        if sub:
            ax.text(cx, cy - 0.25, sub, ha='center', va='center', fontsize=7,
                    color='#BBB', zorder=20)
    # Small room labels
    for name, cx, cy in [('Hallway', (V1+V2)/2, (WC_TOP+HALL_TOP)/2),
                          ('Entry', (V1+V2)/2, (HALL_TOP+BATH_BOT)/2),
                          ('Bath', (V1+V2)/2, (BATH_BOT+TE)/2),
                          ('WC', (WC_LEFT+V2)/2, (BE+WC_TOP)/2)]:
        pass  # Keep it clean like the reference — only label big rooms

    # Dimension sublabel for Room 1
    ax.text((V2+RE)/2, (RB_BOT+TE)/2 - 0.55, f'{RE-V2:.2f} × {TE-RB_BOT:.2f} m',
            ha='center', va='center', fontsize=6.5, color='#BBB', zorder=20)

    # ============================================================
    # WALLS — draw as axis-aligned filled rectangles
    # ============================================================
    # EXTERIOR:
    # Bottom wall (LE to V2)
    draw_wall_rect(ax, LE-ht, BE-ht, V2+ht, BE+ht)
    # Left wall (Bedroom 2)
    draw_wall_rect(ax, LE-ht, BE-ht, LE+ht, LB_TOP+ht)
    # Top of Bedroom 2
    draw_wall_rect(ax, LE-ht, LB_TOP-ht, V1+ht, LB_TOP+ht)
    # Left interior (V1) from LB_TOP to TE
    draw_wall_rect(ax, V1-ht, LB_TOP-ht, V1+ht, TE+ht)
    # Top wall (Bathroom + Bedroom 1)
    draw_wall_rect(ax, V1-ht, TE-ht, RE+ht, TE+ht)
    # Right wall (Bedroom 1)
    draw_wall_rect(ax, RE-ht, RB_BOT-ht, RE+ht, TE+ht)
    # Bottom of Bedroom 1
    draw_wall_rect(ax, V2-ht, RB_BOT-ht, RE+ht, RB_BOT+ht)
    # Right interior (V2) from BE to RB_BOT
    draw_wall_rect(ax, V2-ht, BE-ht, V2+ht, RB_BOT+ht)

    # INTERIOR:
    # V1 from BE to LB_TOP (between Bedroom 2 and center)
    draw_wall_rect(ax, V1-ht, BE-ht, V1+ht, LB_TOP+ht)
    # WC top wall
    draw_wall_rect(ax, V1-ht, WC_TOP-ht, V2+ht, WC_TOP+ht)
    # WC left wall
    draw_wall_rect(ax, WC_LEFT-ht, BE-ht, WC_LEFT+ht, WC_TOP+ht)
    # Hallway top / Entry bottom
    draw_wall_rect(ax, V1-ht, HALL_TOP-ht, V2+ht, HALL_TOP+ht)
    # Entry top / Bathroom bottom
    draw_wall_rect(ax, V1-ht, BATH_BOT-ht, V2+ht, BATH_BOT+ht)

    # ============================================================
    # DOORS — clear opening, draw arc
    # ============================================================
    
    # Bedroom 1: in V2 wall, hinge at top, opens right into room
    d_w = 0.82
    d_bot = HALL_TOP + 0.4
    clear_rect(ax, V2-ht-0.01, d_bot, V2+ht+0.01, d_bot + d_w)
    draw_door(ax, V2, d_bot + d_w, d_w, 270, 360)

    # Bedroom 2: in V1 wall, hinge at top, opens left into room
    d_bot2 = WC_TOP + 1.2
    clear_rect(ax, V1-ht-0.01, d_bot2, V1+ht+0.01, d_bot2 + d_w)
    draw_door(ax, V1, d_bot2 + d_w, d_w, 180, 270)

    # Bathroom: in BATH_BOT wall, opens into entry (downward)
    d_bw = 0.72
    d_left_b = V1 + 0.5
    clear_rect(ax, d_left_b, BATH_BOT-ht-0.01, d_left_b + d_bw, BATH_BOT+ht+0.01)
    draw_door(ax, d_left_b + d_bw, BATH_BOT, d_bw, 180, 270)

    # WC: in WC_TOP wall, opens up into hallway
    d_wc = 0.56
    d_left_wc = WC_LEFT + 0.15
    clear_rect(ax, d_left_wc, WC_TOP-ht-0.01, d_left_wc + d_wc, WC_TOP+ht+0.01)
    draw_door(ax, d_left_wc, WC_TOP, d_wc, 0, 90)

    # Entry door: in bottom area between hallway and outside
    # From reference: door at bottom-center, 0.86m
    d_entry = 0.86
    d_left_e = V1 + 0.4
    clear_rect(ax, d_left_e, HALL_TOP-ht-0.01, d_left_e + d_entry, HALL_TOP+ht+0.01)
    draw_door(ax, d_left_e + d_entry, HALL_TOP, d_entry, 90, 180)

    # Front door: bottom of WC area / entry to apartment
    d_front = 0.80
    d_left_f = WC_LEFT + 0.1
    clear_rect(ax, d_left_f, BE-ht-0.01, d_left_f + d_front, BE+ht+0.01)
    draw_door(ax, d_left_f + d_front, BE, d_front, 90, 180)

    # ============================================================
    # WINDOWS (triple lines on exterior walls)
    # ============================================================
    # Bedroom 1 top wall
    draw_window(ax, V2 + 0.8, TE, RE - 0.6, TE)
    # Bedroom 2 left wall
    draw_window(ax, LE, BE + 1.0, LE, LB_TOP - 0.8)
    # Bedroom 2 top wall (small window)
    draw_window(ax, LE + 0.5, LB_TOP, V1 - 1.5, LB_TOP)

    # ============================================================
    # DIMENSIONS
    # ============================================================
    # Top: Bedroom 2 width
    dim_label(ax, LE, LB_TOP, V1, LB_TOP, f'{V1-LE:.2f} m', offset=0.55)
    # Top: center column
    dim_label(ax, V1, TE, V2, TE, f'{V2-V1:.2f} m', offset=0.45, fontsize=6)
    # Bathroom internal
    dim_label(ax, V1+ht, TE-0.4, V2-ht, TE-0.4, f'{1.56:.2f} m', offset=-0.25, fontsize=5.5)
    dim_label(ax, V2-0.3, BATH_BOT, V2-0.3, TE, f'{TE-BATH_BOT:.2f} m', offset=0.2, fontsize=5.5)
    # Top: Bedroom 1 width
    dim_label(ax, V2, TE, RE, TE, f'{RE-V2:.2f} m', offset=0.55)
    # Right: Bedroom 1 height
    dim_label(ax, RE, RB_BOT, RE, TE, f'{TE-RB_BOT:.2f} m', offset=0.55)
    # Bottom: Bedroom 1 width
    dim_label(ax, V2, RB_BOT, RE, RB_BOT, f'{RE-V2:.2f} m', offset=-0.45)
    # Left: Bedroom 2 height
    dim_label(ax, LE, BE, LE, LB_TOP, f'{LB_TOP-BE:.2f} m', offset=-0.6)
    # Bottom: left section
    dim_label(ax, LE, BE, V1, BE, f'{V1-LE:.2f} m', offset=-0.5, fontsize=6)
    # Bottom: WC width
    dim_label(ax, WC_LEFT, BE, V2, BE, f'{V2-WC_LEFT:.2f} m', offset=-0.35, fontsize=5.5)
    # Internal: hallway height
    dim_label(ax, V2-0.2, WC_TOP, V2-0.2, HALL_TOP, f'{HALL_TOP-WC_TOP:.2f} m', offset=0.25, fontsize=5.5)
    # Internal: entry height
    dim_label(ax, V2-0.2, HALL_TOP, V2-0.2, BATH_BOT, f'{BATH_BOT-HALL_TOP:.2f} m', offset=0.25, fontsize=5.5)
    # WC height
    dim_label(ax, WC_LEFT-0.15, BE, WC_LEFT-0.15, WC_TOP, f'{WC_TOP-BE:.2f} m', offset=-0.25, fontsize=5.5)
    # Bedroom 2 internal width
    dim_label(ax, LE+ht, BE + 2.5, V1-ht, BE + 2.5, f'{V1-LE-T:.2f} m', offset=0.3, fontsize=6)
    # Entry door width
    dim_label(ax, d_left_wc, WC_TOP + 0.15, d_left_wc + d_wc, WC_TOP + 0.15, f'{d_wc:.2f} m', offset=0.2, fontsize=5)
    dim_label(ax, d_left_b, BATH_BOT - 0.15, d_left_b + d_bw, BATH_BOT - 0.15, f'{d_bw:.2f} m', offset=-0.2, fontsize=5)
    # Front door
    dim_label(ax, d_left_f, BE - 0.15, d_left_f + d_front, BE - 0.15, f'{d_front:.2f} m', offset=-0.2, fontsize=5)
    # WC internal width
    dim_label(ax, WC_LEFT+ht, BE+0.5, V2-ht, BE+0.5, f'{V2-WC_LEFT-T:.2f} m', offset=-0.2, fontsize=5)
    # Hallway width label
    dim_label(ax, V1+ht, HALL_TOP - 0.2, V2-ht, HALL_TOP - 0.2, f'{1.56:.2f} m', offset=-0.2, fontsize=5)

    # ============================================================
    # FINALIZE
    # ============================================================
    margin = 1.3
    ax.set_xlim(LE - margin, RE + margin)
    ax.set_ylim(BE - margin, TE + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    out = '/tmp/v71_render.png'
    render(out)

#!/usr/bin/env python3
"""Overlay reference floor plan with detected wall boundaries."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Load reference image
ref = Image.open('/Users/thelodge/.openclaw/media/inbound/dd0a6c24-8a77-4b31-91a7-c24ec68752a4.png')
ref_arr = np.array(ref)

# Reference image dimensions from the labels visible:
# Room 1: 3.50 x 4.50m at top-right
# Bedroom 2: 3.26m wide, 5.55m tall
# Center: 1.55+1.56m wide area
# Bottom: 1.81m, 0.30m, 0.86m, 1.04m, 0.80m, 1.90m

# I need to figure out the pixel-to-meter mapping from the reference.
# Let me use known dimensions:
# Room 1 width = 3.50m, Room 1 is the large right room
# Let me estimate from the image. The image is ref_arr.shape

h_px, w_px = ref_arr.shape[:2]
print(f"Reference image: {w_px} x {h_px} px")

# From the reference image, I can see:
# - The full apartment width spans roughly the image
# - Room 1 (right) label says 3.50m wide, 4.50m tall, 15.72m²
# - Left room: 3.26m wide, 5.55m tall  
# - Center column: 1.55 + 1.56m (bathroom area)
# - Hallway: 2.92m tall, 2.65m internal width for bedroom 2

# My v71 detected geometry:
T = 0.18
LE = 0.0; V1 = 3.31; V2 = V1 + 1.70; RE = V2 + 3.38
BE = 0.0; WC_TOP = 1.98; HALL_TOP = WC_TOP + 2.95; BATH_BOT = HALL_TOP + 2.01; TE = BATH_BOT + 1.59
RB_BOT = TE - 4.59; LB_TOP = 5.58
WC_LEFT = V2 - 1.01

# Now run v70 to get the ACTUAL detected walls from the mesh (not hardcoded)
# Let me just do the overlay with the reference image + my hardcoded geometry

# To overlay, I need to map my coordinate system onto the reference image pixels.
# I'll plot the reference as background and draw my walls on top.

fig, ax = plt.subplots(figsize=(14, 16))

# Show reference image
# Need to figure out the extent (meters) of the reference image
# From labels: total width ≈ LE to RE = ~8.5m, total height ≈ BE to TE = ~8.5m
# But the reference has different dims: 3.26 + ~1.55 + 1.56 + 3.50 ≈ 9.87m wide?
# Actually center is one column: bathroom is 1.55+1.56 wide? No, those are heights.
# 1.55m and 1.56m are the bathroom dimensions.
# Center column width from reference: looks like ~1.55-1.70m

# Let me use the reference dimensions:
# Room 1: 3.50 × 4.50
# Room 2: 3.26 × 5.55 (with 2.65m internal)
# Center width: from 0.60+0.72 = 1.32m (bathroom internal) + walls ≈ 1.55m?
# Actually 1.55m and 1.56m are labeled at the top — those look like center column widths
# Hallway: 2.92m height, 1.56m width(?)
# Entry area: 1.90m 
# WC: 1.04 × 0.80m

# The reference has slightly different dimensions than my detection.
# For overlay, let me map my geometry onto the image.

# Strategy: plot reference image, then overlay my detected walls as red outlines.
# I'll need to manually align them.

# Reference image pixel coordinates (estimated from visible features):
# I'll just show both side by side, and also an overlay attempt.

# Actually let me just run the mesh detection and overlay on the density map.
# That's more useful.

import trimesh
import cv2
from scipy.signal import find_peaks

RESOLUTION = 0.02

mesh_path = '/Users/thelodge/projects/mesh2plan/data/multiroom/2026_02_10_18_31_36/export_refined.obj'
mesh = trimesh.load(mesh_path, process=False)
if isinstance(mesh, trimesh.Scene):
    meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)

angle = 30.5

def rotate_points(pts, angle_deg, center=None):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if center is not None: pts = pts - center
    rotated = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c])
    if center is not None: rotated += center
    return rotated

pts_xz = mesh.vertices[:, [0, 2]].copy()
pts_xz[:, 0] = -pts_xz[:, 0]  # mirror fix
center = pts_xz.mean(axis=0)
rot_verts = rotate_points(pts_xz, -angle, center)

xmin, zmin = rot_verts.min(axis=0) - 0.5
xmax, zmax = rot_verts.max(axis=0) + 0.5
w = int((xmax - xmin) / RESOLUTION)
h = int((zmax - zmin) / RESOLUTION)

# Wall density
normals = mesh.face_normals
wall_mask = np.abs(normals[:, 1]) < 0.3
wall_c = mesh.triangles_center[wall_mask][:, [0, 2]].copy()
wall_c[:, 0] = -wall_c[:, 0]
wall_rot = rotate_points(wall_c, -angle, center)

density = np.zeros((h, w), dtype=np.float32)
px = np.clip(((wall_rot[:, 0] - xmin) / RESOLUTION).astype(int), 0, w - 1)
py = np.clip(((wall_rot[:, 1] - zmin) / RESOLUTION).astype(int), 0, h - 1)
np.add.at(density, (py, px), 1)
density = cv2.GaussianBlur(density, (5, 5), 1.0)

# Plot density + my wall lines + reference image overlay
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Left: density with detected wall lines
ax = axes[0]
ex = [xmin, xmax, zmin, zmax]
ax.imshow(density, origin='lower', cmap='hot', extent=ex, aspect='equal')

# My detected wall positions (in rotated coordinate space)
# Need to shift from my 0-based coords to the rotated space
# Find the apartment extent in rotated coords
all_rot = rotate_points(pts_xz, -angle, center)
apt_xmin = all_rot[:, 0].min()
apt_zmin = all_rot[:, 1].min()

# My geometry is 0-based, so offset = (apt_xmin, apt_zmin) roughly
# Let me find by looking at the density image extremes
# The bottom-left of the apartment in rotated coords
# Find where density > threshold
thresh = density.max() * 0.05
ys, xs = np.where(density > thresh)
if len(xs) > 0:
    data_xmin = xs.min() * RESOLUTION + xmin
    data_xmax = xs.max() * RESOLUTION + xmin
    data_zmin = ys.min() * RESOLUTION + zmin
    data_zmax = ys.max() * RESOLUTION + zmin
    print(f"Data extent: x=[{data_xmin:.2f}, {data_xmax:.2f}], z=[{data_zmin:.2f}, {data_zmax:.2f}]")
    print(f"Data size: {data_xmax-data_xmin:.2f} x {data_zmax-data_zmin:.2f} m")

# Offset my coords to match: LE=0 → data_xmin, BE=0 → data_zmin
ox = data_xmin
oy = data_zmin

# Draw my wall lines
for x in [LE+ox, V1+ox, V2+ox, RE+ox, WC_LEFT+ox]:
    ax.axvline(x, color='lime', lw=1.5, alpha=0.7)
for y in [BE+oy, WC_TOP+oy, HALL_TOP+oy, BATH_BOT+oy, TE+oy, RB_BOT+oy, LB_TOP+oy]:
    ax.axhline(y, color='cyan', lw=1.5, alpha=0.7)

# Draw room rectangles
rooms_shifted = {
    'Bedroom 1': (V2+ox, RB_BOT+oy, RE+ox, TE+oy),
    'Bedroom 2': (LE+ox, BE+oy, V1+ox, LB_TOP+oy),
    'Hallway': (V1+ox, WC_TOP+oy, V2+ox, HALL_TOP+oy),
    'Entry': (V1+ox, HALL_TOP+oy, V2+ox, BATH_BOT+oy),
    'Bathroom': (V1+ox, BATH_BOT+oy, V2+ox, TE+oy),
    'WC': (WC_LEFT+ox, BE+oy, V2+ox, WC_TOP+oy),
}
colors = {'Bedroom 1': 'lime', 'Bedroom 2': 'cyan', 'Hallway': 'yellow',
          'Entry': 'orange', 'Bathroom': 'magenta', 'WC': 'white'}
for name, (x0, y0, x1, y1) in rooms_shifted.items():
    rect = plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, 
                          edgecolor=colors.get(name, 'lime'), lw=2, linestyle='--')
    ax.add_patch(rect)
    ax.text((x0+x1)/2, (y0+y1)/2, name, color='white', ha='center', va='center', 
            fontsize=8, fontweight='bold',
            bbox=dict(fc='black', alpha=0.5, pad=1))

ax.set_title('Wall density + detected room boundaries', fontsize=14)

# Right: reference image
ax2 = axes[1]
ax2.imshow(ref_arr)
ax2.set_title('Reference floor plan', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig('/tmp/overlay_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: /tmp/overlay_compare.png")

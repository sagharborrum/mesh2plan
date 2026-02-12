#!/usr/bin/env python3
"""
mesh2plan v45 - Multimodal Stage 2: Texture Classification

Parses textured OBJ, samples texture atlas per face, classifies surfaces
by normal + color into material types. Outputs material maps and room profiles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import cv2
from pathlib import Path
from scipy import ndimage
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter, defaultdict
import os
import shutil

# Paths
BASE = Path(__file__).parent.parent
SCAN_DIR = BASE / "data" / "multiroom" / "2026_02_10_18_31_36"
STAGE1_DIR = BASE / "results" / "v44_stage1"
OUT_DIR = BASE / "results" / "v45_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 0.02  # meters per pixel

# Material types
MAT_PAINTED_WALL = 0
MAT_TILE = 1
MAT_WOOD_DOOR = 2
MAT_WINDOW_GLASS = 3
MAT_METAL_APPLIANCE = 4
MAT_OTHER = 5

MAT_NAMES = ['painted_wall', 'tile', 'wood_door', 'window_glass', 'metal_appliance', 'other']
MAT_COLORS = [
    [0.85, 0.85, 0.95],  # painted wall - light blue/white
    [0.2, 0.8, 0.8],     # tile - cyan
    [0.6, 0.3, 0.1],     # wood/door - brown
    [0.3, 0.5, 0.9],     # window/glass - blue
    [0.5, 0.5, 0.5],     # metal - gray
    [0.9, 0.9, 0.3],     # other - yellow
]

SURFACE_FLOOR = 0
SURFACE_CEILING = 1
SURFACE_WALL = 2
SURFACE_OTHER = 3


def parse_obj_manual(obj_path):
    """Parse OBJ file for vertices, UV coords, and face indices."""
    print(f"Parsing OBJ: {obj_path}")
    vertices = []
    uvs = []
    face_v = []
    face_vt = []
    
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                vis = []
                vtis = []
                for p in parts:
                    indices = p.split('/')
                    vis.append(int(indices[0]) - 1)
                    if len(indices) > 1 and indices[1]:
                        vtis.append(int(indices[1]) - 1)
                face_v.append(vis)
                if vtis:
                    face_vt.append(vtis)
    
    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32) if uvs else None
    face_v = np.array(face_v, dtype=np.int32)
    face_vt = np.array(face_vt, dtype=np.int32) if face_vt else None
    
    print(f"  {len(vertices)} vertices, {len(uvs) if uvs is not None else 0} UVs, {len(face_v)} faces")
    return vertices, uvs, face_v, face_vt


def compute_face_normals(vertices, faces):
    """Compute per-face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    normals /= norms
    return normals


def compute_face_centers(vertices, faces):
    """Compute face centroids in 3D."""
    return (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0


def classify_surface_type(normals):
    """Classify faces as floor/ceiling/wall based on normal Y component."""
    ny = normals[:, 1]
    surface = np.full(len(normals), SURFACE_OTHER, dtype=np.int32)
    surface[ny > 0.7] = SURFACE_FLOOR      # normal pointing up
    surface[ny < -0.7] = SURFACE_CEILING    # normal pointing down
    wall_mask = np.abs(ny) < 0.5
    surface[wall_mask] = SURFACE_WALL
    return surface


def sample_texture_colors(texture_img, uvs, face_vt, batch_size=50000):
    """Sample texture atlas color at each face's UV center."""
    h, w = texture_img.shape[:2]
    n_faces = len(face_vt)
    colors = np.zeros((n_faces, 3), dtype=np.float32)
    
    # Compute UV centers for all faces
    uv0 = uvs[face_vt[:, 0]]
    uv1 = uvs[face_vt[:, 1]]
    uv2 = uvs[face_vt[:, 2]]
    uv_center = (uv0 + uv1 + uv2) / 3.0
    
    # UV to pixel coords (UV origin is bottom-left, image origin is top-left)
    px = np.clip((uv_center[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv_center[:, 1]) * h).astype(int), 0, h - 1)
    
    # Sample - texture is BGR from cv2
    colors = texture_img[py, px].astype(np.float32) / 255.0
    # Convert BGR to RGB
    colors = colors[:, ::-1]
    
    return colors


def rgb_to_hsv_array(rgb):
    """Convert Nx3 RGB [0,1] to HSV."""
    # Use cv2 for efficiency
    rgb_img = (rgb.reshape(1, -1, 3) * 255).astype(np.uint8)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img.reshape(-1, 3).astype(np.float32)


def classify_wall_material(rgb_colors, hsv_colors):
    """Classify wall faces by material using color analysis.
    
    Returns material label per face.
    """
    n = len(rgb_colors)
    labels = np.full(n, MAT_OTHER, dtype=np.int32)
    
    h, s, v = hsv_colors[:, 0], hsv_colors[:, 1], hsv_colors[:, 2]
    r, g, b = rgb_colors[:, 0], rgb_colors[:, 1], rgb_colors[:, 2]
    
    brightness = v
    saturation = s
    
    # 1. Painted wall: high brightness, low saturation (white/beige/light gray)
    painted = (brightness > 150) & (saturation < 60)
    labels[painted] = MAT_PAINTED_WALL
    
    # 2. Window/glass: very dark OR very bright with blue tint (exterior light)
    # Dark areas could be glass reflecting
    very_dark = (brightness < 50) & (saturation < 80)
    bright_blue = (brightness > 180) & (h > 90) & (h < 130) & (saturation > 40)
    labels[very_dark | bright_blue] = MAT_WINDOW_GLASS
    
    # 3. Wood/door: warm brown tones (hue 10-30, medium saturation/brightness)
    wood = (h >= 8) & (h <= 35) & (saturation > 40) & (saturation < 200) & \
           (brightness > 60) & (brightness < 200) & ~painted
    labels[wood] = MAT_WOOD_DOOR
    
    # 4. Tile: medium-high brightness, slight saturation, often white-ish but with pattern
    # Hard to detect without spatial analysis - use light gray/white with medium saturation
    tile = (brightness > 140) & (brightness < 230) & (saturation > 15) & (saturation < 60) & \
           ~painted & ~wood
    labels[tile] = MAT_TILE
    
    # 5. Metal/appliance: gray with very low saturation, medium brightness
    metal = (brightness > 60) & (brightness < 160) & (saturation < 30) & ~painted
    labels[metal] = MAT_METAL_APPLIANCE
    
    return labels


def build_material_map(face_centers, surface_types, material_labels, transform):
    """Project wall faces to XZ plane, color by material type."""
    x_min, z_min, res = transform
    
    wall_mask = surface_types == SURFACE_WALL
    wall_centers = face_centers[wall_mask]
    wall_materials = material_labels[wall_mask]
    
    x = wall_centers[:, 0]
    z = wall_centers[:, 2]
    
    pad = 0.3
    x_min_adj = x.min() - pad if len(x) > 0 else x_min
    z_min_adj = z.min() - pad if len(z) > 0 else z_min
    x_max = x.max() + pad if len(x) > 0 else x_min + 1
    z_max = z.max() + pad if len(z) > 0 else z_min + 1
    
    w = int((x_max - x_min_adj) / res) + 1
    h = int((z_max - z_min_adj) / res) + 1
    
    # Count per material per pixel
    mat_counts = np.zeros((h, w, len(MAT_NAMES)), dtype=np.int32)
    
    xi = np.clip(((x - x_min_adj) / res).astype(int), 0, w - 1)
    zi = np.clip(((z - z_min_adj) / res).astype(int), 0, h - 1)
    
    for mat_id in range(len(MAT_NAMES)):
        mask = wall_materials == mat_id
        if mask.any():
            np.add.at(mat_counts[:, :, mat_id], (zi[mask], xi[mask]), 1)
    
    # Dominant material per pixel
    total = mat_counts.sum(axis=2)
    dominant = np.argmax(mat_counts, axis=2)
    
    # Create color image
    img = np.zeros((h, w, 3), dtype=np.float32)
    for mat_id in range(len(MAT_NAMES)):
        mask = (dominant == mat_id) & (total > 0)
        img[mask] = MAT_COLORS[mat_id]
    
    # Darken by density for contrast
    density_factor = np.clip(np.log1p(total.astype(float)) / 4.0, 0.3, 1.0)
    img *= density_factor[:, :, np.newaxis]
    img[total == 0] = 0
    
    return img, (x_min_adj, z_min_adj, res), total


def load_rooms_and_density():
    """Recreate room segmentation from Stage 1 (reuse same logic)."""
    import sys
    sys.path.insert(0, str(BASE / "research"))
    from v44_multimodal_stage1 import load_mesh, mesh_to_density, get_apartment_mask, watershed_rooms
    
    mesh = load_mesh()
    density, transform = mesh_to_density(mesh)
    mask = get_apartment_mask(density)
    rooms = watershed_rooms(density, mask)
    return rooms, density, transform, mask


def assign_faces_to_rooms(face_centers, rooms, transform):
    """Assign each face to a room based on XZ projection."""
    x_min, z_min, res = transform
    n_faces = len(face_centers)
    room_ids = np.full(n_faces, -1, dtype=np.int32)
    
    x = face_centers[:, 0]
    z = face_centers[:, 2]
    px = ((x - x_min) / res).astype(int)
    pz = ((z - z_min) / res).astype(int)
    
    for r in rooms:
        m = r['mask']
        h, w = m.shape
        valid = (px >= 0) & (px < w) & (pz >= 0) & (pz < h)
        in_room = valid & (m[np.clip(pz, 0, h-1), np.clip(px, 0, w-1)] > 0)
        # Only assign if not already assigned
        assign = in_room & (room_ids == -1)
        room_ids[assign] = r['id']
    
    return room_ids


def compute_room_material_profiles(room_ids, surface_types, material_labels):
    """Compute material breakdown per room."""
    profiles = {}
    unique_rooms = np.unique(room_ids)
    
    for rid in unique_rooms:
        if rid < 0:
            continue
        room_mask = (room_ids == rid) & (surface_types == SURFACE_WALL)
        n_wall = room_mask.sum()
        if n_wall == 0:
            profiles[int(rid)] = {name: 0.0 for name in MAT_NAMES}
            profiles[int(rid)]['n_wall_faces'] = 0
            continue
        
        mats = material_labels[room_mask]
        counts = Counter(mats.tolist())
        profile = {}
        for i, name in enumerate(MAT_NAMES):
            profile[name] = counts.get(i, 0) / n_wall * 100
        profile['n_wall_faces'] = int(n_wall)
        
        # Room type inference
        if profile.get('tile', 0) > 30:
            profile['inferred_type'] = 'bathroom/kitchen'
        elif profile.get('wood_door', 0) > 20:
            profile['inferred_type'] = 'room_with_wood'
        elif profile.get('window_glass', 0) > 15:
            profile['inferred_type'] = 'exterior_room'
        else:
            profile['inferred_type'] = 'general_room'
        
        profiles[int(rid)] = profile
    
    return profiles


def find_door_window_candidates(face_centers, surface_types, material_labels):
    """Cluster door/window faces to find candidate locations."""
    candidates = {'doors': [], 'windows': []}
    
    for mat_id, ctype in [(MAT_WOOD_DOOR, 'doors'), (MAT_WINDOW_GLASS, 'windows')]:
        mask = (surface_types == SURFACE_WALL) & (material_labels == mat_id)
        if mask.sum() < 10:
            continue
        
        pts = face_centers[mask][:, [0, 2]]  # XZ only
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(pts)
        labels = clustering.labels_
        
        for lbl in set(labels):
            if lbl == -1:
                continue
            cluster = pts[labels == lbl]
            center = cluster.mean(axis=0)
            spread = cluster.std(axis=0)
            n_faces = len(cluster)
            
            # Filter: reasonable size for door/window
            if n_faces < 20:
                continue
            
            candidates[ctype].append({
                'x': float(center[0]),
                'z': float(center[1]),
                'n_faces': int(n_faces),
                'spread_x': float(spread[0]),
                'spread_z': float(spread[1]),
            })
    
    return candidates


def make_texture_samples_grid(rgb_colors, material_labels, surface_types):
    """Create a grid showing sample texture patches per material class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    wall_mask = surface_types == SURFACE_WALL
    
    for mat_id in range(len(MAT_NAMES)):
        ax = axes[mat_id]
        mask = wall_mask & (material_labels == mat_id)
        n = mask.sum()
        
        if n == 0:
            ax.set_facecolor('black')
            ax.set_title(f"{MAT_NAMES[mat_id]}\n(0 faces)", fontsize=10)
            ax.axis('off')
            continue
        
        # Show color distribution as scatter in RGB space
        sample_idx = np.where(mask)[0]
        if len(sample_idx) > 1000:
            sample_idx = np.random.choice(sample_idx, 1000, replace=False)
        
        colors = rgb_colors[sample_idx]
        
        # Create a color swatch grid
        grid_size = min(int(np.sqrt(len(colors))), 30)
        swatch = np.zeros((grid_size, grid_size, 3))
        for i in range(min(grid_size * grid_size, len(colors))):
            r, c = divmod(i, grid_size)
            if r < grid_size:
                swatch[r, c] = colors[i]
        
        ax.imshow(swatch)
        ax.set_title(f"{MAT_NAMES[mat_id]}\n({n} faces, {n*100/wall_mask.sum():.1f}%)", fontsize=10)
        ax.axis('off')
    
    fig.suptitle("Texture Samples by Material Class (Wall Faces)", fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / "texture_samples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_door_window_candidates(candidates, density, transform, rooms):
    """Plot door/window candidates on floor plan."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    d_vis = np.log1p(density)
    ax.imshow(d_vis, cmap='gray_r', origin='upper')
    
    x_min, z_min, res = transform
    
    # Room boundaries
    colors_room = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 1)))
    for r in rooms:
        contours, _ = cv2.findContours(r['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.reshape(-1, 2)
            ax.plot(pts[:, 0], pts[:, 1], color=colors_room[r['id'] % len(colors_room)], 
                    alpha=0.4, linewidth=1.5)
    
    # Plot candidates
    for cand in candidates.get('doors', []):
        px = (cand['x'] - x_min) / res
        pz = (cand['z'] - z_min) / res
        ax.plot(px, pz, 's', color='brown', markersize=12, markeredgecolor='black',
                markeredgewidth=2, label='Door' if cand == candidates['doors'][0] else '')
        ax.annotate(f"D({cand['n_faces']})", (px, pz), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    
    for cand in candidates.get('windows', []):
        px = (cand['x'] - x_min) / res
        pz = (cand['z'] - z_min) / res
        ax.plot(px, pz, 'D', color='deepskyblue', markersize=12, markeredgecolor='black',
                markeredgewidth=2, label='Window' if cand == candidates['windows'][0] else '')
        ax.annotate(f"W({cand['n_faces']})", (px, pz), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f"Door/Window Candidates from Texture\n"
                 f"{len(candidates.get('doors', []))} doors, {len(candidates.get('windows', []))} windows",
                 fontsize=13)
    ax.set_aspect('equal')
    fig.tight_layout()
    out_path = OUT_DIR / "door_window_candidates.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("mesh2plan v45 - Multimodal Stage 2: Texture Classification")
    print("=" * 60)
    
    # Step 1: Parse textured OBJ
    obj_path = SCAN_DIR / "textured_output.obj"
    vertices, uvs, face_v, face_vt = parse_obj_manual(obj_path)
    
    # Step 2: Compute face normals and centers
    print("\nComputing face normals and centers...")
    normals = compute_face_normals(vertices, face_v)
    centers = compute_face_centers(vertices, face_v)
    
    # Step 3: Classify surface type
    surface_types = classify_surface_type(normals)
    for st, name in [(SURFACE_FLOOR, 'floor'), (SURFACE_CEILING, 'ceiling'),
                     (SURFACE_WALL, 'wall'), (SURFACE_OTHER, 'other')]:
        print(f"  {name}: {(surface_types == st).sum()} faces")
    
    # Step 4: Load texture atlas and sample colors
    print("\nLoading texture atlas...")
    tex_path = SCAN_DIR / "textured_output.jpg"
    texture_img = cv2.imread(str(tex_path))
    print(f"  Texture: {texture_img.shape}")
    
    print("Sampling texture colors per face...")
    rgb_colors = sample_texture_colors(texture_img, uvs, face_vt)
    print(f"  Sampled {len(rgb_colors)} face colors")
    
    # Step 5: Classify wall materials
    print("\nClassifying wall materials...")
    hsv_colors = rgb_to_hsv_array(rgb_colors)
    material_labels = classify_wall_material(rgb_colors, hsv_colors)
    
    # Stats
    wall_mask = surface_types == SURFACE_WALL
    wall_mats = material_labels[wall_mask]
    for i, name in enumerate(MAT_NAMES):
        n = (wall_mats == i).sum()
        print(f"  {name}: {n} ({n*100/len(wall_mats):.1f}%)")
    
    # Step 6: Load rooms from Stage 1
    print("\nLoading room segmentation...")
    rooms, density, transform, mask = load_rooms_and_density()
    
    # Step 7: Build material map
    print("\nBuilding material map...")
    mat_img, mat_transform, mat_total = build_material_map(
        centers, surface_types, material_labels, transform)
    
    # Save material map
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: density
    ax = axes[0]
    d_vis = np.log1p(density)
    ax.imshow(d_vis, cmap='gray_r', origin='upper')
    ax.set_title("Point Density (Floor Plan)", fontsize=12)
    ax.set_aspect('equal')
    
    # Right: material map
    ax = axes[1]
    ax.imshow(mat_img, origin='upper')
    # Legend
    patches = [mpatches.Patch(color=MAT_COLORS[i], label=MAT_NAMES[i]) for i in range(len(MAT_NAMES))]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.set_title("Wall Material Classification", fontsize=12)
    ax.set_aspect('equal')
    
    fig.suptitle("Stage 2: Texture-Based Material Map", fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / "material_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    
    # Copy to workspace
    shutil.copy2(out_path, os.path.expanduser("~/.openclaw/workspace/v45_materials.png"))
    print("Copied material_map.png to workspace")
    
    # Step 8: Texture samples grid
    print("\nCreating texture samples grid...")
    make_texture_samples_grid(rgb_colors, material_labels, surface_types)
    
    # Step 9: Room material profiles
    print("\nComputing room material profiles...")
    room_ids = assign_faces_to_rooms(centers, rooms, transform)
    profiles = compute_room_material_profiles(room_ids, surface_types, material_labels)
    
    print("\nRoom material profiles:")
    for rid, prof in sorted(profiles.items()):
        print(f"  Room {rid}: {prof.get('inferred_type', '?')}")
        for name in MAT_NAMES:
            pct = prof.get(name, 0)
            if pct > 1:
                print(f"    {name}: {pct:.1f}%")
    
    with open(OUT_DIR / "room_materials.json", 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved: {OUT_DIR / 'room_materials.json'}")
    
    # Step 10: Door/window candidates
    print("\nFinding door/window candidates from texture...")
    candidates = find_door_window_candidates(centers, surface_types, material_labels)
    print(f"  Found {len(candidates['doors'])} door clusters, {len(candidates['windows'])} window clusters")
    
    for ctype in ['doors', 'windows']:
        for c in candidates[ctype]:
            print(f"    {ctype}: ({c['x']:.2f}, {c['z']:.2f}) n={c['n_faces']}")
    
    plot_door_window_candidates(candidates, density, transform, rooms)
    
    # Save candidates
    with open(OUT_DIR / "door_window_candidates.json", 'w') as f:
        json.dump(candidates, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Stage 2 complete! Outputs in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()

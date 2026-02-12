#!/usr/bin/env python3
"""
mesh2plan v56 - Camera Trajectory Room Segmentation

FUNDAMENTALLY DIFFERENT from all previous approaches.
Instead of analyzing walls, analyze the camera's PATH through the apartment.

When a person scans an apartment with a phone:
- They spend time in each room (looping, panning)
- They pass quickly through doorways/hallways
- Sharp turns often happen at room boundaries

Pipeline:
1. Load all 556 camera poses → trajectory in XZ
2. Compute speed + angular velocity at each frame
3. Slow + looping = "in a room"; fast + straight = "in transit"
4. Cluster trajectory segments by spatial proximity → room regions
5. For each room cluster, find the convex hull or bounding polygon
6. Use cross-section walls to refine room boundaries
"""

import numpy as np
import json
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


def load_trajectory(data_dir, max_frames=600):
    """Load camera positions from frame JSONs."""
    positions = []
    times = []
    for idx in range(max_frames):
        fp = data_dir / f'frame_{idx:05d}.json'
        if not fp.exists():
            continue
        frame = json.load(open(fp))
        pose = np.array(frame['cameraPoseARFrame']).reshape(4, 4, order='C')
        pos = pose[:3, 3]  # Camera position in world coords
        positions.append(pos)
        times.append(frame.get('time', idx * 0.033))
    
    return np.array(positions), np.array(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--mesh', default=None)
    parser.add_argument('--output-dir', default='results/v56')
    parser.add_argument('--resolution', type=float, default=0.02)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = args.resolution
    
    # ── Phase 1: Load trajectory ──
    print("Loading camera trajectory...")
    positions, times = load_trajectory(data_dir)
    print(f"  {len(positions)} frames")
    
    # XZ positions (top-down)
    xz = positions[:, [0, 2]]
    
    # ── Phase 2: Compute motion features ──
    # Speed (displacement between frames)
    dt = np.diff(times)
    dt[dt < 1e-6] = 1e-6
    displacements = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    speeds = displacements / dt
    
    # Angular velocity (change in heading)
    headings = np.arctan2(np.diff(xz[:, 1]), np.diff(xz[:, 0]))
    angular_vel = np.abs(np.diff(headings))
    angular_vel = np.minimum(angular_vel, 2*np.pi - angular_vel)  # wrap
    angular_vel_per_s = angular_vel / dt[:-1]
    
    # Dwell time estimation: how long camera stays near each position
    # Use a sliding window — count frames within 0.5m
    window = 30  # frames
    dwell_score = np.zeros(len(xz))
    for i in range(len(xz)):
        lo, hi = max(0, i-window), min(len(xz), i+window)
        dists = np.linalg.norm(xz[lo:hi] - xz[i], axis=1)
        dwell_score[i] = (dists < 0.5).sum()
    
    print(f"  Speed: {speeds.mean():.2f} ± {speeds.std():.2f} m/s")
    print(f"  Dwell score: {dwell_score.mean():.1f} ± {dwell_score.std():.1f}")
    
    # ── Phase 3: Classify frames as "in room" vs "in transit" ──
    # High dwell + low speed = in a room
    speed_padded = np.concatenate([[speeds[0]], speeds])
    is_room = (dwell_score > np.percentile(dwell_score, 30)) & (speed_padded < np.percentile(speed_padded, 80))
    
    room_pts = xz[is_room]
    print(f"  Room frames: {is_room.sum()}/{len(is_room)} ({is_room.mean()*100:.0f}%)")
    
    # ── Phase 4: Cluster room points ──
    # DBSCAN with eps=0.8m (rooms are separated by walls ~0.15m thick + doorway width)
    clustering = DBSCAN(eps=0.8, min_samples=15).fit(room_pts)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  {n_clusters} room clusters (DBSCAN)")
    
    # ── Phase 5: Build room polygons ──
    # For each cluster, compute convex hull in XZ
    rooms = []
    for c in range(n_clusters):
        pts = room_pts[labels == c]
        if len(pts) < 5:
            continue
        
        # Centroid
        cx, cz = pts.mean(axis=0)
        
        # Convex hull area
        try:
            hull = ConvexHull(pts)
            area = hull.volume  # 2D: volume = area
            hull_pts = pts[hull.vertices]
        except:
            continue
        
        if area < 1.0:
            continue
        
        # Bounding box for aspect ratio
        ptp = pts.max(axis=0) - pts.min(axis=0)
        aspect = max(ptp) / max(min(ptp), 0.1)
        
        if area > 8: name = "Room"
        elif aspect > 2.5: name = "Hallway"
        elif area < 3: name = "Closet"
        else: name = "Bathroom"
        
        rooms.append({
            'cx': cx, 'cz': cz,
            'area': area,
            'aspect': aspect,
            'hull_pts': hull_pts,
            'all_pts': pts,
            'n_frames': len(pts),
            'name': name,
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    
    # Deduplicate names
    name_counts = {}
    for r in rooms:
        n = r['name']
        name_counts[n] = name_counts.get(n, 0) + 1
        if name_counts[n] > 1:
            r['name'] = f"{n} {name_counts[n]}"
    
    total = sum(r['area'] for r in rooms)
    print(f"\n  {len(rooms)} rooms, {total:.1f}m² total")
    for r in rooms:
        print(f"    {r['name']}: {r['area']:.1f}m² ({r['n_frames']} frames, aspect={r['aspect']:.1f})")
    
    # ── Phase 6: Load cross-section for wall overlay ──
    xsection = None
    if args.mesh:
        print("\n  Loading mesh for cross-section overlay...")
        mesh = trimesh.load(args.mesh, process=False)
        verts = np.array(mesh.vertices)
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        y_range = y_max - y_min
        
        x_min_w = min(xz[:, 0].min(), verts[:, 0].min()) - 0.5
        x_max_w = max(xz[:, 0].max(), verts[:, 0].max()) + 0.5
        z_min_w = min(xz[:, 1].min(), verts[:, 2].min()) - 0.5
        z_max_w = max(xz[:, 1].max(), verts[:, 2].max()) + 0.5
        
        W_img = int((x_max_w - x_min_w) / res) + 1
        H_img = int((z_max_w - z_min_w) / res) + 1
        
        xsection = np.zeros((H_img, W_img), dtype=np.float32)
        slice_heights = np.linspace(y_min + 0.3*y_range, y_min + 0.8*y_range, 10)
        
        for y_h in slice_heights:
            try:
                section = mesh.section(plane_origin=[0, y_h, 0], plane_normal=[0, 1, 0])
                if section is None:
                    continue
            except:
                continue
            for entity in section.entities:
                pts_e = section.vertices[entity.points]
                for i in range(len(pts_e) - 1):
                    x0 = int((pts_e[i][0] - x_min_w) / res)
                    z0 = int((pts_e[i][2] - z_min_w) / res)
                    x1 = int((pts_e[i+1][0] - x_min_w) / res)
                    z1 = int((pts_e[i+1][2] - z_min_w) / res)
                    if (0 <= x0 < W_img and 0 <= z0 < H_img and
                        0 <= x1 < W_img and 0 <= z1 < H_img):
                        cv2.line(xsection, (x0, z0), (x1, z1), 1.0, thickness=1)
    
    # ── Phase 7: Render ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 1. Camera trajectory colored by dwell score
    ax = axes[0]
    ax.set_title('Camera Trajectory (color=dwell)')
    scatter = ax.scatter(xz[:, 0], xz[:, 1], c=dwell_score, cmap='YlOrRd', s=2, alpha=0.5)
    ax.plot(xz[:, 0], xz[:, 1], 'b-', alpha=0.1, linewidth=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Dwell score')
    
    # 2. Room clusters
    ax = axes[1]
    ax.set_title(f'Room Clusters ({n_clusters})')
    colors_cluster = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
    
    # Plot noise points gray
    noise = room_pts[labels == -1]
    if len(noise) > 0:
        ax.scatter(noise[:, 0], noise[:, 1], c='gray', s=1, alpha=0.2)
    
    for c in range(n_clusters):
        pts = room_pts[labels == c]
        ax.scatter(pts[:, 0], pts[:, 1], c=[colors_cluster[c % len(colors_cluster)]], s=3, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    
    # 3. Room polygons on cross-section
    ax = axes[2]
    ax.set_title(f'v56 Trajectory Rooms — {len(rooms)} rooms, {total:.1f}m²')
    
    if xsection is not None:
        ax.imshow(xsection, cmap='gray_r', origin='lower',
                  extent=[x_min_w, x_max_w, z_min_w, z_max_w], alpha=0.4)
    
    colors_room = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 3)))
    for i, r in enumerate(rooms):
        hull = r['hull_pts']
        hull_closed = np.vstack([hull, hull[0]])
        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color=colors_room[i], alpha=0.4)
        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                color=colors_room[i], linewidth=2)
        ax.text(r['cx'], r['cz'], f"{r['name']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    
    # Trajectory overlay
    ax.plot(xz[:, 0], xz[:, 1], 'k-', alpha=0.05, linewidth=0.3)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    
    # Scale bar
    ax.plot([-3.5, -2.5], [-4.5, -4.5], 'k-', linewidth=3)
    ax.text(-3.0, -4.3, '1m', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir / 'floorplan.png'}")


if __name__ == '__main__':
    main()

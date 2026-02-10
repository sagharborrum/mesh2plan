#!/usr/bin/env python3
"""
mesh2plan v3 — Improved classification + face-based RANSAC
Key improvements:
1. Use face normals instead of vertex-only RANSAC (preserves mesh structure)
2. Better wall detection by height filtering
3. Manhattan world detection (find dominant orthogonal directions)
4. Wall line fitting with proper extent from face geometry
"""

import time
import json
import numpy as np
import trimesh
from pathlib import Path

SCAN_DIR = Path("data/2026_01_13_14_47_59")
RESULTS_DIR = Path("results/v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_mesh(name):
    path = SCAN_DIR / name
    mesh = trimesh.load(str(path), process=True)
    print(f"Loaded {name}: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    print(f"  Bounds: {mesh.bounds[0].round(2)} → {mesh.bounds[1].round(2)}")
    print(f"  Size: {mesh.extents.round(2)}m")
    return mesh

def face_based_plane_detection(mesh, angle_threshold=8.0, min_area=0.5):
    """
    Group faces by normal similarity using region growing on the face adjacency graph.
    Returns plane groups with proper face membership.
    """
    print("\n--- Face-Based Plane Detection ---")
    t0 = time.time()
    
    cos_thresh = np.cos(np.radians(angle_threshold))
    normals = mesh.face_normals
    areas = mesh.area_faces
    n_faces = len(mesh.faces)
    
    # Build adjacency
    adj = {}
    for i, j in mesh.face_adjacency:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    
    visited = np.zeros(n_faces, dtype=bool)
    planes = []
    
    # Sort by area descending (start with biggest faces)
    order = np.argsort(-areas)
    
    for seed in order:
        if visited[seed]:
            continue
        
        seed_normal = normals[seed]
        region = [seed]
        visited[seed] = True
        queue = [seed]
        
        # Running average normal for the growing region
        avg_normal = seed_normal.copy()
        count = 1
        
        while queue:
            current = queue.pop(0)
            for nb in adj.get(current, []):
                if visited[nb]:
                    continue
                dot = np.dot(normals[nb], avg_normal / np.linalg.norm(avg_normal))
                if dot >= cos_thresh:
                    visited[nb] = True
                    region.append(nb)
                    queue.append(nb)
                    avg_normal = avg_normal + normals[nb]
                    count += 1
        
        total_area = areas[region].sum()
        if total_area < min_area:
            continue
        
        face_indices = np.array(region)
        final_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # Get all vertices of the region
        vert_indices = np.unique(mesh.faces[face_indices].flatten())
        pts = mesh.vertices[vert_indices]
        centroid = np.average(mesh.triangles_center[face_indices], weights=areas[face_indices], axis=0)
        
        # Compute height range
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        
        planes.append({
            'normal': final_normal,
            'centroid': centroid,
            'face_indices': face_indices,
            'vert_indices': vert_indices,
            'points': pts,
            'area': total_area,
            'n_faces': len(region),
            'y_range': (y_min, y_max),
            'height': y_max - y_min,
        })
    
    dt = time.time() - t0
    print(f"  Found {len(planes)} planar regions in {dt:.2f}s")
    print(f"  Total area covered: {sum(p['area'] for p in planes):.1f}m²")
    return planes, dt

def detect_up_axis(mesh):
    """Detect which axis is 'up' by finding the axis with most vertical normal variation."""
    normals = mesh.face_normals
    areas = mesh.area_faces
    
    # Weight normals by face area
    weighted = normals * areas[:, None]
    
    # The up axis should have the most area in faces with normals aligned to it
    axis_scores = []
    for ax in range(3):
        up = np.zeros(3)
        up[ax] = 1.0
        dots = np.abs(normals @ up)
        score = (dots * areas).sum()
        axis_scores.append(score)
    
    up_axis = np.argmax(axis_scores)
    axis_names = ['X', 'Y', 'Z']
    print(f"  Detected up axis: {axis_names[up_axis]} (scores: X={axis_scores[0]:.1f}, Y={axis_scores[1]:.1f}, Z={axis_scores[2]:.1f})")
    return up_axis

def classify_planes(planes, up_axis=1, wall_angle_tol=25, horiz_angle_tol=20):
    """
    Classify planes with better heuristics:
    - Walls: normal roughly perpendicular to up axis AND has significant height
    - Floor: normal points up, at lower elevation
    - Ceiling: normal points down, at upper elevation
    """
    up = np.zeros(3)
    up[up_axis] = 1.0
    
    cos_horiz = np.cos(np.radians(90 - wall_angle_tol))  # how horizontal the normal must be for a wall
    cos_vert = np.cos(np.radians(horiz_angle_tol))  # how vertical the normal must be for floor/ceiling
    
    # Find elevation range of the whole scene
    all_y = np.concatenate([p['points'][:, up_axis] for p in planes])
    y_min, y_max = all_y.min(), all_y.max()
    y_mid = (y_min + y_max) / 2
    total_height = y_max - y_min
    
    walls, floors, ceilings, others = [], [], [], []
    
    for p in planes:
        normal = p['normal']
        dot_up = abs(np.dot(normal, up))
        
        centroid_y = p['centroid'][up_axis]
        
        if dot_up < cos_horiz:
            # Normal is roughly horizontal → wall candidate
            # Additional check: wall should span significant height
            if p['height'] > total_height * 0.3:
                p['type'] = 'wall'
                walls.append(p)
            else:
                p['type'] = 'other'
                others.append(p)
        elif dot_up > cos_vert:
            # Normal is roughly vertical → floor or ceiling
            if np.dot(normal, up) > 0:
                p['type'] = 'floor'
                floors.append(p)
            else:
                p['type'] = 'ceiling'
                ceilings.append(p)
        else:
            p['type'] = 'other'
            others.append(p)
    
    return walls, floors, ceilings, others

def detect_manhattan_directions(walls, up_axis=1):
    """Find dominant orthogonal wall directions (Manhattan world assumption)."""
    if not walls:
        return []
    
    # Collect wall normals projected to horizontal plane
    normals_2d = []
    weights = []
    for w in walls:
        n = w['normal'].copy()
        n[up_axis] = 0
        n_len = np.linalg.norm(n)
        if n_len > 0.1:
            normals_2d.append(n / n_len)
            weights.append(w['area'])
    
    if not normals_2d:
        return []
    
    normals_2d = np.array(normals_2d)
    weights = np.array(weights)
    
    # Cluster normals into opposing pairs (n and -n are the same wall direction)
    # Use the angle to X axis
    h_axes = [0, 2] if up_axis == 1 else [0, 1]
    angles = np.arctan2(normals_2d[:, h_axes[1]], normals_2d[:, h_axes[0]])
    # Normalize to [0, pi) since n and -n are same direction
    angles = angles % np.pi
    
    # Simple clustering: find peaks in angle histogram
    from scipy.ndimage import gaussian_filter1d
    n_bins = 180
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, np.pi), weights=weights)
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(hist_smooth, height=hist_smooth.max() * 0.15, distance=10)
    
    directions = []
    for peak in peaks:
        angle = (bin_edges[peak] + bin_edges[peak+1]) / 2
        direction = np.zeros(3)
        direction[h_axes[0]] = np.cos(angle)
        direction[h_axes[1]] = np.sin(angle)
        directions.append({
            'angle_deg': np.degrees(angle),
            'direction': direction,
            'normal': np.cross(direction, np.array([0, 1, 0]) if up_axis == 1 else np.array([0, 0, 1])),
            'weight': hist_smooth[peak],
        })
    
    print(f"\n  Manhattan directions: {len(directions)}")
    for d in directions:
        print(f"    {d['angle_deg']:.1f}° (weight={d['weight']:.1f})")
    
    return directions

def extract_wall_lines(walls, up_axis=1):
    """Extract wall lines as 2D line segments in the horizontal plane."""
    h_axes = [0, 2] if up_axis == 1 else [0, 1]
    
    wall_lines = []
    for w in walls:
        pts = w['points']
        normal = w['normal']
        centroid = w['centroid']
        
        # Project normal to horizontal
        n_2d = np.array([normal[h_axes[0]], normal[h_axes[1]]])
        n_len = np.linalg.norm(n_2d)
        if n_len < 0.1:
            continue
        n_2d = n_2d / n_len
        
        # Wall direction (tangent)
        wall_dir = np.array([-n_2d[1], n_2d[0]])
        
        # Project points to 2D
        pts_2d = pts[:, h_axes]
        c_2d = np.array([centroid[h_axes[0]], centroid[h_axes[1]]])
        
        # Project onto wall direction
        projections = (pts_2d - c_2d) @ wall_dir
        p_min, p_max = projections.min(), projections.max()
        
        start = c_2d + wall_dir * p_min
        end = c_2d + wall_dir * p_max
        length = p_max - p_min
        
        if length < 0.3:  # Skip tiny walls
            continue
        
        wall_lines.append({
            'start': start,
            'end': end,
            'normal_2d': n_2d,
            'wall_dir': wall_dir,
            'centroid_2d': c_2d,
            'length': length,
            'height': w['height'],
            'area': w['area'],
            'n_faces': w['n_faces'],
        })
    
    return wall_lines

def merge_wall_lines(wall_lines, dist_thresh=0.25, angle_thresh=12):
    """Merge wall lines that belong to the same physical wall."""
    if not wall_lines:
        return []
    
    cos_thresh = np.cos(np.radians(angle_thresh))
    used = set()
    merged = []
    
    # Sort by area descending
    indexed = sorted(enumerate(wall_lines), key=lambda x: -x[1]['area'])
    
    for idx, wl in indexed:
        if idx in used:
            continue
        
        group = [wl]
        used.add(idx)
        
        for idx2, wl2 in indexed:
            if idx2 in used:
                continue
            
            dot = abs(np.dot(wl['normal_2d'], wl2['normal_2d']))
            if dot < cos_thresh:
                continue
            
            # Distance between wall planes
            diff = wl2['centroid_2d'] - wl['centroid_2d']
            dist = abs(np.dot(diff, wl['normal_2d']))
            if dist < dist_thresh:
                group.append(wl2)
                used.add(idx2)
        
        # Merge: recompute from all endpoints
        avg_normal = np.mean([g['normal_2d'] for g in group], axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        wall_dir = np.array([-avg_normal[1], avg_normal[0]])
        
        all_pts = np.array([g['start'] for g in group] + [g['end'] for g in group])
        centroid = all_pts.mean(axis=0)
        proj = (all_pts - centroid) @ wall_dir
        
        start = centroid + wall_dir * proj.min()
        end = centroid + wall_dir * proj.max()
        
        merged.append({
            'start': start,
            'end': end,
            'normal_2d': avg_normal,
            'wall_dir': wall_dir,
            'centroid_2d': centroid,
            'length': np.linalg.norm(end - start),
            'height': max(g['height'] for g in group),
            'area': sum(g['area'] for g in group),
            'n_faces': sum(g['n_faces'] for g in group),
            'merged_count': len(group),
        })
    
    # Sort by length
    merged.sort(key=lambda x: -x['length'])
    return merged

def generate_svg(wall_lines, floor_planes, mesh_bounds, output_path, up_axis=1, title="Floorplan"):
    """Generate SVG floorplan."""
    h_axes = [0, 2] if up_axis == 1 else [0, 1]
    
    min_h = [mesh_bounds[0][h_axes[0]], mesh_bounds[0][h_axes[1]]]
    max_h = [mesh_bounds[1][h_axes[0]], mesh_bounds[1][h_axes[1]]]
    
    pad = 1.5
    scale = 60
    svg_w = (max_h[0] - min_h[0] + 2*pad) * scale
    svg_h = (max_h[1] - min_h[1] + 2*pad) * scale
    
    def tx(v): return (v - min_h[0] + pad) * scale
    def ty(v): return (v - min_h[1] + pad) * scale
    
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}" width="{svg_w:.0f}" height="{svg_h:.0f}">',
        '<defs><style>',
        '  text { font-family: "SF Mono", monospace; fill: #999; font-size: 10px; }',
        '  .dim { font-size: 11px; fill: #ff6b35; font-weight: bold; }',
        '  .title { font-size: 16px; fill: #e2e8f0; font-weight: bold; }',
        '  .info { font-size: 10px; fill: #666; }',
        '</style></defs>',
        f'<rect width="100%" height="100%" fill="#0a0a14"/>',
    ]
    
    # Grid (1m spacing)
    for x in np.arange(np.floor(min_h[0]), np.ceil(max_h[0]) + 1, 1.0):
        lines.append(f'<line x1="{tx(x):.0f}" y1="0" x2="{tx(x):.0f}" y2="{svg_h:.0f}" stroke="#151520" stroke-width="0.5"/>')
    for z in np.arange(np.floor(min_h[1]), np.ceil(max_h[1]) + 1, 1.0):
        lines.append(f'<line x1="0" y1="{ty(z):.0f}" x2="{svg_w:.0f}" y2="{ty(z):.0f}" stroke="#151520" stroke-width="0.5"/>')
    
    # Floor fills
    for fp in floor_planes:
        pts_2d = fp['points'][:, h_axes]
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts_2d)
            hp = pts_2d[hull.vertices]
            path = " ".join([f"{'M' if i==0 else 'L'}{tx(p[0]):.1f},{ty(p[1]):.1f}" for i, p in enumerate(hp)])
            lines.append(f'<path d="{path} Z" fill="#0d1a0d" stroke="#1a3a1a" stroke-width="0.5" opacity="0.6"/>')
        except:
            pass
    
    # Wall lines (thick)
    colors = ['#ff6b35', '#00d4ff', '#ff3366', '#66ff99', '#ffcc00', '#cc66ff', '#ff9966', '#66ccff',
              '#ff4488', '#44ffcc', '#ffaa00', '#aa66ff']
    
    for i, wl in enumerate(wall_lines):
        color = colors[i % len(colors)]
        x1, y1 = tx(wl['start'][0]), ty(wl['start'][1])
        x2, y2 = tx(wl['end'][0]), ty(wl['end'][1])
        
        # Wall thickness
        w = max(3, min(8, wl['area'] / 2))
        
        # Wall line
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" opacity="0.9"/>')
        
        # Endpoints
        lines.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="3" fill="{color}" opacity="0.6"/>')
        lines.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="3" fill="{color}" opacity="0.6"/>')
        
        # Dimension
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Offset label perpendicular to wall
        offset = 14
        nx = wl['normal_2d'][0] * offset
        ny = wl['normal_2d'][1] * offset
        
        mc = wl.get('merged_count', 1)
        label = f"{wl['length']:.2f}m"
        if mc > 1:
            label += f" ({mc})"
        
        lines.append(f'<text x="{mid_x + nx:.0f}" y="{mid_y + ny:.0f}" class="dim" text-anchor="middle" dominant-baseline="middle">{label}</text>')
    
    # Title + stats
    lines.append(f'<text x="12" y="24" class="title">{title}</text>')
    total_wall = sum(wl['length'] for wl in wall_lines)
    lines.append(f'<text x="12" y="42" class="info">{len(wall_lines)} walls, total length: {total_wall:.1f}m</text>')
    
    # Scale bar
    bar_x = 12
    bar_y = svg_h - 20
    bar_len = scale  # 1 meter
    lines.append(f'<line x1="{bar_x}" y1="{bar_y}" x2="{bar_x + bar_len}" y2="{bar_y}" stroke="white" stroke-width="2"/>')
    lines.append(f'<line x1="{bar_x}" y1="{bar_y-4}" x2="{bar_x}" y2="{bar_y+4}" stroke="white" stroke-width="1.5"/>')
    lines.append(f'<line x1="{bar_x+bar_len}" y1="{bar_y-4}" x2="{bar_x+bar_len}" y2="{bar_y+4}" stroke="white" stroke-width="1.5"/>')
    lines.append(f'<text x="{bar_x + bar_len/2}" y="{bar_y - 8}" text-anchor="middle" fill="white" font-size="11px">1m</text>')
    
    lines.append('</svg>')
    output_path.write_text('\n'.join(lines))
    print(f"  Saved: {output_path}")

def export_colored_ply(mesh, planes, output_path):
    """Export colored PLY with plane assignments."""
    face_colors = np.full((len(mesh.faces), 4), [60, 60, 60, 255], dtype=np.uint8)
    
    wall_colors = [
        [255, 107, 53], [0, 212, 255], [255, 51, 102], [102, 255, 153],
        [255, 204, 0], [204, 102, 255], [255, 153, 102], [102, 204, 255],
    ]
    
    wi = 0
    for p in planes:
        t = p.get('type', 'other')
        if t == 'wall':
            c = wall_colors[wi % len(wall_colors)]
            wi += 1
        elif t == 'floor':
            c = [40, 180, 40]
        elif t == 'ceiling':
            c = [40, 40, 180]
        else:
            c = [100, 100, 100]
        
        face_colors[p['face_indices']] = [c[0], c[1], c[2], 255]
    
    colored = mesh.copy()
    colored.visual.face_colors = face_colors
    colored.export(str(output_path))
    print(f"  Saved: {output_path}")

def main():
    print("mesh2plan v3 — Improved Pipeline")
    print("=" * 60)
    
    mesh = load_mesh("export.obj")
    
    # Detect up axis
    up_axis = detect_up_axis(mesh)
    
    # Face-based plane detection
    planes, dt_detect = face_based_plane_detection(mesh, angle_threshold=8.0, min_area=0.3)
    
    # Classify
    walls, floors, ceilings, others = classify_planes(planes, up_axis=up_axis)
    
    print(f"\n  Classification:")
    print(f"    Walls:    {len(walls):>4} (area: {sum(w['area'] for w in walls):.1f}m²)")
    print(f"    Floors:   {len(floors):>4} (area: {sum(f['area'] for f in floors):.1f}m²)")
    print(f"    Ceilings: {len(ceilings):>4} (area: {sum(c['area'] for c in ceilings):.1f}m²)")
    print(f"    Other:    {len(others):>4} (area: {sum(o['area'] for o in others):.1f}m²)")
    
    print(f"\n  Top walls by area:")
    for i, w in enumerate(sorted(walls, key=lambda x: -x['area'])[:10]):
        print(f"    Wall {i}: area={w['area']:.2f}m², height={w['height']:.2f}m, normal={w['normal'].round(3)}, faces={w['n_faces']:,}")
    
    # Manhattan directions
    directions = detect_manhattan_directions(walls, up_axis)
    
    # Extract wall lines
    print("\n--- Wall Line Extraction ---")
    wall_lines = extract_wall_lines(walls, up_axis)
    print(f"  Raw wall lines: {len(wall_lines)}")
    
    merged = merge_wall_lines(wall_lines, dist_thresh=0.3, angle_thresh=15)
    print(f"  After merging: {len(merged)}")
    
    for i, wl in enumerate(merged[:15]):
        mc = wl.get('merged_count', 1)
        print(f"    Wall {i}: {wl['length']:.2f}m × {wl['height']:.2f}m, area={wl['area']:.1f}m², faces={wl['n_faces']:,}" + (f" (merged {mc})" if mc > 1 else ""))
    
    # Generate floorplan
    print("\n--- SVG Floorplan ---")
    # Filter to significant walls only (>1m or >1m² area)
    significant = [wl for wl in merged if wl['length'] > 0.8 or wl['area'] > 1.0]
    print(f"  Significant walls (>0.8m or >1m²): {len(significant)}")
    
    generate_svg(significant, floors, mesh.bounds, RESULTS_DIR / "floorplan.svg",
                up_axis=up_axis,
                title=f"mesh2plan v3 — {len(significant)} walls detected")
    
    # Also all walls
    generate_svg(merged, floors, mesh.bounds, RESULTS_DIR / "floorplan_all.svg",
                up_axis=up_axis,
                title=f"mesh2plan v3 (all) — {len(merged)} walls")
    
    # Export colored mesh
    print("\n--- Colored Mesh ---")
    export_colored_ply(mesh, planes, RESULTS_DIR / "colored.ply")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Scene size: {mesh.extents.round(2)}m")
    print(f"  Detection time: {dt_detect:.2f}s")
    print(f"  Walls: {len(significant)} significant ({len(merged)} total)")
    print(f"  Total wall length: {sum(wl['length'] for wl in significant):.1f}m")
    print(f"  Floors: {len(floors)}, Ceilings: {len(ceilings)}")
    if directions:
        print(f"  Manhattan directions: {[f'{d['angle_deg']:.0f}°' for d in directions]}")
    print(f"\n  Output: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
mesh2plan v2 — Better visualization + wall boundary extraction
Renders colored plane overlays and a proper 2D floorplan with wall lines.
"""

import time
import json
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict

SCAN_DIR = Path("data/2026_01_13_14_47_59")
RESULTS_DIR = Path("results/v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_mesh(name):
    path = SCAN_DIR / name
    mesh = trimesh.load(str(path), process=False)
    print(f"Loaded {name}: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    print(f"  Bounds: {mesh.bounds[0].round(2)} → {mesh.bounds[1].round(2)}")
    print(f"  Extents: {mesh.extents.round(2)} meters")
    return mesh

def ransac_planes(points, max_planes=30, threshold=0.02, min_inliers=100):
    """Sequential RANSAC with better parameters."""
    import pyransac3d as pyrsc
    
    remaining = np.arange(len(points))
    planes = []
    
    for i in range(max_planes):
        if len(remaining) < min_inliers:
            break
        plane = pyrsc.Plane()
        eq, inliers = plane.fit(points[remaining], thresh=threshold, maxIteration=2000)
        if len(inliers) < min_inliers:
            break
        
        actual_inliers = remaining[inliers]
        normal = np.array(eq[:3])
        normal = normal / np.linalg.norm(normal)
        
        plane_points = points[actual_inliers]
        centroid = plane_points.mean(axis=0)
        
        # Compute plane extent (bounding box in plane-local coords)
        # Project points onto plane axes
        if abs(normal[1]) > 0.9:  # Horizontal plane
            ax1 = np.array([1, 0, 0])
        else:
            ax1 = np.cross(normal, [0, 1, 0])
            ax1 = ax1 / np.linalg.norm(ax1)
        ax2 = np.cross(normal, ax1)
        ax2 = ax2 / np.linalg.norm(ax2)
        
        local = plane_points - centroid
        proj1 = local @ ax1
        proj2 = local @ ax2
        
        extent1 = proj1.max() - proj1.min()
        extent2 = proj2.max() - proj2.min()
        area = extent1 * extent2
        
        planes.append({
            'equation': eq,
            'normal': normal,
            'centroid': centroid,
            'inlier_indices': actual_inliers,
            'n_inliers': len(inliers),
            'extent': (extent1, extent2),
            'area': area,
            'ax1': ax1, 'ax2': ax2,
            'proj1_range': (proj1.min(), proj1.max()),
            'proj2_range': (proj2.min(), proj2.max()),
            'points': plane_points,
        })
        
        remaining = np.delete(remaining, inliers)
    
    return planes

def classify_plane(normal, up_thresh=15):
    """Classify a plane by its normal direction."""
    cos_t = np.cos(np.radians(up_thresh))
    dot_up = normal[1]  # Y is up
    
    if dot_up > cos_t:
        return 'floor'
    elif dot_up < -cos_t:
        return 'ceiling'
    elif abs(dot_up) < np.sin(np.radians(up_thresh + 15)):
        return 'wall'
    else:
        return 'other'

def extract_wall_boundaries(wall, points_2d, resolution=0.05):
    """Extract the 2D boundary of a wall's footprint using alpha shapes."""
    from scipy.spatial import ConvexHull
    
    if len(points_2d) < 3:
        return None
    
    try:
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]
    except:
        return None

def find_wall_lines(walls, mesh_points):
    """
    For each wall, project its points onto the XZ plane and find 
    the wall line (intersection of wall plane with floor).
    """
    wall_lines = []
    
    for wall in walls:
        pts = wall['points']
        normal = wall['normal']
        centroid = wall['centroid']
        
        # Project wall points to XZ (top-down view)
        xz_points = pts[:, [0, 2]]  # X, Z
        
        # The wall line direction in XZ is perpendicular to the wall normal projected to XZ
        n_xz = np.array([normal[0], normal[2]])
        n_xz_len = np.linalg.norm(n_xz)
        if n_xz_len < 0.1:
            continue  # Nearly horizontal normal, skip
        n_xz = n_xz / n_xz_len
        
        # Wall direction (tangent) in XZ
        wall_dir = np.array([-n_xz[1], n_xz[0]])
        
        # Project all XZ points onto wall direction to find extent
        centered = xz_points - np.array([centroid[0], centroid[2]])
        projections = centered @ wall_dir
        
        p_min, p_max = projections.min(), projections.max()
        
        # Wall line endpoints
        c_xz = np.array([centroid[0], centroid[2]])
        start = c_xz + wall_dir * p_min
        end = c_xz + wall_dir * p_max
        
        # Height range
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        height = y_max - y_min
        length = p_max - p_min
        
        wall_lines.append({
            'start': start,
            'end': end,
            'normal_xz': n_xz,
            'centroid_xz': c_xz,
            'length': length,
            'height': height,
            'y_range': (y_min, y_max),
            'n_points': len(pts),
            'wall': wall,
        })
    
    return wall_lines

def merge_coplanar_walls(wall_lines, dist_threshold=0.15, angle_threshold=10):
    """Merge wall lines that are nearly coplanar (same orientation, close distance)."""
    if not wall_lines:
        return []
    
    cos_thresh = np.cos(np.radians(angle_threshold))
    merged = []
    used = set()
    
    for i, wl in enumerate(wall_lines):
        if i in used:
            continue
        
        group = [wl]
        used.add(i)
        
        for j, wl2 in enumerate(wall_lines):
            if j in used:
                continue
            
            # Check if normals are similar
            dot = abs(np.dot(wl['normal_xz'], wl2['normal_xz']))
            if dot < cos_thresh:
                continue
            
            # Check distance between wall planes
            diff = wl2['centroid_xz'] - wl['centroid_xz']
            dist = abs(np.dot(diff, wl['normal_xz']))
            if dist < dist_threshold:
                group.append(wl2)
                used.add(j)
        
        # Merge group: combine all points and recompute line
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Average normal
            avg_normal = np.mean([g['normal_xz'] for g in group], axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            wall_dir = np.array([-avg_normal[1], avg_normal[0]])
            
            # Combine all endpoints
            all_starts = [g['start'] for g in group]
            all_ends = [g['end'] for g in group]
            all_pts = np.array(all_starts + all_ends)
            
            centroid = all_pts.mean(axis=0)
            projections = (all_pts - centroid) @ wall_dir
            
            start = centroid + wall_dir * projections.min()
            end = centroid + wall_dir * projections.max()
            
            merged.append({
                'start': start,
                'end': end,
                'normal_xz': avg_normal,
                'centroid_xz': centroid,
                'length': np.linalg.norm(end - start),
                'height': max(g['height'] for g in group),
                'y_range': (min(g['y_range'][0] for g in group), max(g['y_range'][1] for g in group)),
                'n_points': sum(g['n_points'] for g in group),
                'merged_count': len(group),
            })
    
    return merged

def generate_floorplan_svg(wall_lines, floor_planes, mesh_bounds, output_path, title="Floorplan"):
    """Generate a proper 2D floorplan SVG with wall lines and dimensions."""
    
    min_x, min_z = mesh_bounds[0][0], mesh_bounds[0][2]
    max_x, max_z = mesh_bounds[1][0], mesh_bounds[1][2]
    padding = 1.0
    
    scale = 80  # px per meter
    svg_w = (max_x - min_x + 2*padding) * scale
    svg_h = (max_z - min_z + 2*padding) * scale
    
    def tx(x): return (x - min_x + padding) * scale
    def tz(z): return (z - min_z + padding) * scale
    
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}" width="{svg_w:.0f}" height="{svg_h:.0f}">',
        '<defs>',
        '  <style>',
        '    text { font-family: monospace; fill: #888; font-size: 10px; }',
        '    .dim { font-size: 9px; fill: #ff6b35; }',
        '    .title { font-size: 14px; fill: #e2e8f0; font-weight: bold; }',
        '  </style>',
        '</defs>',
        f'<rect width="100%" height="100%" fill="#0f0f1a"/>',
        f'<text x="10" y="20" class="title">{title}</text>',
    ]
    
    # Draw grid
    for x in np.arange(np.floor(min_x), np.ceil(max_x) + 1, 1.0):
        lines.append(f'<line x1="{tx(x):.0f}" y1="0" x2="{tx(x):.0f}" y2="{svg_h:.0f}" stroke="#1a1a2e" stroke-width="0.5"/>')
    for z in np.arange(np.floor(min_z), np.ceil(max_z) + 1, 1.0):
        lines.append(f'<line x1="0" y1="{tz(z):.0f}" x2="{svg_w:.0f}" y2="{tz(z):.0f}" stroke="#1a1a2e" stroke-width="0.5"/>')
    
    # Draw floor polygons (if any)
    for fp in floor_planes:
        pts = fp['points'][:, [0, 2]]
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            path = " ".join([f"{'M' if i==0 else 'L'}{tx(p[0]):.1f},{tz(p[1]):.1f}" for i, p in enumerate(hull_pts)])
            lines.append(f'<path d="{path} Z" fill="#1a2a1a" stroke="#2a4a2a" stroke-width="1" opacity="0.5"/>')
        except:
            pass
    
    # Draw wall lines
    colors = ['#ff6b35', '#00d4ff', '#ff3366', '#66ff66', '#ffcc00', '#cc66ff', '#ff9966', '#66ccff']
    
    for i, wl in enumerate(wall_lines):
        color = colors[i % len(colors)]
        x1, z1 = tx(wl['start'][0]), tz(wl['start'][1])
        x2, z2 = tx(wl['end'][0]), tz(wl['end'][1])
        
        # Wall thickness based on number of points
        width = max(2, min(6, wl['n_points'] / 200))
        
        lines.append(f'<line x1="{x1:.1f}" y1="{z1:.1f}" x2="{x2:.1f}" y2="{z2:.1f}" stroke="{color}" stroke-width="{width:.1f}" stroke-linecap="round"/>')
        
        # Draw normal indicator
        mid_x = (x1 + x2) / 2
        mid_z = (z1 + z2) / 2
        n_len = 15
        nx = mid_x + wl['normal_xz'][0] * n_len
        nz = mid_z + wl['normal_xz'][1] * n_len
        lines.append(f'<line x1="{mid_x:.1f}" y1="{mid_z:.1f}" x2="{nx:.1f}" y2="{nz:.1f}" stroke="{color}" stroke-width="1" opacity="0.4" stroke-dasharray="3,2"/>')
        
        # Dimension label
        length = wl['length']
        label_x = (x1 + x2) / 2
        label_z = (z1 + z2) / 2 - 8
        merged = wl.get('merged_count', 1)
        label = f"{length:.2f}m" + (f" [{merged}]" if merged > 1 else "")
        lines.append(f'<text x="{label_x:.0f}" y="{label_z:.0f}" class="dim" text-anchor="middle">{label}</text>')
    
    # Scale bar
    bar_x = tx(min_x + padding)
    bar_y = svg_h - 30
    bar_len = 1.0 * scale  # 1 meter
    lines.append(f'<line x1="{bar_x:.0f}" y1="{bar_y:.0f}" x2="{bar_x + bar_len:.0f}" y2="{bar_y:.0f}" stroke="white" stroke-width="2"/>')
    lines.append(f'<text x="{bar_x + bar_len/2:.0f}" y="{bar_y + 15:.0f}" text-anchor="middle" fill="white" font-size="11px">1 meter</text>')
    
    # Stats
    lines.append(f'<text x="{svg_w - 10:.0f}" y="20" text-anchor="end" fill="#666" font-size="10px">{len(wall_lines)} walls detected</text>')
    
    lines.append('</svg>')
    output_path.write_text('\n'.join(lines))
    print(f"  Saved: {output_path}")

def export_colored_obj(mesh, planes, output_path):
    """Export mesh with planes colored for visualization."""
    
    # Assign colors to faces based on plane membership
    face_colors = np.ones((len(mesh.faces), 4)) * 0.3  # dark gray default
    face_colors[:, 3] = 1.0
    
    palette = [
        [1.0, 0.42, 0.21, 1.0],  # orange - wall
        [0.0, 0.83, 1.0, 1.0],   # cyan - wall
        [1.0, 0.2, 0.4, 1.0],    # pink - wall
        [0.4, 1.0, 0.4, 1.0],    # green - floor
        [0.4, 0.4, 1.0, 1.0],    # blue - ceiling
        [1.0, 0.8, 0.0, 1.0],    # yellow
        [0.8, 0.4, 1.0, 1.0],    # purple
        [1.0, 0.6, 0.4, 1.0],    # salmon
    ]
    
    # For each plane, find which faces contain its inlier vertices
    for pi, plane in enumerate(planes):
        ptype = classify_plane(plane['normal'])
        if ptype == 'floor':
            color = [0.2, 0.8, 0.2, 1.0]
        elif ptype == 'ceiling':
            color = [0.2, 0.2, 0.8, 1.0]
        elif ptype == 'wall':
            color = palette[pi % len(palette)]
        else:
            color = [0.5, 0.5, 0.5, 1.0]
        
        inlier_set = set(plane['inlier_indices'].tolist())
        for fi, face in enumerate(mesh.faces):
            if any(v in inlier_set for v in face):
                face_colors[fi] = color
    
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)
    colored_mesh.export(str(output_path))
    print(f"  Saved colored mesh: {output_path}")

def main():
    print("mesh2plan v2 — Wall Extraction + Floorplan Generation")
    print("=" * 60)
    
    # Use refined mesh for faster iteration
    mesh = load_mesh("export_refined.obj")
    
    # RANSAC with tuned params
    print("\n--- RANSAC Plane Detection ---")
    t0 = time.time()
    planes = ransac_planes(mesh.vertices, max_planes=30, threshold=0.025, min_inliers=80)
    dt = time.time() - t0
    print(f"Found {len(planes)} planes in {dt:.2f}s")
    
    # Classify
    walls, floors, ceilings, others = [], [], [], []
    for p in planes:
        ptype = classify_plane(p['normal'])
        p['type'] = ptype
        if ptype == 'wall': walls.append(p)
        elif ptype == 'floor': floors.append(p)
        elif ptype == 'ceiling': ceilings.append(p)
        else: others.append(p)
    
    print(f"\nClassification: {len(walls)} walls, {len(floors)} floors, {len(ceilings)} ceilings, {len(others)} other")
    
    for i, w in enumerate(walls):
        print(f"  Wall {i}: normal={w['normal'].round(3)}, area={w['area']:.1f}m², {w['n_inliers']:,} pts, extent={w['extent'][0]:.1f}x{w['extent'][1]:.1f}m")
    for i, f in enumerate(floors):
        print(f"  Floor {i}: normal={f['normal'].round(3)}, area={f['area']:.1f}m², {f['n_inliers']:,} pts")
    
    # Extract wall lines
    print("\n--- Wall Line Extraction ---")
    wall_lines = find_wall_lines(walls, mesh.vertices)
    print(f"Extracted {len(wall_lines)} wall lines")
    
    # Merge coplanar walls
    merged = merge_coplanar_walls(wall_lines, dist_threshold=0.2, angle_threshold=12)
    print(f"After merging: {len(merged)} wall lines")
    for i, wl in enumerate(merged):
        mc = wl.get('merged_count', 1)
        print(f"  Wall {i}: length={wl['length']:.2f}m, height={wl['height']:.2f}m, pts={wl['n_points']:,}" + (f" (merged {mc})" if mc > 1 else ""))
    
    # Generate SVG floorplan
    print("\n--- Floorplan Generation ---")
    generate_floorplan_svg(merged, floors, mesh.bounds, 
                          RESULTS_DIR / "floorplan.svg",
                          title=f"mesh2plan — {len(merged)} walls, {len(floors)} floors")
    
    # Also generate unmerged version
    generate_floorplan_svg(wall_lines, floors, mesh.bounds,
                          RESULTS_DIR / "floorplan_unmerged.svg", 
                          title=f"mesh2plan (unmerged) — {len(wall_lines)} walls")
    
    # Export colored mesh
    print("\n--- Colored Mesh Export ---")
    export_colored_obj(mesh, planes, RESULTS_DIR / "colored_planes.ply")
    
    # Now test on full-res mesh
    print("\n\n" + "=" * 60)
    print("FULL RESOLUTION TEST")
    mesh_full = load_mesh("export.obj")
    
    t0 = time.time()
    planes_full = ransac_planes(mesh_full.vertices, max_planes=30, threshold=0.02, min_inliers=200)
    dt_full = time.time() - t0
    print(f"Found {len(planes_full)} planes in {dt_full:.2f}s")
    
    walls_f, floors_f = [], []
    for p in planes_full:
        ptype = classify_plane(p['normal'])
        p['type'] = ptype
        if ptype == 'wall': walls_f.append(p)
        elif ptype == 'floor': floors_f.append(p)
    
    print(f"Classification: {len(walls_f)} walls, {len(floors_f)} floors")
    
    wall_lines_f = find_wall_lines(walls_f, mesh_full.vertices)
    merged_f = merge_coplanar_walls(wall_lines_f, dist_threshold=0.2, angle_threshold=12)
    
    generate_floorplan_svg(merged_f, floors_f, mesh_full.bounds,
                          RESULTS_DIR / "floorplan_fullres.svg",
                          title=f"mesh2plan (full-res) — {len(merged_f)} walls")
    
    export_colored_obj(mesh_full, planes_full, RESULTS_DIR / "colored_planes_fullres.ply")
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"  Refined mesh: {len(merged)} walls, {len(floors)} floors")
    print(f"  Full-res:     {len(merged_f)} walls, {len(floors_f)} floors")
    print(f"\nOutput files in {RESULTS_DIR}/:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    main()

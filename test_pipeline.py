#!/usr/bin/env python3
"""
mesh2plan test pipeline
Tests plane detection approaches on a LiDAR scan mesh.
"""

import time
import json
import numpy as np
import trimesh
from pathlib import Path

SCAN_DIR = Path("data/2026_01_13_14_47_59")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Load meshes ──────────────────────────────────────────────
def load_mesh(name):
    path = SCAN_DIR / name
    print(f"\n{'='*60}")
    print(f"Loading: {name}")
    t0 = time.time()
    mesh = trimesh.load(str(path), process=False)
    dt = time.time() - t0
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces:    {len(mesh.faces):,}")
    print(f"  Bounds:   {mesh.bounds[0]} → {mesh.bounds[1]}")
    print(f"  Size:     {mesh.extents}")
    print(f"  Loaded in {dt:.2f}s")
    return mesh

# ─── Approach 1: RANSAC (pyransac3d) ─────────────────────────
def test_ransac(mesh, max_planes=20, threshold=0.02):
    """Sequential RANSAC plane fitting on mesh vertices."""
    import pyransac3d as pyrsc
    
    print(f"\n{'='*60}")
    print(f"APPROACH 1: Sequential RANSAC (pyransac3d)")
    print(f"  threshold={threshold}, max_planes={max_planes}")
    
    points = mesh.vertices.copy()
    remaining = np.arange(len(points))
    planes = []
    
    t0 = time.time()
    for i in range(max_planes):
        if len(remaining) < 100:
            break
        
        plane = pyrsc.Plane()
        eq, inliers = plane.fit(points[remaining], thresh=threshold, maxIteration=1000)
        
        if len(inliers) < 50:
            break
        
        # Get actual indices
        actual_inliers = remaining[inliers]
        normal = np.array(eq[:3])
        normal = normal / np.linalg.norm(normal)
        
        planes.append({
            'equation': eq,
            'normal': normal.tolist(),
            'n_inliers': len(inliers),
            'inlier_indices': actual_inliers.tolist(),
            'centroid': points[actual_inliers].mean(axis=0).tolist(),
        })
        
        # Remove inliers
        remaining = np.delete(remaining, inliers)
        print(f"  Plane {i}: normal={normal.round(3)}, inliers={len(inliers):,}, remaining={len(remaining):,}")
    
    dt = time.time() - t0
    print(f"  Found {len(planes)} planes in {dt:.2f}s")
    print(f"  Points explained: {sum(p['n_inliers'] for p in planes):,}/{len(mesh.vertices):,} ({sum(p['n_inliers'] for p in planes)/len(mesh.vertices)*100:.1f}%)")
    
    return planes, dt

# ─── Approach 2: Normal Clustering ───────────────────────────
def test_normal_clustering(mesh, n_clusters=10):
    """Cluster faces by normal direction, then fit planes to clusters."""
    from sklearn.cluster import KMeans
    
    print(f"\n{'='*60}")
    print(f"APPROACH 2: Normal Clustering (KMeans)")
    print(f"  n_clusters={n_clusters}")
    
    t0 = time.time()
    
    # Get face normals
    face_normals = mesh.face_normals
    
    # Cluster by normal direction
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(face_normals)
    
    planes = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_faces = mask.sum()
        if n_faces < 10:
            continue
        
        # Get vertices of faces in this cluster
        face_indices = np.where(mask)[0]
        cluster_faces = mesh.faces[face_indices]
        vertex_indices = np.unique(cluster_faces.flatten())
        cluster_points = mesh.vertices[vertex_indices]
        
        # Average normal
        avg_normal = face_normals[mask].mean(axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # Centroid
        centroid = cluster_points.mean(axis=0)
        
        planes.append({
            'normal': avg_normal.tolist(),
            'centroid': centroid.tolist(),
            'n_faces': int(n_faces),
            'n_vertices': len(vertex_indices),
        })
        print(f"  Cluster {cluster_id}: normal={avg_normal.round(3)}, faces={n_faces:,}")
    
    dt = time.time() - t0
    print(f"  Found {len(planes)} clusters in {dt:.2f}s")
    
    return planes, dt

# ─── Approach 3: Region Growing ──────────────────────────────
def test_region_growing(mesh, angle_threshold=10.0, min_region_size=50):
    """Region growing from seed faces based on normal similarity."""
    
    print(f"\n{'='*60}")
    print(f"APPROACH 3: Region Growing")
    print(f"  angle_threshold={angle_threshold}°, min_region_size={min_region_size}")
    
    t0 = time.time()
    
    cos_threshold = np.cos(np.radians(angle_threshold))
    face_normals = mesh.face_normals
    n_faces = len(mesh.faces)
    
    # Build face adjacency
    # trimesh gives us face adjacency
    adjacency = mesh.face_adjacency
    adj_dict = {}
    for i, j in adjacency:
        adj_dict.setdefault(i, []).append(j)
        adj_dict.setdefault(j, []).append(i)
    
    visited = np.zeros(n_faces, dtype=bool)
    regions = []
    
    # Sort faces by area (start from largest)
    face_areas = mesh.area_faces
    sorted_faces = np.argsort(-face_areas)
    
    for seed in sorted_faces:
        if visited[seed]:
            continue
        
        # Grow region from seed
        region = [seed]
        visited[seed] = True
        queue = [seed]
        seed_normal = face_normals[seed]
        
        while queue:
            current = queue.pop(0)
            for neighbor in adj_dict.get(current, []):
                if visited[neighbor]:
                    continue
                # Check normal similarity
                dot = np.dot(face_normals[neighbor], seed_normal)
                if dot >= cos_threshold:
                    visited[neighbor] = True
                    region.append(neighbor)
                    queue.append(neighbor)
        
        if len(region) >= min_region_size:
            face_indices = np.array(region)
            avg_normal = face_normals[face_indices].mean(axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            
            vertex_indices = np.unique(mesh.faces[face_indices].flatten())
            centroid = mesh.vertices[vertex_indices].mean(axis=0)
            
            regions.append({
                'normal': avg_normal.tolist(),
                'centroid': centroid.tolist(),
                'n_faces': len(region),
                'n_vertices': len(vertex_indices),
            })
    
    dt = time.time() - t0
    print(f"  Found {len(regions)} regions in {dt:.2f}s")
    for i, r in enumerate(regions[:15]):
        print(f"  Region {i}: normal={np.array(r['normal']).round(3)}, faces={r['n_faces']:,}")
    if len(regions) > 15:
        print(f"  ... and {len(regions)-15} more")
    
    return regions, dt

# ─── Classify planes as wall/floor/ceiling ────────────────────
def classify_planes(planes, up_axis=1, angle_tolerance=15):
    """Classify detected planes as wall/floor/ceiling based on normal direction."""
    
    cos_tol = np.cos(np.radians(angle_tolerance))
    up = np.zeros(3)
    up[up_axis] = 1.0
    
    walls, floors, ceilings, other = [], [], [], []
    
    for p in planes:
        normal = np.array(p['normal'])
        dot_up = np.dot(normal, up)
        
        if dot_up > cos_tol:  # Normal points up → floor
            p['type'] = 'floor'
            floors.append(p)
        elif dot_up < -cos_tol:  # Normal points down → ceiling
            p['type'] = 'ceiling'
            ceilings.append(p)
        elif abs(dot_up) < np.sin(np.radians(angle_tolerance)):  # Roughly horizontal normal → wall
            p['type'] = 'wall'
            walls.append(p)
        else:
            p['type'] = 'other'
            other.append(p)
    
    print(f"\n  Classification: {len(walls)} walls, {len(floors)} floors, {len(ceilings)} ceilings, {len(other)} other")
    return walls, floors, ceilings, other

# ─── Generate 2D floorplan from walls ─────────────────────────
def generate_floorplan_svg(walls, mesh_bounds, output_path):
    """Project wall planes to 2D and generate SVG floorplan."""
    
    if not walls:
        print("  No walls detected, skipping floorplan generation")
        return
    
    # Get mesh XZ bounds for SVG viewport
    min_x, min_z = mesh_bounds[0][0], mesh_bounds[0][2]
    max_x, max_z = mesh_bounds[1][0], mesh_bounds[1][2]
    padding = 0.5
    width = max_x - min_x + 2*padding
    height = max_z - min_z + 2*padding
    
    scale = 100  # pixels per meter
    svg_w = width * scale
    svg_h = height * scale
    
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}" width="{svg_w:.0f}" height="{svg_h:.0f}">',
        f'<rect width="100%" height="100%" fill="#1a1a2e"/>',
        f'<g transform="translate({padding*scale},{padding*scale})">',
    ]
    
    for i, wall in enumerate(walls):
        cx = (wall['centroid'][0] - min_x) * scale
        cz = (wall['centroid'][2] - min_z) * scale
        nx = wall['normal'][0]
        nz = wall['normal'][2]
        
        # Draw wall as a line perpendicular to normal
        length = 2.0 * scale  # 2m default line length
        dx, dz = -nz * length/2, nx * length/2
        
        n_key = 'n_inliers' if 'n_inliers' in wall else 'n_faces'
        opacity = min(1.0, wall.get(n_key, 100) / 500)
        
        svg_lines.append(
            f'<line x1="{cx-dx:.1f}" y1="{cz-dz:.1f}" x2="{cx+dx:.1f}" y2="{cz+dz:.1f}" '
            f'stroke="#ff6b35" stroke-width="3" opacity="{opacity:.2f}"/>'
        )
        svg_lines.append(
            f'<circle cx="{cx:.1f}" cy="{cz:.1f}" r="4" fill="#00ff88" opacity="0.8"/>'
        )
    
    svg_lines.append('</g></svg>')
    
    svg_content = '\n'.join(svg_lines)
    output_path.write_text(svg_content)
    print(f"  SVG floorplan saved to {output_path}")

# ─── Main ─────────────────────────────────────────────────────
def main():
    print("mesh2plan — Test Pipeline")
    print("=" * 60)
    
    # Test with refined mesh (smaller, faster iteration)
    mesh = load_mesh("export_refined.obj")
    
    results = {}
    
    # Test 1: RANSAC
    planes_ransac, t_ransac = test_ransac(mesh, max_planes=20, threshold=0.03)
    walls_r, floors_r, ceilings_r, other_r = classify_planes(planes_ransac)
    generate_floorplan_svg(walls_r, mesh.bounds, RESULTS_DIR / "floorplan_ransac.svg")
    results['ransac'] = {
        'time': t_ransac,
        'n_planes': len(planes_ransac),
        'n_walls': len(walls_r),
        'n_floors': len(floors_r),
        'n_ceilings': len(ceilings_r),
    }
    
    # Test 2: Normal Clustering
    planes_cluster, t_cluster = test_normal_clustering(mesh, n_clusters=15)
    walls_c, floors_c, ceilings_c, other_c = classify_planes(planes_cluster)
    generate_floorplan_svg(walls_c, mesh.bounds, RESULTS_DIR / "floorplan_clustering.svg")
    results['normal_clustering'] = {
        'time': t_cluster,
        'n_planes': len(planes_cluster),
        'n_walls': len(walls_c),
        'n_floors': len(floors_c),
        'n_ceilings': len(ceilings_c),
    }
    
    # Test 3: Region Growing
    planes_region, t_region = test_region_growing(mesh, angle_threshold=10, min_region_size=30)
    walls_rg, floors_rg, ceilings_rg, other_rg = classify_planes(planes_region)
    generate_floorplan_svg(walls_rg, mesh.bounds, RESULTS_DIR / "floorplan_region_growing.svg")
    results['region_growing'] = {
        'time': t_region,
        'n_planes': len(planes_region),
        'n_walls': len(walls_rg),
        'n_floors': len(floors_rg),
        'n_ceilings': len(ceilings_rg),
    }
    
    # Also test on the full-res mesh with RANSAC
    mesh_full = load_mesh("export.obj")
    planes_full, t_full = test_ransac(mesh_full, max_planes=20, threshold=0.03)
    walls_f, floors_f, ceilings_f, other_f = classify_planes(planes_full)
    generate_floorplan_svg(walls_f, mesh_full.bounds, RESULTS_DIR / "floorplan_ransac_fullres.svg")
    results['ransac_fullres'] = {
        'time': t_full,
        'n_planes': len(planes_full),
        'n_walls': len(walls_f),
        'n_floors': len(floors_f),
        'n_ceilings': len(ceilings_f),
    }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Time':>8} {'Planes':>8} {'Walls':>8} {'Floors':>8} {'Ceilings':>8}")
    print("-" * 73)
    for name, r in results.items():
        print(f"{name:<25} {r['time']:>7.2f}s {r['n_planes']:>8} {r['n_walls']:>8} {r['n_floors']:>8} {r['n_ceilings']:>8}")
    
    # Save results
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()

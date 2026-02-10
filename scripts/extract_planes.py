#!/usr/bin/env python3
"""Extract planes (floor, walls, ceiling) from OBJ mesh using RANSAC."""

import json
import sys
import numpy as np
import trimesh
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN
from pathlib import Path

def detect_up_axis(points):
    """Detect which axis is 'up' based on bounding box — shortest extent is usually vertical for room scans."""
    bbox = points.max(axis=0) - points.min(axis=0)
    # For room scans, the vertical axis typically has the smallest range (room height < width/length)
    # But also consider: gravity direction in iOS is -Y in ARKit
    # Heuristic: check if Y-range looks like room height (2-4m) 
    # For this scan: X=1.74, Y=2.70, Z=0.52 → Z is thinnest (depth of scan), Y is tallest
    # ARKit convention: Y is up
    return 1  # Y-up for ARKit data

def classify_plane(normal, up_axis=1, threshold=0.3):
    """Classify plane by normal direction relative to up axis."""
    up = np.zeros(3)
    up[up_axis] = 1.0
    
    dot = abs(np.dot(normal, up))
    
    if dot > (1 - threshold):  # Normal aligned with up = horizontal plane
        return "floor" if normal[up_axis] > 0 else "ceiling"
    elif dot < threshold:  # Normal perpendicular to up = vertical plane
        return "wall"
    return "unknown"

def extract_planes(mesh_path, max_planes=20, min_points=500, ransac_threshold=0.02):
    """Extract planes from mesh vertices using iterative RANSAC."""
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    
    # Sample points from mesh surface for better coverage
    points, face_idx = trimesh.sample.sample_surface(mesh, count=200000)
    normals = mesh.face_normals[face_idx]
    
    print(f"Sampled {len(points)} points from {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Determine up axis - check bounding box
    bbox = points.max(axis=0) - points.min(axis=0)
    print(f"Bounding box: {bbox}")
    print(f"Min: {points.min(axis=0)}, Max: {points.max(axis=0)}")
    
    planes = []
    remaining = points.copy()
    remaining_normals = normals.copy()
    
    plane_fitter = pyrsc.Plane()
    
    for i in range(max_planes):
        if len(remaining) < min_points:
            break
            
        # Fit plane with RANSAC
        eq, inliers = plane_fitter.fit(remaining, thresh=ransac_threshold, maxIteration=1000)
        
        if len(inliers) < min_points:
            print(f"  Plane {i}: only {len(inliers)} inliers, stopping")
            break
        
        plane_normal = np.array(eq[:3])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        plane_d = eq[3]
        
        inlier_points = remaining[inliers]
        plane_type = classify_plane(plane_normal, up_axis=1)
        
        # Compute plane bounds (project to 2D on plane)
        centroid = inlier_points.mean(axis=0)
        
        # Get plane basis vectors
        if abs(plane_normal[1]) > 0.9:  # Horizontal plane
            u = np.array([1, 0, 0])
        else:
            u = np.cross(plane_normal, np.array([0, 1, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(plane_normal, u)
        v = v / np.linalg.norm(v)
        
        # Project points onto plane basis
        local = inlier_points - centroid
        proj_u = local @ u
        proj_v = local @ v
        
        # Compute convex hull in 2D for boundary
        from scipy.spatial import ConvexHull
        try:
            pts_2d = np.column_stack([proj_u, proj_v])
            hull = ConvexHull(pts_2d)
            hull_indices = hull.vertices
            # Convert back to 3D
            boundary_3d = []
            for idx in hull_indices:
                p = centroid + proj_u[idx] * u + proj_v[idx] * v
                boundary_3d.append(p.tolist())
            boundary_3d.append(boundary_3d[0])  # Close loop
        except:
            boundary_3d = []
        
        area = ConvexHull(pts_2d).volume if len(pts_2d) >= 3 else 0  # 2D "volume" = area
        
        plane_info = {
            "id": i,
            "type": plane_type,
            "normal": plane_normal.tolist(),
            "d": float(plane_d),
            "centroid": centroid.tolist(),
            "num_points": len(inliers),
            "area": float(area),
            "boundary": boundary_3d,
            "basis_u": u.tolist(),
            "basis_v": v.tolist(),
        }
        
        planes.append(plane_info)
        pct = 100 * len(inliers) / len(remaining)
        print(f"  Plane {i}: {plane_type} | {len(inliers)} pts ({pct:.1f}%) | area={area:.2f} | normal={plane_normal.round(3)}")
        
        # Remove inliers
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
        remaining_normals = remaining_normals[mask]
    
    # Summary
    by_type = {}
    for p in planes:
        by_type.setdefault(p["type"], []).append(p)
    
    print(f"\nFound {len(planes)} planes:")
    for t, ps in by_type.items():
        print(f"  {t}: {len(ps)} planes, {sum(p['num_points'] for p in ps)} points")
    
    return planes, mesh

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/2026_01_13_14_47_59/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/planes.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    planes, mesh = extract_planes(mesh_path)
    
    # Also export mesh stats
    result = {
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds_min": mesh.vertices.min(axis=0).tolist(),
            "bounds_max": mesh.vertices.max(axis=0).tolist(),
        },
        "planes": planes,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved {len(planes)} planes to {output_path}")

if __name__ == "__main__":
    main()

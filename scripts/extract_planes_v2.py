#!/usr/bin/env python3
"""v2: Extract planes, cluster walls, detect openings, generate 2D floor plan."""

import json
import sys
import numpy as np
import trimesh
import pyransac3d as pyrsc
from scipy.spatial import ConvexHull
from scipy.ndimage import label as ndlabel
from sklearn.cluster import DBSCAN
from pathlib import Path
from collections import defaultdict

UP_AXIS = 1  # Y-up (ARKit convention)

def classify_plane(normal, threshold=0.3):
    """Classify plane by normal direction relative to Y-up."""
    up = np.zeros(3)
    up[UP_AXIS] = 1.0
    dot = abs(np.dot(normal, up))
    if dot > (1 - threshold):
        return "floor" if normal[UP_AXIS] > 0 else "ceiling"
    elif dot < threshold:
        return "wall"
    return "unknown"

def extract_planes(mesh_path, max_planes=30, min_points=300, ransac_threshold=0.015):
    """Extract planes using iterative RANSAC with tighter threshold."""
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    
    points, face_idx = trimesh.sample.sample_surface(mesh, count=300000)
    normals = mesh.face_normals[face_idx]
    
    print(f"Sampled {len(points)} points | {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    bbox = points.max(axis=0) - points.min(axis=0)
    print(f"Bounding box: {bbox.round(3)}")
    
    planes = []
    remaining = points.copy()
    plane_fitter = pyrsc.Plane()
    
    for i in range(max_planes):
        if len(remaining) < min_points:
            break
        eq, inliers = plane_fitter.fit(remaining, thresh=ransac_threshold, maxIteration=1500)
        if len(inliers) < min_points:
            break
        
        normal = np.array(eq[:3])
        normal = normal / np.linalg.norm(normal)
        inlier_pts = remaining[inliers]
        centroid = inlier_pts.mean(axis=0)
        ptype = classify_plane(normal)
        
        # Plane basis
        if abs(normal[UP_AXIS]) > 0.9:
            u = np.array([1, 0, 0], dtype=float)
        else:
            u = np.cross(normal, np.array([0, 1, 0], dtype=float))
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        
        # Project to 2D
        local = inlier_pts - centroid
        proj_u = local @ u
        proj_v = local @ v
        pts_2d = np.column_stack([proj_u, proj_v])
        
        try:
            hull = ConvexHull(pts_2d)
            boundary_3d = []
            for idx in hull.vertices:
                p = centroid + proj_u[idx] * u + proj_v[idx] * v
                boundary_3d.append(p.tolist())
            boundary_3d.append(boundary_3d[0])
            area = hull.volume
        except:
            boundary_3d = []
            area = 0
        
        # Detect holes/openings in walls using occupancy grid
        openings = []
        if ptype == "wall" and len(inlier_pts) > 500:
            openings = detect_openings(pts_2d, centroid, u, v, normal)
        
        planes.append({
            "id": i, "type": ptype,
            "normal": normal.tolist(), "d": float(eq[3]),
            "centroid": centroid.tolist(),
            "num_points": len(inliers), "area": float(area),
            "boundary": boundary_3d,
            "basis_u": u.tolist(), "basis_v": v.tolist(),
            "openings": openings,
        })
        
        pct = 100 * len(inliers) / len(remaining)
        n_open = len(openings)
        print(f"  Plane {i}: {ptype:8s} | {len(inliers):6d} pts ({pct:4.1f}%) | area={area:.2f} | openings={n_open}")
        
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
    
    return planes, mesh

def detect_openings(pts_2d, centroid, u, v, normal, grid_res=0.03, min_opening_area=0.05):
    """Detect openings (doors/windows) as gaps in wall point distribution."""
    openings = []
    
    u_min, v_min = pts_2d.min(axis=0)
    u_max, v_max = pts_2d.max(axis=0)
    
    nu = max(int((u_max - u_min) / grid_res), 2)
    nv = max(int((v_max - v_min) / grid_res), 2)
    
    if nu > 500 or nv > 500:
        return []
    
    # Create occupancy grid
    grid = np.zeros((nv, nu), dtype=bool)
    ui = ((pts_2d[:, 0] - u_min) / grid_res).astype(int).clip(0, nu - 1)
    vi = ((pts_2d[:, 1] - v_min) / grid_res).astype(int).clip(0, nv - 1)
    grid[vi, ui] = True
    
    # Find convex hull mask
    from matplotlib.path import Path as MplPath
    try:
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        hull_path = MplPath(hull_pts)
        
        gv, gu = np.mgrid[0:nv, 0:nu]
        grid_pts = np.column_stack([
            gu.ravel() * grid_res + u_min + grid_res/2,
            gv.ravel() * grid_res + v_min + grid_res/2,
        ])
        inside = hull_path.contains_points(grid_pts).reshape(nv, nu)
    except:
        return []
    
    # Holes = inside hull but no points
    holes = inside & ~grid
    
    # Dilate grid slightly to fill small gaps, then re-detect
    from scipy.ndimage import binary_dilation, binary_erosion
    filled = binary_dilation(grid, iterations=2)
    holes = inside & ~filled
    
    # Label connected components
    labeled, n_labels = ndlabel(holes)
    
    for label_id in range(1, n_labels + 1):
        mask = labeled == label_id
        area = mask.sum() * grid_res * grid_res
        if area < min_opening_area:
            continue
        
        # Get bounding box of opening in local coords
        ys, xs = np.where(mask)
        u_lo = xs.min() * grid_res + u_min
        u_hi = (xs.max() + 1) * grid_res + u_min
        v_lo = ys.min() * grid_res + v_min
        v_hi = (ys.max() + 1) * grid_res + v_min
        
        # Convert to 3D
        center = centroid + ((u_lo + u_hi) / 2) * u + ((v_lo + v_hi) / 2) * v
        width = u_hi - u_lo
        height = v_hi - v_lo
        
        # Classify: doors touch the bottom, windows don't
        touches_bottom = v_lo < (v_min + grid_res * 3)
        
        # Build 3D corners
        corners = []
        for cu, cv in [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi), (u_lo, v_lo)]:
            p = centroid + cu * u + cv * v
            corners.append(p.tolist())
        
        openings.append({
            "type": "door" if touches_bottom else "window",
            "center": center.tolist(),
            "width": float(width),
            "height": float(height),
            "area": float(area),
            "corners": corners,
        })
    
    return openings

def cluster_walls(planes, angle_thresh=15, dist_thresh=0.1):
    """Merge wall planes that are nearly coplanar."""
    walls = [p for p in planes if p["type"] == "wall"]
    others = [p for p in planes if p["type"] != "wall"]
    
    if len(walls) <= 1:
        return planes
    
    # Group by similar normal direction
    used = set()
    clusters = []
    
    for i, w1 in enumerate(walls):
        if i in used:
            continue
        cluster = [i]
        n1 = np.array(w1["normal"])
        c1 = np.array(w1["centroid"])
        
        for j, w2 in enumerate(walls):
            if j <= i or j in used:
                continue
            n2 = np.array(w2["normal"])
            c2 = np.array(w2["centroid"])
            
            # Check angle between normals
            cos_angle = abs(np.dot(n1, n2))
            if cos_angle < np.cos(np.radians(angle_thresh)):
                continue
            
            # Check distance between planes (project centroid onto plane normal)
            dist = abs(np.dot(c2 - c1, n1))
            if dist > dist_thresh:
                continue
            
            cluster.append(j)
            used.add(j)
        
        used.add(i)
        clusters.append(cluster)
    
    # Merge clusters
    merged_walls = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged_walls.append(walls[cluster[0]])
            continue
        
        # Weighted average by num_points
        total_pts = sum(walls[i]["num_points"] for i in cluster)
        avg_normal = np.zeros(3)
        avg_centroid = np.zeros(3)
        all_boundary = []
        all_openings = []
        total_area = 0
        
        for i in cluster:
            w = walls[i]
            weight = w["num_points"] / total_pts
            avg_normal += np.array(w["normal"]) * weight
            avg_centroid += np.array(w["centroid"]) * weight
            all_boundary.extend(w["boundary"][:-1])  # skip closing point
            all_openings.extend(w.get("openings", []))
            total_area += w["area"]
        
        avg_normal /= np.linalg.norm(avg_normal)
        
        # Recompute convex hull of merged boundary
        if len(all_boundary) >= 3:
            pts_3d = np.array(all_boundary)
            if abs(avg_normal[UP_AXIS]) > 0.9:
                u = np.array([1, 0, 0], dtype=float)
            else:
                u = np.cross(avg_normal, np.array([0, 1, 0], dtype=float))
            u /= np.linalg.norm(u)
            v = np.cross(avg_normal, u)
            v /= np.linalg.norm(v)
            
            local = pts_3d - avg_centroid
            proj = np.column_stack([local @ u, local @ v])
            try:
                hull = ConvexHull(proj)
                boundary = []
                for idx in hull.vertices:
                    p = avg_centroid + proj[idx, 0] * u + proj[idx, 1] * v
                    boundary.append(p.tolist())
                boundary.append(boundary[0])
                total_area = hull.volume
            except:
                boundary = all_boundary + [all_boundary[0]]
        else:
            boundary = all_boundary + [all_boundary[0]] if all_boundary else []
        
        merged = {
            "id": walls[cluster[0]]["id"],
            "type": "wall",
            "normal": avg_normal.tolist(),
            "d": float(np.dot(avg_centroid, avg_normal)),
            "centroid": avg_centroid.tolist(),
            "num_points": total_pts,
            "area": float(total_area),
            "boundary": boundary,
            "basis_u": u.tolist() if len(all_boundary) >= 3 else walls[cluster[0]]["basis_u"],
            "basis_v": v.tolist() if len(all_boundary) >= 3 else walls[cluster[0]]["basis_v"],
            "openings": all_openings,
            "merged_from": cluster,
        }
        merged_walls.append(merged)
    
    merged_count = len(walls) - len(merged_walls)
    if merged_count > 0:
        print(f"\nClustered {len(walls)} walls → {len(merged_walls)} ({merged_count} merged)")
    
    return others + merged_walls

def generate_floor_plan(planes):
    """Project wall bases to 2D floor plan."""
    walls = [p for p in planes if p["type"] == "wall"]
    floors = [p for p in planes if p["type"] == "floor"]
    
    if not walls:
        return None
    
    # Get floor Y level
    if floors:
        floor_y = min(np.array(f["centroid"])[UP_AXIS] for f in floors)
    else:
        # Estimate from wall boundaries
        all_y = []
        for w in walls:
            for pt in w["boundary"]:
                all_y.append(pt[UP_AXIS])
        floor_y = min(all_y) if all_y else 0
    
    # Project each wall to a line on the floor plane
    # For Y-up: floor plan is in XZ plane
    h_axes = [0, 2]  # X, Z
    
    wall_lines = []
    for w in walls:
        boundary = np.array(w["boundary"])
        if len(boundary) < 2:
            continue
        
        # Project boundary to horizontal plane
        pts_2d = boundary[:, h_axes]
        
        # Get the two extreme points (longest span)
        from scipy.spatial.distance import pdist, squareform
        if len(pts_2d) < 2:
            continue
        dists = squareform(pdist(pts_2d))
        i, j = np.unravel_index(dists.argmax(), dists.shape)
        
        wall_lines.append({
            "start": pts_2d[i].tolist(),
            "end": pts_2d[j].tolist(),
            "wall_id": w["id"],
            "normal_2d": [w["normal"][h_axes[0]], w["normal"][h_axes[1]]],
        })
    
    # Also project openings
    opening_rects = []
    for w in walls:
        for opening in w.get("openings", []):
            corners_3d = np.array(opening["corners"])
            corners_2d = corners_3d[:, h_axes]
            opening_rects.append({
                "type": opening["type"],
                "corners": corners_2d.tolist(),
                "width": opening["width"],
                "wall_id": w["id"],
            })
    
    return {
        "floor_y": float(floor_y),
        "wall_lines": wall_lines,
        "openings": opening_rects,
        "axes": ["x", "z"],
    }

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/2026_01_13_14_47_59/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/planes_v2.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    planes, mesh = extract_planes(mesh_path, ransac_threshold=0.012)
    planes = cluster_walls(planes)
    floor_plan = generate_floor_plan(planes)
    
    # Re-id after merge
    for i, p in enumerate(planes):
        p["id"] = i
    
    result = {
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds_min": mesh.vertices.min(axis=0).tolist(),
            "bounds_max": mesh.vertices.max(axis=0).tolist(),
        },
        "planes": planes,
        "floor_plan": floor_plan,
    }
    
    # Summary
    by_type = defaultdict(int)
    n_openings = 0
    for p in planes:
        by_type[p["type"]] += 1
        n_openings += len(p.get("openings", []))
    
    print(f"\nFinal: {len(planes)} planes ({dict(by_type)}), {n_openings} openings")
    if floor_plan:
        print(f"Floor plan: {len(floor_plan['wall_lines'])} wall lines, {len(floor_plan['openings'])} openings")
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v3: Face-based plane extraction, wall thickness, room detection, cleaner boundaries."""

import json
import sys
import numpy as np
import trimesh
import pyransac3d as pyrsc
from scipy.spatial import ConvexHull
from scipy.ndimage import label as ndlabel, binary_dilation, binary_erosion
from sklearn.cluster import DBSCAN
from pathlib import Path
from collections import defaultdict
from matplotlib.path import Path as MplPath

UP_AXIS = 1  # Y-up (ARKit)
WALL_THICKNESS = 0.1  # meters, for floor plan rendering

def classify_plane(normal, threshold=0.3):
    up = np.zeros(3); up[UP_AXIS] = 1.0
    dot = abs(np.dot(normal, up))
    if dot > (1 - threshold):
        return "floor" if normal[UP_AXIS] > 0 else "ceiling"
    elif dot < threshold:
        return "wall"
    return "unknown"

def extract_planes_from_faces(mesh, max_planes=25, min_faces=50, ransac_threshold=0.012):
    """Use face centroids + normals for more accurate plane extraction."""
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    
    print(f"Total mesh area: {face_areas.sum():.2f} m²")
    
    planes = []
    remaining_idx = np.arange(len(face_centers))
    plane_fitter = pyrsc.Plane()
    
    for i in range(max_planes):
        if len(remaining_idx) < min_faces:
            break
        
        pts = face_centers[remaining_idx]
        eq, inliers = plane_fitter.fit(pts, thresh=ransac_threshold, maxIteration=1500)
        
        if len(inliers) < min_faces:
            break
        
        normal = np.array(eq[:3])
        normal = normal / np.linalg.norm(normal)
        
        # Get actual face indices
        face_indices = remaining_idx[inliers]
        inlier_pts = face_centers[face_indices]
        inlier_areas = face_areas[face_indices]
        total_area = inlier_areas.sum()
        
        # Area-weighted centroid
        centroid = np.average(inlier_pts, axis=0, weights=inlier_areas)
        ptype = classify_plane(normal)
        
        # Plane basis
        if abs(normal[UP_AXIS]) > 0.9:
            u = np.array([1, 0, 0], dtype=float)
        else:
            u = np.cross(normal, np.array([0, 1, 0], dtype=float))
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        
        # Project to 2D for boundary
        local = inlier_pts - centroid
        proj_u = local @ u
        proj_v = local @ v
        pts_2d = np.column_stack([proj_u, proj_v])
        
        # Alpha shape / concave hull would be better, but convex hull for now
        try:
            hull = ConvexHull(pts_2d)
            boundary_3d = []
            for idx in hull.vertices:
                p = centroid + proj_u[idx] * u + proj_v[idx] * v
                boundary_3d.append(p.tolist())
            boundary_3d.append(boundary_3d[0])
        except:
            boundary_3d = []
        
        # For walls: compute height span
        height_min = height_max = None
        if ptype == "wall":
            heights = inlier_pts[:, UP_AXIS]
            height_min = float(heights.min())
            height_max = float(heights.max())
        
        # Detect openings for walls
        openings = []
        if ptype == "wall" and len(inlier_pts) > 200:
            openings = detect_openings_v3(pts_2d, centroid, u, v, normal, inlier_areas)
        
        planes.append({
            "id": i, "type": ptype,
            "normal": normal.tolist(), "d": float(eq[3]),
            "centroid": centroid.tolist(),
            "num_faces": len(inliers),
            "area": float(total_area),
            "boundary": boundary_3d,
            "basis_u": u.tolist(), "basis_v": v.tolist(),
            "openings": openings,
            "height_range": [height_min, height_max] if height_min is not None else None,
        })
        
        pct = 100 * len(inliers) / len(remaining_idx)
        opens = len(openings)
        h_str = f" h=[{height_min:.2f},{height_max:.2f}]" if height_min is not None else ""
        print(f"  Plane {i}: {ptype:8s} | {len(inliers):5d} faces ({pct:4.1f}%) | area={total_area:.2f}m²{h_str} | openings={opens}")
        
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[inliers] = False
        remaining_idx = remaining_idx[mask]
    
    return planes

def detect_openings_v3(pts_2d, centroid, u, v, normal, areas, grid_res=0.025, min_opening_area=0.04):
    """Detect openings with area-weighted occupancy grid."""
    openings = []
    
    u_min, v_min = pts_2d.min(axis=0) - grid_res
    u_max, v_max = pts_2d.max(axis=0) + grid_res
    
    nu = min(int((u_max - u_min) / grid_res) + 1, 600)
    nv = min(int((v_max - v_min) / grid_res) + 1, 600)
    if nu < 3 or nv < 3:
        return []
    
    # Occupancy grid weighted by face area
    grid = np.zeros((nv, nu))
    ui = ((pts_2d[:, 0] - u_min) / grid_res).astype(int).clip(0, nu - 1)
    vi = ((pts_2d[:, 1] - v_min) / grid_res).astype(int).clip(0, nv - 1)
    np.add.at(grid, (vi, ui), areas)
    
    occupied = grid > 0
    
    # Hull mask
    try:
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        hull_path = MplPath(hull_pts)
        gv, gu = np.mgrid[0:nv, 0:nu]
        grid_pts = np.column_stack([
            gu.ravel() * grid_res + u_min + grid_res / 2,
            gv.ravel() * grid_res + v_min + grid_res / 2,
        ])
        inside = hull_path.contains_points(grid_pts).reshape(nv, nu)
    except:
        return []
    
    # Fill small gaps
    filled = binary_dilation(occupied, iterations=2)
    holes = inside & ~filled
    
    # Erode to remove noise
    holes = binary_erosion(holes, iterations=1)
    holes = binary_dilation(holes, iterations=1)
    
    labeled, n_labels = ndlabel(holes)
    
    for label_id in range(1, n_labels + 1):
        mask = labeled == label_id
        area = mask.sum() * grid_res * grid_res
        if area < min_opening_area:
            continue
        
        ys, xs = np.where(mask)
        u_lo = xs.min() * grid_res + u_min
        u_hi = (xs.max() + 1) * grid_res + u_min
        v_lo = ys.min() * grid_res + v_min
        v_hi = (ys.max() + 1) * grid_res + v_min
        
        center = centroid + ((u_lo + u_hi) / 2) * u + ((v_lo + v_hi) / 2) * v
        width = u_hi - u_lo
        height = v_hi - v_lo
        
        # Door: tall opening that reaches near the bottom of the wall
        v_range = pts_2d[:, 1].max() - pts_2d[:, 1].min()
        touches_bottom = (v_lo - pts_2d[:, 1].min()) < v_range * 0.15
        is_tall = height > v_range * 0.5
        
        corners = []
        for cu, cv in [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi), (u_lo, v_lo)]:
            p = centroid + cu * u + cv * v
            corners.append(p.tolist())
        
        opening_type = "door" if (touches_bottom and is_tall) else "window"
        
        openings.append({
            "type": opening_type,
            "center": center.tolist(),
            "width": float(width),
            "height": float(height),
            "area": float(area),
            "corners": corners,
        })
    
    return openings

def cluster_walls(planes, angle_thresh=12, dist_thresh=0.08):
    """Merge coplanar walls with tighter thresholds."""
    walls = [p for p in planes if p["type"] == "wall"]
    others = [p for p in planes if p["type"] != "wall"]
    
    if len(walls) <= 1:
        return planes
    
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
            
            cos_angle = abs(np.dot(n1, n2))
            if cos_angle < np.cos(np.radians(angle_thresh)):
                continue
            dist = abs(np.dot(c2 - c1, n1))
            if dist > dist_thresh:
                continue
            
            cluster.append(j)
            used.add(j)
        used.add(i)
        clusters.append(cluster)
    
    merged_walls = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged_walls.append(walls[cluster[0]])
            continue
        
        total_area = sum(walls[i]["area"] for i in cluster)
        avg_normal = np.zeros(3)
        avg_centroid = np.zeros(3)
        all_boundary = []
        all_openings = []
        
        for i in cluster:
            w = walls[i]
            weight = w["area"] / total_area
            avg_normal += np.array(w["normal"]) * weight
            avg_centroid += np.array(w["centroid"]) * weight
            all_boundary.extend(w["boundary"][:-1])
            all_openings.extend(w.get("openings", []))
        
        avg_normal /= np.linalg.norm(avg_normal)
        
        if abs(avg_normal[UP_AXIS]) > 0.9:
            u = np.array([1, 0, 0], dtype=float)
        else:
            u = np.cross(avg_normal, np.array([0, 1, 0], dtype=float))
        u /= np.linalg.norm(u)
        v_vec = np.cross(avg_normal, u)
        v_vec /= np.linalg.norm(v_vec)
        
        if len(all_boundary) >= 3:
            pts_3d = np.array(all_boundary)
            local = pts_3d - avg_centroid
            proj = np.column_stack([local @ u, local @ v_vec])
            try:
                hull = ConvexHull(proj)
                boundary = []
                for idx in hull.vertices:
                    p = avg_centroid + proj[idx, 0] * u + proj[idx, 1] * v_vec
                    boundary.append(p.tolist())
                boundary.append(boundary[0])
                total_area = hull.volume
            except:
                boundary = all_boundary + [all_boundary[0]]
        else:
            boundary = all_boundary + [all_boundary[0]] if all_boundary else []
        
        # Merge height ranges
        h_ranges = [walls[i].get("height_range") for i in cluster if walls[i].get("height_range")]
        height_range = None
        if h_ranges:
            height_range = [min(h[0] for h in h_ranges), max(h[1] for h in h_ranges)]
        
        merged_walls.append({
            "id": walls[cluster[0]]["id"],
            "type": "wall",
            "normal": avg_normal.tolist(),
            "d": float(np.dot(avg_centroid, avg_normal)),
            "centroid": avg_centroid.tolist(),
            "num_faces": sum(walls[i]["num_faces"] for i in cluster),
            "area": float(total_area),
            "boundary": boundary,
            "basis_u": u.tolist(),
            "basis_v": v_vec.tolist(),
            "openings": all_openings,
            "height_range": height_range,
            "merged_count": len(cluster),
        })
    
    merged = len(walls) - len(merged_walls)
    if merged > 0:
        print(f"\nClustered {len(walls)} walls → {len(merged_walls)} ({merged} merged)")
    
    return others + merged_walls

def generate_floor_plan_v3(planes, mesh):
    """Generate 2D floor plan with wall thickness and room outlines."""
    walls = [p for p in planes if p["type"] == "wall"]
    floors = [p for p in planes if p["type"] == "floor"]
    
    if not walls:
        return None
    
    # Floor Y level
    if floors:
        floor_y = min(np.array(f["centroid"])[UP_AXIS] for f in floors)
    else:
        all_y = []
        for w in walls:
            if w.get("height_range"):
                all_y.append(w["height_range"][0])
        floor_y = min(all_y) if all_y else 0
    
    h_axes = [0, 2]  # XZ plane for Y-up
    
    wall_segments = []
    for w in walls:
        boundary = np.array(w["boundary"])
        if len(boundary) < 3:
            continue
        
        pts_2d = boundary[:, h_axes]
        normal_2d = np.array([w["normal"][h_axes[0]], w["normal"][h_axes[1]]])
        n_len = np.linalg.norm(normal_2d)
        if n_len > 0.01:
            normal_2d /= n_len
        
        # Get the wall line (longest span of projected boundary)
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(pts_2d))
        i, j = np.unravel_index(dists.argmax(), dists.shape)
        
        start = pts_2d[i].tolist()
        end = pts_2d[j].tolist()
        
        # Wall thickness rectangle
        offset = normal_2d * (WALL_THICKNESS / 2)
        thick_rect = [
            (start[0] + offset[0], start[1] + offset[1]),
            (end[0] + offset[0], end[1] + offset[1]),
            (end[0] - offset[0], end[1] - offset[1]),
            (start[0] - offset[0], start[1] - offset[1]),
        ]
        
        wall_segments.append({
            "start": start, "end": end,
            "normal_2d": normal_2d.tolist(),
            "thickness_rect": [list(p) for p in thick_rect],
            "wall_id": w["id"],
            "height": w.get("height_range", [0, 2.5]),
            "area": w["area"],
        })
    
    # Project openings
    opening_rects = []
    for w in walls:
        for opening in w.get("openings", []):
            corners_3d = np.array(opening["corners"])
            corners_2d = corners_3d[:, h_axes].tolist()
            opening_rects.append({
                "type": opening["type"],
                "corners": corners_2d,
                "width": opening["width"],
                "wall_id": w["id"],
            })
    
    # Room detection: find closed polygon from wall intersections
    rooms = detect_rooms(wall_segments)
    
    return {
        "floor_y": float(floor_y),
        "wall_segments": wall_segments,
        "openings": opening_rects,
        "rooms": rooms,
        "axes": ["x", "z"],
        "wall_thickness": WALL_THICKNESS,
    }

def detect_rooms(wall_segments):
    """Try to find closed rooms from wall segment intersections."""
    # Simple approach: find wall line intersections
    if len(wall_segments) < 2:
        return []
    
    lines = []
    for ws in wall_segments:
        s = np.array(ws["start"])
        e = np.array(ws["end"])
        lines.append((s, e))
    
    # Find all pairwise intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt.tolist())
    
    if len(intersections) < 3:
        return []
    
    # Try to form a room polygon from intersections
    pts = np.array(intersections)
    try:
        hull = ConvexHull(pts)
        room_boundary = pts[hull.vertices].tolist()
        room_boundary.append(room_boundary[0])
        room_area = hull.volume
        return [{
            "boundary": room_boundary,
            "area": float(room_area),
        }]
    except:
        return []

def line_intersection(l1, l2):
    """Find intersection of two 2D line segments (extended to infinite lines)."""
    p1, p2 = l1
    p3, p4 = l2
    
    d1 = p2 - p1
    d2 = p4 - p3
    
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None  # Parallel
    
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    
    pt = p1 + t * d1
    
    # Check if intersection is reasonably close to both segments
    # Allow some extension beyond segment endpoints
    margin = 0.5  # meters
    for line, param_d in [(l1, d1), (l2, d2)]:
        seg_len = np.linalg.norm(param_d)
        if seg_len < 0.01:
            return None
    
    return pt

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/2026_01_13_14_47_59/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/planes_v3.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    mesh = trimesh.load(mesh_path, process=False)
    planes = extract_planes_from_faces(mesh, ransac_threshold=0.012)
    planes = cluster_walls(planes)
    floor_plan = generate_floor_plan_v3(planes, mesh)
    
    for i, p in enumerate(planes):
        p["id"] = i
    
    result = {
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds_min": mesh.vertices.min(axis=0).tolist(),
            "bounds_max": mesh.vertices.max(axis=0).tolist(),
            "total_area": float(mesh.area),
        },
        "planes": planes,
        "floor_plan": floor_plan,
    }
    
    by_type = defaultdict(int)
    n_openings = 0
    for p in planes:
        by_type[p["type"]] += 1
        n_openings += len(p.get("openings", []))
    
    print(f"\nFinal: {len(planes)} planes ({dict(by_type)}), {n_openings} openings")
    if floor_plan:
        n_rooms = len(floor_plan.get("rooms", []))
        print(f"Floor plan: {len(floor_plan['wall_segments'])} walls, {len(floor_plan['openings'])} openings, {n_rooms} rooms")
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

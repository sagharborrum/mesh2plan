#!/usr/bin/env python3
"""v6: Manhattan-regularized floor plan from cross-sections.
Fits axis-aligned wall segments to cross-section points for clean rectangular outlines."""

import json, sys, warnings
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc

warnings.filterwarnings("ignore", category=DeprecationWarning)

UP_AXIS = 1

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def slice_mesh_3d(mesh, height):
    """Slice mesh and return 3D points on the slice plane."""
    normal = np.zeros(3); normal[UP_AXIS] = 1.0
    origin = np.zeros(3); origin[UP_AXIS] = height
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None: return None
        # Get 3D vertices directly from section (Path3D)
        # section.vertices are already in mesh coordinates
        pts_3d = section.vertices
        return pts_3d
    except:
        return None

def collect_wall_points(mesh, n_slices=20):
    """Collect XZ points from multiple horizontal slices."""
    bounds = mesh.bounds
    y_min, y_max = bounds[0][UP_AXIS], bounds[1][UP_AXIS]
    y_range = y_max - y_min
    
    heights = np.linspace(y_min + y_range * 0.15, y_max - y_range * 0.15, n_slices)
    
    all_xz = []
    slice_data = []
    
    for h in heights:
        pts_3d = slice_mesh_3d(mesh, h)
        if pts_3d is None: continue
        
        xz = pts_3d[:, [0, 2]]  # Project to XZ
        all_xz.append(xz)
        
        slice_data.append({
            "height": float(h),
            "points_xz": xz.tolist(),
            "n_points": len(xz),
        })
        print(f"  y={h:.3f}: {len(xz)} pts", flush=True)
    
    if all_xz:
        all_xz = np.vstack(all_xz)
    else:
        all_xz = np.empty((0, 2))
    
    return all_xz, slice_data

def fit_manhattan_walls(points_xz, angle_step=1, min_inliers=20, dist_thresh=0.04):
    """Find dominant wall directions and fit axis-aligned segments.
    
    1. Find dominant angle (most points align to it)
    2. Rotate to align with axes
    3. Cluster points into X-aligned and Z-aligned walls
    4. Fit wall segments
    """
    if len(points_xz) < 50:
        return [], 0
    
    # 1. Find dominant angle using Hough-like voting
    best_angle = 0
    best_score = 0
    
    for angle_deg in range(0, 180, angle_step):
        angle = np.radians(angle_deg)
        rotated = rotate_points(points_xz, -angle)
        
        # Score: how many points cluster tightly in X or Z
        # Use histogram binning
        x_hist, _ = np.histogram(rotated[:, 0], bins=100)
        z_hist, _ = np.histogram(rotated[:, 1], bins=100)
        
        # Sharp peaks = walls aligned with this angle
        score = np.sum(x_hist ** 2) + np.sum(z_hist ** 2)
        
        if score > best_score:
            best_score = score
            best_angle = angle_deg
    
    print(f"  Dominant angle: {best_angle}°", flush=True)
    
    # 2. Rotate all points to align with axes
    angle_rad = np.radians(best_angle)
    rotated = rotate_points(points_xz, -angle_rad)
    
    # 3. Find wall lines by clustering in X (vertical walls) and Z (horizontal walls)
    walls = []
    
    for axis in [0, 1]:  # 0=X-const walls, 1=Z-const walls
        coords = rotated[:, axis]
        
        # Find peaks in histogram (wall positions)
        n_bins = max(50, int((coords.max() - coords.min()) / 0.02))
        hist, bin_edges = np.histogram(coords, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Find peaks (bins with many points)
        threshold = max(np.median(hist) * 3, min_inliers)
        peak_mask = hist > threshold
        
        # Cluster adjacent peak bins
        peak_positions = []
        in_peak = False
        peak_start = 0
        peak_weight = 0
        peak_sum = 0
        
        for i in range(len(hist)):
            if peak_mask[i]:
                if not in_peak:
                    in_peak = True
                    peak_start = i
                    peak_weight = 0
                    peak_sum = 0
                peak_weight += hist[i]
                peak_sum += bin_centers[i] * hist[i]
            else:
                if in_peak:
                    peak_positions.append(peak_sum / peak_weight)
                    in_peak = False
        if in_peak:
            peak_positions.append(peak_sum / peak_weight)
        
        # For each peak, extract the wall segment
        other_axis = 1 - axis
        for wall_pos in peak_positions:
            mask = np.abs(rotated[:, axis] - wall_pos) < dist_thresh * 2
            wall_pts = rotated[mask]
            if len(wall_pts) < min_inliers:
                continue
            
            # Wall extent along other axis
            other_coords = wall_pts[:, other_axis]
            seg_start = np.percentile(other_coords, 2)
            seg_end = np.percentile(other_coords, 98)
            
            if seg_end - seg_start < 0.3:  # Skip tiny segments
                continue
            
            # Build wall segment in rotated coords
            if axis == 0:  # X-const wall (vertical line)
                p1_rot = np.array([wall_pos, seg_start])
                p2_rot = np.array([wall_pos, seg_end])
            else:  # Z-const wall (horizontal line)
                p1_rot = np.array([seg_start, wall_pos])
                p2_rot = np.array([seg_end, wall_pos])
            
            # Rotate back to original coords
            p1 = rotate_points(p1_rot.reshape(1, -1), angle_rad)[0]
            p2 = rotate_points(p2_rot.reshape(1, -1), angle_rad)[0]
            
            length = np.linalg.norm(p2 - p1)
            walls.append({
                "start": p1.tolist(),
                "end": p2.tolist(),
                "length": float(length),
                "position": float(wall_pos),
                "axis": "x" if axis == 0 else "z",
                "n_points": int(mask.sum()),
            })
    
    return walls, best_angle

def rotate_points(pts, angle):
    """Rotate 2D points by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    if pts.ndim == 1:
        return R @ pts
    return (R @ pts.T).T

def build_room_polygon(walls, margin=0.1):
    """Build room polygon from wall segments by finding intersections."""
    if len(walls) < 3:
        return None
    
    # Collect all wall endpoints
    all_pts = []
    for w in walls:
        all_pts.append(w["start"])
        all_pts.append(w["end"])
    all_pts = np.array(all_pts)
    
    # Find intersections of all wall pairs
    intersections = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            pt = line_intersection(
                np.array(walls[i]["start"]), np.array(walls[i]["end"]),
                np.array(walls[j]["start"]), np.array(walls[j]["end"]),
                extend=0.5  # Allow some extension
            )
            if pt is not None:
                intersections.append(pt)
    
    if len(intersections) < 3:
        # Fallback: convex hull of wall endpoints
        try:
            hull = ConvexHull(all_pts)
            room_pts = all_pts[hull.vertices]
            poly = list(room_pts.tolist())
            poly.append(poly[0])
            return {
                "exterior": poly,
                "area": float(hull.volume),
                "perimeter": float(sum(np.linalg.norm(room_pts[(i+1)%len(room_pts)] - room_pts[i]) for i in range(len(room_pts)))),
            }
        except:
            return None
    
    # Build polygon from intersections
    pts = np.array(intersections)
    try:
        hull = ConvexHull(pts)
        room_pts = pts[hull.vertices]
        poly = list(room_pts.tolist())
        poly.append(poly[0])
        return {
            "exterior": poly,
            "area": float(hull.volume),
            "perimeter": float(sum(np.linalg.norm(room_pts[(i+1)%len(room_pts)] - room_pts[i]) for i in range(len(room_pts)))),
        }
    except:
        return None

def line_intersection(p1, p2, p3, p4, extend=0.3):
    """Find intersection of two line segments, with optional extension."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10: return None
    
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    s = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    
    # Check if intersection is near both segments
    if t < -extend or t > 1 + extend: return None
    if s < -extend or s > 1 + extend: return None
    
    return (p1 + t * d1).tolist()

def extract_planes(mesh, max_planes=15):
    """Quick plane extraction for 3D overlay."""
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    planes = []
    remaining = np.arange(len(face_centers))
    fitter = pyrsc.Plane()
    
    for i in range(max_planes):
        if len(remaining) < 200: break
        eq, inliers = fitter.fit(face_centers[remaining], thresh=0.015, maxIteration=1500)
        if len(inliers) < 200: break
        
        normal = np.array(eq[:3]); normal /= np.linalg.norm(normal)
        fidx = remaining[inliers]
        centroid = np.average(face_centers[fidx], axis=0, weights=face_areas[fidx])
        area = face_areas[fidx].sum()
        
        up = np.zeros(3); up[UP_AXIS] = 1.0
        dot = abs(np.dot(normal, up))
        ptype = "floor" if dot > 0.7 and normal[UP_AXIS] > 0 else ("ceiling" if dot > 0.7 else ("wall" if dot < 0.3 else "unknown"))
        
        if abs(normal[UP_AXIS]) > 0.9: u = np.array([1,0,0], dtype=float)
        else: u = np.cross(normal, np.array([0,1,0], dtype=float))
        u /= np.linalg.norm(u); v = np.cross(normal, u); v /= np.linalg.norm(v)
        
        pts = face_centers[fidx]
        local = pts - centroid
        pu, pv = local @ u, local @ v
        try:
            hull = ConvexHull(np.column_stack([pu, pv]))
            boundary = [(centroid + pu[j]*u + pv[j]*v).tolist() for j in hull.vertices]
            boundary.append(boundary[0])
        except: boundary = []
        
        planes.append({
            "id": i, "type": ptype, "normal": normal.tolist(),
            "centroid": centroid.tolist(), "area": float(area),
            "boundary": boundary, "basis_u": u.tolist(), "basis_v": v.tolist(),
        })
        
        mask = np.ones(len(remaining), dtype=bool); mask[inliers] = False
        remaining = remaining[mask]
    
    return planes

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/room_scan/2026_02_09_19_03_38/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/floorplan_v6.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {mesh_path}", flush=True)
    mesh = trimesh.load(mesh_path, process=True)
    print(f"Mesh: {len(mesh.vertices)}v, {len(mesh.faces)}f, area={mesh.area:.1f}m²", flush=True)
    
    # 1. Planes
    print("\n── Planes ──", flush=True)
    planes = extract_planes(mesh)
    by_type = defaultdict(int)
    for p in planes: by_type[p["type"]] += 1
    print(f"  {len(planes)} planes: {dict(by_type)}", flush=True)
    
    # 2. Collect cross-section points
    print("\n── Cross-Sections ──", flush=True)
    all_xz, slice_data = collect_wall_points(mesh, n_slices=20)
    print(f"  Total: {len(all_xz)} XZ points from {len(slice_data)} slices", flush=True)
    
    # 3. Manhattan wall fitting
    print("\n── Manhattan Walls ──", flush=True)
    walls, dominant_angle = fit_manhattan_walls(all_xz)
    print(f"  Found {len(walls)} wall segments:", flush=True)
    for w in walls:
        print(f"    {w['axis']}-wall: {w['length']:.2f}m ({w['n_points']} pts)", flush=True)
    
    # 4. Room polygon
    print("\n── Room Polygon ──", flush=True)
    room = build_room_polygon(walls)
    if room:
        print(f"  Area: {room['area']:.2f}m², perimeter: {room['perimeter']:.2f}m", flush=True)
        bounds = np.array(room["exterior"])
        room["dimensions"] = [float(bounds[:,0].max() - bounds[:,0].min()), float(bounds[:,1].max() - bounds[:,1].min())]
    
    # 5. Measurements from walls
    measurements = []
    for w in walls:
        if w["length"] > 0.5:
            mid = [(w["start"][0]+w["end"][0])/2, (w["start"][1]+w["end"][1])/2]
            measurements.append({"start": w["start"], "end": w["end"], "mid": mid, "length": w["length"]})
    
    result = {
        "mesh": {
            "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "total_area": float(mesh.area),
            "bounds_min": mesh.bounds[0].tolist(), "bounds_max": mesh.bounds[1].tolist(),
        },
        "planes": planes,
        "slices": slice_data,
        "walls": walls,
        "room": room,
        "measurements": measurements,
        "dominant_angle": float(dominant_angle),
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {output_path}", flush=True)

if __name__ == "__main__":
    main()

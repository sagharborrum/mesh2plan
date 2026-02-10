#!/usr/bin/env python3
"""v5: Multi-slice composite floor plan with simplified outlines and room detection."""

import json
import sys
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, polygonize
import pyransac3d as pyrsc

UP_AXIS = 1

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return super().default(obj)

def slice_mesh(mesh, height):
    """Slice mesh and return 2D paths + the 3D→2D transform."""
    normal = np.zeros(3); normal[UP_AXIS] = 1.0
    origin = np.zeros(3); origin[UP_AXIS] = height
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None: return None, None
        planar, to_3d = section.to_planar()
        return planar, to_3d
    except:
        return None, None

def simplify_path(vertices, tolerance=0.02):
    """Simplify a path using Douglas-Peucker."""
    if len(vertices) < 3: return vertices
    try:
        line = LineString(vertices)
        simplified = line.simplify(tolerance, preserve_topology=True)
        return np.array(simplified.coords)
    except:
        return vertices

def extract_room_polygon(mesh, n_slices=20, simplify_tol=0.03):
    """Extract room outline by compositing multiple horizontal slices."""
    bounds = mesh.bounds
    y_min, y_max = bounds[0][UP_AXIS], bounds[1][UP_AXIS]
    y_range = y_max - y_min
    
    # Sample at wall heights (skip near floor/ceiling)
    heights = np.linspace(y_min + y_range * 0.15, y_max - y_range * 0.15, n_slices)
    
    all_lines = []
    all_polygons = []
    slice_data = []
    
    print(f"Slicing mesh at {n_slices} heights ({y_min:.2f} to {y_max:.2f})")
    
    for h in heights:
        planar, to_3d = slice_mesh(mesh, h)
        if planar is None: continue
        
        paths_2d = []  # in planar space (for floor plan canvas)
        paths_3d_xz = []  # in mesh XZ coords (for 3D overlay + room polygon)
        
        for entity in planar.entities:
            pts_2d = planar.vertices[entity.points]
            if len(pts_2d) < 2: continue
            
            # Transform 2D planar points back to 3D using to_3d matrix
            pts_homo = np.column_stack([pts_2d, np.zeros(len(pts_2d)), np.ones(len(pts_2d))])
            pts_3d = (to_3d @ pts_homo.T).T[:, :3]
            
            # Project to XZ plane (mesh coordinates)
            pts_xz = pts_3d[:, [0, 2]]  # X, Z
            
            simplified_xz = simplify_path(pts_xz, simplify_tol)
            simplified_2d = simplify_path(pts_2d, simplify_tol)
            paths_2d.append(simplified_2d)
            paths_3d_xz.append(simplified_xz)
            
            if len(simplified_xz) >= 2:
                all_lines.append(LineString(simplified_xz))
        
        # Skip polygons_full — it hangs on complex meshes. Use lines only.
        
        n_pts = sum(len(p) for p in paths_3d_xz)
        slice_data.append({
            "height": float(h),
            "paths": [p.tolist() for p in paths_3d_xz],  # Store XZ coords
            "n_paths": len(paths_3d_xz),
            "n_points": n_pts,
        })
        print(f"  y={h:.3f}: {len(paths_3d_xz)} paths, {n_pts} pts")
    
    # Build room polygon from best slice's convex hull of all points
    room_polygon = None
    room_outline = None
    
    # Collect all XZ points from all slices
    all_pts_xz = []
    for line in all_lines:
        all_pts_xz.extend(list(line.coords))
    
    if len(all_pts_xz) > 10:
        all_pts_xz = np.array(all_pts_xz)
        try:
            # Use concave hull (alpha shape) via buffered convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(all_pts_xz)
            hull_pts = all_pts_xz[hull.vertices]
            room_polygon = Polygon(hull_pts)
            if not room_polygon.is_valid:
                room_polygon = room_polygon.buffer(0)
            room_polygon = room_polygon.simplify(simplify_tol)
            print(f"\nConvex hull room: {room_polygon.area:.2f}m²")
        except Exception as e:
            print(f"Hull failed: {e}")
    
    # Simplify room polygon
    if room_polygon:
        room_polygon = room_polygon.simplify(simplify_tol)
        exterior = np.array(room_polygon.exterior.coords).tolist()
        interiors = [np.array(ring.coords).tolist() for ring in room_polygon.interiors]
        room_outline = {
            "exterior": exterior,
            "interiors": interiors,
            "area": float(room_polygon.area),
            "perimeter": float(room_polygon.length),
        }
        print(f"\nRoom polygon: {room_polygon.area:.2f}m², perimeter={room_polygon.length:.2f}m")
        
        # Compute dimensions (bounding box of room)
        minx, miny, maxx, maxy = room_polygon.bounds
        room_outline["bounds"] = {"min": [minx, miny], "max": [maxx, maxy]}
        room_outline["dimensions"] = [float(maxx - minx), float(maxy - miny)]
    
    # Find best single slice (most vertices at mid-height)
    mid_slices = [s for s in slice_data if abs(s["height"] - (y_min + y_range/2)) < y_range * 0.3]
    best_slice = max(mid_slices, key=lambda s: s["n_points"]) if mid_slices else (slice_data[len(slice_data)//2] if slice_data else None)
    
    return slice_data, room_outline, best_slice

def extract_walls(mesh, max_planes=20, min_faces=200):
    """Quick RANSAC wall extraction for 3D overlay."""
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    planes = []
    remaining = np.arange(len(face_centers))
    fitter = pyrsc.Plane()
    
    for i in range(max_planes):
        if len(remaining) < min_faces: break
        eq, inliers = fitter.fit(face_centers[remaining], thresh=0.015, maxIteration=1500)
        if len(inliers) < min_faces: break
        
        normal = np.array(eq[:3]); normal /= np.linalg.norm(normal)
        fidx = remaining[inliers]
        centroid = np.average(face_centers[fidx], axis=0, weights=face_areas[fidx])
        area = face_areas[fidx].sum()
        
        up = np.zeros(3); up[UP_AXIS] = 1.0
        dot = abs(np.dot(normal, up))
        ptype = "floor" if dot > 0.7 and normal[UP_AXIS] > 0 else ("ceiling" if dot > 0.7 else ("wall" if dot < 0.3 else "unknown"))
        
        if abs(normal[UP_AXIS]) > 0.9:
            u = np.array([1,0,0], dtype=float)
        else:
            u = np.cross(normal, np.array([0,1,0], dtype=float))
        u /= np.linalg.norm(u); v = np.cross(normal, u); v /= np.linalg.norm(v)
        
        pts = face_centers[fidx]
        local = pts - centroid
        pu, pv = local @ u, local @ v
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(np.column_stack([pu, pv]))
            boundary = [(centroid + pu[j]*u + pv[j]*v).tolist() for j in hull.vertices]
            boundary.append(boundary[0])
        except: boundary = []
        
        planes.append({
            "id": i, "type": ptype, "normal": normal.tolist(),
            "centroid": centroid.tolist(), "area": float(area),
            "boundary": boundary, "basis_u": u.tolist(), "basis_v": v.tolist(),
            "num_faces": len(inliers),
        })
        
        mask = np.ones(len(remaining), dtype=bool); mask[inliers] = False
        remaining = remaining[mask]
    
    return planes

def compute_measurements(room_outline):
    """Compute wall-to-wall measurements for the room."""
    if not room_outline: return []
    
    exterior = np.array(room_outline["exterior"])
    if len(exterior) < 4: return []
    
    measurements = []
    for i in range(len(exterior) - 1):
        p1 = exterior[i]
        p2 = exterior[i + 1]
        dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if dist > 0.3:  # Skip tiny segments
            mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
            measurements.append({
                "start": p1.tolist(),
                "end": p2.tolist(),
                "mid": mid,
                "length": float(dist),
            })
    
    return measurements

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/room_scan/2026_02_09_19_03_38/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/floorplan_v5.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=True)
    print(f"Mesh: {len(mesh.vertices)}v, {len(mesh.faces)}f, area={mesh.area:.1f}m²")
    
    # 1. Planes for 3D
    print("\n── Plane Extraction ──")
    planes = extract_walls(mesh)
    by_type = defaultdict(int)
    for p in planes:
        by_type[p["type"]] += 1
    print(f"  {len(planes)} planes: {dict(by_type)}")
    
    # 2. Cross-section floor plan
    print("\n── Cross-Section Floor Plan ──")
    slices, room_outline, best_slice = extract_room_polygon(mesh, n_slices=25, simplify_tol=0.03)
    
    # 3. Measurements
    measurements = compute_measurements(room_outline)
    if measurements:
        print(f"\nMeasurements: {len(measurements)} wall segments")
        for m in measurements:
            print(f"  {m['length']:.2f}m")
    
    result = {
        "mesh": {
            "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "total_area": float(mesh.area),
            "bounds_min": mesh.bounds[0].tolist(), "bounds_max": mesh.bounds[1].tolist(),
        },
        "planes": planes,
        "slices": slices,
        "best_slice": best_slice,
        "room": room_outline,
        "measurements": measurements,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()

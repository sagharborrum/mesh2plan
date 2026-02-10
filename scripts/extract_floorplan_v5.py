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
    """Slice mesh and return 2D paths."""
    normal = np.zeros(3); normal[UP_AXIS] = 1.0
    origin = np.zeros(3); origin[UP_AXIS] = height
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None: return None
        planar, to_3d = section.to_planar()
        return planar
    except:
        return None

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
        planar = slice_mesh(mesh, h)
        if planar is None: continue
        
        paths = []
        for entity in planar.entities:
            pts = planar.vertices[entity.points]
            if len(pts) < 2: continue
            simplified = simplify_path(pts, simplify_tol)
            paths.append(simplified)
            
            # Create shapely lines
            if len(simplified) >= 2:
                all_lines.append(LineString(simplified))
        
        # Try to get closed polygons
        try:
            for poly in planar.polygons_full:
                ext = np.array(poly.exterior.coords)
                simplified_ext = simplify_path(ext, simplify_tol)
                if len(simplified_ext) >= 4:
                    try:
                        sp = Polygon(simplified_ext)
                        if sp.is_valid and sp.area > 0.1:
                            all_polygons.append(sp)
                    except: pass
        except: pass
        
        n_pts = sum(len(p) for p in paths)
        slice_data.append({
            "height": float(h),
            "paths": [p.tolist() for p in paths],
            "n_paths": len(paths),
            "n_points": n_pts,
        })
        print(f"  y={h:.3f}: {len(paths)} paths, {n_pts} pts")
    
    # Composite: merge all polygons
    room_polygon = None
    room_outline = None
    
    if all_polygons:
        try:
            merged = unary_union(all_polygons)
            if isinstance(merged, Polygon) and merged.area > 0.5:
                room_polygon = merged
            elif isinstance(merged, MultiPolygon):
                # Take largest
                largest = max(merged.geoms, key=lambda p: p.area)
                if largest.area > 0.5:
                    room_polygon = largest
        except: pass
    
    # If no polygon from slices, try polygonizing the lines
    if room_polygon is None and all_lines:
        try:
            merged_lines = unary_union(all_lines)
            # Buffer lines slightly and take convex hull as fallback
            buffered = merged_lines.buffer(0.05)
            if isinstance(buffered, Polygon) and buffered.area > 0.5:
                room_polygon = buffered.simplify(simplify_tol * 2)
            elif isinstance(buffered, MultiPolygon):
                largest = max(buffered.geoms, key=lambda p: p.area)
                if largest.area > 0.5:
                    room_polygon = largest.simplify(simplify_tol * 2)
        except: pass
    
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

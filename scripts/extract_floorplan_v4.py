#!/usr/bin/env python3
"""v4: Cross-section based floor plan — slice mesh at floor height for accurate room boundary."""

import json
import sys
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict

UP_AXIS = 1  # Y-up (ARKit)

def slice_mesh_at_height(mesh, height, axis=1):
    """Slice mesh with a horizontal plane and return 2D cross-section paths."""
    # Create slicing plane
    normal = np.zeros(3)
    normal[axis] = 1.0
    origin = np.zeros(3)
    origin[axis] = height
    
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return None
        
        # Get 2D paths
        planar, to_3d = section.to_planar()
        return planar, to_3d, section
    except Exception as e:
        print(f"  Slice at {height:.3f} failed: {e}")
        return None

def extract_floor_plan_from_slices(mesh, n_slices=10):
    """Take multiple horizontal slices and find the best one for floor plan."""
    bounds = mesh.bounds
    y_min, y_max = bounds[0][UP_AXIS], bounds[1][UP_AXIS]
    y_range = y_max - y_min
    
    print(f"Mesh Y range: {y_min:.3f} to {y_max:.3f} ({y_range:.3f}m)")
    
    # Slice at multiple heights
    best_slice = None
    best_area = 0
    best_height = None
    all_slices = []
    
    heights = np.linspace(y_min + y_range * 0.05, y_max - y_range * 0.05, n_slices)
    
    for h in heights:
        result = slice_mesh_at_height(mesh, h)
        if result is None:
            continue
        
        planar, to_3d, section = result
        
        # Compute total area (may be 0 for open meshes)
        try:
            area = float(planar.area)
        except:
            area = 0.0
        
        try:
            n_paths = len(planar.polygons_full)
        except:
            n_paths = 0
        
        # Count entities instead of paths
        n_entities = len(planar.entities) if hasattr(planar, 'entities') else 0
        total_len = sum(len(e.points) for e in planar.entities) if n_entities else 0
        
        print(f"  Slice y={h:.3f}: area={area:.3f}m², polygons={n_paths}, entities={n_entities}, verts={len(planar.vertices)}")
        
        all_slices.append({
            "height": float(h),
            "area": float(area),
            "n_paths": n_paths,
            "planar": planar,
            "to_3d": to_3d,
        })
        
        # Best = largest area (likely at counter/table height for most coverage)
        if area > best_area:
            best_area = area
            best_slice = planar
            best_height = h
    
    return all_slices, best_slice, best_height

def planar_to_json(planar, height, h_axes=[0, 2]):
    """Convert trimesh Path2D to JSON-serializable format."""
    paths = []
    
    for i, entity in enumerate(planar.entities):
        pts_idx = entity.points
        pts = planar.vertices[pts_idx]
        paths.append({
            "id": i,
            "points": pts.tolist(),
            "closed": entity.closed if hasattr(entity, 'closed') else False,
        })
    
    # Also extract polygons
    polygons = []
    if hasattr(planar, 'polygons_full'):
        for i, poly in enumerate(planar.polygons_full):
            exterior = np.array(poly.exterior.coords).tolist()
            interiors = [np.array(ring.coords).tolist() for ring in poly.interiors]
            polygons.append({
                "id": i,
                "exterior": exterior,
                "interiors": interiors,
                "area": float(poly.area),
            })
    
    return {
        "height": float(height),
        "paths": paths,
        "polygons": polygons,
        "total_area": float(planar.area) if hasattr(planar, 'area') else 0,
    }

def run_plane_extraction(mesh):
    """Quick plane extraction for wall/floor/ceiling classification."""
    import pyransac3d as pyrsc
    from scipy.spatial import ConvexHull
    
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    
    planes = []
    remaining_idx = np.arange(len(face_centers))
    plane_fitter = pyrsc.Plane()
    
    for i in range(15):
        if len(remaining_idx) < 100:
            break
        pts = face_centers[remaining_idx]
        eq, inliers = plane_fitter.fit(pts, thresh=0.012, maxIteration=1500)
        if len(inliers) < 100:
            break
        
        normal = np.array(eq[:3])
        normal /= np.linalg.norm(normal)
        
        face_idx = remaining_idx[inliers]
        centroid = np.average(face_centers[face_idx], axis=0, weights=face_areas[face_idx])
        area = face_areas[face_idx].sum()
        
        # Classify
        up = np.zeros(3); up[UP_AXIS] = 1.0
        dot = abs(np.dot(normal, up))
        if dot > 0.7:
            ptype = "floor" if normal[UP_AXIS] > 0 else "ceiling"
        elif dot < 0.3:
            ptype = "wall"
        else:
            ptype = "unknown"
        
        # Boundary
        if abs(normal[UP_AXIS]) > 0.9:
            u = np.array([1, 0, 0], dtype=float)
        else:
            u = np.cross(normal, np.array([0, 1, 0], dtype=float))
        u /= np.linalg.norm(u)
        v = np.cross(normal, u); v /= np.linalg.norm(v)
        
        inlier_pts = face_centers[face_idx]
        local = inlier_pts - centroid
        proj_u = local @ u
        proj_v = local @ v
        pts_2d = np.column_stack([proj_u, proj_v])
        
        try:
            hull = ConvexHull(pts_2d)
            boundary = []
            for idx in hull.vertices:
                p = centroid + proj_u[idx] * u + proj_v[idx] * v
                boundary.append(p.tolist())
            boundary.append(boundary[0])
        except:
            boundary = []
        
        # Height range for walls
        heights = inlier_pts[:, UP_AXIS]
        h_range = [float(heights.min()), float(heights.max())] if ptype == "wall" else None
        
        planes.append({
            "id": i, "type": ptype,
            "normal": normal.tolist(),
            "centroid": centroid.tolist(),
            "area": float(area),
            "boundary": boundary,
            "basis_u": u.tolist(), "basis_v": v.tolist(),
            "height_range": h_range,
            "num_faces": len(inliers),
        })
        
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[inliers] = False
        remaining_idx = remaining_idx[mask]
    
    return planes

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/2026_01_13_14_47_59/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/floorplan_v4.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=True)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, area={mesh.area:.2f}m²")
    
    # 1. Extract planes for 3D view
    print("\n── Plane Extraction ──")
    planes = run_plane_extraction(mesh)
    by_type = defaultdict(int)
    for p in planes:
        by_type[p["type"]] += 1
        print(f"  Plane {p['id']}: {p['type']:8s} | area={p['area']:.2f}m² | faces={p['num_faces']}")
    print(f"  Total: {dict(by_type)}")
    
    # 2. Cross-section floor plan
    print("\n── Cross-Section Slicing ──")
    all_slices, best_slice, best_height = extract_floor_plan_from_slices(mesh, n_slices=15)
    
    # Convert best slice + a few others to JSON
    slice_data = []
    for s in all_slices:
        if s["planar"] is not None:
            sd = planar_to_json(s["planar"], s["height"])
            sd["is_best"] = bool(abs(s["height"] - best_height) < 0.001) if best_height else False
            slice_data.append(sd)
    
    print(f"\nBest slice at y={best_height:.3f} (area={best_slice.area:.3f}m²)")
    
    # 3. Build result
    result = {
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "total_area": float(mesh.area),
            "bounds_min": mesh.bounds[0].tolist(),
            "bounds_max": mesh.bounds[1].tolist(),
        },
        "planes": planes,
        "slices": slice_data,
        "best_slice_height": float(best_height) if best_height else None,
    }
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.bool_): return bool(obj)
            return super().default(obj)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    
    print(f"\nSaved to {output_path}")
    print(f"  {len(planes)} planes, {len(slice_data)} slices")

if __name__ == "__main__":
    main()

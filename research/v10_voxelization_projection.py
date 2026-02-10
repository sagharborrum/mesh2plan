#!/usr/bin/env python3
"""
v10: Voxelization + 2D Projection Approach
===========================================

New approach that doesn't rely on Manhattan assumptions:
1. Convert mesh to voxel grid
2. Project occupied voxels to 2D top-down view  
3. Use morphological operations to clean up walls
4. Extract contours for room boundaries
5. Detect openings (doors/windows) in the boundary

This should be more robust for non-axis-aligned rooms and complex geometries.
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
from scipy import ndimage
from sklearn.cluster import DBSCAN
import cv2

warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def voxelize_mesh(mesh, voxel_size=0.05):
    """Convert mesh to voxel grid"""
    print(f"Voxelizing mesh with size {voxel_size}m...")
    
    # Get mesh bounds
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    # Create voxel grid dimensions
    x_size = int(np.ceil((x_max - x_min) / voxel_size))
    y_size = int(np.ceil((y_max - y_min) / voxel_size))  
    z_size = int(np.ceil((z_max - z_min) / voxel_size))
    
    print(f"Grid size: {x_size} x {y_size} x {z_size} = {x_size*y_size*z_size:,} voxels")
    
    # Initialize voxel grid
    voxels = np.zeros((x_size, y_size, z_size), dtype=bool)
    
    # Convert vertices to voxel coordinates
    vertices = mesh.vertices.copy()
    vertices[:, 0] = (vertices[:, 0] - x_min) / voxel_size
    vertices[:, 1] = (vertices[:, 1] - y_min) / voxel_size  
    vertices[:, 2] = (vertices[:, 2] - z_min) / voxel_size
    
    # Mark voxels that contain vertices
    for v in vertices:
        xi, yi, zi = int(v[0]), int(v[1]), int(v[2])
        if 0 <= xi < x_size and 0 <= yi < y_size and 0 <= zi < z_size:
            voxels[xi, yi, zi] = True
    
    # Also voxelize faces to fill gaps
    print("Filling face voxels...")
    for face in mesh.faces:
        # Get triangle vertices
        tri_verts = vertices[face]
        
        # Rasterize triangle in 3D (simplified - use bounding box)
        min_coords = np.floor(tri_verts.min(axis=0)).astype(int)
        max_coords = np.ceil(tri_verts.max(axis=0)).astype(int)
        
        for xi in range(max(0, min_coords[0]), min(x_size, max_coords[0]+1)):
            for yi in range(max(0, min_coords[1]), min(y_size, max_coords[1]+1)):
                for zi in range(max(0, min_coords[2]), min(z_size, max_coords[2]+1)):
                    # Simple check - if point is close to triangle, mark as occupied
                    point = np.array([xi, yi, zi])
                    if np.linalg.norm(tri_verts - point, axis=1).min() < 1.5:
                        voxels[xi, yi, zi] = True
    
    occupied_count = np.sum(voxels)
    print(f"Occupied voxels: {occupied_count:,} ({100*occupied_count/(x_size*y_size*z_size):.1f}%)")
    
    return voxels, (x_min, y_min, z_min), voxel_size

def project_to_2d(voxels, wall_height_range=(0.3, 0.8)):
    """Project voxel grid to 2D, focusing on wall heights"""
    print("Projecting to 2D...")
    
    # Get Y dimension bounds for walls (assume Y is up)
    y_size = voxels.shape[1]
    y_start = int(wall_height_range[0] * y_size) 
    y_end = int(wall_height_range[1] * y_size)
    
    print(f"Using Y slice {y_start}:{y_end} (heights {wall_height_range[0]:.1f}-{wall_height_range[1]:.1f})")
    
    # Project by taking maximum occupancy in the wall height range
    wall_slice = voxels[:, y_start:y_end, :]
    projection = np.max(wall_slice, axis=1)  # Max over Y axis
    
    print(f"2D projection size: {projection.shape}")
    print(f"Occupied pixels: {np.sum(projection):,}")
    
    return projection

def clean_projection(projection, kernel_size=3):
    """Clean up 2D projection using morphological operations"""
    print("Cleaning projection...")
    
    # Convert to uint8
    img = (projection * 255).astype(np.uint8)
    
    # Morphological opening to remove noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Morphological closing to connect gaps
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to boolean
    cleaned_bool = cleaned > 128
    
    print(f"Cleaning removed {np.sum(projection) - np.sum(cleaned_bool)} pixels")
    
    return cleaned_bool

def extract_room_boundary(projection):
    """Extract room boundary contours"""
    print("Extracting boundaries...")
    
    # Convert to uint8 for contour detection
    img = (projection * 255).astype(np.uint8)
    
    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    # Get the largest contour (main room boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to regular array
    boundary_points = simplified.reshape(-1, 2)
    
    print(f"Found {len(contours)} contours, main boundary has {len(boundary_points)} points")
    
    return boundary_points, contours

def detect_openings(boundary_points, projection, min_opening_size=5):
    """Detect doors and windows in the boundary"""
    print("Detecting openings...")
    
    openings = []
    
    # For each edge of the boundary, check for gaps
    for i in range(len(boundary_points)):
        p1 = boundary_points[i]
        p2 = boundary_points[(i+1) % len(boundary_points)]
        
        # Sample points along the edge
        num_samples = int(np.linalg.norm(p2 - p1) * 2)  # 2 samples per pixel
        if num_samples < 2:
            continue
            
        edge_points = []
        for t in np.linspace(0, 1, num_samples):
            point = p1 + t * (p2 - p1)
            edge_points.append(point.astype(int))
        
        # Check occupancy along the edge
        gap_start = None
        for j, point in enumerate(edge_points):
            x, y = point
            if 0 <= x < projection.shape[0] and 0 <= y < projection.shape[1]:
                occupied = projection[x, y]
                
                if not occupied and gap_start is None:
                    gap_start = j
                elif occupied and gap_start is not None:
                    # End of gap
                    gap_length = j - gap_start
                    if gap_length >= min_opening_size:
                        gap_center = (gap_start + j) // 2
                        center_point = edge_points[gap_center] if gap_center < len(edge_points) else edge_points[-1]
                        opening_type = "door" if gap_length < 25 else "window"  # Rough heuristic
                        openings.append({
                            "type": opening_type,
                            "position": center_point,
                            "size": gap_length
                        })
                    gap_start = None
    
    print(f"Found {len(openings)} openings")
    return openings

def analyze_mesh_voxelization(mesh_path, voxel_size=0.05):
    """Main analysis function"""
    print(f"\n=== Voxelization Analysis: {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if hasattr(mesh, 'vertices'):
        print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    else:
        print("Error loading mesh")
        return None
    
    # Step 1: Voxelize
    voxels, origin, voxel_size = voxelize_mesh(mesh, voxel_size)
    
    # Step 2: Project to 2D
    projection = project_to_2d(voxels)
    
    # Step 3: Clean up
    cleaned_projection = clean_projection(projection)
    
    # Step 4: Extract boundary
    boundary_points, all_contours = extract_room_boundary(cleaned_projection)
    
    # Step 5: Detect openings
    openings = detect_openings(boundary_points, cleaned_projection)
    
    # Convert back to world coordinates
    world_boundary = []
    for point in boundary_points:
        world_x = origin[0] + point[0] * voxel_size
        world_z = origin[2] + point[1] * voxel_size  # Note: Y and Z mapping
        world_boundary.append([float(world_x), float(world_z)])
    
    world_openings = []
    for opening in openings:
        world_x = origin[0] + opening["position"][0] * voxel_size
        world_z = origin[2] + opening["position"][1] * voxel_size
        world_openings.append({
            "type": opening["type"],
            "position": [float(world_x), float(world_z)],
            "size_pixels": opening["size"],
            "size_meters": float(opening["size"] * voxel_size)
        })
    
    # Calculate room area
    if len(world_boundary) >= 3:
        # Simple polygon area calculation
        area = 0.0
        for i in range(len(world_boundary)):
            j = (i + 1) % len(world_boundary)
            area += world_boundary[i][0] * world_boundary[j][1]
            area -= world_boundary[j][0] * world_boundary[i][1]
        area = abs(area) / 2.0
    else:
        area = 0.0
    
    return {
        "method": "voxelization_projection_v10",
        "mesh_file": str(mesh_path),
        "parameters": {
            "voxel_size": voxel_size,
            "wall_height_range": [0.3, 0.8]
        },
        "room_boundary": world_boundary,
        "openings": world_openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "voxel_stats": {
            "grid_size": voxels.shape,
            "occupied_voxels": int(np.sum(voxels)),
            "projection_pixels": int(np.sum(projection))
        }
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python v10_voxelization_projection.py <mesh.obj> <output.json>")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_voxelization(mesh_path)
        if result:
            # Save results
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NpEncoder)
            print(f"\n✅ Results saved to {output_path}")
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Boundary points: {len(result['room_boundary'])}")
            print(f"   Openings: {len(result['openings'])}")
            for opening in result['openings']:
                print(f"     - {opening['type']}: {opening['size_meters']:.2f}m")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main()
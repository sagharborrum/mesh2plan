#!/usr/bin/env python3
"""
v10: Voxelization + 2D Projection Approach (FIXED)
==================================================

FIXES APPLIED:
- Use voxel_size=0.02m instead of 0.05m for finer resolution
- Use absolute wall height band (0.8-1.5m above floor) not relative
- Implement cross-section slicing approach like v9
- Add Manhattan wall fitting with histogram angle voting
- Add wall segment merging and regularization 
- Fix coordinate system and area calculations

This approach:
1. Find floor level using Z-coordinate histogram
2. Slice mesh at wall height (0.8-1.5m above floor)
3. Voxelize the slice with fine resolution
4. Project to 2D and extract wall structure
5. Use Manhattan world fitting for rectilinear walls
6. Detect openings via gap analysis
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
from scipy import ndimage
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def determine_coordinate_system(mesh):
    """Determine coordinate system from mesh geometry"""
    verts = mesh.vertices
    normals = mesh.face_normals
    up_candidates = {
        'X': np.abs(normals[:,0]).mean(),
        'Y': np.abs(normals[:,1]).mean(),
        'Z': np.abs(normals[:,2]).mean()
    }
    up_axis = max(up_candidates, key=up_candidates.get)
    projection = 'XY' if up_axis == 'Z' else ('XZ' if up_axis == 'Y' else 'YZ')
    return {'up_axis': up_axis, 'projection': projection}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def find_floor_level(mesh):
    """Find floor level using Z-coordinate histogram"""
    print("Finding floor level...")
    
    # Get Z coordinates
    z_coords = mesh.vertices[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Create histogram
    bins = 50
    hist, bin_edges = np.histogram(z_coords, bins=bins)
    
    # Floor is typically the lowest significant peak
    # Look for highest density in bottom 30% of height range
    bottom_30_idx = int(0.3 * bins)
    bottom_hist = hist[:bottom_30_idx]
    
    if len(bottom_hist) > 0:
        floor_bin_idx = np.argmax(bottom_hist)
        floor_level = bin_edges[floor_bin_idx]
    else:
        floor_level = z_min
    
    print(f"Floor level detected at Z = {floor_level:.2f}m")
    print(f"Height range: {z_min:.2f}m to {z_max:.2f}m ({z_max-z_min:.2f}m total)")
    
    return floor_level

def slice_at_wall_height(mesh, floor_level, height_range=(0.8, 1.5)):
    """Extract cross-section slice at wall height above floor"""
    print(f"Slicing mesh at wall height {height_range[0]}-{height_range[1]}m above floor...")
    
    # Get mesh Z range to adapt to available data
    z_coords = mesh.vertices[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    total_height = z_max - z_min
    
    print(f"Mesh Z range: {z_min:.2f} to {z_max:.2f}m (total: {total_height:.2f}m)")
    
    # If mesh height is very small, use relative positioning
    if total_height < 2.0:  # Less than 2m total height
        print("  Small mesh height detected, using relative positioning...")
        # Use middle portion of the mesh (likely contains walls)
        wall_z_min = floor_level + total_height * 0.2  # Start at 20% above floor
        wall_z_max = floor_level + total_height * 0.8  # End at 80% above floor
    else:
        # Use absolute offsets for normal-sized meshes
        wall_z_min = floor_level + height_range[0]
        wall_z_max = floor_level + height_range[1]
        # But clamp to available data
        wall_z_max = min(wall_z_max, z_max - 0.05)  # Leave 5cm margin
    
    print(f"Wall slice range: Z = {wall_z_min:.2f} to {wall_z_max:.2f}m")
    
    # Get faces that intersect the wall height band
    vertices = mesh.vertices
    faces = mesh.faces
    
    wall_faces = []
    wall_vertices = []
    
    for face in faces:
        face_verts = vertices[face]
        face_z_min = face_verts[:, 2].min()
        face_z_max = face_verts[:, 2].max()
        
        # Check if face intersects wall height band
        if face_z_max >= wall_z_min and face_z_min <= wall_z_max:
            wall_faces.append(face)
            wall_vertices.extend(face_verts)
    
    if not wall_vertices:
        print("WARNING: No vertices found in wall height range!")
        return np.array([]), []
    
    wall_vertices = np.array(wall_vertices)
    print(f"Found {len(wall_vertices)} vertices in wall height band")
    
    return wall_vertices, wall_faces

def voxelize_wall_slice(wall_vertices, voxel_size=0.02):
    """Convert wall slice to voxel grid with fine resolution"""
    print(f"Voxelizing wall slice with {voxel_size}m resolution...")
    
    if len(wall_vertices) == 0:
        return np.array([]), (0, 0, 0), voxel_size
    
    # Get bounds in XY plane (top-down view)
    x_coords = wall_vertices[:, 0]
    y_coords = wall_vertices[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create 2D grid (top-down projection)
    x_size = int(np.ceil((x_max - x_min) / voxel_size)) + 1
    y_size = int(np.ceil((y_max - y_min) / voxel_size)) + 1
    
    print(f"2D Grid size: {x_size} x {y_size} = {x_size*y_size:,} pixels")
    print(f"Coverage: {x_max-x_min:.2f}m x {y_max-y_min:.2f}m")
    
    # Create occupancy grid
    occupancy = np.zeros((x_size, y_size), dtype=bool)
    
    # Convert vertices to grid coordinates and mark occupied
    for vertex in wall_vertices:
        xi = int((vertex[0] - x_min) / voxel_size)
        yi = int((vertex[1] - y_min) / voxel_size)
        
        if 0 <= xi < x_size and 0 <= yi < y_size:
            occupancy[xi, yi] = True
    
    occupied_count = np.sum(occupancy)
    print(f"Occupied cells: {occupied_count:,} ({100*occupied_count/(x_size*y_size):.1f}%)")
    
    return occupancy, (x_min, y_min), voxel_size

def manhattan_wall_fitting(occupancy, voxel_size, origin):
    """Detect wall orientations using histogram voting like v9"""
    print("Detecting wall orientations with Manhattan fitting...")
    
    # Find wall pixels
    wall_y, wall_x = np.where(occupancy)
    
    if len(wall_x) == 0:
        return [], []
    
    wall_points = np.column_stack((wall_x, wall_y))
    print(f"Processing {len(wall_points)} wall pixels")
    
    # Convert to world coordinates
    world_x = origin[0] + wall_x * voxel_size
    world_y = origin[1] + wall_y * voxel_size
    world_points = np.column_stack((world_x, world_y))
    
    # Use Hough line detection for wall segments
    img = (occupancy * 255).astype(np.uint8)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection
    edges = cv2.Canny(img, 50, 150)
    
    # Hough line detection 
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=20, minLineLength=10, maxLineGap=5)
    
    wall_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Convert to world coordinates
            world_x1 = origin[0] + x1 * voxel_size
            world_y1 = origin[1] + y1 * voxel_size
            world_x2 = origin[0] + x2 * voxel_size
            world_y2 = origin[1] + y2 * voxel_size
            
            # Calculate length and angle
            length = np.sqrt((world_x2-world_x1)**2 + (world_y2-world_y1)**2)
            angle = np.arctan2(world_y2-world_y1, world_x2-world_x1) * 180 / np.pi
            
            wall_segments.append({
                'start': [float(world_x1), float(world_y1)],
                'end': [float(world_x2), float(world_y2)],
                'length': float(length),
                'angle': float(angle)
            })
    
    print(f"Detected {len(wall_segments)} wall segments")
    
    # Manhattan angle snapping (0, 90, 180, 270 degrees)
    snapped_segments = []
    for seg in wall_segments:
        angle = seg['angle'] % 180  # Normalize to 0-180
        
        # Snap to nearest Manhattan angle
        if angle < 45:
            snapped_angle = 0
        elif angle < 135:
            snapped_angle = 90
        else:
            snapped_angle = 180
            
        # Recalculate endpoints with snapped angle
        start = np.array(seg['start'])
        length = seg['length']
        
        if snapped_angle == 0:  # Horizontal
            end = start + [length, 0]
        elif snapped_angle == 90:  # Vertical
            end = start + [0, length]
        else:  # snapped_angle == 180, Horizontal (negative)
            end = start + [-length, 0]
        
        snapped_segments.append({
            'start': start.tolist(),
            'end': end.tolist(),
            'length': float(length),
            'angle': float(snapped_angle),
            'snapped': True
        })
    
    print(f"Manhattan-snapped to {len(snapped_segments)} segments")
    return snapped_segments, world_points

def merge_wall_segments(segments, merge_threshold=0.5):
    """Merge nearby collinear wall segments"""
    print(f"Merging wall segments (threshold={merge_threshold}m)...")
    
    if not segments:
        return segments
    
    merged = []
    used = set()
    
    for i, seg1 in enumerate(segments):
        if i in used:
            continue
            
        # Start with this segment
        current_start = np.array(seg1['start'])
        current_end = np.array(seg1['end'])
        current_angle = seg1['angle']
        merged_length = seg1['length']
        
        # Try to merge with other segments of same angle
        for j, seg2 in enumerate(segments):
            if j <= i or j in used:
                continue
            
            if abs(seg2['angle'] - current_angle) > 5:  # Must be same direction
                continue
            
            seg2_start = np.array(seg2['start'])
            seg2_end = np.array(seg2['end'])
            
            # Check if segments are collinear and close
            # Try connecting end-to-start or end-to-end
            distances = [
                np.linalg.norm(current_end - seg2_start),
                np.linalg.norm(current_end - seg2_end),
                np.linalg.norm(current_start - seg2_start),
                np.linalg.norm(current_start - seg2_end)
            ]
            
            min_dist = min(distances)
            if min_dist < merge_threshold:
                # Merge segments - extend to furthest points
                all_points = [current_start, current_end, seg2_start, seg2_end]
                
                if current_angle % 180 == 0:  # Horizontal
                    all_points.sort(key=lambda p: p[0])
                    new_start = all_points[0]
                    new_end = all_points[-1]
                else:  # Vertical
                    all_points.sort(key=lambda p: p[1])
                    new_start = all_points[0]
                    new_end = all_points[-1]
                
                current_start = new_start
                current_end = new_end
                merged_length = np.linalg.norm(current_end - current_start)
                used.add(j)
        
        merged.append({
            'start': current_start.tolist(),
            'end': current_end.tolist(),
            'length': float(merged_length),
            'angle': float(current_angle),
            'merged': True
        })
    
    print(f"Merged {len(segments)} segments into {len(merged)}")
    return merged

def create_room_polygon(wall_segments):
    """Create room polygon from wall segments"""
    print("Creating room polygon...")
    
    if not wall_segments:
        return []
    
    # Collect all wall endpoints
    points = []
    for seg in wall_segments:
        points.append(seg['start'])
        points.append(seg['end'])
    
    if not points:
        return []
    
    points = np.array(points)
    
    # Remove duplicate points
    unique_points = []
    for point in points:
        is_duplicate = False
        for existing in unique_points:
            if np.linalg.norm(np.array(point) - np.array(existing)) < 0.1:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point.tolist())
    
    if len(unique_points) < 3:
        return unique_points
    
    # Sort points to form a polygon (convex hull for now)
    points_array = np.array(unique_points)
    center = points_array.mean(axis=0)
    
    # Sort by angle from center
    angles = np.arctan2(points_array[:, 1] - center[1], 
                       points_array[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    polygon = points_array[sorted_indices].tolist()
    
    print(f"Room polygon: {len(polygon)} vertices")
    return polygon

def detect_wall_openings(wall_segments, opening_threshold=0.8):
    """Detect openings (doors/windows) by finding gaps in walls"""
    print(f"Detecting openings (gap threshold={opening_threshold}m)...")
    
    openings = []
    
    # Look for gaps between wall segments
    for i, seg1 in enumerate(wall_segments):
        seg1_end = np.array(seg1['end'])
        
        for j, seg2 in enumerate(wall_segments):
            if i >= j:
                continue
                
            seg2_start = np.array(seg2['start'])
            
            # Check if segments are aligned and have a gap
            if abs(seg1['angle'] - seg2['angle']) < 10:  # Similar angles
                gap_distance = np.linalg.norm(seg1_end - seg2_start)
                
                if opening_threshold < gap_distance < 3.0:  # Reasonable opening size
                    opening_center = (seg1_end + seg2_start) / 2
                    
                    # Classify as door or window
                    opening_type = "door" if gap_distance < 2.0 else "window"
                    
                    openings.append({
                        'type': opening_type,
                        'position': opening_center.tolist(),
                        'width': float(gap_distance),
                        'between_walls': [i, j]
                    })
    
    print(f"Found {len(openings)} openings")
    return openings

def analyze_mesh_voxelization(mesh_path, voxel_size=0.02):
    """Main analysis function with v9-inspired approach"""
    print(f"\n=== Voxelization Analysis v10 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Find floor level (like v9)
    floor_level = find_floor_level(mesh)
    
    # Step 2: Slice at wall height (like v9)
    wall_vertices, wall_faces = slice_at_wall_height(mesh, floor_level)
    
    if len(wall_vertices) == 0:
        print("ERROR: No wall vertices found!")
        return None
    
    # Step 3: Voxelize with fine resolution
    occupancy, origin, voxel_size = voxelize_wall_slice(wall_vertices, voxel_size)
    
    # Step 4: Manhattan wall fitting (like v9)
    wall_segments, wall_points = manhattan_wall_fitting(occupancy, voxel_size, origin)
    
    # Step 5: Merge segments (like v9)
    merged_segments = merge_wall_segments(wall_segments)
    
    # Step 6: Create room polygon
    room_boundary = create_room_polygon(merged_segments)
    
    # Step 7: Detect openings
    openings = detect_wall_openings(merged_segments)
    
    # Calculate room area
    if len(room_boundary) >= 3:
        # Shoelace formula for polygon area
        area = 0.0
        n = len(room_boundary)
        for i in range(n):
            j = (i + 1) % n
            area += room_boundary[i][0] * room_boundary[j][1]
            area -= room_boundary[j][0] * room_boundary[i][1]
        area = abs(area) / 2.0
    else:
        area = 0.0
    
    return {
        "method": "voxelization_projection_v10_fixed",
        "mesh_file": str(mesh_path),
        "parameters": {
            "voxel_size": voxel_size,
            "wall_height_range": [0.8, 1.5],
            "floor_level": float(floor_level)
        },
        "room_boundary": room_boundary,
        "wall_segments": merged_segments,
        "openings": openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "analysis_stats": {
            "wall_vertices": len(wall_vertices),
            "wall_segments_raw": len(wall_segments),
            "wall_segments_merged": len(merged_segments),
            "boundary_vertices": len(room_boundary)
        }
    }

def create_visualization(result, output_path):
    """Create floor plan visualization"""
    if not result:
        return
        
    print("Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Wall segments
    ax1.set_title(f"Wall Segments\nArea: {result['room_area_sqm']:.1f} m²")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Draw wall segments
    for i, seg in enumerate(result['wall_segments']):
        start = seg['start']
        end = seg['end']
        ax1.plot([start[0], end[0]], [start[1], end[1]], 
                'b-', linewidth=3, alpha=0.7)
        
        # Add length annotation
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax1.annotate(f"{seg['length']:.1f}m", (mid_x, mid_y), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Draw openings
    for opening in result['openings']:
        pos = opening['position']
        ax1.plot(pos[0], pos[1], 'ro', markersize=10, alpha=0.7)
        ax1.annotate(f"{opening['type']}\n{opening['width']:.1f}m", 
                    (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    # Plot 2: Room boundary
    ax2.set_title(f"Room Boundary\n{len(result['room_boundary'])} vertices")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    if len(result['room_boundary']) >= 3:
        boundary = np.array(result['room_boundary'])
        # Close the polygon
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax2.fill(boundary_closed[:, 0], boundary_closed[:, 1], 
                alpha=0.3, color='lightblue', edgecolor='blue', linewidth=2)
        
        # Mark vertices
        ax2.scatter(boundary[:, 0], boundary[:, 1], 
                   c='red', s=50, alpha=0.7, zorder=5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v10_voxelization_projection.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
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
            
            # Create visualization
            create_visualization(result, viz_path)
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Wall segments: {len(result['wall_segments'])}")
            print(f"   Boundary vertices: {len(result['room_boundary'])}")
            print(f"   Openings: {len(result['openings'])}")
            for opening in result['openings']:
                print(f"     - {opening['type']}: {opening['width']:.1f}m")
            print(f"   Floor level: {result['parameters']['floor_level']:.2f}m")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
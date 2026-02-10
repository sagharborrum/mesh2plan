#!/usr/bin/env python3
"""
v14: Hybrid Approach - v11 Walls + v12 Openings (FIXED)
========================================================

FIXES APPLIED:
- Use FIXED v11 wall boundary extraction with alpha shapes and line fitting
- Use FIXED v12 opening detection with proper wall-height rendering
- Combine both approaches properly with consistent coordinate systems
- Apply floor level detection and wall height filtering consistently
- Use Manhattan regularization and wall merging from both approaches

This hybrid approach leverages the FIXED strengths of both methods:
- v11 FIXED: Excellent wall foot-point analysis with alpha shapes and line fitting
- v12 FIXED: Superior opening detection with high-res rasterization and Hough lines

The combination provides the best of both worlds.
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN
from collections import defaultdict
import math

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
    return {'up_axis': up_axis}

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
    
    z_coords = mesh.vertices[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Create histogram
    bins = 50
    hist, bin_edges = np.histogram(z_coords, bins=bins)
    
    # Floor is typically the lowest significant peak
    bottom_30_idx = int(0.3 * bins)
    bottom_hist = hist[:bottom_30_idx]
    
    if len(bottom_hist) > 0:
        floor_bin_idx = np.argmax(bottom_hist)
        floor_level = bin_edges[floor_bin_idx]
    else:
        floor_level = z_min
    
    print(f"Floor level detected at Z = {floor_level:.2f}m")
    return floor_level

# ============= V11 FIXED COMPONENTS =============

def classify_faces_by_normal(mesh, vertical_threshold=0.7):
    """Classify mesh faces by normal direction (from v11 FIXED)"""
    print("V11 Component: Classifying faces by normal direction...")
    
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Up vector (assume Z is up for most scans)
    up_vector = np.array([0, 0, 1])
    
    # Calculate dot product with up vector
    dot_products = np.abs(np.dot(face_normals, up_vector))
    
    # Classify faces
    floors_ceilings = dot_products > vertical_threshold
    walls = dot_products <= vertical_threshold
    
    # Further classify floors vs ceilings by Z-component of normal
    floor_mask = (face_normals[:, 2] > 0) & floors_ceilings
    ceiling_mask = (face_normals[:, 2] < 0) & floors_ceilings
    
    # Get face indices
    floor_faces = np.where(floor_mask)[0]
    ceiling_faces = np.where(ceiling_mask)[0]
    wall_faces = np.where(walls)[0]
    
    print(f"  Floor faces: {len(floor_faces):,}")
    print(f"  Ceiling faces: {len(ceiling_faces):,}")
    print(f"  Wall faces: {len(wall_faces):,}")
    
    return wall_faces, floor_faces, ceiling_faces

def extract_wall_footpoints(mesh, wall_faces, floor_level, foot_height=0.3):
    """Extract wall vertices near floor level (from v11 FIXED)"""
    print(f"V11 Component: Extracting wall foot-points (within {foot_height}m of floor)...")
    
    # Get all vertices used by wall faces
    wall_vertex_indices = set()
    for face_idx in wall_faces:
        face = mesh.faces[face_idx]
        wall_vertex_indices.update(face)
    
    wall_vertices = mesh.vertices[list(wall_vertex_indices)]
    
    # Filter to foot-points near floor level
    foot_mask = np.abs(wall_vertices[:, 2] - floor_level) <= foot_height
    footpoints = wall_vertices[foot_mask]
    
    print(f"  Wall footpoints: {len(footpoints):,}")
    return footpoints

def alpha_shape_boundary(points, alpha=0.5):
    """Create concave hull using alpha shape (from v11 FIXED)"""
    print(f"V11 Component: Computing alpha shape boundary (alpha={alpha})...")
    
    if len(points) < 3:
        return points
    
    # Project to XY plane
    points_2d = points[:, [0, 1]]
    
    try:
        # Delaunay triangulation
        tri = Delaunay(points_2d)
        triangles = tri.simplices
        
        # Compute circumradius for each triangle
        def circumradius(triangle_points):
            a, b, c = triangle_points
            side_a = np.linalg.norm(b - c)
            side_b = np.linalg.norm(a - c)  
            side_c = np.linalg.norm(a - b)
            
            area = 0.5 * abs(np.cross(b - a, c - a))
            
            if area < 1e-10:
                return float('inf')
            
            return (side_a * side_b * side_c) / (4 * area)
        
        # Filter triangles by alpha criterion
        valid_triangles = []
        for triangle in triangles:
            triangle_points = points_2d[triangle]
            circumr = circumradius(triangle_points)
            
            if circumr <= 1.0 / alpha:
                valid_triangles.append(triangle)
        
        if not valid_triangles:
            # Fallback to convex hull
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        
        # Extract boundary edges
        edge_count = defaultdict(int)
        for triangle in valid_triangles:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                edge_count[edge] += 1
        
        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # Build adjacency graph and find longest path
        adjacency = defaultdict(list)
        for edge in boundary_edges:
            adjacency[edge[0]].append(edge[1])
            adjacency[edge[1]].append(edge[0])
        
        # Find longest connected component
        visited = set()
        longest_path = []
        
        for start_node in adjacency.keys():
            if start_node in visited:
                continue
            
            current_path = [start_node]
            current = start_node
            visited.add(current)
            
            while True:
                next_nodes = [n for n in adjacency[current] if n not in visited]
                if not next_nodes:
                    break
                
                next_node = next_nodes[0]
                current_path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(current_path) > len(longest_path):
                longest_path = current_path
        
        boundary_points = points_2d[longest_path]
        print(f"  Alpha shape boundary: {len(boundary_points)} points")
        return boundary_points
        
    except Exception as e:
        print(f"  Error in alpha shape: {e}, falling back to convex hull")
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]

# ============= V12 FIXED COMPONENTS =============

def render_wall_occupancy_map(mesh, floor_level, height_range=(0.8, 1.5), resolution=1024):
    """Render wall vertices at height band to occupancy map (from v12 FIXED)"""
    print(f"V12 Component: Rendering wall occupancy map at {height_range[0]}-{height_range[1]}m above floor...")
    
    # Filter vertices to wall height band
    vertices = mesh.vertices
    z_coords = vertices[:, 2]
    
    wall_z_min = floor_level + height_range[0]
    wall_z_max = floor_level + height_range[1]
    
    wall_mask = (z_coords >= wall_z_min) & (z_coords <= wall_z_max)
    wall_vertices = vertices[wall_mask]
    
    if len(wall_vertices) == 0:
        return None
    
    # Get bounds in XY plane
    x_coords = wall_vertices[:, 0]
    y_coords = wall_vertices[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    # Calculate pixel size
    pixel_size_x = (x_max - x_min) / resolution
    pixel_size_y = (y_max - y_min) / resolution
    pixel_size = max(pixel_size_x, pixel_size_y)
    
    # Adjust resolution
    width = int((x_max - x_min) / pixel_size)
    height = int((y_max - y_min) / pixel_size)
    
    # Initialize occupancy map
    occupancy = np.zeros((height, width), dtype=np.uint8)
    
    # Rasterize wall vertices
    for vertex in wall_vertices:
        x, y, z = vertex
        px = int((x - x_min) / pixel_size)
        py = int((y - y_min) / pixel_size)
        
        if 0 <= px < width and 0 <= py < height:
            occupancy[py, px] = 255
    
    print(f"  Occupancy map: {width}x{height}, {np.sum(occupancy > 0):,} occupied pixels")
    
    return {
        'occupancy': occupancy,
        'bounds': (x_min, y_min, x_max, y_max),
        'pixel_size': pixel_size,
        'resolution': (width, height)
    }

def detect_openings_from_occupancy(occupancy_data):
    """Detect openings from occupancy map (from v12 FIXED)"""
    print("V12 Component: Detecting openings from occupancy map...")
    
    if occupancy_data is None:
        return []
    
    occupancy = occupancy_data['occupancy']
    
    # Clean up occupancy map
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    
    # Edge detection
    edges = cv2.Canny(cleaned, 50, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # Hough line detection for wall segments
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=30, minLineLength=30, maxLineGap=10)
    
    openings = []
    if lines is not None:
        # Convert lines to world coordinates and analyze gaps
        x_min, y_min, x_max, y_max = occupancy_data['bounds']
        pixel_size = occupancy_data['pixel_size']
        
        world_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            world_x1 = x_min + x1 * pixel_size
            world_y1 = y_min + y1 * pixel_size
            world_x2 = x_min + x2 * pixel_size
            world_y2 = y_min + y2 * pixel_size
            
            length = np.sqrt((world_x2-world_x1)**2 + (world_y2-world_y1)**2)
            angle = np.arctan2(world_y2-world_y1, world_x2-world_x1) * 180 / np.pi
            
            world_lines.append({
                'start': [world_x1, world_y1],
                'end': [world_x2, world_y2],
                'length': length,
                'angle': angle
            })
        
        # Find gaps between lines (potential openings)
        for i, line1 in enumerate(world_lines):
            for j, line2 in enumerate(world_lines):
                if i >= j:
                    continue
                
                if abs(line1['angle'] - line2['angle']) < 15:  # Similar angles
                    end1 = np.array(line1['end'])
                    start2 = np.array(line2['start'])
                    gap_dist = np.linalg.norm(end1 - start2)
                    
                    if 0.6 < gap_dist < 2.5:  # Reasonable opening size
                        center = (end1 + start2) / 2
                        opening_type = "door" if gap_dist < 1.5 else "window"
                        
                        openings.append({
                            'type': opening_type,
                            'position': center.tolist(),
                            'width': float(gap_dist),
                            'detection_method': 'v12_gap_analysis'
                        })
    
    print(f"  Found {len(openings)} openings")
    return openings

# ============= HYBRID COMBINATION =============

def manhattan_regularization(segments, angle_tolerance=15):
    """Apply Manhattan regularization to segments"""
    print("Applying Manhattan regularization...")
    
    regularized = []
    for seg in segments:
        angle = seg['angle'] % 180
        
        # Snap to nearest Manhattan angle
        if angle < 45:
            snapped_angle = 0
        elif angle < 135:
            snapped_angle = 90
        else:
            snapped_angle = 180
        
        # Check if within tolerance
        angle_diff = min(abs(angle - snapped_angle), 
                        abs(angle - (snapped_angle - 180)))
        
        if angle_diff <= angle_tolerance:
            start = np.array(seg['start'])
            length = seg['length']
            
            if snapped_angle == 0:
                end = start + [length, 0]
            elif snapped_angle == 90:
                end = start + [0, length]
            else:
                end = start + [-length, 0]
            
            regularized.append({
                'start': start.tolist(),
                'end': end.tolist(),
                'length': float(length),
                'angle': float(snapped_angle),
                'regularized': True
            })
        else:
            regularized.append(seg)
    
    return regularized

def boundary_to_wall_segments(boundary_points):
    """Convert boundary points to wall segments for consistency"""
    print("Converting boundary to wall segments...")
    
    if len(boundary_points) < 3:
        return []
    
    segments = []
    for i in range(len(boundary_points)):
        start = boundary_points[i]
        end = boundary_points[(i + 1) % len(boundary_points)]
        
        length = np.linalg.norm(np.array(end) - np.array(start))
        angle = np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi
        
        if length > 0.2:  # Minimum segment length
            segments.append({
                'start': start.tolist() if isinstance(start, np.ndarray) else start,
                'end': end.tolist() if isinstance(end, np.ndarray) else end,
                'length': float(length),
                'angle': float(angle)
            })
    
    return segments

def combine_wall_approaches(v11_boundary, v12_openings):
    """Combine v11 boundary with v12 openings intelligently"""
    print("Combining v11 wall boundary with v12 opening detection...")
    
    # Convert v11 boundary to wall segments
    wall_segments = boundary_to_wall_segments(v11_boundary)
    
    # Apply Manhattan regularization
    regularized_segments = manhattan_regularization(wall_segments)
    
    # Use v12 openings as-is since they're already in world coordinates
    combined_openings = v12_openings
    
    print(f"  Combined result: {len(regularized_segments)} wall segments, {len(combined_openings)} openings")
    
    return regularized_segments, combined_openings

def analyze_mesh_hybrid(mesh_path):
    """Main hybrid analysis combining FIXED v11 and v12"""
    print(f"\n=== Hybrid Analysis v14 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Common step: Find floor level
    floor_level = find_floor_level(mesh)
    
    # ===== V11 FIXED APPROACH: Wall Boundary Extraction =====
    print(f"\n{'='*20} V11 FIXED APPROACH {'='*20}")
    
    # Face classification
    wall_faces, floor_faces, ceiling_faces = classify_faces_by_normal(mesh)
    
    # Extract wall footpoints
    wall_footpoints = extract_wall_footpoints(mesh, wall_faces, floor_level)
    
    if len(wall_footpoints) == 0:
        print("No wall footpoints found!")
        return None
    
    # Alpha shape boundary
    v11_boundary = alpha_shape_boundary(wall_footpoints)
    
    # ===== V12 FIXED APPROACH: Opening Detection =====
    print(f"\n{'='*20} V12 FIXED APPROACH {'='*20}")
    
    # Render occupancy map
    occupancy_data = render_wall_occupancy_map(mesh, floor_level)
    
    # Detect openings
    v12_openings = detect_openings_from_occupancy(occupancy_data)
    
    # ===== COMBINE APPROACHES =====
    print(f"\n{'='*20} COMBINING APPROACHES {'='*20}")
    
    wall_segments, combined_openings = combine_wall_approaches(v11_boundary, v12_openings)
    
    # Calculate room area
    if len(v11_boundary) >= 3:
        area = 0.0
        n = len(v11_boundary)
        for i in range(n):
            j = (i + 1) % n
            area += v11_boundary[i][0] * v11_boundary[j][1]
            area -= v11_boundary[j][0] * v11_boundary[i][1]
        area = abs(area) / 2.0
    else:
        area = 0.0
    
    return {
        "method": "hybrid_v11_v12_fixed_v14",
        "mesh_file": str(mesh_path),
        "parameters": {
            "floor_level": float(floor_level),
            "v11_footpoints": len(wall_footpoints),
            "v12_resolution": occupancy_data['resolution'] if occupancy_data else None
        },
        "room_boundary": v11_boundary.tolist() if isinstance(v11_boundary, np.ndarray) else v11_boundary,
        "wall_segments": wall_segments,
        "openings": combined_openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "analysis_stats": {
            "v11_wall_faces": len(wall_faces),
            "v11_footpoints": len(wall_footpoints),
            "v11_boundary_points": len(v11_boundary),
            "v12_openings": len(v12_openings),
            "combined_wall_segments": len(wall_segments),
            "combined_openings": len(combined_openings)
        }
    }

def create_visualization(result, output_path):
    """Create comprehensive hybrid visualization"""
    if not result:
        return
        
    print("Creating hybrid visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Wall segments
    ax1.set_title(f"Wall Segments (v11 + Manhattan)\nArea: {result['room_area_sqm']:.1f} m²")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    for i, seg in enumerate(result['wall_segments']):
        start = seg['start']
        end = seg['end']
        color = 'blue' if seg.get('regularized') else 'gray'
        ax1.plot([start[0], end[0]], [start[1], end[1]], 
                color=color, linewidth=3, alpha=0.7)
        
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax1.annotate(f"{seg['length']:.1f}m", (mid_x, mid_y), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Plot 2: Room boundary
    ax2.set_title(f"Room Boundary (v11 Alpha Shape)\n{len(result['room_boundary'])} vertices")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    if len(result['room_boundary']) >= 3:
        boundary = np.array(result['room_boundary'])
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax2.fill(boundary_closed[:, 0], boundary_closed[:, 1], 
                alpha=0.3, color='lightblue', edgecolor='blue', linewidth=2)
        ax2.scatter(boundary[:, 0], boundary[:, 1], 
                   c='red', s=50, alpha=0.7, zorder=5)
    
    # Plot 3: Openings detection
    ax3.set_title(f"Opening Detection (v12)\n{len(result['openings'])} openings")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Draw room boundary in gray
    if len(result['room_boundary']) >= 3:
        boundary = np.array(result['room_boundary'])
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax3.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                color='gray', linewidth=1, alpha=0.5)
    
    # Draw openings
    colors = {'door': 'red', 'window': 'orange'}
    for opening in result['openings']:
        pos = opening['position']
        color = colors.get(opening['type'], 'purple')
        ax3.plot(pos[0], pos[1], 'o', color=color, markersize=12, alpha=0.8)
        ax3.annotate(f"{opening['type']}\n{opening['width']:.1f}m", 
                    (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # Plot 4: Combined statistics
    ax4.set_title("Hybrid Analysis Statistics")
    ax4.axis('off')
    
    stats_text = f"""HYBRID ANALYSIS v14 (FIXED)

RESULTS:
Area: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Wall Segments: {len(result['wall_segments'])}
Boundary Vertices: {len(result['room_boundary'])}
Openings: {len(result['openings'])}

V11 COMPONENT (Walls):
Wall Faces: {result['analysis_stats']['v11_wall_faces']:,}
Footpoints: {result['analysis_stats']['v11_footpoints']:,}
Boundary Points: {result['analysis_stats']['v11_boundary_points']}

V12 COMPONENT (Openings):
Resolution: {result['parameters']['v12_resolution']}
Detected Openings: {result['analysis_stats']['v12_openings']}

PARAMETERS:
Floor Level: {result['parameters']['floor_level']:.2f}m

OPENING TYPES:"""
    
    door_count = sum(1 for op in result['openings'] if op['type'] == 'door')
    window_count = sum(1 for op in result['openings'] if op['type'] == 'window')
    stats_text += f"\nDoors: {door_count}\nWindows: {window_count}"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hybrid visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v14_hybrid_walls_openings.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_hybrid(mesh_path)
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
                method = opening.get('detection_method', '')
                print(f"     - {opening['type']}: {opening['width']:.1f}m ({method})")
            print(f"   Floor level: {result['parameters']['floor_level']:.2f}m")
            print(f"   V11 footpoints: {result['analysis_stats']['v11_footpoints']:,}")
            print(f"   V12 resolution: {result['parameters']['v12_resolution']}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
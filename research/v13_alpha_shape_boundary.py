#!/usr/bin/env python3
"""
v13: Alpha Shape Based Room Boundary Extraction
===============================================

Approach:
1. Extract mesh vertices at a specific height range (floor level)
2. Project points to 2D (X-Z plane)
3. Use Alpha Shapes to create a boundary that can handle concave shapes
4. Refine the boundary to remove interior details
5. Detect openings as concave regions in the alpha shape

Alpha shapes are a generalization of convex hulls that can capture 
concave boundaries by controlling the "alpha" parameter.
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from collections import defaultdict
import math

warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def extract_floor_points(mesh, floor_height_range=None, sample_ratio=0.1):
    """Extract points at floor level"""
    print("Extracting floor-level points...")
    
    vertices = mesh.vertices
    
    # Determine floor height range
    if floor_height_range is None:
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        height_range = y_max - y_min
        floor_height_range = (y_min, y_min + height_range * 0.1)  # Bottom 10%
    
    print(f"  Floor height range: {floor_height_range[0]:.2f} - {floor_height_range[1]:.2f} m")
    
    # Filter vertices by height
    floor_mask = ((vertices[:, 1] >= floor_height_range[0]) & 
                  (vertices[:, 1] <= floor_height_range[1]))
    floor_points = vertices[floor_mask]
    
    print(f"  Floor points before sampling: {len(floor_points):,}")
    
    # Sample points to reduce density for alpha shape computation
    if len(floor_points) > 1000:
        n_samples = max(1000, int(len(floor_points) * sample_ratio))
        indices = np.random.choice(len(floor_points), n_samples, replace=False)
        floor_points = floor_points[indices]
    
    print(f"  Floor points after sampling: {len(floor_points):,}")
    
    return floor_points

def points_to_2d(points_3d):
    """Project 3D points to 2D (X-Z plane)"""
    return points_3d[:, [0, 2]]  # Take X and Z coordinates

def compute_alpha_shape(points_2d, alpha=1.0):
    """Compute alpha shape (concave hull) of 2D points"""
    print(f"Computing alpha shape (alpha = {alpha})...")
    
    if len(points_2d) < 3:
        print("  Not enough points for triangulation")
        return [], []
    
    # Delaunay triangulation
    try:
        tri = Delaunay(points_2d)
    except Exception as e:
        print(f"  Error in triangulation: {e}")
        return [], []
    
    # Find triangles and edges
    triangles = tri.simplices
    
    # Compute circumradius for each triangle
    def circumradius(triangle_points):
        """Compute circumradius of a triangle"""
        a, b, c = triangle_points
        
        # Side lengths
        side_a = np.linalg.norm(b - c)
        side_b = np.linalg.norm(a - c)  
        side_c = np.linalg.norm(a - b)
        
        # Area using cross product
        area = 0.5 * abs(np.cross(b - a, c - a))
        
        if area < 1e-10:  # Degenerate triangle
            return float('inf')
        
        # Circumradius formula
        return (side_a * side_b * side_c) / (4 * area)
    
    # Filter triangles by alpha criterion
    valid_triangles = []
    for triangle in triangles:
        triangle_points = points_2d[triangle]
        circumr = circumradius(triangle_points)
        
        if circumr <= 1.0 / alpha:
            valid_triangles.append(triangle)
    
    print(f"  Valid triangles: {len(valid_triangles)} / {len(triangles)}")
    
    if not valid_triangles:
        print("  No valid triangles found - alpha may be too small")
        return [], []
    
    # Extract boundary edges
    edge_count = defaultdict(int)
    
    for triangle in valid_triangles:
        for i in range(3):
            edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
            edge_count[edge] += 1
    
    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    print(f"  Boundary edges: {len(boundary_edges)}")
    
    return boundary_edges, valid_triangles

def edges_to_boundary_polygon(boundary_edges, points_2d):
    """Convert boundary edges to ordered polygon"""
    print("Converting edges to boundary polygon...")
    
    if not boundary_edges:
        return []
    
    # Build adjacency graph
    adjacency = defaultdict(list)
    for edge in boundary_edges:
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
    
    # Find the longest connected component
    visited = set()
    longest_path = []
    
    for start_node in adjacency.keys():
        if start_node in visited:
            continue
            
        # Trace path from this starting node
        current_path = [start_node]
        current = start_node
        visited.add(current)
        
        while True:
            # Find next unvisited neighbor
            next_nodes = [n for n in adjacency[current] if n not in visited]
            if not next_nodes:
                break
            
            next_node = next_nodes[0]
            current_path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        if len(current_path) > len(longest_path):
            longest_path = current_path
    
    # Convert indices to coordinates
    boundary_points = [points_2d[i] for i in longest_path]
    
    print(f"  Boundary polygon has {len(boundary_points)} points")
    
    return boundary_points

def detect_alpha_shape_openings(boundary_points, concavity_threshold=0.5):
    """Detect openings based on concave regions in alpha shape"""
    print("Detecting openings from alpha shape concavities...")
    
    openings = []
    
    if len(boundary_points) < 4:
        return openings
    
    # Analyze angles between consecutive boundary segments
    for i in range(len(boundary_points)):
        p1 = np.array(boundary_points[(i-1) % len(boundary_points)])
        p2 = np.array(boundary_points[i])
        p3 = np.array(boundary_points[(i+1) % len(boundary_points)])
        
        # Vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.acos(cos_angle)
        
        # Check for sharp concave angles (potential openings)
        if angle < math.pi - concavity_threshold:  # Sharp inward turn
            gap_length = np.linalg.norm(p3 - p1)
            
            if 0.5 < gap_length < 5.0:  # Reasonable opening size
                opening_type = "door" if gap_length < 2.0 else "window"
                center = (p1 + p3) / 2
                
                openings.append({
                    "type": opening_type,
                    "position": center.tolist(),
                    "size_meters": float(gap_length),
                    "angle_radians": float(angle)
                })
    
    print(f"  Found {len(openings)} potential openings")
    return openings

def refine_boundary(boundary_points, simplification_threshold=0.1):
    """Simplify boundary by removing redundant points"""
    print("Refining boundary...")
    
    if len(boundary_points) < 3:
        return boundary_points
    
    refined_points = [boundary_points[0]]  # Always keep first point
    
    for i in range(1, len(boundary_points) - 1):
        p1 = np.array(refined_points[-1])
        p2 = np.array(boundary_points[i])
        p3 = np.array(boundary_points[i + 1])
        
        # Check if middle point is close to the line between p1 and p3
        # Distance from point to line
        line_vec = p3 - p1
        point_vec = p2 - p1
        
        if np.linalg.norm(line_vec) < 1e-10:
            continue
            
        # Project point onto line
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        projection = p1 + t * line_vec
        distance = np.linalg.norm(p2 - projection)
        
        # Keep point if it's far enough from the line
        if distance > simplification_threshold:
            refined_points.append(boundary_points[i])
    
    refined_points.append(boundary_points[-1])  # Always keep last point
    
    print(f"  Refined from {len(boundary_points)} to {len(refined_points)} points")
    return refined_points

def analyze_mesh_alpha_shape(mesh_path, alpha=0.5, floor_height_range=None):
    """Main analysis function"""
    print(f"\n=== Alpha Shape Boundary Extraction: {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if hasattr(mesh, 'vertices'):
        print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    else:
        print("Error loading mesh")
        return None
    
    # Step 1: Extract floor-level points
    floor_points_3d = extract_floor_points(mesh, floor_height_range)
    
    if len(floor_points_3d) < 3:
        print("Not enough floor points found!")
        return None
    
    # Step 2: Project to 2D
    floor_points_2d = points_to_2d(floor_points_3d)
    
    # Step 3: Compute alpha shape
    boundary_edges, triangles = compute_alpha_shape(floor_points_2d, alpha)
    
    if not boundary_edges:
        print("No boundary found - trying larger alpha...")
        alpha *= 2
        boundary_edges, triangles = compute_alpha_shape(floor_points_2d, alpha)
        
        if not boundary_edges:
            print("Still no boundary found!")
            return None
    
    # Step 4: Convert edges to polygon
    boundary_points = edges_to_boundary_polygon(boundary_edges, floor_points_2d)
    
    if not boundary_points:
        print("Could not form boundary polygon!")
        return None
    
    # Step 5: Refine boundary
    refined_boundary = refine_boundary(boundary_points)
    
    # Step 6: Detect openings
    openings = detect_alpha_shape_openings(refined_boundary)
    
    # Calculate room area
    if len(refined_boundary) >= 3:
        area = 0.0
        for i in range(len(refined_boundary)):
            j = (i + 1) % len(refined_boundary)
            area += refined_boundary[i][0] * refined_boundary[j][1]
            area -= refined_boundary[j][0] * refined_boundary[i][1]
        area = abs(area) / 2.0
    else:
        area = 0.0
    
    return {
        "method": "alpha_shape_boundary_v13",
        "mesh_file": str(mesh_path),
        "parameters": {
            "alpha": alpha,
            "floor_height_range": floor_height_range,
            "floor_points_3d": len(floor_points_3d)
        },
        "room_boundary": refined_boundary,
        "openings": openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "alpha_shape_stats": {
            "boundary_edges": len(boundary_edges),
            "triangles": len(triangles),
            "boundary_points": len(boundary_points),
            "refined_points": len(refined_boundary)
        }
    }

def create_visualization(result, floor_points_2d, triangles, boundary_edges, output_path):
    """Create visualization of alpha shape analysis"""
    print(f"Creating visualization: {output_path}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Floor points
    ax1.scatter(floor_points_2d[:, 0], floor_points_2d[:, 1], 
                alpha=0.5, s=1, c='blue', label='Floor Points')
    ax1.set_title('Floor Points (2D Projection)')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Z (meters)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Alpha shape triangulation
    ax2.scatter(floor_points_2d[:, 0], floor_points_2d[:, 1], 
                alpha=0.3, s=1, c='lightblue')
    
    # Draw triangles
    for triangle in triangles:
        triangle_points = floor_points_2d[triangle]
        triangle_closed = np.vstack([triangle_points, triangle_points[0]])
        ax2.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'g-', alpha=0.3, linewidth=0.5)
    
    ax2.set_title('Alpha Shape Triangulation')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Boundary edges
    ax3.scatter(floor_points_2d[:, 0], floor_points_2d[:, 1], 
                alpha=0.3, s=1, c='lightblue')
    
    # Draw boundary edges
    for edge in boundary_edges:
        p1, p2 = floor_points_2d[edge[0]], floor_points_2d[edge[1]]
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
    
    ax3.set_title('Alpha Shape Boundary Edges')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Z (meters)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Final boundary polygon
    if result['room_boundary']:
        boundary = np.array(result['room_boundary'] + [result['room_boundary'][0]])
        ax4.plot(boundary[:, 0], boundary[:, 1], 'b-', linewidth=3, label='Room Boundary')
        ax4.fill(boundary[:, 0], boundary[:, 1], alpha=0.3, color='lightblue')
    
    # Plot openings
    for opening in result['openings']:
        pos = opening['position']
        ax4.plot(pos[0], pos[1], 'ro', markersize=10, label=f'{opening["type"].title()}')
    
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Z (meters)')
    ax4.set_title('Final Room Boundary')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Add stats text
    stats_text = f"""ANALYSIS RESULTS (V13)
    
Area: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Boundary Points: {len(result['room_boundary'])}
Openings: {len(result['openings'])}

PARAMETERS:
Alpha: {result['parameters']['alpha']}
Floor Points: {result['parameters']['floor_points_3d']:,}

ALPHA SHAPE STATS:
Boundary Edges: {result['alpha_shape_stats']['boundary_edges']}
Triangles: {result['alpha_shape_stats']['triangles']}
Original Boundary: {result['alpha_shape_stats']['boundary_points']} points
Refined Boundary: {result['alpha_shape_stats']['refined_points']} points
"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python v13_alpha_shape_boundary.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_alpha_shape(mesh_path)
        if result:
            # Save results
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NpEncoder)
            print(f"\n✅ Results saved to {output_path}")
            
            # Create visualization if requested
            if viz_path:
                # Re-run intermediate steps for visualization
                mesh = trimesh.load(mesh_path)
                floor_points_3d = extract_floor_points(mesh, result['parameters']['floor_height_range'])
                floor_points_2d = points_to_2d(floor_points_3d)
                boundary_edges, triangles = compute_alpha_shape(floor_points_2d, result['parameters']['alpha'])
                create_visualization(result, floor_points_2d, triangles, boundary_edges, viz_path)
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Boundary points: {len(result['room_boundary'])}")
            print(f"   Alpha shape triangles: {result['alpha_shape_stats']['triangles']}")
            print(f"   Openings: {len(result['openings'])}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main()
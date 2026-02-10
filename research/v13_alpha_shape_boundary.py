#!/usr/bin/env python3
"""
v13: Alpha Shape Based Room Boundary Extraction (FIXED)
=======================================================

FIXES APPLIED:
- Expand floor detection band to use proper wall-height range like v9
- Auto-tune alpha parameter based on data characteristics
- Reduce boundary refinement aggressiveness
- Use floor level detection like other fixed approaches
- Better opening detection using gap analysis
- Improve alpha shape computation for better concave boundaries

This approach:
1. Find floor level using histogram like v9
2. Extract points at wall height (foot-points)  
3. Auto-tune alpha parameter for optimal boundary
4. Use Alpha Shapes to handle concave room shapes
5. Apply gentle boundary refinement
6. Detect openings via concavity and gap analysis
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

def extract_wall_footpoints(mesh, floor_level, height_range=(0.0, 0.8), sample_ratio=0.3):
    """Extract points near floor level (wall foot-points) - EXPANDED RANGE"""
    print(f"Extracting wall foot-points {height_range[0]}-{height_range[1]}m above floor...")
    
    vertices = mesh.vertices
    total_height = vertices[:, 2].max() - vertices[:, 2].min()
    
    # Adapt to small meshes
    if total_height < 2.0:
        print(f"  Small mesh height ({total_height:.2f}m), using adaptive range...")
        z_min = floor_level + total_height * 0.1
        z_max = floor_level + total_height * 0.7
    else:
        z_min = floor_level + height_range[0]
        z_max = floor_level + height_range[1]
    
    print(f"  Height range: {z_min:.2f} - {z_max:.2f} m")
    
    # Filter vertices by height
    height_mask = (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
    wall_points = vertices[height_mask]
    
    print(f"  Wall points before sampling: {len(wall_points):,}")
    
    # Sample points but less aggressively
    if len(wall_points) > 2000:
        n_samples = max(2000, int(len(wall_points) * sample_ratio))
        indices = np.random.choice(len(wall_points), n_samples, replace=False)
        wall_points = wall_points[indices]
    
    print(f"  Wall points after sampling: {len(wall_points):,}")
    
    return wall_points

def points_to_2d(points_3d):
    """Project 3D points to 2D (X-Y plane for top-down view)"""
    return points_3d[:, [0, 1]]  # Take X and Y coordinates

def auto_tune_alpha(points_2d, alpha_candidates=None):
    """Auto-tune alpha parameter for optimal boundary"""
    print("Auto-tuning alpha parameter...")
    
    if alpha_candidates is None:
        # Create range of alpha values to test
        alpha_candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    best_alpha = alpha_candidates[0]
    best_score = -1
    best_boundary_length = 0
    
    for alpha in alpha_candidates:
        try:
            boundary_edges, triangles = compute_alpha_shape(points_2d, alpha, verbose=False)
            
            if not boundary_edges:
                continue
            
            # Score based on boundary complexity and reasonableness
            boundary_points = edges_to_boundary_polygon(boundary_edges, points_2d, verbose=False)
            
            if len(boundary_points) < 4:  # Too simple
                continue
            
            if len(boundary_points) > 50:  # Too complex
                continue
            
            # Calculate boundary length
            boundary_length = 0
            for i in range(len(boundary_points)):
                p1 = np.array(boundary_points[i])
                p2 = np.array(boundary_points[(i + 1) % len(boundary_points)])
                boundary_length += np.linalg.norm(p2 - p1)
            
            # Score: prefer moderate complexity with reasonable perimeter
            complexity_score = 1.0 / (1 + abs(len(boundary_points) - 12))  # Target ~12 points
            length_reasonableness = 1.0 / (1 + abs(boundary_length - 20))  # Target ~20m perimeter
            
            score = complexity_score * length_reasonableness
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_boundary_length = boundary_length
            
            print(f"    Alpha {alpha}: {len(boundary_points)} points, {boundary_length:.1f}m perimeter, score {score:.3f}")
            
        except Exception as e:
            print(f"    Alpha {alpha}: Failed ({e})")
            continue
    
    print(f"  Best alpha: {best_alpha} (score: {best_score:.3f}, perimeter: {best_boundary_length:.1f}m)")
    return best_alpha

def compute_alpha_shape(points_2d, alpha=1.0, verbose=True):
    """Compute alpha shape (concave hull) of 2D points"""
    if verbose:
        print(f"Computing alpha shape (alpha = {alpha})...")
    
    if len(points_2d) < 3:
        if verbose:
            print("  Not enough points for triangulation")
        return [], []
    
    # Delaunay triangulation
    try:
        tri = Delaunay(points_2d)
    except Exception as e:
        if verbose:
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
    
    if verbose:
        print(f"  Valid triangles: {len(valid_triangles)} / {len(triangles)}")
    
    if not valid_triangles:
        if verbose:
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
    
    if verbose:
        print(f"  Boundary edges: {len(boundary_edges)}")
    
    return boundary_edges, valid_triangles

def edges_to_boundary_polygon(boundary_edges, points_2d, verbose=True):
    """Convert boundary edges to ordered polygon"""
    if verbose:
        print("Converting edges to boundary polygon...")
    
    if not boundary_edges:
        return []
    
    # Build adjacency graph
    adjacency = defaultdict(list)
    for edge in boundary_edges:
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
    
    # Find the longest connected component (main boundary)
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
    boundary_points = [points_2d[i].tolist() for i in longest_path]
    
    if verbose:
        print(f"  Boundary polygon has {len(boundary_points)} points")
    
    return boundary_points

def gentle_boundary_refinement(boundary_points, simplification_threshold=0.05):
    """Gently simplify boundary - LESS AGGRESSIVE"""
    print("Gently refining boundary...")
    
    if len(boundary_points) < 4:
        return boundary_points
    
    refined_points = []
    
    for i in range(len(boundary_points)):
        current = np.array(boundary_points[i])
        prev_pt = np.array(boundary_points[(i - 1) % len(boundary_points)])
        next_pt = np.array(boundary_points[(i + 1) % len(boundary_points)])
        
        # Check if current point is roughly on line between prev and next
        if np.linalg.norm(next_pt - prev_pt) < 1e-6:
            refined_points.append(boundary_points[i])
            continue
        
        # Distance from current point to line prev->next
        line_vec = next_pt - prev_pt
        point_vec = current - prev_pt
        
        # Project point onto line
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        projection = prev_pt + t * line_vec
        distance = np.linalg.norm(current - projection)
        
        # Keep point if it's far enough from the line (gentler threshold)
        if distance > simplification_threshold:
            refined_points.append(boundary_points[i])
    
    print(f"  Gently refined from {len(boundary_points)} to {len(refined_points)} points")
    
    # Ensure we keep reasonable complexity
    if len(refined_points) < 4:
        print("  Refinement too aggressive, keeping original boundary")
        return boundary_points
    
    return refined_points

def detect_openings_improved(boundary_points, gap_threshold=0.8, concavity_threshold=0.3):
    """Improved opening detection using both gaps and concavity"""
    print("Detecting openings using improved gap and concavity analysis...")
    
    openings = []
    
    if len(boundary_points) < 4:
        return openings
    
    # Method 1: Gap analysis - look for unusually long edges
    edge_lengths = []
    for i in range(len(boundary_points)):
        p1 = np.array(boundary_points[i])
        p2 = np.array(boundary_points[(i + 1) % len(boundary_points)])
        edge_lengths.append(np.linalg.norm(p2 - p1))
    
    # Find edges significantly longer than median (potential openings)
    median_length = np.median(edge_lengths)
    
    for i, length in enumerate(edge_lengths):
        if length > max(gap_threshold, median_length * 2):  # Either absolute or relative threshold
            p1 = np.array(boundary_points[i])
            p2 = np.array(boundary_points[(i + 1) % len(boundary_points)])
            
            center = (p1 + p2) / 2
            opening_type = "door" if length < 2.0 else "window"
            
            openings.append({
                "type": opening_type,
                "position": center.tolist(),
                "width": float(length),
                "detection_method": "gap_analysis"
            })
    
    # Method 2: Concavity analysis for sharp inward angles
    for i in range(len(boundary_points)):
        p1 = np.array(boundary_points[(i-1) % len(boundary_points)])
        p2 = np.array(boundary_points[i])
        p3 = np.array(boundary_points[(i+1) % len(boundary_points)])
        
        # Vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
        
        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.acos(cos_angle)
        
        # Cross product to determine if it's concave (inward)
        cross = np.cross(v1, v2)
        
        # Check for sharp concave angles (inward turns)
        if cross < 0 and angle < math.pi - concavity_threshold:  # Sharp inward turn
            gap_length = np.linalg.norm(p3 - p1)
            
            if 0.5 < gap_length < 3.0:  # Reasonable opening size
                # Check if we already found this opening via gap analysis
                center = (p1 + p3) / 2
                is_duplicate = any(np.linalg.norm(np.array(op['position']) - center) < 0.5 
                                 for op in openings)
                
                if not is_duplicate:
                    opening_type = "door" if gap_length < 1.8 else "window"
                    
                    openings.append({
                        "type": opening_type,
                        "position": center.tolist(),
                        "width": float(gap_length),
                        "detection_method": "concavity_analysis",
                        "angle_radians": float(angle)
                    })
    
    print(f"  Found {len(openings)} potential openings")
    return openings

def analyze_mesh_alpha_shape(mesh_path, auto_tune=True):
    """Main analysis function with fixes applied"""
    print(f"\n=== Alpha Shape Boundary Extraction v13 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Find floor level (like v9)
    floor_level = find_floor_level(mesh)
    
    # Step 2: Extract wall foot-points with expanded range
    wall_points_3d = extract_wall_footpoints(mesh, floor_level)
    
    if len(wall_points_3d) < 10:
        print("Not enough wall points found!")
        return None
    
    # Step 3: Project to 2D
    wall_points_2d = points_to_2d(wall_points_3d)
    
    # Step 4: Auto-tune alpha parameter
    if auto_tune:
        alpha = auto_tune_alpha(wall_points_2d)
    else:
        alpha = 1.0  # Default fallback
    
    # Step 5: Compute alpha shape
    boundary_edges, triangles = compute_alpha_shape(wall_points_2d, alpha)
    
    if not boundary_edges:
        print("No boundary found - trying fallback alpha values...")
        for fallback_alpha in [0.5, 2.0, 5.0]:
            boundary_edges, triangles = compute_alpha_shape(wall_points_2d, fallback_alpha)
            if boundary_edges:
                alpha = fallback_alpha
                break
        
        if not boundary_edges:
            print("Still no boundary found!")
            return None
    
    # Step 6: Convert edges to polygon
    boundary_points = edges_to_boundary_polygon(boundary_edges, wall_points_2d)
    
    if not boundary_points:
        print("Could not form boundary polygon!")
        return None
    
    # Step 7: Gentle boundary refinement
    refined_boundary = gentle_boundary_refinement(boundary_points)
    
    # Step 8: Improved opening detection
    openings = detect_openings_improved(refined_boundary)
    
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
        "method": "alpha_shape_boundary_v13_fixed",
        "mesh_file": str(mesh_path),
        "parameters": {
            "alpha": alpha,
            "floor_level": float(floor_level),
            "wall_height_range": [0.0, 0.8],
            "auto_tuned": auto_tune,
            "wall_points_3d": len(wall_points_3d)
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

def create_visualization(result, output_path):
    """Create floor plan visualization"""
    if not result:
        return
        
    print("Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Room boundary with alpha shape details
    ax1.set_title(f"Alpha Shape Boundary\nArea: {result['room_area_sqm']:.1f} m² (α={result['parameters']['alpha']})")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Draw boundary
    if len(result['room_boundary']) >= 3:
        boundary = np.array(result['room_boundary'])
        # Close the polygon
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax1.fill(boundary_closed[:, 0], boundary_closed[:, 1], 
                alpha=0.3, color='lightblue', edgecolor='blue', linewidth=2)
        
        # Mark vertices
        ax1.scatter(boundary[:, 0], boundary[:, 1], 
                   c='red', s=50, alpha=0.7, zorder=5)
        
        # Number the vertices
        for i, point in enumerate(boundary):
            ax1.annotate(str(i), point, xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
    
    # Draw openings
    for opening in result['openings']:
        pos = opening['position']
        ax1.plot(pos[0], pos[1], 'ro', markersize=12, alpha=0.8)
        ax1.annotate(f"{opening['type']}\n{opening['width']:.1f}m", 
                    (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: Analysis statistics
    ax2.set_title("Analysis Statistics")
    ax2.axis('off')
    
    stats_text = f"""ALPHA SHAPE ANALYSIS v13 (FIXED)

RESULTS:
Area: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Boundary Points: {len(result['room_boundary'])}
Openings: {len(result['openings'])}

PARAMETERS:
Alpha: {result['parameters']['alpha']} {'(auto-tuned)' if result['parameters']['auto_tuned'] else '(manual)'}
Floor Level: {result['parameters']['floor_level']:.2f}m
Wall Height Range: {result['parameters']['wall_height_range'][0]:.1f}-{result['parameters']['wall_height_range'][1]:.1f}m
Wall Points: {result['parameters']['wall_points_3d']:,}

ALPHA SHAPE STATS:
Boundary Edges: {result['alpha_shape_stats']['boundary_edges']}
Triangles: {result['alpha_shape_stats']['triangles']}
Original Points: {result['alpha_shape_stats']['boundary_points']}
Refined Points: {result['alpha_shape_stats']['refined_points']}

OPENING DETAILS:"""
    
    for i, opening in enumerate(result['openings']):
        method = opening.get('detection_method', 'unknown')
        stats_text += f"\n{i+1}. {opening['type']}: {opening['width']:.1f}m ({method})"
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v13_alpha_shape_boundary.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
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
            
            # Create visualization
            create_visualization(result, viz_path)
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Boundary points: {len(result['room_boundary'])}")
            print(f"   Alpha parameter: {result['parameters']['alpha']} {'(auto-tuned)' if result['parameters']['auto_tuned'] else ''}")
            print(f"   Triangles: {result['alpha_shape_stats']['triangles']}")
            print(f"   Openings: {len(result['openings'])}")
            for opening in result['openings']:
                method = opening.get('detection_method', '')
                print(f"     - {opening['type']}: {opening['width']:.1f}m ({method})")
            print(f"   Floor level: {result['parameters']['floor_level']:.2f}m")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
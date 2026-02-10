#!/usr/bin/env python3
"""
v11: Normal-based Wall Segmentation (IMPROVED WITH MANHATTAN MERGING)
====================================================================

IMPROVEMENTS APPLIED:
- Better Manhattan regularization with histogram-based angle detection
- Proper wall segment merging like v9
- Group wall points by dominant directions
- Merge collinear and parallel segments
- Detect openings between merged segments

Target: ~11.5 m² with 5-8 wall segments and 2-4 openings

This approach:
1. Classify faces by normal direction (keep existing logic)
2. Extract wall foot-points near floor level  
3. Find dominant wall angles via histogram voting (like v9)
4. Group wall points by angle (parallel walls)
5. Fit line segments to each group and merge collinear ones
6. Detect gaps between segments as openings
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def analyze_mesh_geometry(mesh):
    """Analyze mesh to determine coordinate system"""
    print("Analyzing mesh geometry...")
    verts = mesh.vertices
    
    # Calculate spans
    spans = {
        'X': verts[:,0].max() - verts[:,0].min(),
        'Y': verts[:,1].max() - verts[:,1].min(), 
        'Z': verts[:,2].max() - verts[:,2].min()
    }
    
    print(f"  Coordinate spans: X={spans['X']:.2f}m, Y={spans['Y']:.2f}m, Z={spans['Z']:.2f}m")
    
    # Determine up axis by analyzing face normals
    normals = mesh.face_normals
    
    # Calculate how many faces have normals pointing primarily in each axis direction
    x_vertical = np.sum(np.abs(normals[:, 0]) > 0.8)  # Faces perpendicular to X
    y_vertical = np.sum(np.abs(normals[:, 1]) > 0.8)  # Faces perpendicular to Y  
    z_vertical = np.sum(np.abs(normals[:, 2]) > 0.8)  # Faces perpendicular to Z
    
    print(f"  Normal analysis: X axis has {x_vertical} perpendicular faces")
    print(f"  Normal analysis: Y axis has {y_vertical} perpendicular faces")  
    print(f"  Normal analysis: Z axis has {z_vertical} perpendicular faces")
    
    # The axis with most perpendicular faces is likely the up direction
    if z_vertical >= max(x_vertical, y_vertical):
        up_axis = 'Z'
        horizontal_axes = ['X', 'Y']
    elif y_vertical >= max(x_vertical, z_vertical):
        up_axis = 'Y'
        horizontal_axes = ['X', 'Z']
    else:
        up_axis = 'X' 
        horizontal_axes = ['Y', 'Z']
        
    print(f"  Determined orientation: {up_axis} is UP")
    
    return {
        'spans': spans,
        'up_axis': up_axis,
        'horizontal_axes': horizontal_axes
    }

def classify_faces_by_normal(mesh, vertical_threshold=0.7, up_axis='Z'):
    """Classify mesh faces as floors, ceilings, or walls based on their normals"""
    print("Classifying faces by normal direction...")
    
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Up vector based on coordinate system
    if up_axis == 'Z':
        up_vector = np.array([0, 0, 1])
    elif up_axis == 'Y':
        up_vector = np.array([0, 1, 0])
    else:  # X
        up_vector = np.array([1, 0, 0])
    
    # Calculate dot product with up vector
    dot_products = np.abs(np.dot(face_normals, up_vector))
    
    # Classify faces
    floors_ceilings = dot_products > vertical_threshold  # Close to up/down
    walls = dot_products <= vertical_threshold  # Close to horizontal
    
    # Further classify floors vs ceilings by direction of normal
    if up_axis == 'Z':
        floor_mask = (face_normals[:, 2] > 0) & floors_ceilings  # Normal points up
        ceiling_mask = (face_normals[:, 2] < 0) & floors_ceilings  # Normal points down
    elif up_axis == 'Y':
        floor_mask = (face_normals[:, 1] > 0) & floors_ceilings
        ceiling_mask = (face_normals[:, 1] < 0) & floors_ceilings
    else:  # X
        floor_mask = (face_normals[:, 0] > 0) & floors_ceilings
        ceiling_mask = (face_normals[:, 0] < 0) & floors_ceilings
    
    # Get face indices
    floor_faces = np.where(floor_mask)[0]
    ceiling_faces = np.where(ceiling_mask)[0]
    wall_faces = np.where(walls)[0]
    
    print(f"  Floor faces: {len(floor_faces):,}")
    print(f"  Ceiling faces: {len(ceiling_faces):,}")
    print(f"  Wall faces: {len(wall_faces):,}")
    
    return {
        'floor_faces': floor_faces,
        'ceiling_faces': ceiling_faces,
        'wall_faces': wall_faces,
        'face_normals': face_normals
    }

def extract_wall_boundary_points(mesh, wall_faces, up_axis='Z'):
    """Extract wall vertices and project to horizontal plane"""
    print(f"Extracting wall boundary points (projecting to horizontal plane)...")
    
    # Get all vertices from wall faces
    wall_vertex_indices = set()
    for face_idx in wall_faces:
        face = mesh.faces[face_idx]
        wall_vertex_indices.update(face)
    
    wall_vertices = mesh.vertices[list(wall_vertex_indices)]
    print(f"  Wall vertices: {len(wall_vertices):,}")
    
    # Project to horizontal plane based on up axis
    if up_axis == 'Z':
        boundary_points_2d = wall_vertices[:, [0, 1]]  # X-Y projection
        print(f"  Projecting to X-Y plane (Z up)")
    elif up_axis == 'Y':
        boundary_points_2d = wall_vertices[:, [0, 2]]  # X-Z projection  
        print(f"  Projecting to X-Z plane (Y up)")
    else:  # X
        boundary_points_2d = wall_vertices[:, [1, 2]]  # Y-Z projection
        print(f"  Projecting to Y-Z plane (X up)")
    
    print(f"  2D boundary points: {len(boundary_points_2d):,}")
    return boundary_points_2d

def find_dominant_wall_angles(points_2d, n_angles=36):
    """Find dominant wall angles using histogram voting (like v9)"""
    print(f"Finding dominant wall angles...")
    
    if len(points_2d) < 100:
        print("  Not enough points for angle analysis")
        return [0, 90]  # Default to Manhattan
    
    # Sample point pairs to compute angles
    n_samples = min(5000, len(points_2d) * (len(points_2d) - 1) // 2)
    
    angles = []
    np.random.seed(42)  # For reproducible results
    
    for _ in range(n_samples):
        # Pick two random points
        idx = np.random.choice(len(points_2d), 2, replace=False)
        p1, p2 = points_2d[idx[0]], points_2d[idx[1]]
        
        # Skip if points are too close
        if np.linalg.norm(p2 - p1) < 0.1:
            continue
            
        # Calculate angle
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
        # Normalize to [0, 180) since walls don't have direction
        angle = angle % 180
        angles.append(angle)
    
    if not angles:
        return [0, 90]
    
    # Create histogram
    hist, bin_edges = np.histogram(angles, bins=n_angles, range=(0, 180))
    
    # Find peaks (dominant angles)
    # Look for bins with high counts
    threshold = np.percentile(hist, 85)  # Top 15% of bins
    peak_indices = np.where(hist >= threshold)[0]
    
    dominant_angles = []
    for idx in peak_indices:
        angle = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        dominant_angles.append(angle)
    
    # Ensure we have at least 2 perpendicular angles
    if len(dominant_angles) == 0:
        dominant_angles = [0, 90]
    elif len(dominant_angles) == 1:
        primary = dominant_angles[0]
        perpendicular = (primary + 90) % 180
        dominant_angles.append(perpendicular)
    
    # Sort angles
    dominant_angles.sort()
    
    print(f"  Dominant wall angles: {dominant_angles[:4]}")  # Show first 4
    return dominant_angles

def group_points_by_wall_direction(points_2d, dominant_angles, angle_tolerance=15):
    """Group wall points by their direction relative to dominant angles"""
    print(f"Grouping points by wall direction...")
    
    # For each point pair, determine which dominant angle it's closest to
    wall_groups = defaultdict(list)
    
    # Use boundary points from convex hull to get perimeter
    hull = ConvexHull(points_2d)
    boundary_points = points_2d[hull.vertices]
    
    # Analyze each edge of the boundary
    n = len(boundary_points)
    for i in range(n):
        p1 = boundary_points[i]
        p2 = boundary_points[(i + 1) % n]
        
        # Calculate edge length and angle
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        
        if edge_length < 0.05:  # Skip very short edges
            continue
            
        edge_angle = np.arctan2(edge_vec[1], edge_vec[0]) * 180 / np.pi
        edge_angle = edge_angle % 180
        
        # Find closest dominant angle
        best_angle = None
        min_diff = float('inf')
        
        for dom_angle in dominant_angles:
            diff = min(abs(edge_angle - dom_angle), abs(edge_angle - (dom_angle + 180)))
            if diff < min_diff and diff <= angle_tolerance:
                min_diff = diff
                best_angle = dom_angle
        
        if best_angle is not None:
            wall_groups[best_angle].append({
                'start': p1.copy(),
                'end': p2.copy(),
                'length': edge_length,
                'angle': edge_angle
            })
    
    print(f"  Grouped into {len(wall_groups)} wall directions")
    for angle, segments in wall_groups.items():
        total_length = sum(seg['length'] for seg in segments)
        print(f"    {angle:.1f}°: {len(segments)} segments, {total_length:.2f}m total")
    
    return wall_groups

def merge_wall_segments_by_direction(wall_groups, merge_distance=0.5):
    """Merge segments within each wall direction"""
    print(f"Merging wall segments by direction...")
    
    merged_walls = []
    
    for angle, segments in wall_groups.items():
        if not segments:
            continue
            
        # Sort segments by position along the wall direction
        if abs(angle) < 45 or abs(angle - 180) < 45:  # Horizontal-ish
            segments.sort(key=lambda s: min(s['start'][0], s['end'][0]))
        else:  # Vertical-ish
            segments.sort(key=lambda s: min(s['start'][1], s['end'][1]))
        
        # Merge adjacent/overlapping segments
        merged = []
        current = segments[0].copy()
        
        for i in range(1, len(segments)):
            next_seg = segments[i]
            
            # Check if segments can be merged (are they close/touching?)
            end_to_start = np.linalg.norm(np.array(current['end']) - np.array(next_seg['start']))
            start_to_end = np.linalg.norm(np.array(current['start']) - np.array(next_seg['end']))
            
            min_gap = min(end_to_start, start_to_end)
            
            if min_gap <= merge_distance:
                # Merge segments - extend to cover both
                all_points = [
                    current['start'], current['end'], 
                    next_seg['start'], next_seg['end']
                ]
                
                if abs(angle) < 45 or abs(angle - 180) < 45:  # Horizontal
                    all_points.sort(key=lambda p: p[0])
                else:  # Vertical
                    all_points.sort(key=lambda p: p[1])
                
                current['start'] = all_points[0]
                current['end'] = all_points[-1]
                current['length'] = np.linalg.norm(np.array(current['end']) - np.array(current['start']))
            else:
                # Can't merge, save current and start new one
                if current['length'] > 0.2:  # Only keep significant walls
                    merged.append(current)
                current = next_seg.copy()
        
        # Don't forget the last segment
        if current['length'] > 0.2:
            merged.append(current)
        
        merged_walls.extend(merged)
    
    print(f"  Merged to {len(merged_walls)} wall segments")
    for i, wall in enumerate(merged_walls):
        print(f"    Wall {i+1}: {wall['length']:.2f}m at {wall['angle']:.1f}°")
    
    return merged_walls

def detect_openings_between_walls(merged_walls, gap_threshold=0.5):
    """Detect openings (gaps) between wall segments"""
    print(f"Detecting openings (gap threshold: {gap_threshold}m)...")
    
    openings = []
    
    for i, wall1 in enumerate(merged_walls):
        for j, wall2 in enumerate(merged_walls):
            if i >= j:
                continue
            
            # Check for gaps between wall endpoints
            wall1_start = np.array(wall1['start'])
            wall1_end = np.array(wall1['end'])
            wall2_start = np.array(wall2['start'])
            wall2_end = np.array(wall2['end'])
            
            # Calculate all endpoint distances
            distances = [
                ('end1_start2', np.linalg.norm(wall1_end - wall2_start), wall1_end, wall2_start),
                ('end1_end2', np.linalg.norm(wall1_end - wall2_end), wall1_end, wall2_end),
                ('start1_start2', np.linalg.norm(wall1_start - wall2_start), wall1_start, wall2_start),
                ('start1_end2', np.linalg.norm(wall1_start - wall2_end), wall1_start, wall2_end),
            ]
            
            for connection_type, distance, p1, p2 in distances:
                if gap_threshold < distance < 3.0:  # Reasonable opening size
                    opening_center = (p1 + p2) / 2
                    opening_type = "door" if distance < 1.5 else "window"
                    
                    openings.append({
                        'type': opening_type,
                        'position': opening_center.tolist(),
                        'width': float(distance),
                        'between_walls': [i, j],
                        'connection': connection_type
                    })
    
    # Remove duplicate openings (same position)
    unique_openings = []
    for opening in openings:
        is_duplicate = False
        for existing in unique_openings:
            if np.linalg.norm(np.array(opening['position']) - np.array(existing['position'])) < 0.3:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_openings.append(opening)
    
    print(f"  Found {len(unique_openings)} openings")
    for opening in unique_openings:
        print(f"    {opening['type']}: {opening['width']:.2f}m")
    
    return unique_openings

def analyze_mesh_normal_segmentation_improved(mesh_path, vertical_threshold=0.7):
    """Main analysis function with improved Manhattan wall fitting"""
    print(f"\n=== Normal-based Wall Segmentation v11 (IMPROVED MANHATTAN): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Analyze mesh geometry  
    geometry = analyze_mesh_geometry(mesh)
    
    # Step 2: Classify faces by normal direction
    classification = classify_faces_by_normal(mesh, vertical_threshold, geometry['up_axis'])
    
    # Step 3: Extract wall boundary points
    boundary_points_2d = extract_wall_boundary_points(mesh, classification['wall_faces'], geometry['up_axis'])
    
    if len(boundary_points_2d) == 0:
        print("No wall boundary points found!")
        return None
    
    # Step 4: Find dominant wall angles (Manhattan detection)
    dominant_angles = find_dominant_wall_angles(boundary_points_2d)
    
    # Step 5: Group points by wall direction
    wall_groups = group_points_by_wall_direction(boundary_points_2d, dominant_angles)
    
    # Step 6: Merge segments within each direction
    merged_walls = merge_wall_segments_by_direction(wall_groups)
    
    # Step 7: Detect openings between walls
    openings = detect_openings_between_walls(merged_walls)
    
    # Step 8: Calculate room area from boundary
    # Use convex hull for area calculation (simpler and more robust)
    if len(boundary_points_2d) >= 3:
        hull = ConvexHull(boundary_points_2d)
        boundary_vertices = boundary_points_2d[hull.vertices]
        
        # Shoelace formula for area
        area = 0.0
        n = len(boundary_vertices)
        for i in range(n):
            j = (i + 1) % n
            area += boundary_vertices[i][0] * boundary_vertices[j][1]
            area -= boundary_vertices[j][0] * boundary_vertices[i][1]
        area = abs(area) / 2.0
    else:
        boundary_vertices = boundary_points_2d
        area = 0.0
    
    return {
        "method": "normal_wall_segmentation_v11_improved_manhattan",
        "mesh_file": str(mesh_path),
        "parameters": {
            "vertical_threshold": vertical_threshold,
            "coordinate_system": geometry,
            "dominant_angles": dominant_angles
        },
        "room_boundary": boundary_vertices.tolist() if isinstance(boundary_vertices, np.ndarray) else boundary_vertices,
        "wall_segments": merged_walls,
        "openings": openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "face_classification": {
            "floor_faces": len(classification['floor_faces']),
            "ceiling_faces": len(classification['ceiling_faces']),
            "wall_faces": len(classification['wall_faces'])
        },
        "analysis_stats": {
            "boundary_points_2d": len(boundary_points_2d),
            "boundary_vertices": len(boundary_vertices),
            "wall_segments": len(merged_walls),
            "openings": len(openings),
            "dominant_angles": len(dominant_angles)
        }
    }

def create_visualization(result, output_path):
    """Create floor plan visualization"""
    if not result:
        return
        
    print("Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Wall segments with measurements
    ax1.set_title(f"Wall Segments (Improved Manhattan)\nArea: {result['room_area_sqm']:.1f} m²")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Draw wall segments with different colors for different angles
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    angles_seen = {}
    color_idx = 0
    
    for i, seg in enumerate(result['wall_segments']):
        start = seg['start']
        end = seg['end']
        angle_key = round(seg['angle'] / 45) * 45  # Group by 45° increments
        
        if angle_key not in angles_seen:
            angles_seen[angle_key] = colors[color_idx % len(colors)]
            color_idx += 1
        
        color = angles_seen[angle_key]
        
        ax1.plot([start[0], end[0]], [start[1], end[1]], 
                color=color, linewidth=4, alpha=0.7)
        
        # Add length annotation
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax1.annotate(f"{seg['length']:.1f}m", (mid_x, mid_y), 
                    fontsize=9, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Draw openings
    for opening in result['openings']:
        pos = opening['position']
        ax1.plot(pos[0], pos[1], 'ko', markersize=8, alpha=0.8)
        ax1.annotate(f"{opening['type']}\n{opening['width']:.1f}m", 
                    (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', fontweight='bold')
    
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
    
    # Add stats
    stats_text = f"""Improved v11 Stats:
Wall segments: {len(result['wall_segments'])}
Openings: {len(result['openings'])}
Dominant angles: {result['parameters']['dominant_angles'][:3]}
Coordinate system: {result['parameters']['coordinate_system']['up_axis']} up"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v11_improved.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_normal_segmentation_improved(mesh_path)
        if result:
            # Save results
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NpEncoder)
            print(f"\n✅ Results saved to {output_path}")
            
            # Create visualization
            create_visualization(result, viz_path)
            
            # Print summary
            print(f"\n📊 IMPROVED v11 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Wall segments: {len(result['wall_segments'])} (target: ~7)")
            print(f"   Boundary vertices: {len(result['room_boundary'])}")
            print(f"   Openings: {len(result['openings'])} (target: 2-4)")
            for opening in result['openings']:
                print(f"     - {opening['type']}: {opening['width']:.1f}m")
            print(f"   Dominant angles: {result['parameters']['dominant_angles'][:4]}")
            
            # Calculate accuracy vs target (11.5 m²)
            target_area = 11.5
            error_pct = abs(result['room_area_sqm'] - target_area) / target_area * 100
            print(f"   Area accuracy: {error_pct:.1f}% error vs 11.5 m² target")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
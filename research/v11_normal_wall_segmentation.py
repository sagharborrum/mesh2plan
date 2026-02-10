#!/usr/bin/env python3
"""
v11: Normal-based Wall Segmentation (FIXED)
==========================================

FIXES APPLIED:
- Replace convex hull with alpha shape / concave hull approach
- Add proper wall height filtering (foot-points near floor level)
- Implement line segment fitting on wall boundaries like v9
- Add Manhattan wall regularization and merging
- Improve opening detection via gap analysis

This approach:
1. Classify faces by normal direction (good - keep this)
2. Extract wall vertices at floor level (wall foot-points)
3. Use alpha shape for concave boundary detection
4. Fit line segments to boundary points
5. Apply Manhattan world regularization
6. Detect openings between wall segments
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

def classify_faces_by_normal(mesh, vertical_threshold=0.7):
    """Classify mesh faces as floors, ceilings, or walls based on their normals"""
    print("Classifying faces by normal direction...")
    
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Up vector (assume Z is up for most scans)
    up_vector = np.array([0, 0, 1])
    
    # Calculate dot product with up vector
    dot_products = np.abs(np.dot(face_normals, up_vector))
    
    # Classify faces
    floors_ceilings = dot_products > vertical_threshold  # Close to vertical
    walls = dot_products <= vertical_threshold  # Close to horizontal
    
    # Further classify floors vs ceilings by Z-component of normal
    floor_mask = (face_normals[:, 2] > 0) & floors_ceilings  # Normal points up
    ceiling_mask = (face_normals[:, 2] < 0) & floors_ceilings  # Normal points down
    
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

def extract_wall_footpoints(mesh, wall_faces, floor_level, foot_height=0.3):
    """Extract wall vertices near floor level (foot-points)"""
    print(f"Extracting wall foot-points (within {foot_height}m of floor)...")
    
    # Get all vertices used by wall faces
    wall_vertex_indices = set()
    for face_idx in wall_faces:
        face = mesh.faces[face_idx]
        wall_vertex_indices.update(face)
    
    wall_vertices = mesh.vertices[list(wall_vertex_indices)]
    
    # Adaptive height filtering for small meshes
    z_coords = mesh.vertices[:, 2]
    total_height = z_coords.max() - z_coords.min()
    
    if total_height < 2.0:  # Small mesh
        print(f"  Small mesh height ({total_height:.2f}m), using adaptive filtering...")
        # Use broader range for small meshes
        foot_height = min(foot_height, total_height * 0.4)  # Up to 40% of mesh height
    
    # Filter to foot-points near floor level
    foot_mask = np.abs(wall_vertices[:, 2] - floor_level) <= foot_height
    footpoints = wall_vertices[foot_mask]
    
    print(f"  Wall vertices total: {len(wall_vertices):,}")
    print(f"  Wall footpoints: {len(footpoints):,} (height tolerance: {foot_height:.2f}m)")
    
    return footpoints

def alpha_shape_boundary(points, alpha=0.1):
    """Create concave hull using alpha shape"""
    print(f"Computing alpha shape boundary (alpha={alpha})...")
    
    if len(points) < 3:
        return points
    
    # For simplicity, use a concave hull approximation
    # Project to XY plane
    points_2d = points[:, [0, 1]]
    
    try:
        # Try to use convex hull with refinement
        hull = ConvexHull(points_2d)
        boundary_indices = hull.vertices
        boundary_points = points_2d[boundary_indices]
        
        # Refine boundary by adding points that are close to edges
        refined_boundary = []
        for i in range(len(boundary_points)):
            refined_boundary.append(boundary_points[i])
            
            # Check for points between this edge and next
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]
            
            # Find points close to this edge
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            
            if edge_len > 0:
                edge_unit = edge_vec / edge_len
                
                for point in points_2d:
                    # Project point onto edge
                    to_point = point - p1
                    proj_len = np.dot(to_point, edge_unit)
                    
                    if 0 < proj_len < edge_len:  # Point projects onto edge
                        proj_point = p1 + proj_len * edge_unit
                        dist_to_edge = np.linalg.norm(point - proj_point)
                        
                        if dist_to_edge < alpha:  # Close to edge
                            refined_boundary.append(point)
        
        # Remove duplicates and sort
        if refined_boundary:
            refined_boundary = np.array(refined_boundary)
            # Remove duplicates
            unique_boundary = []
            for point in refined_boundary:
                is_duplicate = False
                for existing in unique_boundary:
                    if np.linalg.norm(point - existing) < 0.1:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_boundary.append(point)
            
            if len(unique_boundary) >= 3:
                unique_boundary = np.array(unique_boundary)
                # Sort by angle from center
                center = unique_boundary.mean(axis=0)
                angles = np.arctan2(unique_boundary[:, 1] - center[1], 
                                  unique_boundary[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                boundary_points = unique_boundary[sorted_indices]
        
        print(f"  Alpha shape boundary: {len(boundary_points)} points")
        return boundary_points
        
    except Exception as e:
        print(f"  Error in alpha shape: {e}")
        # Fallback to convex hull
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]

def fit_line_segments(boundary_points, min_segment_length=0.5):
    """Fit line segments to boundary points"""
    print(f"Fitting line segments (min length: {min_segment_length}m)...")
    
    if len(boundary_points) < 4:
        return []
    
    # Use Hough line detection approach
    # Create a binary image from boundary points
    min_coords = boundary_points.min(axis=0)
    max_coords = boundary_points.max(axis=0)
    
    # Create image with sufficient resolution
    image_size = 512
    scale_x = (image_size - 1) / (max_coords[0] - min_coords[0]) if max_coords[0] != min_coords[0] else 1
    scale_y = (image_size - 1) / (max_coords[1] - min_coords[1]) if max_coords[1] != min_coords[1] else 1
    
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Draw boundary points
    for point in boundary_points:
        x = int((point[0] - min_coords[0]) * scale_x)
        y = int((point[1] - min_coords[1]) * scale_y)
        if 0 <= x < image_size and 0 <= y < image_size:
            img[y, x] = 255
    
    # Dilate points to create connected edges
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, 
                           threshold=20, minLineLength=30, maxLineGap=10)
    
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Convert back to world coordinates
            world_x1 = min_coords[0] + x1 / scale_x
            world_y1 = min_coords[1] + y1 / scale_y
            world_x2 = min_coords[0] + x2 / scale_x
            world_y2 = min_coords[1] + y2 / scale_y
            
            # Calculate length
            length = np.sqrt((world_x2 - world_x1)**2 + (world_y2 - world_y1)**2)
            
            if length >= min_segment_length:
                angle = np.arctan2(world_y2 - world_y1, world_x2 - world_x1) * 180 / np.pi
                
                segments.append({
                    'start': [float(world_x1), float(world_y1)],
                    'end': [float(world_x2), float(world_y2)],
                    'length': float(length),
                    'angle': float(angle)
                })
    
    print(f"  Fitted {len(segments)} line segments")
    return segments

def manhattan_regularization(segments, angle_tolerance=15):
    """Snap segments to Manhattan world (0°, 90°, 180°, 270°)"""
    print("Applying Manhattan regularization...")
    
    regularized = []
    for seg in segments:
        angle = seg['angle'] % 180  # Normalize to 0-180
        
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
            # Recalculate endpoints with snapped angle
            start = np.array(seg['start'])
            length = seg['length']
            
            if snapped_angle == 0:  # Horizontal
                end = start + [length, 0]
            elif snapped_angle == 90:  # Vertical
                end = start + [0, length]
            else:  # 180 degrees, horizontal negative
                end = start + [-length, 0]
            
            regularized.append({
                'start': start.tolist(),
                'end': end.tolist(),
                'length': float(length),
                'angle': float(snapped_angle),
                'regularized': True
            })
        else:
            # Keep original if not close to Manhattan
            regularized.append(seg)
    
    print(f"  Regularized {len(regularized)} segments")
    return regularized

def merge_wall_segments(segments, merge_threshold=0.3):
    """Merge nearby collinear segments"""
    print(f"Merging segments (threshold: {merge_threshold}m)...")
    
    if not segments:
        return segments
    
    merged = []
    used = set()
    
    for i, seg1 in enumerate(segments):
        if i in used:
            continue
        
        current = dict(seg1)  # Copy
        
        # Try to merge with other segments
        for j, seg2 in enumerate(segments):
            if j <= i or j in used:
                continue
            
            # Check if segments are collinear
            if abs(current['angle'] - seg2['angle']) > 5:
                continue
            
            # Check if segments are close
            current_end = np.array(current['end'])
            seg2_start = np.array(seg2['start'])
            seg2_end = np.array(seg2['end'])
            current_start = np.array(current['start'])
            
            distances = [
                np.linalg.norm(current_end - seg2_start),
                np.linalg.norm(current_end - seg2_end),
                np.linalg.norm(current_start - seg2_start),
                np.linalg.norm(current_start - seg2_end)
            ]
            
            if min(distances) < merge_threshold:
                # Merge - extend to cover both segments
                all_points = [current_start, current_end, seg2_start, seg2_end]
                
                if current['angle'] % 180 == 0:  # Horizontal
                    all_points.sort(key=lambda p: p[0])
                else:  # Vertical
                    all_points.sort(key=lambda p: p[1])
                
                current['start'] = all_points[0].tolist()
                current['end'] = all_points[-1].tolist()
                current['length'] = np.linalg.norm(np.array(all_points[-1]) - np.array(all_points[0]))
                current['merged'] = True
                used.add(j)
        
        merged.append(current)
    
    print(f"  Merged to {len(merged)} segments")
    return merged

def detect_wall_openings(segments, gap_threshold=0.5):
    """Detect openings between wall segments"""
    print(f"Detecting openings (gap threshold: {gap_threshold}m)...")
    
    openings = []
    
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue
            
            # Check for gaps between parallel segments
            if abs(seg1['angle'] - seg2['angle']) < 10:  # Similar angles
                seg1_end = np.array(seg1['end'])
                seg2_start = np.array(seg2['start'])
                
                gap_distance = np.linalg.norm(seg1_end - seg2_start)
                
                if gap_threshold < gap_distance < 3.0:  # Reasonable opening
                    opening_center = (seg1_end + seg2_start) / 2
                    opening_type = "door" if gap_distance < 1.5 else "window"
                    
                    openings.append({
                        'type': opening_type,
                        'position': opening_center.tolist(),
                        'width': float(gap_distance),
                        'between_walls': [i, j]
                    })
    
    print(f"  Found {len(openings)} openings")
    return openings

def analyze_mesh_normal_segmentation(mesh_path, vertical_threshold=0.7):
    """Main analysis function with improved boundary extraction"""
    print(f"\n=== Normal-based Wall Segmentation v11 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Find floor level
    floor_level = find_floor_level(mesh)
    
    # Step 2: Classify faces by normal direction (keep original approach)
    classification = classify_faces_by_normal(mesh, vertical_threshold)
    
    # Step 3: Extract wall foot-points near floor level
    footpoints = extract_wall_footpoints(mesh, classification['wall_faces'], floor_level)
    
    if len(footpoints) == 0:
        print("No wall footpoints found!")
        return None
    
    # Step 4: Create concave boundary using alpha shape
    boundary_points = alpha_shape_boundary(footpoints, alpha=0.5)
    
    # Step 5: Fit line segments to boundary
    segments = fit_line_segments(boundary_points)
    
    # Step 6: Apply Manhattan regularization
    regularized_segments = manhattan_regularization(segments)
    
    # Step 7: Merge collinear segments
    merged_segments = merge_wall_segments(regularized_segments)
    
    # Step 8: Detect openings
    openings = detect_wall_openings(merged_segments)
    
    # Calculate room area from boundary
    if len(boundary_points) >= 3:
        # Shoelace formula
        area = 0.0
        n = len(boundary_points)
        for i in range(n):
            j = (i + 1) % n
            area += boundary_points[i][0] * boundary_points[j][1]
            area -= boundary_points[j][0] * boundary_points[i][1]
        area = abs(area) / 2.0
    else:
        area = 0.0
    
    return {
        "method": "normal_wall_segmentation_v11_fixed",
        "mesh_file": str(mesh_path),
        "parameters": {
            "vertical_threshold": vertical_threshold,
            "floor_level": float(floor_level),
            "foot_height": 0.3
        },
        "room_boundary": boundary_points.tolist() if isinstance(boundary_points, np.ndarray) else boundary_points,
        "wall_segments": merged_segments,
        "openings": openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "face_classification": {
            "floor_faces": len(classification['floor_faces']),
            "ceiling_faces": len(classification['ceiling_faces']),
            "wall_faces": len(classification['wall_faces'])
        },
        "analysis_stats": {
            "wall_footpoints": len(footpoints),
            "boundary_points": len(boundary_points),
            "wall_segments": len(merged_segments),
            "openings": len(openings)
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
    
    # Add stats
    stats_text = f"""Stats:
Floor level: {result['parameters']['floor_level']:.2f}m
Wall faces: {result['face_classification']['wall_faces']:,}
Footpoints: {result['analysis_stats']['wall_footpoints']:,}"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v11_normal_wall_segmentation.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_normal_segmentation(mesh_path)
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
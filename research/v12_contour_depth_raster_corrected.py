#!/usr/bin/env python3
"""
v12: Contour Detection on Rasterized Depth/Height Maps (FIXED)
==============================================================

FIXES APPLIED:
- Increase resolution to 1024x1024 for better detail
- Use proper wall-height band (0.8-1.5m above floor) like v9
- Add Hough line detection for wall segments
- Improve threshold settings for better wall detection
- Add Manhattan wall regularization
- Better opening detection between wall segments

This approach:
1. Find floor level like v9 does
2. Render mesh at wall height band to high-res depth map
3. Use image processing (edge detection, contours) with proper parameters
4. Extract wall segments using Hough line detection
5. Apply Manhattan fitting and merging
6. Detect openings as gaps between wall segments
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

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

def filter_vertices_by_height(mesh, floor_level, height_range=(0.8, 1.5)):
    """Filter vertices to wall height band like v9"""
    print(f"Filtering vertices to wall height {height_range[0]}-{height_range[1]}m above floor...")
    
    vertices = mesh.vertices
    z_coords = vertices[:, 2]
    total_height = z_coords.max() - z_coords.min()
    
    if total_height < 2.0:  # Small mesh
        print(f"  Small mesh height ({total_height:.2f}m), using adaptive filtering...")
        wall_z_min = floor_level + total_height * 0.2  # Start at 20% above floor
        wall_z_max = floor_level + total_height * 0.8  # End at 80% above floor
    else:
        wall_z_min = floor_level + height_range[0]
        wall_z_max = floor_level + height_range[1]
        wall_z_max = min(wall_z_max, z_coords.max() - 0.05)  # Leave margin
    
    # Filter to wall height band
    wall_mask = (z_coords >= wall_z_min) & (z_coords <= wall_z_max)
    wall_vertices = vertices[wall_mask]
    
    print(f"  Filtered from {len(vertices):,} to {len(wall_vertices):,} vertices")
    print(f"  Wall height range: {wall_z_min:.2f} to {wall_z_max:.2f}m")
    
    return wall_vertices

def render_wall_height_map(wall_vertices, resolution=1024, padding=0.1):
    """Render wall vertices to high-resolution occupancy map"""
    print(f"Rendering wall height map (resolution: {resolution})...")
    
    if len(wall_vertices) == 0:
        print("No wall vertices to render!")
        return None
    
    # Get bounds in XY plane (top-down view)
    x_coords = wall_vertices[:, 0]
    y_coords = wall_vertices[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add padding
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
    
    # Adjust resolution to maintain aspect ratio
    width = int((x_max - x_min) / pixel_size)
    height = int((y_max - y_min) / pixel_size)
    
    print(f"  Map size: {width} x {height} pixels")
    print(f"  Pixel size: {pixel_size:.4f} m/pixel")
    print(f"  Coverage: {x_max-x_min:.2f}m x {y_max-y_min:.2f}m")
    
    # Initialize occupancy map
    occupancy = np.zeros((height, width), dtype=np.uint8)
    
    # Rasterize wall vertices
    for vertex in wall_vertices:
        x, y, z = vertex
        
        # Convert to pixel coordinates
        px = int((x - x_min) / pixel_size)
        py = int((y - y_min) / pixel_size)
        
        if 0 <= px < width and 0 <= py < height:
            occupancy[py, px] = 255  # Mark as occupied
    
    occupied_pixels = np.sum(occupancy > 0)
    print(f"  Occupied pixels: {occupied_pixels:,} ({100*occupied_pixels/(width*height):.1f}%)")
    
    return {
        'occupancy': occupancy,
        'bounds': (x_min, y_min, x_max, y_max),
        'pixel_size': pixel_size,
        'resolution': (width, height)
    }

def process_occupancy_map(height_data):
    """Process occupancy map to extract wall structure"""
    print("Processing occupancy map for wall detection...")
    
    occupancy = height_data['occupancy']
    
    # Morphological operations to clean and thicken walls
    kernel = np.ones((3, 3), np.uint8)
    
    # Close small gaps
    cleaned = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate slightly to ensure connectivity
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    
    # Edge detection with better parameters
    edges = cv2.Canny(cleaned, 50, 150)
    
    # Dilate edges slightly to ensure connectivity
    edge_kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)
    
    print(f"  Cleaned wall pixels: {np.sum(cleaned > 0):,}")
    print(f"  Edge pixels: {np.sum(edges > 0):,}")
    
    return cleaned, edges

def detect_wall_segments_hough(edges, height_data, min_line_length=30, max_line_gap=10):
    """Detect wall segments using Hough line detection"""
    print("Detecting wall segments using Hough transform...")
    
    # Hough line detection with adjusted parameters
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=30, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        print("  No lines detected!")
        return []
    
    # Convert to world coordinates
    x_min, y_min, x_max, y_max = height_data['bounds']
    pixel_size = height_data['pixel_size']
    
    wall_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Convert to world coordinates
        world_x1 = x_min + x1 * pixel_size
        world_y1 = y_min + y1 * pixel_size
        world_x2 = x_min + x2 * pixel_size
        world_y2 = y_min + y2 * pixel_size
        
        # Calculate length and angle
        length = np.sqrt((world_x2-world_x1)**2 + (world_y2-world_y1)**2)
        angle = np.arctan2(world_y2-world_y1, world_x2-world_x1) * 180 / np.pi
        
        # Filter out very short segments
        if length > 0.3:  # Minimum 30cm
            wall_segments.append({
                'start': [float(world_x1), float(world_y1)],
                'end': [float(world_x2), float(world_y2)],
                'length': float(length),
                'angle': float(angle)
            })
    
    print(f"  Detected {len(wall_segments)} wall segments")
    return wall_segments

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
            # Keep original
            regularized.append(seg)
    
    print(f"  Regularized {len(regularized)} segments")
    return regularized

def merge_wall_segments(segments, merge_threshold=0.4):
    """Merge nearby collinear wall segments"""
    print(f"Merging wall segments (threshold: {merge_threshold}m)...")
    
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
            if abs(current['angle'] - seg2['angle']) > 10:
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
                # Merge segments
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

def create_room_boundary(wall_segments):
    """Create room boundary from wall segments"""
    print("Creating room boundary...")
    
    if not wall_segments:
        return []
    
    # Collect all endpoints
    points = []
    for seg in wall_segments:
        points.append(seg['start'])
        points.append(seg['end'])
    
    if not points:
        return []
    
    points = np.array(points)
    
    # Remove duplicates
    unique_points = []
    for point in points:
        is_duplicate = False
        for existing in unique_points:
            if np.linalg.norm(np.array(point) - np.array(existing)) < 0.15:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point.tolist())
    
    if len(unique_points) < 3:
        return unique_points
    
    # Sort points to form polygon
    points_array = np.array(unique_points)
    center = points_array.mean(axis=0)
    
    # Sort by angle from center
    angles = np.arctan2(points_array[:, 1] - center[1], 
                       points_array[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    boundary = points_array[sorted_indices].tolist()
    print(f"  Room boundary: {len(boundary)} vertices")
    
    return boundary

def detect_wall_openings(wall_segments, gap_threshold=0.6):
    """Detect openings between wall segments"""
    print(f"Detecting openings (gap threshold: {gap_threshold}m)...")
    
    openings = []
    
    for i, seg1 in enumerate(wall_segments):
        for j, seg2 in enumerate(wall_segments):
            if i >= j:
                continue
            
            # Check for gaps between aligned segments
            if abs(seg1['angle'] - seg2['angle']) < 15:  # Similar angles
                seg1_end = np.array(seg1['end'])
                seg2_start = np.array(seg2['start'])
                
                gap_distance = np.linalg.norm(seg1_end - seg2_start)
                
                if gap_threshold < gap_distance < 2.5:  # Reasonable opening
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

def analyze_mesh_contour_detection(mesh_path, resolution=1024):
    """Main analysis function with fixes applied"""
    print(f"\n=== Contour Detection v12 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Find floor level (like v9)
    floor_level = find_floor_level(mesh)
    
    # Step 2: Filter vertices to wall height band (like v9)
    wall_vertices = filter_vertices_by_height(mesh, floor_level)
    
    if len(wall_vertices) == 0:
        print("No wall vertices found!")
        return None
    
    # Step 3: Render to high-resolution occupancy map
    height_data = render_wall_height_map(wall_vertices, resolution)
    
    if height_data is None:
        print("Failed to render height map!")
        return None
    
    # Step 4: Process occupancy map
    wall_image, edges = process_occupancy_map(height_data)
    
    # Step 5: Detect wall segments using Hough lines
    wall_segments = detect_wall_segments_hough(edges, height_data)
    
    if not wall_segments:
        print("No wall segments detected!")
        return {
            "method": "contour_depth_raster_v12_fixed",
            "mesh_file": str(mesh_path),
            "room_area_sqm": 0.0,
            "room_area_sqft": 0.0,
            "room_boundary": [],
            "wall_segments": [],
            "openings": []
        }
    
    # Step 6: Apply Manhattan regularization
    regularized_segments = manhattan_regularization(wall_segments)
    
    # Step 7: Merge segments
    merged_segments = merge_wall_segments(regularized_segments)
    
    # Step 8: Create room boundary
    room_boundary = create_room_boundary(merged_segments)
    
    # Step 9: Detect openings
    openings = detect_wall_openings(merged_segments)
    
    # Calculate room area
    if len(room_boundary) >= 3:
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
        "method": "contour_depth_raster_v12_fixed",
        "mesh_file": str(mesh_path),
        "parameters": {
            "resolution": resolution,
            "floor_level": float(floor_level),
            "wall_height_range": [0.8, 1.5],
            "pixel_size": height_data['pixel_size']
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
            "boundary_vertices": len(room_boundary),
            "map_resolution": height_data['resolution']
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
    if 'parameters' in result:
        stats_text = f"""Stats:
Resolution: {result['analysis_stats']['map_resolution']}
Floor level: {result['parameters']['floor_level']:.2f}m
Wall vertices: {result['analysis_stats']['wall_vertices']:,}
Pixel size: {result['parameters']['pixel_size']:.4f}m"""
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v12_contour_depth_raster.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_contour_detection(mesh_path)
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
            if 'parameters' in result:
                print(f"   Floor level: {result['parameters']['floor_level']:.2f}m")
                print(f"   Resolution: {result['analysis_stats']['map_resolution']}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
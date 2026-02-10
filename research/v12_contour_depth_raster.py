#!/usr/bin/env python3
"""
v12: Contour Detection on Rasterized Depth/Height Maps
======================================================

Approach:
1. Render mesh to a top-down depth/height map (rasterization)
2. Process the height map as an image
3. Use image processing techniques (edge detection, contours) to find walls
4. Extract room boundaries from detected contours
5. Detect openings as gaps in contours

This approach treats the problem as computer vision on rendered images.
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def render_top_down_depth_map(mesh, resolution=1024, padding=0.1):
    """Render mesh to top-down depth map"""
    print(f"Rendering top-down depth map (resolution: {resolution})...")
    
    # Get mesh bounds
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    # Add padding
    x_range = x_max - x_min
    z_range = z_max - z_min
    x_min -= x_range * padding
    x_max += x_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding
    
    # Calculate pixel size
    pixel_size_x = (x_max - x_min) / resolution
    pixel_size_z = (z_max - z_min) / resolution
    pixel_size = max(pixel_size_x, pixel_size_z)
    
    # Adjust resolution to maintain aspect ratio
    width = int((x_max - x_min) / pixel_size)
    height = int((z_max - z_min) / pixel_size)
    
    print(f"  Map size: {width} x {height} pixels")
    print(f"  Pixel size: {pixel_size:.4f} m/pixel")
    
    # Initialize height maps
    min_heights = np.full((height, width), np.inf)
    max_heights = np.full((height, width), -np.inf)
    occupancy = np.zeros((height, width), dtype=bool)
    
    # Rasterize mesh vertices
    vertices = mesh.vertices
    for vertex in vertices:
        x, y, z = vertex
        
        # Convert to pixel coordinates
        px = int((x - x_min) / pixel_size)
        pz = int((z - z_min) / pixel_size)
        
        if 0 <= px < width and 0 <= pz < height:
            min_heights[pz, px] = min(min_heights[pz, px], y)
            max_heights[pz, px] = max(max_heights[pz, px], y)
            occupancy[pz, px] = True
    
    # Create height difference map (wall indicator)
    height_diff = np.where(occupancy, max_heights - min_heights, 0)
    
    # Handle infinite values
    min_heights[min_heights == np.inf] = 0
    max_heights[max_heights == -np.inf] = 0
    
    print(f"  Occupied pixels: {np.sum(occupancy):,} ({100*np.sum(occupancy)/(width*height):.1f}%)")
    
    return {
        'min_heights': min_heights,
        'max_heights': max_heights,
        'height_diff': height_diff,
        'occupancy': occupancy,
        'bounds': (x_min, z_min, x_max, z_max),
        'pixel_size': pixel_size,
        'resolution': (width, height)
    }

def process_height_map(height_data, wall_height_threshold=0.5):
    """Process height map to detect walls"""
    print("Processing height map for wall detection...")
    
    height_diff = height_data['height_diff']
    occupancy = height_data['occupancy']
    
    # Create wall mask based on height difference
    wall_mask = (height_diff > wall_height_threshold) & occupancy
    
    # Convert to uint8 for image processing
    wall_image = (wall_mask * 255).astype(np.uint8)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    wall_image = cv2.morphologyEx(wall_image, cv2.MORPH_CLOSE, kernel)
    wall_image = cv2.morphologyEx(wall_image, cv2.MORPH_OPEN, kernel)
    
    # Edge detection
    edges = cv2.Canny(wall_image, 50, 150)
    
    print(f"  Wall pixels: {np.sum(wall_image > 0):,}")
    print(f"  Edge pixels: {np.sum(edges > 0):,}")
    
    return wall_image, edges

def extract_contours_from_edges(edges, min_contour_area=100):
    """Extract contours from edge-detected image"""
    print("Extracting contours...")
    
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            valid_contours.append(contour)
    
    print(f"  Found {len(contours)} total contours, {len(valid_contours)} valid (area > {min_contour_area})")
    
    return valid_contours

def contours_to_world_coordinates(contours, height_data):
    """Convert pixel contours to world coordinates"""
    print("Converting contours to world coordinates...")
    
    x_min, z_min, x_max, z_max = height_data['bounds']
    pixel_size = height_data['pixel_size']
    
    world_contours = []
    
    for contour in contours:
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to world coordinates
        world_points = []
        for point in simplified.reshape(-1, 2):
            px, pz = point
            world_x = x_min + px * pixel_size
            world_z = z_min + pz * pixel_size
            world_points.append([world_x, world_z])
        
        world_contours.append(world_points)
    
    return world_contours

def detect_contour_openings(world_contours, height_data, gap_threshold=1.0):
    """Detect openings in contours"""
    print("Detecting openings in contours...")
    
    openings = []
    
    for contour_idx, contour in enumerate(world_contours):
        if len(contour) < 3:
            continue
            
        # Check for gaps between consecutive points
        for i in range(len(contour)):
            p1 = np.array(contour[i])
            p2 = np.array(contour[(i + 1) % len(contour)])
            
            gap_length = np.linalg.norm(p2 - p1)
            
            if gap_length > gap_threshold:
                # Found a gap - could be an opening
                center = (p1 + p2) / 2
                opening_type = "door" if gap_length < 3.0 else "window"
                
                openings.append({
                    "type": opening_type,
                    "position": center.tolist(),
                    "size_meters": float(gap_length),
                    "contour_index": contour_idx
                })
    
    print(f"  Found {len(openings)} openings")
    return openings

def analyze_mesh_contour_detection(mesh_path, resolution=512, wall_height_threshold=0.5):
    """Main analysis function"""
    print(f"\n=== Contour Detection on Rasterized Depth Map: {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if hasattr(mesh, 'vertices'):
        print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    else:
        print("Error loading mesh")
        return None
    
    # Step 1: Render to depth map
    height_data = render_top_down_depth_map(mesh, resolution)
    
    # Step 2: Process height map for walls
    wall_image, edges = process_height_map(height_data, wall_height_threshold)
    
    # Step 3: Extract contours
    contours = extract_contours_from_edges(edges)
    
    if not contours:
        print("No valid contours found!")
        return None
    
    # Step 4: Convert to world coordinates
    world_contours = contours_to_world_coordinates(contours, height_data)
    
    # Step 5: Detect openings
    openings = detect_contour_openings(world_contours, height_data)
    
    # Select the largest contour as the main room boundary
    if world_contours:
        # Calculate areas and select largest
        contour_areas = []
        for contour in world_contours:
            if len(contour) >= 3:
                area = 0.0
                for i in range(len(contour)):
                    j = (i + 1) % len(contour)
                    area += contour[i][0] * contour[j][1]
                    area -= contour[j][0] * contour[i][1]
                area = abs(area) / 2.0
                contour_areas.append(area)
            else:
                contour_areas.append(0.0)
        
        largest_idx = np.argmax(contour_areas)
        main_boundary = world_contours[largest_idx]
        main_area = contour_areas[largest_idx]
    else:
        main_boundary = []
        main_area = 0.0
    
    return {
        "method": "contour_depth_raster_v12",
        "mesh_file": str(mesh_path),
        "parameters": {
            "resolution": resolution,
            "wall_height_threshold": wall_height_threshold,
            "pixel_size": height_data['pixel_size']
        },
        "room_boundary": main_boundary,
        "all_contours": world_contours,
        "openings": openings,
        "room_area_sqm": float(main_area),
        "room_area_sqft": float(main_area * 10.764),
        "contour_stats": {
            "total_contours": len(world_contours),
            "main_boundary_points": len(main_boundary),
            "map_resolution": height_data['resolution']
        }
    }

def create_visualization(result, height_data, wall_image, edges, output_path):
    """Create comprehensive visualization"""
    print(f"Creating visualization: {output_path}")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Height map
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(height_data['height_diff'], cmap='viridis', origin='lower')
    plt.colorbar(label='Height Difference (m)')
    plt.title('Height Difference Map')
    
    # Plot 2: Wall detection
    ax2 = plt.subplot(2, 3, 2)
    plt.imshow(wall_image, cmap='gray', origin='lower')
    plt.title('Wall Detection')
    
    # Plot 3: Edge detection
    ax3 = plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray', origin='lower')
    plt.title('Edge Detection')
    
    # Plot 4: All contours in world coordinates
    ax4 = plt.subplot(2, 3, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, len(result['all_contours'])))
    for i, contour in enumerate(result['all_contours']):
        if len(contour) > 0:
            contour_array = np.array(contour + [contour[0]])  # Close the contour
            ax4.plot(contour_array[:, 0], contour_array[:, 1], 
                    color=colors[i], linewidth=2, label=f'Contour {i+1}')
    
    # Plot openings
    for opening in result['openings']:
        pos = opening['position']
        ax4.plot(pos[0], pos[1], 'ro', markersize=8)
        ax4.annotate(opening['type'], (pos[0], pos[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Z (meters)')
    ax4.set_title('Detected Contours (World Coordinates)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Plot 5: Main room boundary only
    ax5 = plt.subplot(2, 3, 5)
    if result['room_boundary']:
        boundary = np.array(result['room_boundary'] + [result['room_boundary'][0]])
        ax5.plot(boundary[:, 0], boundary[:, 1], 'b-', linewidth=3, label='Main Boundary')
        ax5.fill(boundary[:, 0], boundary[:, 1], alpha=0.3, color='lightblue')
    
    # Plot openings
    for opening in result['openings']:
        if opening['contour_index'] == 0:  # Only openings in main contour
            pos = opening['position']
            ax5.plot(pos[0], pos[1], 'ro', markersize=10)
    
    ax5.set_xlabel('X (meters)')
    ax5.set_ylabel('Z (meters)')
    ax5.set_title('Main Room Boundary')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # Plot 6: Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""ANALYSIS RESULTS (V12)
    
Area: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Boundary Points: {len(result['room_boundary'])}
Total Contours: {result['contour_stats']['total_contours']}
Openings: {len(result['openings'])}

PARAMETERS:
Resolution: {result['parameters']['resolution']}
Pixel Size: {result['parameters']['pixel_size']:.4f} m
Wall Height Threshold: {result['parameters']['wall_height_threshold']} m

MAP STATISTICS:
Map Size: {result['contour_stats']['map_resolution']}
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python v12_contour_depth_raster.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
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
            
            # Create visualization if requested
            if viz_path:
                # Re-run analysis to get intermediate data for visualization
                mesh = trimesh.load(mesh_path)
                height_data = render_top_down_depth_map(mesh, result['parameters']['resolution'])
                wall_image, edges = process_height_map(height_data, result['parameters']['wall_height_threshold'])
                create_visualization(result, height_data, wall_image, edges, viz_path)
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Boundary points: {len(result['room_boundary'])}")
            print(f"   Total contours: {result['contour_stats']['total_contours']}")
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
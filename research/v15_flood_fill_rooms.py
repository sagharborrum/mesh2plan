#!/usr/bin/env python3
"""
v15: Flood-fill Room Segmentation (FIXED)
=========================================

FIXES APPLIED:
- Use finer grid resolution (0.02m cell size instead of large pixels)
- Proper wall-height occupancy using floor level detection like v9
- Better multi-room seed selection and detection
- Improved opening detection between rooms
- Prevent over-simplification of boundaries (more than 4 points)
- Use proper coordinate system and floor level detection

This approach:
1. Find floor level and create fine occupancy grid at wall height
2. Use improved seed placement for multiple rooms
3. Flood fill from seeds to segment connected spaces
4. Extract detailed room boundaries (not simplified)
5. Detect openings/passages between rooms
6. Handle multi-room layouts properly
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN

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

def create_fine_occupancy_grid(mesh, floor_level, cell_size=0.02, height_range=(0.8, 1.5)):
    """Create fine occupancy grid using wall height band - FIXED RESOLUTION"""
    print(f"Step 1: Creating fine occupancy grid (cell size: {cell_size}m)...")
    
    # Filter vertices to wall height band
    vertices = mesh.vertices
    z_coords = vertices[:, 2]
    
    wall_z_min = floor_level + height_range[0]
    wall_z_max = floor_level + height_range[1]
    
    wall_mask = (z_coords >= wall_z_min) & (z_coords <= wall_z_max)
    wall_vertices = vertices[wall_mask]
    
    print(f"  Wall vertices in height band: {len(wall_vertices):,}")
    
    if len(wall_vertices) == 0:
        print("ERROR: No wall vertices found!")
        return None, None, None
    
    # Get bounds in XY plane
    x_coords = wall_vertices[:, 0]
    y_coords = wall_vertices[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add small padding
    padding = 0.5  # 50cm padding
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    # Calculate grid dimensions based on cell size
    width = int((x_max - x_min) / cell_size) + 1
    height = int((y_max - y_min) / cell_size) + 1
    
    print(f"  Grid dimensions: {width} x {height} = {width*height:,} cells")
    print(f"  Coverage: {x_max-x_min:.2f}m x {y_max-y_min:.2f}m")
    
    # Initialize occupancy grid (0 = free, 255 = occupied/wall)
    occupancy_grid = np.zeros((height, width), dtype=np.uint8)
    
    # Mark occupied cells where wall vertices exist
    for vertex in wall_vertices:
        x, y, z = vertex
        
        grid_x = int((x - x_min) / cell_size)
        grid_y = int((y - y_min) / cell_size)
        
        if 0 <= grid_x < width and 0 <= grid_y < height:
            occupancy_grid[grid_y, grid_x] = 255
    
    # Apply morphological operations to create coherent walls
    kernel = np.ones((3, 3), np.uint8)
    occupancy_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel, iterations=3)
    occupancy_grid = cv2.dilate(occupancy_grid, kernel, iterations=2)  # Thicken walls
    
    occupied_cells = np.sum(occupancy_grid == 255)
    free_cells = np.sum(occupancy_grid == 0)
    
    print(f"  Final grid: {occupied_cells:,} occupied, {free_cells:,} free ({100*occupied_cells/(occupied_cells+free_cells):.1f}% walls)")
    
    return occupancy_grid, (x_min, x_max, y_min, y_max), cell_size

def improved_room_seed_detection(occupancy_grid, cell_size, min_room_size_m2=2.0):
    """Improved seed detection for multiple rooms - BETTER MULTI-ROOM"""
    print("Step 2: Improved room seed detection for multi-room layouts...")
    
    # Free space mask
    free_space = (occupancy_grid == 0)
    
    # Distance transform to find room centers
    distance_map = cv2.distanceTransform(free_space.astype(np.uint8), 
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Convert minimum room size to pixels
    min_room_pixels = int(min_room_size_m2 / (cell_size ** 2))
    
    print(f"  Minimum room size: {min_room_size_m2} m² = {min_room_pixels} pixels")
    
    # Find local maxima using different scales
    seeds = []
    
    # Multiple scales for different room sizes
    for scale in [0.3, 0.5, 0.7, 0.9]:  # Different room center detection scales
        # Threshold based on distance percentile for this scale
        if np.max(distance_map) > 0:
            threshold = np.percentile(distance_map[distance_map > 0], 80 + scale*15)
            local_maxima = distance_map > threshold
            
            # Use peak detection to find isolated maxima
            from scipy import ndimage
            maxima_coords = ndimage.maximum_filter(distance_map, size=max(5, int(10/scale)))
            isolated_maxima = (distance_map == maxima_coords) & local_maxima
            
            scale_seeds = np.column_stack(np.where(isolated_maxima))
            
            # Filter by minimum distance from existing seeds
            for new_seed in scale_seeds:
                too_close = False
                for existing_seed in seeds:
                    dist = np.linalg.norm(new_seed - existing_seed)
                    if dist < 15:  # Minimum separation in pixels
                        too_close = True
                        break
                
                if not too_close:
                    seeds.append(new_seed)
    
    seeds = np.array(seeds) if seeds else np.array([]).reshape(0, 2)
    
    # If no seeds found, use fallback method
    if len(seeds) == 0:
        print("  No seeds found with advanced method, using fallback...")
        # Simple fallback: find center of largest free area
        if np.max(distance_map) > 3:  # At least 3 pixel distance from walls
            max_dist_location = np.unravel_index(np.argmax(distance_map), distance_map.shape)
            seeds = np.array([max_dist_location])
    
    # Limit number of seeds for practical processing
    if len(seeds) > 15:
        print(f"  Limiting seeds from {len(seeds)} to 15...")
        # Keep seeds with highest distance values
        distances = [distance_map[seed[0], seed[1]] for seed in seeds]
        best_indices = np.argsort(distances)[-15:]
        seeds = seeds[best_indices]
    
    print(f"  Found {len(seeds)} room seeds")
    
    return seeds, distance_map

def flood_fill_multi_room(occupancy_grid, seeds, cell_size):
    """Advanced flood fill for multiple rooms - MULTI-ROOM HANDLING"""
    print("Step 3: Multi-room flood fill...")
    
    room_map = np.zeros_like(occupancy_grid, dtype=np.int32)
    rooms = []
    
    for i, (row, col) in enumerate(seeds):
        room_id = i + 1
        
        # Skip if this cell is occupied or already assigned
        if occupancy_grid[row, col] != 0 or room_map[row, col] != 0:
            continue
        
        print(f"  Flood filling room {room_id} from ({row}, {col})...")
        
        # Use OpenCV's floodFill for efficient flood filling
        mask = np.zeros((occupancy_grid.shape[0] + 2, occupancy_grid.shape[1] + 2), np.uint8)
        
        # Create working copy
        temp_grid = occupancy_grid.copy()
        
        # Flood fill
        area, _, _, _ = cv2.floodFill(temp_grid, mask, (col, row), room_id,
                                      loDiff=0, upDiff=0, flags=8)
        
        # Calculate minimum area in pixels for a meaningful room
        min_area_pixels = max(100, int(2.0 / (cell_size ** 2)))  # At least 2 m²
        
        if area > min_area_pixels:
            # Mark this room in the room map
            room_region = (temp_grid == room_id)
            room_map[room_region] = room_id
            
            # Calculate area in m²
            area_m2 = area * (cell_size ** 2)
            
            rooms.append({
                'room_id': room_id,
                'area_pixels': area,
                'area_m2': area_m2,
                'seed_location': [row, col]
            })
            
            print(f"    Room {room_id}: {area} pixels = {area_m2:.1f} m²")
        else:
            print(f"    Room {room_id}: Too small ({area} pixels), skipping")
    
    print(f"  Created {len(rooms)} valid rooms")
    return room_map, rooms

def extract_detailed_room_boundaries(room_map, rooms, grid_bounds, cell_size):
    """Extract detailed room boundaries - PREVENT OVER-SIMPLIFICATION"""
    print("Step 4: Extracting detailed room boundaries...")
    
    x_min, x_max, y_min, y_max = grid_bounds
    
    room_boundaries = []
    
    for room in rooms:
        room_id = room['room_id']
        room_mask = (room_map == room_id)
        
        if not np.any(room_mask):
            continue
        
        # Find contours of this room
        contours, _ = cv2.findContours(room_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Minimum contour size
                # Simplify contour but not too aggressively - KEEP DETAIL
                epsilon = 0.005 * cv2.arcLength(contour, True)  # Very gentle simplification
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to world coordinates
                world_points = []
                for point in simplified_contour.squeeze():
                    if len(point.shape) == 0:  # Handle single point case
                        col, row = point, point
                    else:
                        col, row = point
                    
                    world_x = x_min + (col * cell_size)
                    world_y = y_min + (row * cell_size)
                    world_points.append([world_x, world_y])
                
                if len(world_points) >= 3:  # Need at least 3 points for a room
                    world_boundary = np.array(world_points)
                    
                    # Calculate area using shoelace formula
                    room_area = 0.0
                    n = len(world_boundary)
                    for i in range(n):
                        j = (i + 1) % n
                        room_area += world_boundary[i][0] * world_boundary[j][1]
                        room_area -= world_boundary[j][0] * world_boundary[i][1]
                    room_area = abs(room_area) / 2.0
                    
                    room_boundaries.append({
                        'room_id': room_id,
                        'boundary_points': world_boundary.tolist(),
                        'area_m2': room_area,
                        'area_ft2': room_area * 10.764,
                        'boundary_vertex_count': len(world_boundary)
                    })
                    
                    print(f"  Room {room_id}: {len(world_boundary)} boundary points, {room_area:.1f} m²")
    
    return room_boundaries

def detect_room_openings(room_map, occupancy_grid, grid_bounds, cell_size):
    """Detect openings/passages between rooms"""
    print("Step 5: Detecting openings between rooms...")
    
    openings = []
    x_min, x_max, y_min, y_max = grid_bounds
    
    # Get unique room IDs
    room_ids = np.unique(room_map)
    room_ids = room_ids[room_ids > 0]  # Exclude background
    
    for i, room_id_1 in enumerate(room_ids):
        for room_id_2 in room_ids[i+1:]:
            
            # Create masks for each room
            room1_mask = (room_map == room_id_1)
            room2_mask = (room_map == room_id_2)
            
            # Expand rooms by 1 pixel to find adjacency
            kernel = np.ones((3, 3), np.uint8)
            room1_expanded = cv2.dilate(room1_mask.astype(np.uint8), kernel, iterations=1)
            room2_expanded = cv2.dilate(room2_mask.astype(np.uint8), kernel, iterations=1)
            
            # Find where expanded rooms overlap in free space
            overlap = (room1_expanded & room2_expanded) & (occupancy_grid == 0)
            
            if np.sum(overlap) > 3:  # Minimum opening size
                # Find contours of the overlap (potential openings)
                contours, _ = cv2.findContours(overlap.astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area_pixels = cv2.contourArea(contour)
                    if area_pixels > 2:  # Minimum opening area
                        
                        # Find centroid of opening
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Convert to world coordinates
                            world_x = x_min + (cx * cell_size)
                            world_y = y_min + (cy * cell_size)
                            
                            # Calculate opening dimensions
                            opening_area_m2 = area_pixels * (cell_size ** 2)
                            
                            # Estimate opening type
                            opening_type = "door" if opening_area_m2 < 2.0 else "passage"
                            
                            openings.append({
                                'type': opening_type,
                                'position': [world_x, world_y],
                                'area_m2': opening_area_m2,
                                'between_rooms': [int(room_id_1), int(room_id_2)],
                                'area_pixels': int(area_pixels)
                            })
                            
                            print(f"  Opening between rooms {room_id_1}-{room_id_2}: {opening_area_m2:.2f} m² at ({world_x:.1f}, {world_y:.1f})")
    
    print(f"  Found {len(openings)} openings between rooms")
    return openings

def analyze_mesh_flood_fill(mesh_path, cell_size=0.02):
    """Main flood fill analysis with fixes"""
    print(f"\n=== Flood-fill Room Segmentation v15 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Find floor level
    floor_level = find_floor_level(mesh)
    
    # Step 2: Create fine occupancy grid
    occupancy_grid, grid_bounds, cell_size = create_fine_occupancy_grid(mesh, floor_level, cell_size)
    
    if occupancy_grid is None:
        print("Failed to create occupancy grid!")
        return None
    
    # Step 3: Find room seeds
    seeds, distance_map = improved_room_seed_detection(occupancy_grid, cell_size)
    
    if len(seeds) == 0:
        print("No room seeds found!")
        return None
    
    # Step 4: Multi-room flood fill
    room_map, rooms = flood_fill_multi_room(occupancy_grid, seeds, cell_size)
    
    if len(rooms) == 0:
        print("No rooms found!")
        return None
    
    # Step 5: Extract detailed boundaries
    room_boundaries = extract_detailed_room_boundaries(room_map, rooms, grid_bounds, cell_size)
    
    # Step 6: Detect openings
    openings = detect_room_openings(room_map, occupancy_grid, grid_bounds, cell_size)
    
    # Calculate total area
    total_area_m2 = sum(room['area_m2'] for room in room_boundaries)
    
    # For compatibility, select main room (largest)
    if room_boundaries:
        main_room = max(room_boundaries, key=lambda r: r['area_m2'])
        main_boundary = main_room['boundary_points']
        main_area = main_room['area_m2']
    else:
        main_boundary = []
        main_area = 0.0
    
    return {
        "method": "flood_fill_rooms_v15_fixed",
        "mesh_file": str(mesh_path),
        "parameters": {
            "floor_level": float(floor_level),
            "cell_size": cell_size,
            "wall_height_range": [0.8, 1.5],
            "grid_dimensions": occupancy_grid.shape
        },
        "room_boundary": main_boundary,  # Main room for compatibility
        "all_rooms": room_boundaries,    # All detected rooms
        "openings": openings,
        "room_area_sqm": float(main_area),
        "room_area_sqft": float(main_area * 10.764),
        "total_area_sqm": float(total_area_m2),
        "total_area_sqft": float(total_area_m2 * 10.764),
        "analysis_stats": {
            "rooms_detected": len(room_boundaries),
            "total_openings": len(openings),
            "seeds_used": len(seeds),
            "grid_cells": int(occupancy_grid.size),
            "occupied_cells": int(np.sum(occupancy_grid == 255))
        }
    }

def create_visualization(result, output_path):
    """Create comprehensive visualization"""
    if not result:
        return
        
    print("Creating visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All rooms
    ax1.set_title(f"All Detected Rooms\nTotal: {result['total_area_sqm']:.1f} m²")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(result['all_rooms'])))
    for i, room in enumerate(result['all_rooms']):
        if len(room['boundary_points']) >= 3:
            boundary = np.array(room['boundary_points'])
            boundary_closed = np.vstack([boundary, boundary[0]])
            ax1.fill(boundary_closed[:, 0], boundary_closed[:, 1], 
                    alpha=0.4, color=colors[i], edgecolor='black', linewidth=1)
            ax1.annotate(f"Room {room['room_id']}\n{room['area_m2']:.1f}m²", 
                        boundary.mean(axis=0), ha='center', va='center',
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Main room boundary
    ax2.set_title(f"Main Room Boundary\n{len(result['room_boundary'])} vertices")
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
                   c='red', s=30, alpha=0.7, zorder=5)
    
    # Plot 3: Openings
    ax3.set_title(f"Detected Openings\n{len(result['openings'])} total")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Draw all room boundaries in gray
    for room in result['all_rooms']:
        if len(room['boundary_points']) >= 3:
            boundary = np.array(room['boundary_points'])
            boundary_closed = np.vstack([boundary, boundary[0]])
            ax3.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                    color='gray', linewidth=1, alpha=0.5)
    
    # Draw openings
    for opening in result['openings']:
        pos = opening['position']
        ax3.plot(pos[0], pos[1], 'ro', markersize=8, alpha=0.8)
        ax3.annotate(f"{opening['type']}\n{opening['area_m2']:.2f}m²", 
                    (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    
    # Plot 4: Statistics
    ax4.set_title("Analysis Statistics")
    ax4.axis('off')
    
    stats_text = f"""FLOOD-FILL ANALYSIS v15 (FIXED)

RESULTS:
Main Room: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Total Area: {result['total_area_sqm']:.2f} m² ({result['total_area_sqft']:.1f} ft²)
Rooms Detected: {result['analysis_stats']['rooms_detected']}
Openings: {result['analysis_stats']['total_openings']}

PARAMETERS:
Cell Size: {result['parameters']['cell_size']}m
Floor Level: {result['parameters']['floor_level']:.2f}m
Grid Size: {result['parameters']['grid_dimensions']}
Seeds Used: {result['analysis_stats']['seeds_used']}

ROOM DETAILS:"""
    
    for room in result['all_rooms']:
        stats_text += f"\nRoom {room['room_id']}: {room['area_m2']:.1f}m² ({room['boundary_vertex_count']} vertices)"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v15_flood_fill_rooms.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_flood_fill(mesh_path)
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
            print(f"   Main room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Total area: {result['total_area_sqm']:.1f} m² ({result['total_area_sqft']:.1f} ft²)")
            print(f"   Rooms detected: {result['analysis_stats']['rooms_detected']}")
            print(f"   Openings: {result['analysis_stats']['total_openings']}")
            print(f"   Main boundary vertices: {len(result['room_boundary'])}")
            print(f"   Cell size: {result['parameters']['cell_size']}m")
            print(f"   Floor level: {result['parameters']['floor_level']:.2f}m")
            
            for room in result['all_rooms']:
                print(f"     Room {room['room_id']}: {room['area_m2']:.1f} m² ({room['boundary_vertex_count']} vertices)")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
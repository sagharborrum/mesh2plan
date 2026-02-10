#!/usr/bin/env python3
"""
v15: Flood-fill Room Segmentation
=================================

Approach:
1. Rasterize mesh top-down into occupancy grid
   - Occupied = walls/objects (above threshold height)
   - Free = open space (below threshold height) 
2. Flood fill from open space to find rooms
   - Start from free space and expand
   - Walls are the unfilled boundaries between rooms
3. Extract room boundaries from flood-filled regions
4. Handle multi-room layouts naturally

This approach should work well for complex floor plans with multiple rooms
by treating the problem as a spatial connectivity analysis.
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

def create_occupancy_grid(mesh, resolution=256, wall_height_threshold=0.5, padding=0.1):
    """Create occupancy grid from mesh - occupied cells are walls/objects"""
    print("Step 1: Creating occupancy grid...")
    
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
    
    # Initialize occupancy grid (0 = free, 255 = occupied)
    occupancy_grid = np.zeros((resolution, resolution), dtype=np.uint8)
    height_map = np.zeros((resolution, resolution))
    
    # Project vertices to 2D grid
    vertices = mesh.vertices
    
    # Convert world coordinates to pixel coordinates
    x_pixels = ((vertices[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
    z_pixels = ((vertices[:, 2] - z_min) / (z_max - z_min) * (resolution - 1)).astype(int)
    
    # Filter valid pixels
    valid_mask = ((x_pixels >= 0) & (x_pixels < resolution) & 
                  (z_pixels >= 0) & (z_pixels < resolution))
    
    valid_x = x_pixels[valid_mask]
    valid_z = z_pixels[valid_mask]
    valid_y = vertices[valid_mask, 1]
    
    # Fill height map with maximum height at each pixel
    print(f"  Projecting {len(valid_x)} vertices to occupancy grid...")
    for x, z, y in zip(valid_x, valid_z, valid_y):
        height_map[z, x] = max(height_map[z, x], y - y_min)
    
    # Mark cells as occupied if above wall height threshold
    occupancy_grid[height_map > wall_height_threshold] = 255
    
    # Apply morphological operations to clean up the grid
    kernel = np.ones((3, 3), np.uint8)
    occupancy_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel, iterations=2)
    occupancy_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_OPEN, kernel, iterations=1)
    
    print(f"  Occupancy grid created: {np.sum(occupancy_grid == 255)} occupied cells, {np.sum(occupancy_grid == 0)} free cells")
    
    return occupancy_grid, height_map, (x_min, x_max, z_min, z_max)

def find_room_seeds(occupancy_grid, min_distance=10, max_seeds=20):
    """Find seed points for flood fill in open space areas"""
    print("Step 2: Finding room seed points...")
    
    # Free space mask
    free_space = (occupancy_grid == 0)
    
    # Distance transform to find centers of open areas
    distance_map = cv2.distanceTransform(free_space.astype(np.uint8), 
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Find local maxima as potential seed points
    # Use a higher threshold and limit the number of seeds
    threshold = max(min_distance, np.percentile(distance_map[distance_map > 0], 90))
    local_maxima = (distance_map > threshold)
    
    # Get coordinates of seed points
    potential_seeds = np.column_stack(np.where(local_maxima))
    
    print(f"  Found {len(potential_seeds)} potential seed points with threshold {threshold:.1f}")
    
    # If too many seeds, select the ones with highest distance values
    if len(potential_seeds) > max_seeds:
        print(f"  Reducing to {max_seeds} best seeds...")
        distances = distance_map[potential_seeds[:, 0], potential_seeds[:, 1]]
        best_indices = np.argsort(distances)[-max_seeds:]
        seed_coords = potential_seeds[best_indices]
    else:
        seed_coords = potential_seeds
    
    # If no seeds found, use a lower threshold
    if len(seed_coords) == 0:
        print("  No seeds found, using lower threshold...")
        lower_threshold = np.percentile(distance_map[distance_map > 0], 75)
        local_maxima = (distance_map > lower_threshold)
        potential_seeds = np.column_stack(np.where(local_maxima))
        
        if len(potential_seeds) > max_seeds:
            distances = distance_map[potential_seeds[:, 0], potential_seeds[:, 1]]
            best_indices = np.argsort(distances)[-max_seeds:]
            seed_coords = potential_seeds[best_indices]
        else:
            seed_coords = potential_seeds
    
    print(f"  Final seed count: {len(seed_coords)}")
    
    return seed_coords, distance_map

def flood_fill_rooms(occupancy_grid, seed_coords):
    """Perform flood fill from seed points to segment rooms"""
    print("Step 3: Flood filling rooms...")
    
    room_map = np.zeros_like(occupancy_grid, dtype=np.int32)
    room_boundaries = []
    
    for i, (row, col) in enumerate(seed_coords):
        room_id = i + 1
        
        # Skip if this cell is already assigned or occupied
        if occupancy_grid[row, col] != 0 or room_map[row, col] != 0:
            continue
        
        # Flood fill from this seed point
        print(f"  Flood filling room {room_id} from ({row}, {col})...")
        
        # Use OpenCV's floodFill for efficient flood filling
        mask = np.zeros((occupancy_grid.shape[0] + 2, occupancy_grid.shape[1] + 2), np.uint8)
        
        # Create a copy for flood fill (floodFill modifies the image)
        temp_grid = occupancy_grid.copy()
        
        # Flood fill
        area, _, _, _ = cv2.floodFill(temp_grid, mask, (col, row), room_id,
                                      loDiff=0, upDiff=0, flags=8)
        
        if area > 100:  # Minimum room size in pixels
            # Mark this room in the room map
            room_region = (temp_grid == room_id)
            room_map[room_region] = room_id
            
            # Find contours for this room
            contours, _ = cv2.findContours(room_region.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    room_boundaries.append({
                        'room_id': room_id,
                        'contour': contour,
                        'area_pixels': area
                    })
    
    print(f"  Created {len(room_boundaries)} room regions")
    
    return room_map, room_boundaries

def extract_room_boundaries_world(room_boundaries, grid_bounds, resolution):
    """Convert room boundaries from grid coordinates to world coordinates"""
    print("Step 4: Converting room boundaries to world coordinates...")
    
    x_min, x_max, z_min, z_max = grid_bounds
    
    world_boundaries = []
    
    for room in room_boundaries:
        contour = room['contour']
        room_id = room['room_id']
        
        # Convert contour points to world coordinates
        world_points = []
        for point in contour.squeeze():
            col, row = point  # OpenCV uses (x,y) = (col,row)
            
            # Convert to world coordinates
            world_x = x_min + (col / resolution) * (x_max - x_min)
            world_z = z_min + (row / resolution) * (z_max - z_min)
            world_points.append([world_x, world_z])
        
        world_boundary = np.array(world_points)
        
        # Calculate room area in world units
        room_area = calculate_room_area(world_boundary)
        
        world_boundaries.append({
            'room_id': room_id,
            'boundary_points': world_boundary,
            'area_m2': room_area,
            'area_ft2': room_area * 10.764
        })
    
    print(f"  Converted {len(world_boundaries)} room boundaries to world coordinates")
    
    return world_boundaries

def detect_openings_between_rooms(room_map, occupancy_grid):
    """Detect openings (doors/passages) between rooms"""
    print("Step 5: Detecting openings between rooms...")
    
    openings = []
    
    # Find boundaries between different rooms
    kernel = np.ones((3, 3), np.uint8)
    
    # Get unique room IDs
    room_ids = np.unique(room_map)
    room_ids = room_ids[room_ids > 0]  # Exclude background (0)
    
    for room_id in room_ids:
        room_mask = (room_map == room_id)
        
        # Dilate room to find adjacent areas
        dilated_room = cv2.dilate(room_mask.astype(np.uint8), kernel, iterations=1)
        
        # Find what the dilation touches
        touching_mask = (dilated_room == 1) & (room_mask == 0)
        
        # Check what room IDs this room touches
        touching_rooms = room_map[touching_mask]
        adjacent_room_ids = np.unique(touching_rooms)
        adjacent_room_ids = adjacent_room_ids[adjacent_room_ids > 0]
        
        for adj_room_id in adjacent_room_ids:
            if adj_room_id > room_id:  # Avoid duplicate pairs
                # Find the boundary between these two rooms
                room1_boundary = cv2.dilate((room_map == room_id).astype(np.uint8), kernel, iterations=1)
                room2_boundary = cv2.dilate((room_map == adj_room_id).astype(np.uint8), kernel, iterations=1)
                
                # Find where boundaries overlap (potential openings)
                overlap = (room1_boundary & room2_boundary) & (occupancy_grid == 0)
                
                if np.sum(overlap) > 5:  # Minimum opening size
                    # Find contours of the overlap
                    contours, _ = cv2.findContours(overlap.astype(np.uint8), 
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 3:
                            openings.append({
                                'room1': room_id,
                                'room2': adj_room_id,
                                'contour': contour
                            })
    
    print(f"  Found {len(openings)} potential openings between rooms")
    
    return openings

def calculate_room_area(boundary_points):
    """Calculate room area using shoelace formula"""
    if len(boundary_points) < 3:
        return 0.0
    
    # Shoelace formula
    x = boundary_points[:, 0]
    y = boundary_points[:, 1]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                        for i in range(-1, len(x)-1)))

def create_visualization(mesh, occupancy_grid, room_map, world_boundaries, 
                        distance_map, seed_coords, height_map, output_path):
    """Create comprehensive visualization"""
    print("Creating visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original mesh (top view)
    vertices_2d = mesh.vertices[:, [0, 2]]
    ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='lightgray', s=0.1, alpha=0.5)
    ax1.set_title('Original Mesh (Top View)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Occupancy grid with seed points
    ax2.imshow(occupancy_grid, cmap='gray', origin='lower')
    ax2.scatter(seed_coords[:, 1], seed_coords[:, 0], c='red', s=50, marker='x')
    ax2.set_title(f'Occupancy Grid + Seeds ({len(seed_coords)} seeds)')
    
    # 3. Distance transform
    ax3.imshow(distance_map, cmap='viridis', origin='lower')
    ax3.scatter(seed_coords[:, 1], seed_coords[:, 0], c='red', s=50, marker='x')
    ax3.set_title('Distance Transform (Room Centers)')
    
    # 4. Room segmentation result
    room_display = np.zeros_like(room_map, dtype=float)
    unique_rooms = np.unique(room_map)
    unique_rooms = unique_rooms[unique_rooms > 0]
    
    # Assign different colors to different rooms
    for i, room_id in enumerate(unique_rooms):
        room_display[room_map == room_id] = i + 1
    
    ax4.imshow(room_display, cmap='tab10', origin='lower')
    ax4.set_title(f'Room Segmentation ({len(unique_rooms)} rooms)')
    
    # Add room boundary overlays
    for room in world_boundaries:
        # We need to convert world coordinates back to grid coordinates for display
        # This is approximate for visualization
        boundary = room['boundary_points']
        if len(boundary) > 2:
            boundary_closed = np.vstack([boundary, boundary[0]])
            ax1.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                    linewidth=2, label=f'Room {room["room_id"]} ({room["area_m2"]:.1f}m²)')
    
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def process_mesh_flood_fill(mesh_path, output_path, visualization_path=None):
    """Main processing function for flood-fill room segmentation"""
    print(f"\nProcessing: {mesh_path}")
    print("=" * 60)
    
    # Load mesh
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None
    
    # Step 1: Create occupancy grid
    occupancy_grid, height_map, grid_bounds = create_occupancy_grid(mesh)
    
    # Step 2: Find room seeds
    seed_coords, distance_map = find_room_seeds(occupancy_grid)
    
    if len(seed_coords) == 0:
        print("No room seeds found! Cannot proceed with flood fill.")
        return None
    
    # Step 3: Flood fill rooms
    room_map, room_boundaries = flood_fill_rooms(occupancy_grid, seed_coords)
    
    if len(room_boundaries) == 0:
        print("No rooms found after flood fill!")
        return None
    
    # Step 4: Convert to world coordinates
    world_boundaries = extract_room_boundaries_world(room_boundaries, grid_bounds, 
                                                    occupancy_grid.shape[0])
    
    # Step 5: Detect openings
    openings = detect_openings_between_rooms(room_map, occupancy_grid)
    
    # Calculate total area and metrics
    total_area_m2 = sum(room['area_m2'] for room in world_boundaries)
    total_area_ft2 = total_area_m2 * 10.764
    
    # Prepare results
    results = {
        'method': 'v15_flood_fill_rooms',
        'mesh_file': str(mesh_path),
        'mesh_stats': {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces)
        },
        'occupancy_analysis': {
            'grid_resolution': occupancy_grid.shape[0],
            'occupied_cells': int(np.sum(occupancy_grid == 255)),
            'free_cells': int(np.sum(occupancy_grid == 0))
        },
        'room_segmentation': {
            'seed_points': len(seed_coords),
            'rooms_found': len(world_boundaries),
            'total_area_m2': total_area_m2,
            'total_area_ft2': total_area_ft2
        },
        'opening_detection': {
            'openings_found': len(openings)
        },
        'results': {
            'rooms': world_boundaries,
            'openings': openings,
            'seed_coordinates': seed_coords
        }
    }
    
    # Create visualization
    if visualization_path:
        create_visualization(mesh, occupancy_grid, room_map, world_boundaries, 
                           distance_map, seed_coords, height_map, visualization_path)
    
    # Print results summary
    print(f"\nResults Summary:")
    print(f"Rooms found: {len(world_boundaries)}")
    print(f"Total area: {total_area_m2:.1f} m² ({total_area_ft2:.1f} ft²)")
    print(f"Openings detected: {len(openings)}")
    
    for i, room in enumerate(world_boundaries):
        print(f"  Room {room['room_id']}: {room['area_m2']:.1f} m² ({room['area_ft2']:.1f} ft²), {len(room['boundary_points'])} boundary points")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print(f"Results saved to {output_path}")
    return results

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python v15_flood_fill_rooms.py <mesh.obj> <output.json> [visualization.png]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    output_path = sys.argv[2]
    visualization_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_mesh_flood_fill(mesh_path, output_path, visualization_path)

if __name__ == "__main__":
    main()
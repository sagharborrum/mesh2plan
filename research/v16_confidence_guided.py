#!/usr/bin/env python3
"""
v16: Confidence-Map Guided Extraction
=====================================

Approach:
1. Load confidence maps from LiDAR data: data/*/conf_*.png
2. Load frame poses from JSON files: data/*/frame_*.json
3. Use confidence maps to weight/filter mesh regions:
   - High confidence = structural elements (walls/floors)
   - Low confidence = reflective/transparent surfaces (windows/glass)
4. Extract floor plans guided by confidence information
5. Use camera poses for occlusion analysis and data quality assessment

This approach leverages the rich metadata available in newer scan datasets.
Only works with datasets that have confidence maps and frame metadata.
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import glob
from PIL import Image

warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def analyze_frame_metadata(data_dir):
    """Analyze frame JSON files to understand camera poses and quality"""
    print("Step 1: Analyzing frame metadata...")
    
    frame_files = glob.glob(str(Path(data_dir) / "frame_*.json"))
    
    if not frame_files:
        print("  No frame JSON files found!")
        return None
    
    frame_data = []
    
    for frame_file in sorted(frame_files)[:20]:  # Limit to first 20 for analysis
        try:
            with open(frame_file, 'r') as f:
                data = json.load(f)
                frame_data.append(data)
        except Exception as e:
            print(f"  Error reading {frame_file}: {e}")
    
    if not frame_data:
        print("  No valid frame data loaded!")
        return None
    
    # Extract useful information
    frame_analysis = {
        'total_frames_analyzed': len(frame_data),
        'available_fields': list(frame_data[0].keys()) if frame_data else [],
        'motion_quality_stats': {},
        'velocity_stats': {},
        'camera_info': {}
    }
    
    # Analyze motion quality if available
    motion_qualities = [f.get('motionQuality', 0) for f in frame_data if 'motionQuality' in f]
    if motion_qualities:
        frame_analysis['motion_quality_stats'] = {
            'mean': np.mean(motion_qualities),
            'std': np.std(motion_qualities),
            'min': np.min(motion_qualities),
            'max': np.max(motion_qualities)
        }
    
    # Analyze velocities if available
    avg_velocities = [f.get('averageVelocity', 0) for f in frame_data if 'averageVelocity' in f]
    if avg_velocities:
        frame_analysis['velocity_stats'] = {
            'mean': np.mean(avg_velocities),
            'std': np.std(avg_velocities),
            'min': np.min(avg_velocities),
            'max': np.max(avg_velocities)
        }
    
    # Extract camera intrinsics if available
    if 'intrinsics' in frame_data[0]:
        frame_analysis['camera_info'] = {
            'intrinsics': frame_data[0]['intrinsics'],
            'has_projection_matrix': 'projectionMatrix' in frame_data[0]
        }
    
    print(f"  Analyzed {len(frame_data)} frames")
    print(f"  Available fields: {', '.join(frame_analysis['available_fields'])}")
    
    if motion_qualities:
        print(f"  Motion quality: {frame_analysis['motion_quality_stats']['mean']:.3f} ± {frame_analysis['motion_quality_stats']['std']:.3f}")
    
    return frame_analysis

def load_confidence_maps(data_dir, max_maps=10):
    """Load and analyze confidence maps from PNG files"""
    print("Step 2: Loading confidence maps...")
    
    conf_files = glob.glob(str(Path(data_dir) / "conf_*.png"))
    
    if not conf_files:
        print("  No confidence map files found!")
        return None, None
    
    print(f"  Found {len(conf_files)} confidence maps, loading first {max_maps}...")
    
    confidence_maps = []
    confidence_stats = {
        'file_count': len(conf_files),
        'loaded_count': 0,
        'resolution': None,
        'confidence_distribution': {}
    }
    
    for i, conf_file in enumerate(sorted(conf_files)[:max_maps]):
        try:
            # Load confidence map as grayscale
            conf_map = np.array(Image.open(conf_file).convert('L'))
            confidence_maps.append(conf_map)
            
            if confidence_stats['resolution'] is None:
                confidence_stats['resolution'] = conf_map.shape
                
            confidence_stats['loaded_count'] += 1
            
        except Exception as e:
            print(f"  Error loading {conf_file}: {e}")
    
    if not confidence_maps:
        print("  No confidence maps loaded successfully!")
        return None, None
    
    # Analyze confidence distribution
    all_confidences = np.concatenate([cm.flatten() for cm in confidence_maps])
    confidence_stats['confidence_distribution'] = {
        'mean': float(np.mean(all_confidences)),
        'std': float(np.std(all_confidences)),
        'percentiles': {
            '25th': float(np.percentile(all_confidences, 25)),
            '50th': float(np.percentile(all_confidences, 50)),
            '75th': float(np.percentile(all_confidences, 75)),
            '90th': float(np.percentile(all_confidences, 90))
        }
    }
    
    print(f"  Loaded {len(confidence_maps)} confidence maps")
    print(f"  Resolution: {confidence_stats['resolution']}")
    print(f"  Confidence range: {np.min(all_confidences)} - {np.max(all_confidences)}")
    print(f"  Mean confidence: {confidence_stats['confidence_distribution']['mean']:.1f}")
    
    return confidence_maps, confidence_stats

def create_confidence_weighted_occupancy(mesh, confidence_maps, confidence_stats, resolution=256):
    """Create occupancy grid weighted by confidence information"""
    print("Step 3: Creating confidence-weighted occupancy grid...")
    
    # Get mesh bounds
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    z_range = z_max - z_min
    x_min -= x_range * padding
    x_max += x_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding
    
    # Create base occupancy grid
    occupancy_grid = np.zeros((resolution, resolution))
    confidence_grid = np.zeros((resolution, resolution))
    
    # Project mesh vertices
    vertices = mesh.vertices
    x_pixels = ((vertices[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
    z_pixels = ((vertices[:, 2] - z_min) / (z_max - z_min) * (resolution - 1)).astype(int)
    
    # Filter valid pixels
    valid_mask = ((x_pixels >= 0) & (x_pixels < resolution) & 
                  (z_pixels >= 0) & (z_pixels < resolution))
    
    valid_x = x_pixels[valid_mask]
    valid_z = z_pixels[valid_mask]
    valid_y = vertices[valid_mask, 1]
    
    # Fill occupancy grid
    print(f"  Projecting {len(valid_x)} vertices to occupancy grid...")
    for x, z, y in zip(valid_x, valid_z, valid_y):
        occupancy_grid[z, x] = max(occupancy_grid[z, x], y - y_min)
    
    # Create aggregate confidence map
    if confidence_maps:
        print("  Processing confidence information...")
        # Average all confidence maps
        avg_confidence = np.mean(confidence_maps, axis=0)
        
        # Resize to match occupancy grid
        avg_confidence_resized = cv2.resize(avg_confidence, (resolution, resolution))
        confidence_grid = avg_confidence_resized
        
        # Apply confidence thresholding
        high_conf_threshold = confidence_stats['confidence_distribution']['percentiles']['75th']
        low_conf_threshold = confidence_stats['confidence_distribution']['percentiles']['25th']
        
        # Create masks
        high_confidence_mask = confidence_grid > high_conf_threshold
        low_confidence_mask = confidence_grid < low_conf_threshold
        
        print(f"  High confidence regions: {np.sum(high_confidence_mask)} pixels")
        print(f"  Low confidence regions: {np.sum(low_confidence_mask)} pixels")
        
        # Weight occupancy by confidence
        # High confidence areas are more likely to be structural
        confidence_weighted_occupancy = occupancy_grid.copy()
        confidence_weighted_occupancy[high_confidence_mask] *= 1.5  # Boost structural elements
        confidence_weighted_occupancy[low_confidence_mask] *= 0.5   # Reduce transparent elements
    else:
        confidence_grid = np.ones_like(occupancy_grid)
        confidence_weighted_occupancy = occupancy_grid
        high_confidence_mask = np.ones_like(occupancy_grid, dtype=bool)
        low_confidence_mask = np.zeros_like(occupancy_grid, dtype=bool)
    
    return (confidence_weighted_occupancy, confidence_grid, 
            high_confidence_mask, low_confidence_mask, (x_min, x_max, z_min, z_max))

def extract_structural_elements(confidence_weighted_occupancy, high_confidence_mask, 
                               wall_height_threshold=0.5):
    """Extract walls and structural elements using confidence weighting"""
    print("Step 4: Extracting structural elements...")
    
    # Create wall mask based on height and confidence
    wall_mask = ((confidence_weighted_occupancy > wall_height_threshold) & 
                 high_confidence_mask)
    
    # Apply morphological operations to clean up walls
    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find wall contours
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wall_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum wall area
            wall_regions.append(contour)
    
    print(f"  Found {len(wall_regions)} structural wall regions")
    
    return wall_regions, wall_mask

def detect_transparent_openings(confidence_grid, low_confidence_mask, wall_mask):
    """Detect openings using low-confidence regions"""
    print("Step 5: Detecting transparent openings...")
    
    # Find regions that should be walls but have low confidence
    # These are likely windows or glass doors
    potential_openings = (wall_mask.astype(bool) & low_confidence_mask)
    
    # Find contours of potential openings
    opening_contours, _ = cv2.findContours(potential_openings.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    openings = []
    for contour in opening_contours:
        area = cv2.contourArea(contour)
        if area > 20:  # Minimum opening size
            openings.append(contour)
    
    print(f"  Detected {len(openings)} potential transparent openings")
    
    return openings

def convert_to_world_coordinates(regions, grid_bounds, resolution):
    """Convert grid coordinates to world coordinates"""
    print("Step 6: Converting to world coordinates...")
    
    x_min, x_max, z_min, z_max = grid_bounds
    world_regions = []
    
    for region in regions:
        world_points = []
        for point in region.squeeze():
            col, row = point  # OpenCV uses (x,y) = (col,row)
            
            # Convert to world coordinates
            world_x = x_min + (col / resolution) * (x_max - x_min)
            world_z = z_min + (row / resolution) * (z_max - z_min)
            world_points.append([world_x, world_z])
        
        world_regions.append(np.array(world_points))
    
    return world_regions

def calculate_area(points):
    """Calculate area using shoelace formula"""
    if len(points) < 3:
        return 0.0
    
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                        for i in range(-1, len(x)-1)))

def create_visualization(mesh, confidence_weighted_occupancy, confidence_grid,
                        wall_mask, high_confidence_mask, low_confidence_mask,
                        wall_regions_world, openings_world, output_path):
    """Create comprehensive visualization"""
    print("Creating visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 20))
    
    # 1. Original mesh
    vertices_2d = mesh.vertices[:, [0, 2]]
    ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='lightgray', s=0.1, alpha=0.5)
    ax1.set_title('Original Mesh (Top View)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence grid
    ax2.imshow(confidence_grid, cmap='viridis', origin='lower')
    ax2.set_title('Confidence Map (Aggregated)')
    
    # 3. High/Low confidence regions
    combined_mask = np.zeros_like(confidence_grid)
    combined_mask[high_confidence_mask] = 1  # High confidence = 1
    combined_mask[low_confidence_mask] = -1  # Low confidence = -1
    ax3.imshow(combined_mask, cmap='RdYlGn', origin='lower', vmin=-1, vmax=1)
    ax3.set_title('Confidence Regions (Green=High, Red=Low)')
    
    # 4. Wall detection
    ax4.imshow(wall_mask, cmap='gray', origin='lower')
    ax4.set_title('Detected Walls (Confidence Weighted)')
    
    # 5. Confidence weighted occupancy
    ax5.imshow(confidence_weighted_occupancy, cmap='plasma', origin='lower')
    ax5.set_title('Confidence-Weighted Height Map')
    
    # 6. Final result with boundaries
    ax6.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='lightgray', s=0.1, alpha=0.3)
    
    # Plot wall boundaries
    for i, wall in enumerate(wall_regions_world):
        if len(wall) > 2:
            wall_closed = np.vstack([wall, wall[0]])
            ax6.plot(wall_closed[:, 0], wall_closed[:, 1], 
                    'r-', linewidth=2, label=f'Wall {i+1}' if i < 5 else '')
    
    # Plot openings
    for i, opening in enumerate(openings_world):
        if len(opening) > 2:
            opening_closed = np.vstack([opening, opening[0]])
            ax6.plot(opening_closed[:, 0], opening_closed[:, 1], 
                    'b-', linewidth=2, label=f'Opening {i+1}' if i < 5 else '')
    
    ax6.set_title('Final Result: Walls (Red) & Openings (Blue)')
    ax6.set_aspect('equal')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def process_mesh_confidence_guided(mesh_path, output_path, visualization_path=None):
    """Main processing function for confidence-guided extraction"""
    print(f"\nProcessing: {mesh_path}")
    print("=" * 60)
    
    # Load mesh
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None
    
    # Get data directory (assume it's the parent directory of the mesh file)
    mesh_path = Path(mesh_path)
    data_dir = mesh_path.parent
    
    # Step 1: Analyze frame metadata
    frame_analysis = analyze_frame_metadata(data_dir)
    
    # Step 2: Load confidence maps
    confidence_maps, confidence_stats = load_confidence_maps(data_dir)
    
    if confidence_maps is None:
        print("Warning: No confidence maps available. Using basic occupancy grid.")
        confidence_maps = []
        confidence_stats = {'confidence_distribution': {'percentiles': {'75th': 128, '25th': 64}}}
    
    # Step 3: Create confidence-weighted occupancy grid
    (confidence_weighted_occupancy, confidence_grid, 
     high_confidence_mask, low_confidence_mask, grid_bounds) = create_confidence_weighted_occupancy(
        mesh, confidence_maps, confidence_stats)
    
    # Step 4: Extract structural elements
    wall_regions, wall_mask = extract_structural_elements(
        confidence_weighted_occupancy, high_confidence_mask)
    
    # Step 5: Detect transparent openings
    openings = detect_transparent_openings(confidence_grid, low_confidence_mask, wall_mask)
    
    # Step 6: Convert to world coordinates
    wall_regions_world = convert_to_world_coordinates(wall_regions, grid_bounds, 
                                                     confidence_weighted_occupancy.shape[0])
    openings_world = convert_to_world_coordinates(openings, grid_bounds,
                                                 confidence_weighted_occupancy.shape[0])
    
    # Calculate metrics
    total_wall_area = sum(calculate_area(wall) for wall in wall_regions_world if len(wall) >= 3)
    total_opening_area = sum(calculate_area(opening) for opening in openings_world if len(opening) >= 3)
    
    # Prepare results
    results = {
        'method': 'v16_confidence_guided',
        'mesh_file': str(mesh_path),
        'data_directory': str(data_dir),
        'mesh_stats': {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces)
        },
        'frame_analysis': frame_analysis,
        'confidence_analysis': confidence_stats,
        'structural_extraction': {
            'wall_regions': len(wall_regions_world),
            'total_wall_area_m2': total_wall_area,
            'total_wall_area_ft2': total_wall_area * 10.764,
            'openings_detected': len(openings_world),
            'total_opening_area_m2': total_opening_area,
            'total_opening_area_ft2': total_opening_area * 10.764
        },
        'results': {
            'wall_boundaries': wall_regions_world,
            'openings': openings_world,
            'confidence_metadata': {
                'has_confidence_maps': len(confidence_maps) > 0,
                'confidence_maps_used': len(confidence_maps) if confidence_maps else 0
            }
        }
    }
    
    # Create visualization
    if visualization_path:
        create_visualization(mesh, confidence_weighted_occupancy, confidence_grid,
                           wall_mask, high_confidence_mask, low_confidence_mask,
                           wall_regions_world, openings_world, visualization_path)
    
    # Print results summary
    print(f"\nResults Summary:")
    print(f"Frame metadata: {'Available' if frame_analysis else 'Not available'}")
    print(f"Confidence maps: {len(confidence_maps) if confidence_maps else 0} loaded")
    print(f"Wall regions: {len(wall_regions_world)}")
    print(f"Total wall area: {total_wall_area:.1f} m² ({total_wall_area * 10.764:.1f} ft²)")
    print(f"Openings detected: {len(openings_world)}")
    print(f"Opening area: {total_opening_area:.1f} m² ({total_opening_area * 10.764:.1f} ft²)")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print(f"Results saved to {output_path}")
    return results

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python v16_confidence_guided.py <mesh.obj> <output.json> [visualization.png]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    output_path = sys.argv[2]
    visualization_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_mesh_confidence_guided(mesh_path, output_path, visualization_path)

if __name__ == "__main__":
    main()
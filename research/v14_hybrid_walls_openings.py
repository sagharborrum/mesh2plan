#!/usr/bin/env python3
"""
v14: Hybrid Approach - v11 Walls + v12 Openings
================================================

Approach:
1. Use v11's normal-based wall segmentation for robust boundary extraction
2. Use v12's image processing for door/window detection
3. Combine the results: 
   - Wall boundaries from normal analysis (more accurate)
   - Opening detection from image processing (better gap detection)
4. Project openings back to the wall boundary to create final floor plan

This hybrid approach leverages the strengths of both methods:
- v11's excellent boundary detection
- v12's superior opening detection capabilities
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
from scipy import ndimage

warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def classify_faces_by_normal(mesh, vertical_threshold=0.7):
    """Classify mesh faces as floors, ceilings, or walls based on their normals"""
    print("Phase 1: Classifying faces by normal direction...")
    
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Up vector (assume Y is up)
    up_vector = np.array([0, 1, 0])
    
    # Calculate dot product with up vector
    dot_products = np.abs(np.dot(face_normals, up_vector))
    
    # Classify faces
    wall_faces = dot_products < vertical_threshold
    floor_faces = (dot_products >= vertical_threshold) & (np.dot(face_normals, up_vector) < 0)
    ceiling_faces = (dot_products >= vertical_threshold) & (np.dot(face_normals, up_vector) > 0)
    
    print(f"  Wall faces: {np.sum(wall_faces)}")
    print(f"  Floor faces: {np.sum(floor_faces)}")
    print(f"  Ceiling faces: {np.sum(ceiling_faces)}")
    
    return wall_faces, floor_faces, ceiling_faces

def extract_wall_boundaries(mesh, wall_faces, clustering_eps=0.3, min_samples=5):
    """Extract wall boundaries using face normal classification"""
    print("Phase 1: Extracting wall boundaries...")
    
    if not np.any(wall_faces):
        print("  No wall faces found!")
        return np.array([]), 0
    
    # Get wall face vertices
    wall_face_indices = np.where(wall_faces)[0]
    print(f"  Processing {len(wall_face_indices)} wall faces...")
    
    all_wall_vertices = []
    
    # Sample faces if too many to avoid memory issues
    if len(wall_face_indices) > 10000:
        print(f"  Sampling {10000} faces from {len(wall_face_indices)} total...")
        wall_face_indices = np.random.choice(wall_face_indices, 10000, replace=False)
    
    for i, face_idx in enumerate(wall_face_indices):
        if i % 5000 == 0:
            print(f"    Processing face {i}/{len(wall_face_indices)}")
        face_vertices = mesh.vertices[mesh.faces[face_idx]]
        all_wall_vertices.extend(face_vertices)
    
    wall_vertices = np.array(all_wall_vertices)
    
    if len(wall_vertices) == 0:
        return np.array([]), 0
    
    # Project wall vertices to XZ plane (top-down view)
    wall_points_2d = wall_vertices[:, [0, 2]]  # Remove Y coordinate
    
    print(f"  Total wall vertices: {len(wall_vertices)}")
    
    # Sample points if too many for clustering
    if len(wall_points_2d) > 5000:
        print(f"  Sampling 5000 points from {len(wall_points_2d)} for clustering...")
        indices = np.random.choice(len(wall_points_2d), 5000, replace=False)
        clustering_points = wall_points_2d[indices]
    else:
        clustering_points = wall_points_2d
    
    # Use clustering to group wall points
    print("  Running DBSCAN clustering...")
    if len(clustering_points) > min_samples:
        clustering = DBSCAN(eps=clustering_eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(clustering_points)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"  Wall clusters found: {n_clusters}")
    else:
        cluster_labels = np.zeros(len(clustering_points))
        n_clusters = 1 if len(clustering_points) > 0 else 0
    
    # Create convex hull of all wall points for boundary
    print("  Creating convex hull...")
    if len(wall_points_2d) >= 3:
        # Use all points for hull, not just clustered ones
        hull = ConvexHull(wall_points_2d)
        boundary_points = wall_points_2d[hull.vertices]
        print(f"  Convex hull created with {len(boundary_points)} boundary points")
        return boundary_points, n_clusters
    
    return wall_points_2d, n_clusters

def render_top_down_depth_map(mesh, resolution=256, padding=0.1):
    """Render mesh to top-down depth map using vertex projection (faster than raycasting)"""
    print("Phase 2: Rendering top-down depth map...")
    
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
    
    # Initialize depth map
    depth_map = np.zeros((resolution, resolution))
    
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
    
    # Fill depth map with maximum height at each pixel
    print(f"  Projecting {len(valid_x)} vertices to depth map...")
    for x, z, y in zip(valid_x, valid_z, valid_y):
        depth_map[z, x] = max(depth_map[z, x], y - y_min)
    
    return depth_map, (x_min, x_max, z_min, z_max)

def detect_openings_from_depth_map(depth_map, wall_height_threshold=0.5, 
                                   canny_low=50, canny_high=150):
    """Detect door/window openings using image processing on depth map"""
    print("Phase 2: Detecting openings from depth map...")
    
    # Normalize depth map to 0-255
    normalized_map = np.uint8(255 * (depth_map / np.max(depth_map)) if np.max(depth_map) > 0 else depth_map)
    
    # Create wall mask (areas above threshold height)
    wall_mask = depth_map > wall_height_threshold
    wall_image = (wall_mask * 255).astype(np.uint8)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    wall_image = cv2.morphologyEx(wall_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    wall_image = cv2.morphologyEx(wall_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Edge detection
    edges = cv2.Canny(wall_image, canny_low, canny_high)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detect openings as gaps in the wall contours
    opening_regions = []
    
    for contour in contours:
        # Create contour mask
        contour_mask = np.zeros_like(wall_image)
        cv2.fillPoly(contour_mask, [contour], 255)
        
        # Find gaps: areas that should be walls but aren't filled
        expected_wall = cv2.dilate(contour_mask, kernel, iterations=2)
        actual_wall = wall_image
        gaps = cv2.bitwise_and(expected_wall, cv2.bitwise_not(actual_wall))
        
        # Find gap contours
        gap_contours, _ = cv2.findContours(gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for gap_contour in gap_contours:
            area = cv2.contourArea(gap_contour)
            if area > 50:  # Minimum opening size
                opening_regions.append(gap_contour)
    
    print(f"  Found {len(opening_regions)} potential openings")
    
    return opening_regions, wall_image, edges

def project_openings_to_world(openings, depth_map_bounds, resolution):
    """Project opening regions from image coordinates to world coordinates"""
    print("Phase 3: Projecting openings to world coordinates...")
    
    x_min, x_max, z_min, z_max = depth_map_bounds
    
    opening_world_coords = []
    for opening in openings:
        # Convert contour points from image to world coordinates
        world_points = []
        for point in opening.squeeze():
            col, row = point  # Note: OpenCV uses (x,y) = (col,row)
            
            # Convert to world coordinates
            world_x = x_min + (col / resolution) * (x_max - x_min)
            world_z = z_min + (row / resolution) * (z_max - z_min)
            world_points.append([world_x, world_z])
        
        opening_world_coords.append(np.array(world_points))
    
    return opening_world_coords

def combine_walls_and_openings(boundary_points, openings, buffer_distance=0.2):
    """Combine wall boundaries with detected openings"""
    print("Phase 3: Combining wall boundaries with openings...")
    
    if len(boundary_points) == 0:
        return boundary_points, openings
    
    # For now, keep them separate but ensure they're in the same coordinate system
    # Future improvement: actually modify the boundary to include opening gaps
    
    print(f"  Final boundary points: {len(boundary_points)}")
    print(f"  Final openings: {len(openings)}")
    
    return boundary_points, openings

def calculate_room_area(boundary_points):
    """Calculate room area using shoelace formula"""
    if len(boundary_points) < 3:
        return 0.0
    
    # Shoelace formula
    x = boundary_points[:, 0]
    y = boundary_points[:, 1]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                        for i in range(-1, len(x)-1)))

def create_visualization(mesh, boundary_points, openings, wall_faces, wall_image, 
                        edges, output_path):
    """Create comprehensive visualization"""
    print("Creating visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original mesh (top view)
    vertices_2d = mesh.vertices[:, [0, 2]]
    ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='lightgray', s=0.1, alpha=0.5)
    ax1.set_title('Original Mesh (Top View)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Wall classification and boundary
    if np.any(wall_faces):
        wall_face_indices = np.where(wall_faces)[0]
        wall_vertices = []
        for face_idx in wall_face_indices:
            face_vertices = mesh.vertices[mesh.faces[face_idx]]
            wall_vertices.extend(face_vertices)
        wall_vertices_2d = np.array(wall_vertices)[:, [0, 2]]
        ax2.scatter(wall_vertices_2d[:, 0], wall_vertices_2d[:, 1], 
                   c='red', s=0.1, alpha=0.3, label='Wall vertices')
    
    if len(boundary_points) > 0:
        # Close the boundary for plotting
        boundary_closed = np.vstack([boundary_points, boundary_points[0]])
        ax2.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                'b-', linewidth=3, label='Boundary')
        ax2.scatter(boundary_points[:, 0], boundary_points[:, 1], 
                   c='blue', s=50, label='Boundary points')
    
    ax2.set_title('Wall Segmentation & Boundary')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Depth map and wall detection
    ax3.imshow(wall_image, cmap='gray', origin='lower')
    ax3.set_title('Wall Detection (Image Processing)')
    
    # 4. Detected openings
    ax4.imshow(edges, cmap='gray', origin='lower')
    if openings:
        for i, opening in enumerate(openings):
            if len(opening) > 0:
                opening_closed = np.vstack([opening, opening[0]])
                ax4.plot(opening_closed[:, 0], opening_closed[:, 1], 
                        linewidth=2, label=f'Opening {i+1}')
    ax4.set_title('Detected Openings')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def process_mesh_hybrid(mesh_path, output_path, visualization_path=None):
    """Main processing function for hybrid approach"""
    print(f"\nProcessing: {mesh_path}")
    print("=" * 60)
    
    # Load mesh
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None
    
    # Phase 1: Wall boundary extraction (v11 approach)
    wall_faces, floor_faces, ceiling_faces = classify_faces_by_normal(mesh)
    boundary_points, n_clusters = extract_wall_boundaries(mesh, wall_faces)
    
    # Phase 2: Opening detection (v12 approach)
    depth_map, depth_bounds = render_top_down_depth_map(mesh)
    opening_regions, wall_image, edges = detect_openings_from_depth_map(depth_map)
    openings_world = project_openings_to_world(opening_regions, depth_bounds, 
                                               depth_map.shape[0])
    
    # Phase 3: Combine results
    final_boundary, final_openings = combine_walls_and_openings(boundary_points, 
                                                                openings_world)
    
    # Calculate metrics
    room_area_m2 = calculate_room_area(final_boundary)
    room_area_ft2 = room_area_m2 * 10.764  # Convert to square feet
    
    # Prepare results
    results = {
        'method': 'v14_hybrid_walls_openings',
        'mesh_file': str(mesh_path),
        'mesh_stats': {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces)
        },
        'wall_analysis': {
            'wall_faces': int(np.sum(wall_faces)),
            'floor_faces': int(np.sum(floor_faces)), 
            'ceiling_faces': int(np.sum(ceiling_faces)),
            'wall_clusters': n_clusters
        },
        'boundary_extraction': {
            'boundary_points': len(final_boundary),
            'room_area_m2': room_area_m2,
            'room_area_ft2': room_area_ft2
        },
        'opening_detection': {
            'openings_found': len(final_openings),
            'depth_map_resolution': depth_map.shape[0]
        },
        'results': {
            'boundary_points': final_boundary,
            'openings': final_openings,
            'wall_classification': {
                'wall_face_indices': np.where(wall_faces)[0] if np.any(wall_faces) else []
            }
        }
    }
    
    # Create visualization
    if visualization_path:
        create_visualization(mesh, final_boundary, final_openings, 
                           wall_faces, wall_image, edges, visualization_path)
    
    # Save results
    print(f"\nResults Summary:")
    print(f"Room area: {room_area_m2:.1f} m² ({room_area_ft2:.1f} ft²)")
    print(f"Boundary points: {len(final_boundary)}")
    print(f"Openings detected: {len(final_openings)}")
    print(f"Wall clusters: {n_clusters}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print(f"Results saved to {output_path}")
    return results

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python v14_hybrid_walls_openings.py <mesh.obj> <output.json> [visualization.png]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    output_path = sys.argv[2]
    visualization_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_mesh_hybrid(mesh_path, output_path, visualization_path)

if __name__ == "__main__":
    main()
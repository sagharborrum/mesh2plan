#!/usr/bin/env python3
"""
v16: Confidence-Map Guided Extraction (FIXED)
=============================================

FIXES APPLIED:
- Better confidence threshold tuning for narrow ranges (mean=1.9, range 0-2)
- Use confidence to weight vertices, then apply v11-style analysis
- Proper floor level detection and wall height filtering
- Apply confidence-weighted vertex analysis with alpha shapes
- Combine confidence information with proven geometric analysis
- Handle sensor-specific calibration for confidence values

This approach:
1. Load confidence maps and frame metadata
2. Use confidence to weight mesh vertices by reliability
3. Apply floor level detection like other fixed approaches  
4. Extract wall foot-points weighted by confidence
5. Use alpha shape boundary extraction with confidence weighting
6. Detect openings using confidence-based transparency analysis
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN
from collections import defaultdict
import glob
from PIL import Image
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

def analyze_frame_metadata(data_dir):
    """Analyze frame JSON files for camera poses and quality - ENHANCED"""
    print("Step 1: Analyzing frame metadata...")
    
    frame_files = glob.glob(str(Path(data_dir) / "frame_*.json"))
    
    if not frame_files:
        print("  No frame JSON files found!")
        return None
    
    frame_data = []
    
    for frame_file in sorted(frame_files)[:20]:  # Analyze first 20 frames
        try:
            with open(frame_file, 'r') as f:
                data = json.load(f)
                frame_data.append(data)
        except Exception as e:
            print(f"  Error reading {frame_file}: {e}")
    
    if not frame_data:
        print("  No valid frame data loaded!")
        return None
    
    # Enhanced analysis with all available fields
    frame_analysis = {
        'total_frames_analyzed': len(frame_data),
        'total_frames_available': len(frame_files),
        'available_fields': list(frame_data[0].keys()) if frame_data else []
    }
    
    # Analyze all numerical fields
    numerical_fields = ['motionQuality', 'averageVelocity', 'averageAngularVelocity', 
                       'exposureDuration', 'cameraGrain', 'frame_index', 'time']
    
    for field in numerical_fields:
        values = [f.get(field, 0) for f in frame_data if field in f and f[field] is not None]
        if values:
            frame_analysis[f'{field}_stats'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    # Camera intrinsics analysis
    if 'intrinsics' in frame_data[0]:
        frame_analysis['camera_intrinsics'] = frame_data[0]['intrinsics']
        frame_analysis['has_projection_matrix'] = 'projectionMatrix' in frame_data[0]
    
    print(f"  Analyzed {len(frame_data)} frames (of {len(frame_files)} available)")
    print(f"  Available fields: {', '.join(frame_analysis['available_fields'])}")
    
    # Print key statistics
    if 'motionQuality_stats' in frame_analysis:
        stats = frame_analysis['motionQuality_stats']
        print(f"  Motion quality: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f}-{stats['max']:.3f})")
    
    return frame_analysis

def load_and_analyze_confidence_maps(data_dir, max_maps=10):
    """Load confidence maps with enhanced analysis - BETTER THRESHOLDING"""
    print("Step 2: Loading and analyzing confidence maps...")
    
    conf_files = glob.glob(str(Path(data_dir) / "conf_*.png"))
    
    if not conf_files:
        print("  No confidence map files found!")
        return None, None
    
    print(f"  Found {len(conf_files)} confidence maps, loading {min(max_maps, len(conf_files))}...")
    
    confidence_maps = []
    
    for i, conf_file in enumerate(sorted(conf_files)[:max_maps]):
        try:
            # Load as raw values (no conversion)
            conf_map = np.array(Image.open(conf_file))
            confidence_maps.append(conf_map)
            
        except Exception as e:
            print(f"  Error loading {conf_file}: {e}")
    
    if not confidence_maps:
        print("  No confidence maps loaded successfully!")
        return None, None
    
    # Enhanced confidence analysis
    all_confidences = np.concatenate([cm.flatten() for cm in confidence_maps])
    
    confidence_stats = {
        'file_count': len(conf_files),
        'loaded_count': len(confidence_maps),
        'resolution': confidence_maps[0].shape,
        'value_range': [float(np.min(all_confidences)), float(np.max(all_confidences))],
        'mean': float(np.mean(all_confidences)),
        'std': float(np.std(all_confidences)),
        'percentiles': {
            '10th': float(np.percentile(all_confidences, 10)),
            '25th': float(np.percentile(all_confidences, 25)),
            '50th': float(np.percentile(all_confidences, 50)),
            '75th': float(np.percentile(all_confidences, 75)),
            '90th': float(np.percentile(all_confidences, 90)),
            '95th': float(np.percentile(all_confidences, 95))
        }
    }
    
    print(f"  Loaded {len(confidence_maps)} confidence maps")
    print(f"  Resolution: {confidence_stats['resolution']}")
    print(f"  Value range: {confidence_stats['value_range'][0]:.1f} - {confidence_stats['value_range'][1]:.1f}")
    print(f"  Mean: {confidence_stats['mean']:.2f} ± {confidence_stats['std']:.2f}")
    
    # Auto-tune thresholds for narrow ranges
    value_span = confidence_stats['value_range'][1] - confidence_stats['value_range'][0]
    if value_span < 50:  # Narrow range (like 0-2)
        print(f"  Detected narrow confidence range ({value_span:.1f}), using percentile-based thresholds")
        confidence_stats['auto_thresholds'] = {
            'high_confidence': confidence_stats['percentiles']['75th'],
            'medium_confidence': confidence_stats['percentiles']['50th'], 
            'low_confidence': confidence_stats['percentiles']['25th']
        }
    else:  # Wide range
        confidence_stats['auto_thresholds'] = {
            'high_confidence': confidence_stats['mean'] + 0.5 * confidence_stats['std'],
            'medium_confidence': confidence_stats['mean'],
            'low_confidence': confidence_stats['mean'] - 0.5 * confidence_stats['std']
        }
    
    print(f"  Auto-tuned thresholds: High={confidence_stats['auto_thresholds']['high_confidence']:.2f}, " +
          f"Medium={confidence_stats['auto_thresholds']['medium_confidence']:.2f}, " +
          f"Low={confidence_stats['auto_thresholds']['low_confidence']:.2f}")
    
    return confidence_maps, confidence_stats

def weight_vertices_by_confidence(mesh, confidence_maps, confidence_stats, floor_level):
    """Weight mesh vertices by confidence and extract wall foot-points - KEY FIX"""
    print("Step 3: Weighting vertices by confidence...")
    
    if not confidence_maps or not confidence_stats:
        print("  No confidence data available, using uniform weights")
        # Apply standard wall foot-point extraction
        vertices = mesh.vertices
        z_coords = vertices[:, 2]
        foot_mask = np.abs(z_coords - floor_level) <= 0.3
        wall_footpoints = vertices[foot_mask]
        vertex_weights = np.ones(len(wall_footpoints))
        
        return wall_footpoints, vertex_weights
    
    # Get vertices in wall foot-point range
    vertices = mesh.vertices
    z_coords = vertices[:, 2]
    foot_mask = np.abs(z_coords - floor_level) <= 0.3
    wall_footpoints = vertices[foot_mask]
    
    print(f"  Wall foot-points before confidence filtering: {len(wall_footpoints):,}")
    
    if len(wall_footpoints) == 0:
        print("  No wall foot-points found!")
        return np.array([]), np.array([])
    
    # Create aggregate confidence map
    avg_confidence = np.mean(confidence_maps, axis=0)
    conf_height, conf_width = avg_confidence.shape
    
    # Map 3D wall foot-points to confidence map coordinates
    x_coords = wall_footpoints[:, 0]
    y_coords = wall_footpoints[:, 1]
    
    # Estimate bounds from mesh
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add padding for safety
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    # Map to confidence image coordinates
    conf_x = ((x_coords - x_min) / (x_max - x_min) * (conf_width - 1)).astype(int)
    conf_y = ((y_coords - y_min) / (y_max - y_min) * (conf_height - 1)).astype(int)
    
    # Clamp to valid ranges
    conf_x = np.clip(conf_x, 0, conf_width - 1)
    conf_y = np.clip(conf_y, 0, conf_height - 1)
    
    # Extract confidence values for each vertex
    vertex_confidences = avg_confidence[conf_y, conf_x]
    
    # Convert to weights using auto-tuned thresholds
    thresholds = confidence_stats['auto_thresholds']
    vertex_weights = np.ones(len(vertex_confidences))
    
    # Apply confidence-based weighting
    high_conf_mask = vertex_confidences >= thresholds['high_confidence']
    low_conf_mask = vertex_confidences <= thresholds['low_confidence']
    
    vertex_weights[high_conf_mask] = 2.0  # Boost high-confidence vertices
    vertex_weights[low_conf_mask] = 0.3   # Reduce low-confidence vertices
    
    # Filter out very low confidence vertices
    min_weight_threshold = 0.2
    keep_mask = vertex_weights >= min_weight_threshold
    
    filtered_footpoints = wall_footpoints[keep_mask]
    filtered_weights = vertex_weights[keep_mask]
    
    print(f"  Confidence filtering: {len(wall_footpoints):,} -> {len(filtered_footpoints):,} vertices")
    print(f"  High confidence vertices: {np.sum(high_conf_mask):,}")
    print(f"  Low confidence vertices: {np.sum(low_conf_mask):,}")
    print(f"  Weight range: {np.min(filtered_weights):.2f} - {np.max(filtered_weights):.2f}")
    
    return filtered_footpoints, filtered_weights

def confidence_weighted_alpha_shape(points, weights, alpha=0.5):
    """Create alpha shape with confidence weighting - V11 STYLE WITH CONFIDENCE"""
    print(f"Step 4: Confidence-weighted alpha shape (alpha={alpha})...")
    
    if len(points) < 3:
        return points
    
    # Project to XY plane
    points_2d = points[:, [0, 1]]
    
    try:
        # Use confidence weighting by duplicating high-confidence points
        weighted_points = []
        for i, point in enumerate(points_2d):
            weight = weights[i]
            # Duplicate points based on weight (more weight = more influence)
            num_copies = max(1, int(weight * 3))
            for _ in range(num_copies):
                # Add slight noise to avoid exact duplicates
                noise = np.random.normal(0, 0.01, 2)  # 1cm noise
                weighted_points.append(point + noise)
        
        weighted_points = np.array(weighted_points)
        
        # Delaunay triangulation on weighted points
        tri = Delaunay(weighted_points)
        triangles = tri.simplices
        
        # Compute circumradius for each triangle
        def circumradius(triangle_points):
            a, b, c = triangle_points
            side_a = np.linalg.norm(b - c)
            side_b = np.linalg.norm(a - c)  
            side_c = np.linalg.norm(a - b)
            
            area = 0.5 * abs(np.cross(b - a, c - a))
            
            if area < 1e-10:
                return float('inf')
            
            return (side_a * side_b * side_c) / (4 * area)
        
        # Filter triangles by alpha criterion
        valid_triangles = []
        for triangle in triangles:
            triangle_points = weighted_points[triangle]
            circumr = circumradius(triangle_points)
            
            if circumr <= 1.0 / alpha:
                valid_triangles.append(triangle)
        
        if not valid_triangles:
            # Fallback to convex hull
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        
        # Extract boundary edges
        edge_count = defaultdict(int)
        for triangle in valid_triangles:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                edge_count[edge] += 1
        
        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # Convert edges to boundary polygon using original points
        adjacency = defaultdict(list)
        for edge in boundary_edges:
            adjacency[edge[0]].append(edge[1])
            adjacency[edge[1]].append(edge[0])
        
        # Find longest connected component
        visited = set()
        longest_path = []
        
        for start_node in adjacency.keys():
            if start_node in visited:
                continue
            
            current_path = [start_node]
            current = start_node
            visited.add(current)
            
            while True:
                next_nodes = [n for n in adjacency[current] if n not in visited]
                if not next_nodes:
                    break
                
                next_node = next_nodes[0]
                current_path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(current_path) > len(longest_path):
                longest_path = current_path
        
        # Map back to original points by finding nearest
        boundary_points = []
        for wp_idx in longest_path:
            wp = weighted_points[wp_idx]
            # Find closest original point
            distances = np.linalg.norm(points_2d - wp, axis=1)
            closest_idx = np.argmin(distances)
            boundary_points.append(points_2d[closest_idx])
        
        boundary_points = np.array(boundary_points)
        
        # Remove duplicates
        unique_boundary = []
        for point in boundary_points:
            is_duplicate = False
            for existing in unique_boundary:
                if np.linalg.norm(point - existing) < 0.1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boundary.append(point)
        
        boundary_points = np.array(unique_boundary) if unique_boundary else boundary_points
        
        print(f"  Confidence-weighted alpha shape: {len(boundary_points)} boundary points")
        return boundary_points
        
    except Exception as e:
        print(f"  Error in confidence-weighted alpha shape: {e}")
        # Fallback to convex hull
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]

def detect_transparency_openings(confidence_maps, confidence_stats, boundary_points):
    """Detect openings using confidence-based transparency detection"""
    print("Step 5: Detecting transparency-based openings...")
    
    if not confidence_maps or len(boundary_points) < 3:
        return []
    
    openings = []
    
    # Use low-confidence regions to infer transparent openings (windows/glass doors)
    avg_confidence = np.mean(confidence_maps, axis=0)
    thresholds = confidence_stats['auto_thresholds']
    
    # Find very low confidence regions
    transparent_mask = avg_confidence < thresholds['low_confidence']
    
    # Find contours in transparent regions
    contours, _ = cv2.findContours(transparent_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to world coordinates (approximate)
    conf_height, conf_width = avg_confidence.shape
    
    # Estimate bounds from boundary points
    boundary_array = np.array(boundary_points)
    x_min, x_max = boundary_array[:, 0].min(), boundary_array[:, 0].max()
    y_min, y_max = boundary_array[:, 1].min(), boundary_array[:, 1].max()
    
    for contour in contours:
        area_pixels = cv2.contourArea(contour)
        if area_pixels > 50:  # Minimum opening size
            
            # Find centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Convert to world coordinates (approximate)
                world_x = x_min + (cx / conf_width) * (x_max - x_min)
                world_y = y_min + (cy / conf_height) * (y_max - y_min)
                
                # Estimate size
                opening_area_m2 = area_pixels * ((x_max - x_min) * (y_max - y_min)) / (conf_width * conf_height)
                
                opening_type = "window" if opening_area_m2 < 3.0 else "glass_wall"
                
                openings.append({
                    'type': opening_type,
                    'position': [world_x, world_y],
                    'area_m2': opening_area_m2,
                    'detection_method': 'confidence_transparency',
                    'confidence_based': True
                })
    
    print(f"  Found {len(openings)} transparency-based openings")
    return openings

def analyze_mesh_confidence_guided(mesh_path):
    """Main confidence-guided analysis with fixes applied"""
    print(f"\n=== Confidence-guided Extraction v16 (FIXED): {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error loading mesh")
        return None
    
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Get data directory
    mesh_path = Path(mesh_path)
    data_dir = mesh_path.parent
    
    # Step 1: Find floor level (like other fixed approaches)
    floor_level = find_floor_level(mesh)
    
    # Step 2: Analyze frame metadata
    frame_analysis = analyze_frame_metadata(data_dir)
    
    # Step 3: Load and analyze confidence maps with better thresholding
    confidence_maps, confidence_stats = load_and_analyze_confidence_maps(data_dir)
    
    # Step 4: Weight vertices by confidence and extract wall footpoints
    wall_footpoints, vertex_weights = weight_vertices_by_confidence(
        mesh, confidence_maps, confidence_stats, floor_level)
    
    if len(wall_footpoints) == 0:
        print("No wall footpoints found!")
        return None
    
    # Step 5: Create confidence-weighted alpha shape (v11-style with confidence)
    boundary_points = confidence_weighted_alpha_shape(wall_footpoints, vertex_weights)
    
    # Step 6: Detect transparency-based openings
    openings = detect_transparency_openings(confidence_maps, confidence_stats, boundary_points)
    
    # Calculate room area
    if len(boundary_points) >= 3:
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
        "method": "confidence_guided_v16_fixed",
        "mesh_file": str(mesh_path),
        "data_directory": str(data_dir),
        "parameters": {
            "floor_level": float(floor_level),
            "has_confidence_data": confidence_maps is not None,
            "confidence_maps_used": len(confidence_maps) if confidence_maps else 0,
            "auto_thresholds": confidence_stats['auto_thresholds'] if confidence_stats else None
        },
        "room_boundary": boundary_points.tolist() if isinstance(boundary_points, np.ndarray) else boundary_points,
        "openings": openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "frame_analysis": frame_analysis,
        "confidence_analysis": confidence_stats,
        "analysis_stats": {
            "wall_footpoints": len(wall_footpoints),
            "boundary_vertices": len(boundary_points),
            "confidence_weighted_vertices": len(wall_footpoints),
            "transparency_openings": len(openings)
        }
    }

def create_visualization(result, output_path):
    """Create comprehensive confidence-guided visualization"""
    if not result:
        return
        
    print("Creating confidence-guided visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Confidence-weighted boundary
    ax1.set_title(f"Confidence-Weighted Boundary\nArea: {result['room_area_sqm']:.1f} m²")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    if len(result['room_boundary']) >= 3:
        boundary = np.array(result['room_boundary'])
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax1.fill(boundary_closed[:, 0], boundary_closed[:, 1], 
                alpha=0.3, color='lightblue', edgecolor='blue', linewidth=2)
        ax1.scatter(boundary[:, 0], boundary[:, 1], 
                   c='red', s=50, alpha=0.7, zorder=5)
        
        # Number the vertices
        for i, point in enumerate(boundary):
            ax1.annotate(str(i), point, xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
    
    # Plot 2: Confidence-based openings
    ax2.set_title(f"Confidence-Based Openings\n{len(result['openings'])} detected")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Draw boundary in gray
    if len(result['room_boundary']) >= 3:
        boundary = np.array(result['room_boundary'])
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax2.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                color='gray', linewidth=1, alpha=0.5)
    
    # Draw openings
    for opening in result['openings']:
        pos = opening['position']
        color = 'orange' if opening['type'] == 'window' else 'red'
        ax2.plot(pos[0], pos[1], 'o', color=color, markersize=10, alpha=0.8)
        ax2.annotate(f"{opening['type']}\n{opening['area_m2']:.2f}m²", 
                    (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # Plot 3: Confidence analysis
    ax3.set_title("Confidence Data Analysis")
    ax3.axis('off')
    
    if result['confidence_analysis']:
        conf_stats = result['confidence_analysis']
        conf_text = f"""CONFIDENCE ANALYSIS:

Maps Available: {conf_stats['loaded_count']} of {conf_stats['file_count']}
Resolution: {conf_stats['resolution']}
Value Range: {conf_stats['value_range'][0]:.1f} - {conf_stats['value_range'][1]:.1f}
Mean ± Std: {conf_stats['mean']:.2f} ± {conf_stats['std']:.2f}

AUTO-TUNED THRESHOLDS:
High: {conf_stats['auto_thresholds']['high_confidence']:.2f}
Medium: {conf_stats['auto_thresholds']['medium_confidence']:.2f}  
Low: {conf_stats['auto_thresholds']['low_confidence']:.2f}

PERCENTILES:
90th: {conf_stats['percentiles']['90th']:.2f}
75th: {conf_stats['percentiles']['75th']:.2f}
50th: {conf_stats['percentiles']['50th']:.2f}
25th: {conf_stats['percentiles']['25th']:.2f}"""
    else:
        conf_text = "No confidence data available"
    
    ax3.text(0.05, 0.95, conf_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Frame analysis
    ax4.set_title("Frame Metadata Analysis")
    ax4.axis('off')
    
    if result['frame_analysis']:
        frame_stats = result['frame_analysis']
        frame_text = f"""FRAME ANALYSIS:

Frames: {frame_stats['total_frames_analyzed']} of {frame_stats['total_frames_available']}
Available Fields: {len(frame_stats['available_fields'])}

"""
        
        # Add motion quality stats if available
        if 'motionQuality_stats' in frame_stats:
            mq = frame_stats['motionQuality_stats']
            frame_text += f"""MOTION QUALITY:
Mean: {mq['mean']:.3f} ± {mq['std']:.3f}
Range: {mq['min']:.3f} - {mq['max']:.3f}

"""
        
        # Add other stats
        for field in ['averageVelocity_stats', 'exposureDuration_stats']:
            if field in frame_stats:
                stats = frame_stats[field]
                field_name = field.replace('_stats', '').replace('average', 'Avg ')
                frame_text += f"""{field_name}:
Mean: {stats['mean']:.3f}
Range: {stats['min']:.3f} - {stats['max']:.3f}

"""
    else:
        frame_text = "No frame metadata available"
    
    ax4.text(0.05, 0.95, frame_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confidence-guided visualization saved to {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python v16_confidence_guided.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else output_path.with_suffix('.png')
    
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        return
    
    try:
        result = analyze_mesh_confidence_guided(mesh_path)
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
            print(f"   Boundary vertices: {len(result['room_boundary'])}")
            print(f"   Confidence-weighted vertices: {result['analysis_stats']['confidence_weighted_vertices']:,}")
            print(f"   Transparency openings: {result['analysis_stats']['transparency_openings']}")
            print(f"   Confidence maps used: {result['parameters']['confidence_maps_used']}")
            print(f"   Floor level: {result['parameters']['floor_level']:.2f}m")
            
            if result['parameters']['auto_thresholds']:
                thresholds = result['parameters']['auto_thresholds']
                print(f"   Confidence thresholds: H={thresholds['high_confidence']:.2f}, " +
                      f"M={thresholds['medium_confidence']:.2f}, L={thresholds['low_confidence']:.2f}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
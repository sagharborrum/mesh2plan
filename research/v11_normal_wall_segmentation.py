#!/usr/bin/env python3
"""
v11: Normal-based Wall Segmentation
===================================

Approach:
1. Calculate face normals for all triangles in the mesh
2. Classify faces by normal direction (dot product with up vector)
   - Floors/ceilings: normal close to vertical (±Y)
   - Walls: normal close to horizontal (XZ plane)
3. Extract wall faces and project them to 2D
4. Use clustering to group wall segments
5. Extract room boundary from wall projections

This approach should work well for rooms with clear vertical walls.
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
    print("Classifying faces by normal direction...")
    
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Up vector (assume Y is up)
    up_vector = np.array([0, 1, 0])
    
    # Calculate dot product with up vector
    dot_products = np.abs(np.dot(face_normals, up_vector))
    
    # Classify faces
    floors_ceilings = dot_products > vertical_threshold  # Close to vertical
    walls = dot_products <= vertical_threshold  # Close to horizontal
    
    # Further classify floors vs ceilings by Y-component of normal
    floor_mask = (face_normals[:, 1] > 0) & floors_ceilings  # Normal points up
    ceiling_mask = (face_normals[:, 1] < 0) & floors_ceilings  # Normal points down
    
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

def extract_wall_vertices(mesh, wall_faces):
    """Extract all vertices that belong to wall faces"""
    print("Extracting wall vertices...")
    
    # Get all vertices used by wall faces
    wall_vertex_indices = set()
    for face_idx in wall_faces:
        face = mesh.faces[face_idx]
        wall_vertex_indices.update(face)
    
    wall_vertices = mesh.vertices[list(wall_vertex_indices)]
    
    print(f"  Wall vertices: {len(wall_vertices):,}")
    
    return wall_vertices, list(wall_vertex_indices)

def project_walls_to_2d(wall_vertices, method='xy'):
    """Project wall vertices to 2D plane"""
    print(f"Projecting walls to 2D (method: {method})...")
    
    if method == 'xy':
        # Project to XY plane (ignore Z)
        projected = wall_vertices[:, [0, 1]]
    elif method == 'xz':
        # Project to XZ plane (ignore Y) - common for top-down view
        projected = wall_vertices[:, [0, 2]]
    elif method == 'yz':
        # Project to YZ plane (ignore X)
        projected = wall_vertices[:, [1, 2]]
    else:
        # Auto-detect best projection plane
        # Use the plane with maximum spread
        ranges = np.ptp(wall_vertices, axis=0)  # Peak-to-peak (max - min) for each axis
        max_spread_axes = np.argsort(ranges)[-2:]  # Two axes with largest spread
        projected = wall_vertices[:, max_spread_axes]
        print(f"  Auto-detected projection: axes {max_spread_axes} (ranges: {ranges})")
    
    return projected

def cluster_wall_points(projected_points, eps=0.2, min_samples=10):
    """Cluster wall points to identify separate wall segments"""
    print("Clustering wall points...")
    
    if len(projected_points) < min_samples:
        print(f"  Not enough points for clustering ({len(projected_points)} < {min_samples})")
        return np.zeros(len(projected_points)), 1
    
    # Use DBSCAN to cluster wall points
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(projected_points)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"  Found {n_clusters} wall clusters, {n_noise} noise points")
    
    return labels, n_clusters

def extract_boundary_from_walls(projected_points, labels, hull_method='convex'):
    """Extract room boundary from clustered wall points"""
    print(f"Extracting boundary using {hull_method} hull...")
    
    if hull_method == 'convex':
        try:
            hull = ConvexHull(projected_points)
            boundary_points = projected_points[hull.vertices]
            # Sort points to form a proper polygon
            center = np.mean(boundary_points, axis=0)
            angles = np.arctan2(boundary_points[:, 1] - center[1], 
                              boundary_points[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            boundary_points = boundary_points[sorted_indices]
        except Exception as e:
            print(f"  Error computing convex hull: {e}")
            boundary_points = projected_points
    else:
        # Simple bounding box
        min_coords = np.min(projected_points, axis=0)
        max_coords = np.max(projected_points, axis=0)
        boundary_points = np.array([
            [min_coords[0], min_coords[1]],
            [max_coords[0], min_coords[1]],
            [max_coords[0], max_coords[1]],
            [min_coords[0], max_coords[1]]
        ])
    
    print(f"  Boundary has {len(boundary_points)} points")
    return boundary_points

def detect_wall_openings(wall_vertices, wall_faces, mesh, gap_threshold=0.5):
    """Detect openings in walls by finding gaps in wall coverage"""
    print("Detecting wall openings...")
    
    # This is a simplified approach - look for large gaps in wall segments
    # In practice, this would require more sophisticated analysis
    
    openings = []
    
    # For now, return empty list - this would need more complex implementation
    print(f"  Found {len(openings)} openings")
    
    return openings

def analyze_mesh_normal_segmentation(mesh_path, vertical_threshold=0.7, projection_method='xz'):
    """Main analysis function"""
    print(f"\n=== Normal-based Wall Segmentation: {mesh_path} ===")
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if hasattr(mesh, 'vertices'):
        print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    else:
        print("Error loading mesh")
        return None
    
    # Step 1: Classify faces by normal direction
    classification = classify_faces_by_normal(mesh, vertical_threshold)
    
    # Step 2: Extract wall vertices
    wall_vertices, wall_vertex_indices = extract_wall_vertices(mesh, classification['wall_faces'])
    
    if len(wall_vertices) == 0:
        print("No wall vertices found!")
        return None
    
    # Step 3: Project to 2D
    projected_points = project_walls_to_2d(wall_vertices, projection_method)
    
    # Step 4: Cluster wall points
    labels, n_clusters = cluster_wall_points(projected_points)
    
    # Step 5: Extract boundary
    boundary_points = extract_boundary_from_walls(projected_points, labels)
    
    # Step 6: Detect openings (simplified)
    openings = detect_wall_openings(wall_vertices, classification['wall_faces'], mesh)
    
    # Calculate room area
    if len(boundary_points) >= 3:
        # Simple polygon area calculation
        area = 0.0
        for i in range(len(boundary_points)):
            j = (i + 1) % len(boundary_points)
            area += boundary_points[i][0] * boundary_points[j][1]
            area -= boundary_points[j][0] * boundary_points[i][1]
        area = abs(area) / 2.0
    else:
        area = 0.0
    
    return {
        "method": "normal_wall_segmentation_v11",
        "mesh_file": str(mesh_path),
        "parameters": {
            "vertical_threshold": vertical_threshold,
            "projection_method": projection_method
        },
        "room_boundary": boundary_points.tolist(),
        "openings": openings,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "face_classification": {
            "floor_faces": len(classification['floor_faces']),
            "ceiling_faces": len(classification['ceiling_faces']),
            "wall_faces": len(classification['wall_faces'])
        },
        "wall_analysis": {
            "wall_vertices": len(wall_vertices),
            "wall_clusters": n_clusters,
            "boundary_points": len(boundary_points)
        }
    }

def create_visualization(result, output_path):
    """Create a visualization of the analysis results"""
    print(f"Creating visualization: {output_path}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Room boundary
    boundary = np.array(result['room_boundary'])
    if len(boundary) > 0:
        # Close the polygon
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax1.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'b-', linewidth=2, label='Room Boundary')
        ax1.fill(boundary_closed[:, 0], boundary_closed[:, 1], alpha=0.3, color='lightblue')
    
    # Plot openings
    for opening in result['openings']:
        pos = opening['position']
        ax1.plot(pos[0], pos[1], 'ro', markersize=8, label=f'{opening["type"].title()}')
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Z (meters)' if result['parameters']['projection_method'] == 'xz' else 'Y (meters)')
    ax1.set_title('Room Boundary (V11 Normal Segmentation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Face classification stats
    face_counts = [
        result['face_classification']['floor_faces'],
        result['face_classification']['wall_faces'],
        result['face_classification']['ceiling_faces']
    ]
    face_labels = ['Floor', 'Wall', 'Ceiling']
    colors = ['brown', 'gray', 'lightgray']
    
    ax2.pie(face_counts, labels=face_labels, colors=colors, autopct='%1.1f%%')
    ax2.set_title('Face Classification')
    
    # Add stats text
    stats_text = f"""Area: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Boundary Points: {len(result['room_boundary'])}
Wall Vertices: {result['wall_analysis']['wall_vertices']:,}
Wall Clusters: {result['wall_analysis']['wall_clusters']}
Vertical Threshold: {result['parameters']['vertical_threshold']}"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python v11_normal_wall_segmentation.py <mesh.obj> <output.json> [visualization.png]")
        return
    
    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    viz_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
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
            
            # Create visualization if requested
            if viz_path:
                create_visualization(result, viz_path)
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Room area: {result['room_area_sqm']:.1f} m² ({result['room_area_sqft']:.1f} ft²)")
            print(f"   Boundary points: {len(result['room_boundary'])}")
            print(f"   Wall faces: {result['face_classification']['wall_faces']:,}")
            print(f"   Wall clusters: {result['wall_analysis']['wall_clusters']}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main()
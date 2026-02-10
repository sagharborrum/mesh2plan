#!/usr/bin/env python3
"""
v11: Normal-based Wall Segmentation (CORRECTED AXES)
==================================================

AXIS CORRECTION APPLIED:
- Diagnostic showed XY projection (Z up) is correct orientation
- Current mesh: 1.74m × 2.70m (Y matches expected 2.7m, X is compressed)  
- Expected: ~5.5m × 2.70m = 11.5 m²
- Using XY projection instead of XZ

This approach:
1. Load mesh and determine coordinate system
2. Classify faces by normal direction  
3. Project wall vertices to XY plane (Z as up)
4. Extract room boundary using improved methods
5. Calculate area and detect features
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

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
    
    # Check face normals to identify up axis
    normals = mesh.face_normals
    up_candidates = {
        'X': np.abs(normals[:,0]).mean(),
        'Y': np.abs(normals[:,1]).mean(),
        'Z': np.abs(normals[:,2]).mean()
    }
    
    up_axis = max(up_candidates, key=up_candidates.get)
    
    print(f"  Coordinate spans: X={spans['X']:.2f}m, Y={spans['Y']:.2f}m, Z={spans['Z']:.2f}m")
    print(f"  Normal analysis: {up_axis} axis has most vertical faces (value: {up_candidates[up_axis]:.3f})")
    print(f"  Determined orientation: {up_axis} is UP")
    
    return {
        'spans': spans,
        'up_axis': up_axis,
        'up_strength': up_candidates[up_axis]
    }

def classify_faces_by_normal(mesh, up_axis='Z', vertical_threshold=0.7):
    """Classify mesh faces as floors, ceilings, or walls"""
    print("Classifying faces by normal direction...")
    
    # Define up vector based on detected axis
    up_vectors = {
        'X': np.array([1, 0, 0]),
        'Y': np.array([0, 1, 0]),
        'Z': np.array([0, 0, 1])
    }
    up_vector = up_vectors[up_axis]
    
    face_normals = mesh.face_normals
    
    # Calculate dot product with up vector
    dot_products = np.abs(np.dot(face_normals, up_vector))
    
    # Classify faces
    horizontal_faces = dot_products > vertical_threshold  # Horizontal surfaces
    wall_faces = dot_products <= vertical_threshold  # Vertical surfaces (walls)
    
    # Separate floors from ceilings based on normal direction
    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[up_axis]
    floor_mask = (face_normals[:, axis_idx] > 0) & horizontal_faces  # Normal points up
    ceiling_mask = (face_normals[:, axis_idx] < 0) & horizontal_faces  # Normal points down
    
    floor_faces = np.where(floor_mask)[0]
    ceiling_faces = np.where(ceiling_mask)[0]
    wall_face_indices = np.where(wall_faces)[0]
    
    print(f"  Floor faces: {len(floor_faces):,}")
    print(f"  Ceiling faces: {len(ceiling_faces):,}")
    print(f"  Wall faces: {len(wall_face_indices):,}")
    
    return {
        'floor_faces': floor_faces,
        'ceiling_faces': ceiling_faces,
        'wall_faces': wall_face_indices,
        'face_normals': face_normals
    }

def extract_wall_boundary_points(mesh, wall_faces, up_axis='Z'):
    """Extract 2D boundary points from wall faces"""
    print("Extracting wall boundary points...")
    
    # Get all vertices used by wall faces
    wall_vertex_indices = set()
    for face_idx in wall_faces:
        face = mesh.faces[face_idx]
        wall_vertex_indices.update(face)
    
    wall_vertices = mesh.vertices[list(wall_vertex_indices)]
    
    # Project to 2D based on up axis
    if up_axis == 'X':
        # Project to YZ plane
        boundary_2d = wall_vertices[:, [1, 2]]  # Y, Z
        height_axis = 0
        horizontal_labels = ['Y', 'Z']
    elif up_axis == 'Y': 
        # Project to XZ plane  
        boundary_2d = wall_vertices[:, [0, 2]]  # X, Z
        height_axis = 1
        horizontal_labels = ['X', 'Z']
    else:  # up_axis == 'Z'
        # Project to XY plane
        boundary_2d = wall_vertices[:, [0, 1]]  # X, Y
        height_axis = 2
        horizontal_labels = ['X', 'Y']
    
    print(f"  Projecting to {horizontal_labels[0]}-{horizontal_labels[1]} plane ({up_axis} up)")
    print(f"  Wall vertices: {len(wall_vertices):,}")
    print(f"  2D boundary points: {len(boundary_2d):,}")
    
    return boundary_2d, horizontal_labels

def create_room_boundary(points_2d, method='convex_hull'):
    """Create room boundary from 2D points"""
    print(f"Creating room boundary using {method}...")
    
    if len(points_2d) < 3:
        print("  Not enough points for boundary")
        return np.array([]), 0.0
        
    try:
        # Use convex hull for robustness 
        hull = ConvexHull(points_2d)
        boundary_points = points_2d[hull.vertices]
        
        # Calculate area using shoelace formula
        area = 0.0
        n = len(boundary_points)
        for i in range(n):
            j = (i + 1) % n
            area += boundary_points[i][0] * boundary_points[j][1]
            area -= boundary_points[j][0] * boundary_points[i][1]
        area = abs(area) / 2.0
        
        print(f"  Boundary points: {len(boundary_points)}")
        print(f"  Calculated area: {area:.2f} m²")
        
        return boundary_points, area
        
    except Exception as e:
        print(f"  Error creating boundary: {e}")
        return np.array([]), 0.0

def analyze_room_features(boundary_points, horizontal_labels):
    """Analyze room features from boundary"""
    if len(boundary_points) == 0:
        return []
        
    # Calculate wall segments
    segments = []
    n = len(boundary_points)
    
    for i in range(n):
        start = boundary_points[i]
        end = boundary_points[(i + 1) % n]
        
        length = np.linalg.norm(end - start)
        angle = np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi
        
        segments.append({
            'start': start.tolist(),
            'end': end.tolist(), 
            'length': float(length),
            'angle': float(angle),
            'axis_labels': horizontal_labels
        })
    
    print(f"  Wall segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"    Segment {i+1}: {seg['length']:.2f}m at {seg['angle']:.1f}°")
    
    return segments

def create_detailed_visualization(mesh, result, output_path):
    """Create comprehensive visualization"""
    print("Creating detailed visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    
    # 1. 3D mesh overview
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Sample vertices for 3D plot (performance)
    verts = mesh.vertices
    if len(verts) > 5000:
        indices = np.random.choice(len(verts), 5000, replace=False)
        sample_verts = verts[indices]
    else:
        sample_verts = verts
    
    ax_3d.scatter(sample_verts[:, 0], sample_verts[:, 1], sample_verts[:, 2], 
                 alpha=0.3, s=0.5, c='gray')
    ax_3d.set_title(f'3D Mesh\\n{len(verts):,} vertices')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    
    # 2. 2D Floor plan
    ax_floor = fig.add_subplot(gs[0, 1])
    
    boundary = np.array(result['room_boundary'])
    if len(boundary) > 0:
        # Draw boundary
        boundary_poly = Polygon(boundary, fill=False, edgecolor='blue', linewidth=2)
        ax_floor.add_patch(boundary_poly)
        
        # Draw wall segments
        segments = result.get('wall_segments', [])
        for i, seg in enumerate(segments):
            start, end = np.array(seg['start']), np.array(seg['end'])
            ax_floor.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2, alpha=0.7)
            
            # Add length labels
            mid = (start + end) / 2
            ax_floor.text(mid[0], mid[1], f'{seg["length"]:.1f}m', 
                         fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Set aspect and limits
        ax_floor.set_aspect('equal')
        margin = 0.5
        ax_floor.set_xlim(boundary[:, 0].min() - margin, boundary[:, 0].max() + margin)
        ax_floor.set_ylim(boundary[:, 1].min() - margin, boundary[:, 1].max() + margin)
    
    # Get axis labels from result
    geom = result.get('geometry_analysis', {})
    up_axis = geom.get('up_axis', 'Z')
    if up_axis == 'Z':
        ax_floor.set_xlabel('X (m)')
        ax_floor.set_ylabel('Y (m)')
    elif up_axis == 'Y':
        ax_floor.set_xlabel('X (m)')
        ax_floor.set_ylabel('Z (m)')
    else:  # up_axis == 'X'
        ax_floor.set_xlabel('Y (m)')
        ax_floor.set_ylabel('Z (m)')
        
    ax_floor.set_title(f'Floor Plan\\nArea: {result["room_area_sqm"]:.2f} m²')
    ax_floor.grid(True, alpha=0.3)
    
    # 3. Coordinate analysis
    ax_coords = fig.add_subplot(gs[0, 2])
    ax_coords.axis('off')
    
    # Text summary
    info_text = f"""COORDINATE ANALYSIS
    
Mesh File: {Path(result['mesh_file']).name}
Vertices: {result.get('mesh_stats', {}).get('vertices', 'N/A'):,}
Faces: {result.get('mesh_stats', {}).get('faces', 'N/A'):,}

GEOMETRY:
Up Axis: {up_axis}
Coordinate Spans:
"""
    
    spans = geom.get('spans', {})
    for axis, span in spans.items():
        info_text += f"  {axis}: {span:.2f}m\n"
    
    info_text += f"""
ROOM MEASUREMENTS:
Area: {result['room_area_sqm']:.2f} m² ({result['room_area_sqft']:.1f} ft²)
Perimeter segments: {len(result.get('wall_segments', []))}
Boundary points: {len(result['room_boundary'])}

FACE CLASSIFICATION:
Floor faces: {result.get('face_classification', {}).get('floor_faces', 0):,}
Wall faces: {result.get('face_classification', {}).get('wall_faces', 0):,}
Ceiling faces: {result.get('face_classification', {}).get('ceiling_faces', 0):,}

METHOD: {result['method']}
"""
    
    ax_coords.text(0.05, 0.95, info_text, transform=ax_coords.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # 4. Bottom row: Projections comparison
    projections = ['XY', 'XZ', 'YZ']
    proj_axes = [(0, 1), (0, 2), (1, 2)]
    proj_labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
    
    for i, (proj_name, (ax1, ax2), (label1, label2)) in enumerate(zip(projections, proj_axes, proj_labels)):
        ax = fig.add_subplot(gs[1, i])
        
        # Project vertices
        coords_2d = verts[:, [ax1, ax2]]
        
        # Sample for performance
        if len(coords_2d) > 3000:
            indices = np.random.choice(len(coords_2d), 3000, replace=False)
            plot_coords = coords_2d[indices]
        else:
            plot_coords = coords_2d
            
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1], alpha=0.2, s=0.5, c='gray')
        
        # Add convex hull
        try:
            hull = ConvexHull(coords_2d)
            for simplex in hull.simplices:
                ax.plot(coords_2d[simplex, 0], coords_2d[simplex, 1], 'b-', alpha=0.5)
            
            # Calculate hull area
            hull_area = hull.volume  # In 2D, volume is area
            ax.set_title(f'{proj_name} Projection\\nArea: {hull_area:.2f} m²')
            
        except:
            ax.set_title(f'{proj_name} Projection')
            
        ax.set_xlabel(f'{label1} (m)')
        ax.set_ylabel(f'{label2} (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Highlight the one we used
        if ((up_axis == 'Z' and proj_name == 'XY') or 
            (up_axis == 'Y' and proj_name == 'XZ') or 
            (up_axis == 'X' and proj_name == 'YZ')):
            ax.patch.set_facecolor('lightgreen')
            ax.patch.set_alpha(0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved: {output_path}")

def analyze_mesh_corrected(mesh_path):
    """Main analysis with corrected coordinate system"""
    print(f"\\n=== v11 Normal Segmentation (CORRECTED AXES): {mesh_path} ===")
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, 'vertices'):
        print("Error: Could not load mesh")
        return None
    
    print(f"Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Step 1: Analyze geometry to determine coordinate system
    geometry = analyze_mesh_geometry(mesh)
    
    # Step 2: Classify faces
    classification = classify_faces_by_normal(mesh, geometry['up_axis'])
    
    # Step 3: Extract wall boundary in 2D
    boundary_2d, horizontal_labels = extract_wall_boundary_points(
        mesh, classification['wall_faces'], geometry['up_axis'])
    
    if len(boundary_2d) == 0:
        print("Error: No wall boundary points found")
        return None
    
    # Step 4: Create room boundary
    boundary_points, area = create_room_boundary(boundary_2d)
    
    # Step 5: Analyze room features
    wall_segments = analyze_room_features(boundary_points, horizontal_labels)
    
    # Compile results
    result = {
        "method": "normal_wall_segmentation_v11_corrected",
        "mesh_file": str(mesh_path),
        "mesh_stats": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces)
        },
        "geometry_analysis": geometry,
        "face_classification": {
            "floor_faces": len(classification['floor_faces']),
            "ceiling_faces": len(classification['ceiling_faces']),
            "wall_faces": len(classification['wall_faces'])
        },
        "room_boundary": boundary_points.tolist() if len(boundary_points) > 0 else [],
        "wall_segments": wall_segments,
        "room_area_sqm": float(area),
        "room_area_sqft": float(area * 10.764),
        "coordinate_system": {
            "up_axis": geometry['up_axis'],
            "horizontal_axes": horizontal_labels,
            "projection_plane": f"{horizontal_labels[0]}-{horizontal_labels[1]}"
        }
    }
    
    print(f"\\n✅ CORRECTED RESULT:")
    print(f"   Room area: {area:.2f} m² ({area * 10.764:.1f} ft²)")
    print(f"   Coordinate system: {geometry['up_axis']} up, {horizontal_labels[0]}-{horizontal_labels[1]} projection")
    print(f"   Wall segments: {len(wall_segments)}")
    
    return result

if __name__ == "__main__":
    # Test both meshes
    test_meshes = [
        "data/2026_01_13_14_47_59/export_refined.obj",
        "data/gdrive_sample/2026_02_09_19_03_38/export_refined.obj"
    ]
    
    output_dir = Path("results/v11_tests")
    output_dir.mkdir(exist_ok=True)
    
    for i, mesh_path in enumerate(test_meshes, 1):
        if not Path(mesh_path).exists():
            print(f"Skipping {mesh_path} - file not found")
            continue
            
        result = analyze_mesh_corrected(mesh_path)
        if result:
            # Save results
            result_file = output_dir / f"test{i}_corrected_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=NpEncoder)
            
            # Create visualization  
            viz_file = output_dir / f"test{i}_corrected_visualization.png"
            mesh = trimesh.load(mesh_path)
            create_detailed_visualization(mesh, result, viz_file)
            
            print(f"   Results saved: {result_file}")
            print(f"   Visualization: {viz_file}")
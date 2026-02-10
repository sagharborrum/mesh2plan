#!/usr/bin/env python3
"""
Axis Diagnostic Script - Find correct coordinate system for mesh2plan
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_all_projections(mesh_path):
    """Analyze all possible 2D projections to find the correct one"""
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    verts = mesh.vertices
    
    print(f"Vertices: {len(verts):,}")
    print(f"X range: {verts[:,0].min():.3f} to {verts[:,0].max():.3f} = {verts[:,0].max() - verts[:,0].min():.3f}m")
    print(f"Y range: {verts[:,1].min():.3f} to {verts[:,1].max():.3f} = {verts[:,1].max() - verts[:,1].min():.3f}m")
    print(f"Z range: {verts[:,2].min():.3f} to {verts[:,2].max():.3f} = {verts[:,2].max() - verts[:,2].min():.3f}m")
    
    # Check face normals to determine which axis is "up"
    normals = mesh.face_normals
    up_candidates = [np.abs(normals[:,i]).mean() for i in range(3)]
    print(f"Mean |normal| per axis: X={up_candidates[0]:.3f} Y={up_candidates[1]:.3f} Z={up_candidates[2]:.3f}")
    print(f"Most vertical faces point along: {['X', 'Y', 'Z'][np.argmax(up_candidates)]}-axis")
    
    # Calculate all possible 2D projections
    projections = {
        'XY': (0, 1, 2),  # Project to XY, Z is up
        'XZ': (0, 2, 1),  # Project to XZ, Y is up
        'YZ': (1, 2, 0),  # Project to YZ, X is up
    }
    
    results = {}
    
    for name, (axis1, axis2, up_axis) in projections.items():
        coords_2d = verts[:, [axis1, axis2]]
        
        # Calculate bounding box
        width = coords_2d[:, 0].max() - coords_2d[:, 0].min()
        height = coords_2d[:, 1].max() - coords_2d[:, 1].min()
        area_bbox = width * height
        
        # Calculate convex hull area
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords_2d)
            hull_area = hull.volume  # In 2D, volume is actually area
        except:
            hull_area = area_bbox
        
        results[name] = {
            'projection': name,
            'horizontal_axes': [['X', 'Y', 'Z'][axis1], ['X', 'Y', 'Z'][axis2]],
            'up_axis': ['X', 'Y', 'Z'][up_axis],
            'dimensions': (width, height),
            'bbox_area': area_bbox,
            'hull_area': hull_area,
            'aspect_ratio': width / height if height > 0 else float('inf')
        }
        
        print(f"\n{name} projection ({['X','Y','Z'][axis1]}-{['X','Y','Z'][axis2]}, {['X','Y','Z'][up_axis]} up):")
        print(f"  Dimensions: {width:.2f} × {height:.2f} m")
        print(f"  BBox Area: {area_bbox:.2f} m²")
        print(f"  Hull Area: {hull_area:.2f} m²")
        print(f"  Aspect ratio: {width/height:.2f}" if height > 0 else "  Aspect ratio: ∞")
    
    # Analyze which projection makes most sense
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"Expected room: ~5.5m × 2.7m = ~14.9 m² (aspect ratio: ~2.0)")
    print(f"Reference measurement: 11.5 m²")
    
    best_match = None
    best_score = float('inf')
    target_area = 11.5
    target_aspect = 5.5 / 2.7
    
    for name, result in results.items():
        area_error = abs(result['hull_area'] - target_area) / target_area
        aspect_error = abs(result['aspect_ratio'] - target_aspect) / target_aspect if result['aspect_ratio'] != float('inf') else 1.0
        
        # Combined score (area error + aspect error)
        score = area_error + aspect_error
        result['score'] = score
        
        print(f"\n{name}: Area={result['hull_area']:.1f}m² (error: {area_error:.1%}), "
              f"Aspect={result['aspect_ratio']:.2f} (error: {aspect_error:.1%}), "
              f"Score={score:.3f}")
        
        if score < best_score:
            best_score = score
            best_match = name
    
    print(f"\n🎯 BEST MATCH: {best_match}")
    print(f"   {results[best_match]}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, (axis1, axis2, up_axis)) in enumerate(projections.items()):
        coords_2d = verts[:, [axis1, axis2]]
        
        # Sample points for plotting (too many points slow down the plot)
        if len(coords_2d) > 10000:
            indices = np.random.choice(len(coords_2d), 10000, replace=False)
            plot_coords = coords_2d[indices]
        else:
            plot_coords = coords_2d
        
        axes[i].scatter(plot_coords[:, 0], plot_coords[:, 1], alpha=0.1, s=0.5)
        axes[i].set_title(f'{name} Projection\\n{results[name]["hull_area"]:.1f} m²')
        axes[i].set_xlabel(f'{["X", "Y", "Z"][axis1]} (m)')
        axes[i].set_ylabel(f'{["X", "Y", "Z"][axis2]} (m)')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        
        # Add convex hull
        try:
            hull = ConvexHull(coords_2d)
            for simplex in hull.simplices:
                axes[i].plot(coords_2d[simplex, 0], coords_2d[simplex, 1], 'r-', alpha=0.8)
        except:
            pass
    
    plt.tight_layout()
    plt.savefig('~/projects/mesh2plan/axis_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\\nVisualization saved to: axis_analysis.png")
    
    return results, best_match

if __name__ == "__main__":
    # Test on the problem mesh
    mesh_path = "data/2026_01_13_14_47_59/export_refined.obj"
    results, best_match = analyze_all_projections(mesh_path)
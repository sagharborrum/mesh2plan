#!/usr/bin/env python3
"""
Quick script to debug mesh coordinate systems and bounds.
"""

import trimesh
import numpy as np
from pathlib import Path

def analyze_mesh_bounds(mesh_file):
    print(f"\n=== {mesh_file} ===")
    try:
        mesh = trimesh.load(mesh_file)
        
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print("Invalid mesh file")
            return
        
        vertices = mesh.vertices
        print(f"Vertices: {len(vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        
        # Bounds for each axis
        for i, axis in enumerate(['X', 'Y', 'Z']):
            coords = vertices[:, i]
            min_val = np.min(coords)
            max_val = np.max(coords)
            range_val = max_val - min_val
            print(f"{axis}: {min_val:.3f} to {max_val:.3f} (range: {range_val:.3f}m)")
        
        # Guess the up axis (tallest dimension, usually)
        ranges = []
        for i in range(3):
            coords = vertices[:, i]
            range_val = np.max(coords) - np.min(coords)
            ranges.append(range_val)
        
        max_idx = np.argmax(ranges)
        axis_names = ['X', 'Y', 'Z']
        print(f"Probable up-axis: {axis_names[max_idx]} (largest range: {ranges[max_idx]:.3f}m)")
        
        # If Y has the largest range, it's probably Y-up (not Z-up)
        if max_idx == 1:
            print("WARNING: This looks like Y-up, not Z-up!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Test both datasets and both mesh types
    datasets = [
        "data/gdrive_sample/2026_02_09_19_03_38/export.obj",
        "data/gdrive_sample/2026_02_09_19_03_38/export_refined.obj",
        "data/2026_01_13_14_47_59/export.obj", 
        "data/2026_01_13_14_47_59/export_refined.obj"
    ]
    
    for dataset in datasets:
        analyze_mesh_bounds(dataset)
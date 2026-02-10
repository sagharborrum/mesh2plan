#!/usr/bin/env python3
"""
Visualize v10 voxelization results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_v10_results(result_file, output_image):
    """Plot the v10 analysis results"""
    
    # Load results
    with open(result_file) as f:
        results = json.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot boundary
    boundary = np.array(results['room_boundary'])
    if len(boundary) > 0:
        # Close the boundary polygon
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'b-', linewidth=2, label='Room Boundary')
        ax.fill(boundary_closed[:, 0], boundary_closed[:, 1], alpha=0.3, color='lightblue')
    
    # Plot openings
    for i, opening in enumerate(results['openings']):
        pos = opening['position']
        ax.plot(pos[0], pos[1], 'ro', markersize=8, label=f'{opening["type"].title()} ({opening["size_meters"]:.2f}m)')
    
    # Labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title(f'V10 Voxelization Results\n{Path(results["mesh_file"]).name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add stats text
    stats_text = f"""Area: {results['room_area_sqm']:.2f} m² ({results['room_area_sqft']:.1f} ft²)
Boundary Points: {len(results['room_boundary'])}
Openings: {len(results['openings'])}
Voxel Size: {results['parameters']['voxel_size']}m
Grid: {results['voxel_stats']['grid_size']}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_image}")
    
    return fig

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python visualize_v10_results.py <results.json> <output.png>")
        sys.exit(1)
    
    plot_v10_results(sys.argv[1], sys.argv[2])
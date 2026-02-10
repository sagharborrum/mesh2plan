#!/usr/bin/env python3
"""
Create Final Architectural Floor Plan
=====================================

Generate a professional floor plan visualization showing the corrected results
with proper measurements, labels, and architectural styling.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch
import numpy as np
import json
from pathlib import Path

def create_architectural_floorplan():
    """Create professional architectural floor plan"""
    
    # Load the best result (v14 corrected, large mesh)
    result_file = "results/v14_tests/test2_corrected_results.json" 
    if not Path(result_file).exists():
        print(f"Results file not found: {result_file}")
        return
        
    with open(result_file) as f:
        result = json.load(f)
    
    # Create figure with professional styling
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Get room boundary and segments
    boundary = np.array(result['room_boundary'])
    segments = result.get('wall_segments', [])
    area = result['room_area_sqm']
    
    # Drawing settings
    wall_color = '#2C3E50'      # Dark blue-gray
    wall_width = 3
    text_color = '#34495E'      # Medium gray
    grid_color = '#ECF0F1'      # Light gray
    
    # Draw room boundary (walls)
    room_poly = Polygon(boundary, fill=False, edgecolor=wall_color, 
                       linewidth=wall_width, alpha=0.9)
    ax.add_patch(room_poly)
    
    # Fill room interior with light color
    room_fill = Polygon(boundary, fill=True, facecolor='#F8F9FA', 
                       edgecolor='none', alpha=0.3, zorder=0)
    ax.add_patch(room_fill)
    
    # Add wall measurements
    for i, seg in enumerate(segments):
        start = np.array(seg['start'])
        end = np.array(seg['end'])
        length = seg['length']
        
        # Skip very small segments (noise)
        if length < 0.5:
            continue
            
        # Calculate midpoint and label position
        midpoint = (start + end) / 2
        
        # Calculate perpendicular offset for label
        wall_vector = end - start
        wall_length = np.linalg.norm(wall_vector)
        if wall_length > 0:
            wall_unit = wall_vector / wall_length
            perp_vector = np.array([-wall_unit[1], wall_unit[0]]) * 0.15  # Offset distance
            
            label_pos = midpoint + perp_vector
            
            # Add measurement text
            ax.text(label_pos[0], label_pos[1], f'{length:.1f}m', 
                   fontsize=10, ha='center', va='center', 
                   color=text_color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                           edgecolor='none', alpha=0.9))
            
            # Add measurement line (dimension line)
            line_start = start + perp_vector * 0.7
            line_end = end + perp_vector * 0.7
            ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 
                   'k-', alpha=0.5, linewidth=1)
            
            # Add tick marks
            tick_size = 0.05
            tick_perp = np.array([-perp_vector[1], perp_vector[0]]) * tick_size
            ax.plot([line_start[0]-tick_perp[0], line_start[0]+tick_perp[0]], 
                   [line_start[1]-tick_perp[1], line_start[1]+tick_perp[1]], 
                   'k-', linewidth=1)
            ax.plot([line_end[0]-tick_perp[0], line_end[0]+tick_perp[0]], 
                   [line_end[1]-tick_perp[1], line_end[1]+tick_perp[1]], 
                   'k-', linewidth=1)
    
    # Calculate overall room dimensions
    x_span = boundary[:, 0].max() - boundary[:, 0].min()
    y_span = boundary[:, 1].max() - boundary[:, 1].min()
    
    # Add overall dimension lines
    margin = 0.8
    x_min, x_max = boundary[:, 0].min(), boundary[:, 0].max()
    y_min, y_max = boundary[:, 1].min(), boundary[:, 1].max()
    
    # Bottom dimension line (width)
    dim_y = y_min - margin
    ax.plot([x_min, x_max], [dim_y, dim_y], 'k-', linewidth=2)
    ax.plot([x_min, x_min], [dim_y-0.1, dim_y+0.1], 'k-', linewidth=2)
    ax.plot([x_max, x_max], [dim_y-0.1, dim_y+0.1], 'k-', linewidth=2)
    ax.text((x_min + x_max)/2, dim_y - 0.3, f'{x_span:.2f}m', 
           fontsize=12, ha='center', va='center', fontweight='bold', color=text_color)
    
    # Left dimension line (height)  
    dim_x = x_min - margin
    ax.plot([dim_x, dim_x], [y_min, y_max], 'k-', linewidth=2)
    ax.plot([dim_x-0.1, dim_x+0.1], [y_min, y_min], 'k-', linewidth=2)
    ax.plot([dim_x-0.1, dim_x+0.1], [y_max, y_max], 'k-', linewidth=2)
    ax.text(dim_x - 0.3, (y_min + y_max)/2, f'{y_span:.2f}m', 
           fontsize=12, ha='center', va='center', fontweight='bold', color=text_color,
           rotation=90)
    
    # Add title and info box
    title_text = "FLOOR PLAN"
    ax.text(0.5, 0.95, title_text, transform=ax.transAxes, fontsize=18, 
           fontweight='bold', ha='center', va='top', color=text_color)
    
    # Info box
    info_text = f"""ROOM MEASUREMENTS
Area: {area:.1f} m² ({area * 10.764:.1f} ft²)
Dimensions: {x_span:.2f} × {y_span:.2f} m
Perimeter: {len(segments)} segments

METHOD: v14 Hybrid (Corrected)
COORDINATE SYSTEM: XY Projection (Z-up)
REFERENCE: 11.5 m² (Error: {abs(area-11.5)/11.5*100:.1f}%)"""
    
    # Create info box
    info_box = FancyBboxPatch((0.02, 0.02), 0.35, 0.25, 
                             boxstyle="round,pad=0.02", 
                             facecolor='white', edgecolor=text_color, 
                             linewidth=1, transform=ax.transAxes)
    ax.add_patch(info_box)
    
    ax.text(0.04, 0.24, info_text, transform=ax.transAxes, fontsize=10,
           va='top', ha='left', color=text_color, fontfamily='monospace')
    
    # Add north arrow
    north_x, north_y = 0.9, 0.85
    arrow_props = dict(arrowstyle='->', lw=2, color=text_color)
    ax.annotate('', xy=(0.9, 0.9), xytext=(0.9, 0.85), 
               xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=arrow_props)
    ax.text(0.92, 0.875, 'N', transform=ax.transAxes, fontsize=12, 
           fontweight='bold', ha='center', va='center', color=text_color)
    
    # Set equal aspect ratio and clean up axes
    ax.set_aspect('equal')
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, color=grid_color, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis labels
    ax.set_xlabel('X (meters)', fontsize=12, color=text_color)
    ax.set_ylabel('Y (meters)', fontsize=12, color=text_color)
    
    # Adjust margins
    ax.margins(0.2)
    
    # Style axes
    ax.tick_params(colors=text_color, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1)
    
    # Save high-quality output
    output_path = "results/final_architectural_floorplan.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Architectural floor plan saved: {output_path}")
    
    # Also save as PDF for vector graphics
    pdf_path = "results/final_architectural_floorplan.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Vector PDF saved: {pdf_path}")
    
    plt.close()

def create_comparison_chart():
    """Create before/after comparison chart"""
    
    approaches = ['v10', 'v11', 'v12', 'v14']
    before_areas = [1.8, 0.4, 8.8, 16.5]  # Approximate from NOTES.md
    after_areas = [7.77, 14.13, 7.87, 13.65]  # From corrected results
    target = 11.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(approaches))
    width = 0.35
    
    # Bar chart
    bars1 = ax1.bar(x - width/2, before_areas, width, label='Before Correction', 
                   color='#E74C3C', alpha=0.7)
    bars2 = ax1.bar(x + width/2, after_areas, width, label='After Correction', 
                   color='#27AE60', alpha=0.7)
    
    # Add target line
    ax1.axhline(y=target, color='#3498DB', linestyle='--', linewidth=2, 
               label=f'Target: {target} m²')
    
    ax1.set_xlabel('Approach')
    ax1.set_ylabel('Room Area (m²)')
    ax1.set_title('Mesh2Plan Area Correction Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Error percentage chart
    before_errors = [abs(a - target) / target * 100 for a in before_areas]
    after_errors = [abs(a - target) / target * 100 for a in after_areas]
    
    bars3 = ax2.bar(x - width/2, before_errors, width, label='Before Correction', 
                   color='#E74C3C', alpha=0.7)
    bars4 = ax2.bar(x + width/2, after_errors, width, label='After Correction', 
                   color='#27AE60', alpha=0.7)
    
    ax2.set_xlabel('Approach')
    ax2.set_ylabel('Error from Target (%)')
    ax2.set_title('Error Reduction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(approaches)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    def add_error_labels(bars, errors):
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{error:.0f}%', ha='center', va='bottom', fontsize=9)
    
    add_error_labels(bars3, before_errors)
    add_error_labels(bars4, after_errors)
    
    plt.tight_layout()
    plt.savefig("results/correction_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Comparison chart saved: results/correction_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating final architectural visualizations...")
    create_architectural_floorplan()
    create_comparison_chart()
    print("✅ All visualizations complete!")
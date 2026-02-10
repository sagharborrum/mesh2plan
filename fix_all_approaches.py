#!/usr/bin/env python3
"""
Apply Axis Correction to All Approaches
======================================

Apply the Z-up, XY-projection fix to v10, v12, v14 approaches based on the successful v11 correction.
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import warnings
import shutil
import subprocess
import sys

warnings.filterwarnings("ignore")

def determine_coordinate_system(mesh):
    """Determine coordinate system like v11_corrected"""
    verts = mesh.vertices
    spans = {
        'X': verts[:,0].max() - verts[:,0].min(),
        'Y': verts[:,1].max() - verts[:,1].min(), 
        'Z': verts[:,2].max() - verts[:,2].min()
    }
    
    normals = mesh.face_normals
    up_candidates = {
        'X': np.abs(normals[:,0]).mean(),
        'Y': np.abs(normals[:,1]).mean(),
        'Z': np.abs(normals[:,2]).mean()
    }
    
    up_axis = max(up_candidates, key=up_candidates.get)
    
    return {
        'up_axis': up_axis,
        'spans': spans,
        'projection': 'XY' if up_axis == 'Z' else ('XZ' if up_axis == 'Y' else 'YZ')
    }

def patch_approach_file(file_path, patches):
    """Apply patches to an approach file"""
    print(f"Patching {file_path}...")
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply patches
    for old_pattern, new_pattern in patches:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"  ✅ Applied patch: {old_pattern[:50]}...")
        else:
            print(f"  ⚠️  Pattern not found: {old_pattern[:50]}...")
    
    # Write patched version
    output_path = file_path.parent / f"{file_path.stem}_corrected.py"
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def patch_v10():
    """Patch v10 voxelization approach"""
    file_path = Path("research/v10_voxelization_projection.py")
    
    patches = [
        # Change projection method detection
        (
            "# Project voxel centers to 2D (XZ plane - top down view)",
            "# Determine correct coordinate system\n    coord_system = determine_coordinate_system(mesh)\n    up_axis = coord_system['up_axis']\n    \n    # Project voxel centers to 2D based on detected coordinate system"
        ),
        (
            "voxel_2d = occupied_centers[:, [0, 2]]  # X, Z coordinates",
            "# Project based on up axis\n    if up_axis == 'Z':  # XY projection\n        voxel_2d = occupied_centers[:, [0, 1]]\n        axis_labels = ['X', 'Y']\n    elif up_axis == 'Y':  # XZ projection\n        voxel_2d = occupied_centers[:, [0, 2]] \n        axis_labels = ['X', 'Z']\n    else:  # YZ projection\n        voxel_2d = occupied_centers[:, [1, 2]]\n        axis_labels = ['Y', 'Z']"
        ),
        (
            'projection_method": "xz"',
            'projection_method": coord_system["projection"]'
        )
    ]
    
    # Add coordinate system function at top
    header_addition = '''
def determine_coordinate_system(mesh):
    """Determine coordinate system from mesh geometry"""
    verts = mesh.vertices
    normals = mesh.face_normals
    up_candidates = {
        'X': np.abs(normals[:,0]).mean(),
        'Y': np.abs(normals[:,1]).mean(),
        'Z': np.abs(normals[:,2]).mean()
    }
    up_axis = max(up_candidates, key=up_candidates.get)
    projection = 'XY' if up_axis == 'Z' else ('XZ' if up_axis == 'Y' else 'YZ')
    return {'up_axis': up_axis, 'projection': projection}
'''
    
    # Read and modify file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add function after imports
    import_end = content.find('warnings.filterwarnings("ignore")')
    if import_end != -1:
        insert_pos = content.find('\n', import_end) + 1
        content = content[:insert_pos] + header_addition + content[insert_pos:]
    
    # Apply patches
    for old, new in patches:
        content = content.replace(old, new)
    
    output_path = file_path.parent / f"{file_path.stem}_corrected.py"
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path

def patch_v12():
    """Patch v12 contour detection approach"""
    file_path = Path("research/v12_contour_depth_raster.py")
    
    # This is more complex as it creates depth maps
    patches = [
        (
            "# Project vertices to top-down view (XZ plane)",
            "# Determine coordinate system\n    coord_system = determine_coordinate_system(mesh)\n    up_axis = coord_system['up_axis']\n    \n    # Project vertices based on coordinate system"
        ),
        (
            "xy_coords = vertices[:, [0, 2]]  # X, Z coordinates\n    heights = vertices[:, 1]  # Y heights",
            "# Project based on up axis\n    if up_axis == 'Z':  # XY projection\n        xy_coords = vertices[:, [0, 1]]\n        heights = vertices[:, 2]\n        axis_labels = ['X', 'Y']\n    elif up_axis == 'Y':  # XZ projection\n        xy_coords = vertices[:, [0, 2]]\n        heights = vertices[:, 1]\n        axis_labels = ['X', 'Z']\n    else:  # YZ projection\n        xy_coords = vertices[:, [1, 2]]\n        heights = vertices[:, 0]\n        axis_labels = ['Y', 'Z']"
        )
    ]
    
    return patch_approach_file(file_path, patches)

def patch_v14():
    """Patch v14 hybrid approach"""
    file_path = Path("research/v14_hybrid_walls_openings.py")
    
    patches = [
        (
            "# Project wall vertices to 2D (XZ plane - Y is up)",
            "# Determine coordinate system\n    coord_system = determine_coordinate_system(mesh)\n    up_axis = coord_system['up_axis']\n    \n    # Project wall vertices to 2D based on detected coordinate system"
        ),
        (
            "wall_vertices_2d = wall_vertices[:, [0, 2]]  # X, Z coordinates",
            "# Project based on up axis\n    if up_axis == 'Z':  # XY projection\n        wall_vertices_2d = wall_vertices[:, [0, 1]]\n        axis_labels = ['X', 'Y']\n    elif up_axis == 'Y':  # XZ projection\n        wall_vertices_2d = wall_vertices[:, [0, 2]]\n        axis_labels = ['X', 'Z']\n    else:  # YZ projection\n        wall_vertices_2d = wall_vertices[:, [1, 2]]\n        axis_labels = ['Y', 'Z']"
        ),
        (
            "# Project all vertices to 2D for depth map (XZ plane)",
            "# Project all vertices to 2D for depth map based on coordinate system"
        ),
        (
            "xy_coords = all_vertices[:, [0, 2]]  # X, Z coordinates\n    heights = all_vertices[:, 1]  # Y heights",
            "# Project for depth map based on up axis\n    if up_axis == 'Z':  # XY projection\n        xy_coords = all_vertices[:, [0, 1]]\n        heights = all_vertices[:, 2]\n    elif up_axis == 'Y':  # XZ projection\n        xy_coords = all_vertices[:, [0, 2]]\n        heights = all_vertices[:, 1]\n    else:  # YZ projection\n        xy_coords = all_vertices[:, [1, 2]]\n        heights = all_vertices[:, 0]"
        )
    ]
    
    return patch_approach_file(file_path, patches)

def main():
    """Apply corrections to all approaches"""
    print("🔧 Applying axis corrections to all approaches...")
    
    # Add coordinate system function to each file
    coord_function = '''
def determine_coordinate_system(mesh):
    """Determine coordinate system from mesh geometry"""
    verts = mesh.vertices
    normals = mesh.face_normals
    up_candidates = {
        'X': np.abs(normals[:,0]).mean(),
        'Y': np.abs(normals[:,1]).mean(),
        'Z': np.abs(normals[:,2]).mean()
    }
    up_axis = max(up_candidates, key=up_candidates.get)
    return {'up_axis': up_axis}
'''
    
    # Patch each approach
    corrected_files = []
    
    # V10 Voxelization
    try:
        v10_corrected = patch_v10()
        corrected_files.append(('v10', v10_corrected))
        print("✅ v10 patched successfully")
    except Exception as e:
        print(f"❌ Error patching v10: {e}")
    
    # V12 Contour Detection  
    try:
        with open("research/v12_contour_depth_raster.py", 'r') as f:
            content = f.read()
        
        # Add coordinate system function
        import_end = content.find('warnings.filterwarnings("ignore")')
        insert_pos = content.find('\n', import_end) + 1
        content = content[:insert_pos] + coord_function + content[insert_pos:]
        
        # Apply patches
        v12_patches = [
            (
                "xy_coords = vertices[:, [0, 2]]  # X, Z coordinates\n    heights = vertices[:, 1]  # Y heights",
                "# Determine coordinate system\n    coord_system = determine_coordinate_system(mesh)\n    up_axis = coord_system['up_axis']\n    \n    # Project based on up axis\n    if up_axis == 'Z':  # XY projection\n        xy_coords = vertices[:, [0, 1]]\n        heights = vertices[:, 2]\n    elif up_axis == 'Y':  # XZ projection\n        xy_coords = vertices[:, [0, 2]]\n        heights = vertices[:, 1]\n    else:  # YZ projection\n        xy_coords = vertices[:, [1, 2]]\n        heights = vertices[:, 0]"
            )
        ]
        
        for old, new in v12_patches:
            content = content.replace(old, new)
        
        v12_corrected = Path("research/v12_contour_depth_raster_corrected.py")
        with open(v12_corrected, 'w') as f:
            f.write(content)
        
        corrected_files.append(('v12', v12_corrected))
        print("✅ v12 patched successfully")
        
    except Exception as e:
        print(f"❌ Error patching v12: {e}")
    
    # V14 Hybrid
    try:
        with open("research/v14_hybrid_walls_openings.py", 'r') as f:
            content = f.read()
        
        # Add coordinate system function
        import_end = content.find('warnings.filterwarnings("ignore")')
        insert_pos = content.find('\n', import_end) + 1
        content = content[:insert_pos] + coord_function + content[insert_pos:]
        
        # Apply patches for wall analysis
        v14_patches = [
            (
                "wall_vertices_2d = wall_vertices[:, [0, 2]]  # X, Z coordinates",
                "# Determine coordinate system\n    coord_system = determine_coordinate_system(mesh)\n    up_axis = coord_system['up_axis']\n    \n    # Project wall vertices based on coordinate system\n    if up_axis == 'Z':  # XY projection\n        wall_vertices_2d = wall_vertices[:, [0, 1]]\n    elif up_axis == 'Y':  # XZ projection\n        wall_vertices_2d = wall_vertices[:, [0, 2]]\n    else:  # YZ projection\n        wall_vertices_2d = wall_vertices[:, [1, 2]]"
            ),
            (
                "xy_coords = all_vertices[:, [0, 2]]  # X, Z coordinates\n    heights = all_vertices[:, 1]  # Y heights",
                "# Project for depth map based on up axis\n    if up_axis == 'Z':  # XY projection\n        xy_coords = all_vertices[:, [0, 1]]\n        heights = all_vertices[:, 2]\n    elif up_axis == 'Y':  # XZ projection\n        xy_coords = all_vertices[:, [0, 2]]\n        heights = all_vertices[:, 1]\n    else:  # YZ projection\n        xy_coords = all_vertices[:, [1, 2]]\n        heights = all_vertices[:, 0]"
            )
        ]
        
        for old, new in v14_patches:
            content = content.replace(old, new)
        
        v14_corrected = Path("research/v14_hybrid_walls_openings_corrected.py")
        with open(v14_corrected, 'w') as f:
            f.write(content)
        
        corrected_files.append(('v14', v14_corrected))
        print("✅ v14 patched successfully")
        
    except Exception as e:
        print(f"❌ Error patching v14: {e}")
    
    # Test all corrected approaches
    test_meshes = [
        "data/2026_01_13_14_47_59/export_refined.obj",
        "data/gdrive_sample/2026_02_09_19_03_38/export_refined.obj"
    ]
    
    results_summary = {}
    
    for approach_name, script_path in corrected_files:
        print(f"\\n🧪 Testing {approach_name}_corrected...")
        
        # Create results directory
        results_dir = Path(f"results/{approach_name}_tests")
        results_dir.mkdir(exist_ok=True)
        
        approach_results = {}
        
        for i, mesh_path in enumerate(test_meshes, 1):
            if not Path(mesh_path).exists():
                print(f"   ⚠️  Skipping {mesh_path} - not found")
                continue
                
            try:
                # Run the corrected script
                result = subprocess.run([
                    sys.executable, str(script_path), mesh_path, 
                    str(results_dir / f"test{i}_corrected_results.json"),
                    str(results_dir / f"test{i}_corrected_visualization.png")
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Load results
                    result_file = results_dir / f"test{i}_corrected_results.json"
                    if result_file.exists():
                        with open(result_file) as f:
                            data = json.load(f)
                            area = data.get('room_area_sqm', 0)
                            approach_results[f'test{i}'] = {
                                'area_sqm': area,
                                'area_sqft': area * 10.764,
                                'mesh': Path(mesh_path).name
                            }
                            print(f"   ✅ Test {i}: {area:.2f} m²")
                    else:
                        print(f"   ❌ Test {i}: No results file")
                else:
                    print(f"   ❌ Test {i}: Script failed - {result.stderr[:200]}")
                    
            except Exception as e:
                print(f"   ❌ Test {i}: Error - {e}")
        
        results_summary[approach_name] = approach_results
    
    # Print summary
    print(f"\\n{'='*60}")
    print("🎯 CORRECTED RESULTS SUMMARY")
    print(f"{'='*60}")
    
    target_area = 11.5
    print(f"Target area: {target_area:.1f} m² (reference measurement)")
    print()
    
    for approach, tests in results_summary.items():
        print(f"{approach.upper()}:")
        for test, result in tests.items():
            area = result['area_sqm']
            error = abs(area - target_area) / target_area * 100 if area > 0 else 100
            print(f"  {test} ({result['mesh']}): {area:.2f} m² (error: {error:.1f}%)")
        print()
    
    # Save summary
    with open("results/axis_correction_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"📊 Summary saved: results/axis_correction_summary.json")

if __name__ == "__main__":
    main()
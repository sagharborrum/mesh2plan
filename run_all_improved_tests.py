#!/usr/bin/env python3
"""
Run all improved approaches on both test meshes
Generates comprehensive results for comparison
"""

import subprocess
import json
from pathlib import Path
import sys

def run_approach(script_path, mesh_path, output_dir, test_name):
    """Run a single approach on a mesh"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json = output_dir / f"{test_name}_results.json"
    visualization_png = output_dir / f"{test_name}_visualization.png"
    
    try:
        print(f"\n🔬 Running {script_path.name} on {mesh_path.name}...")
        
        cmd = [
            sys.executable, str(script_path), 
            str(mesh_path), str(results_json), str(visualization_png)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {script_path.stem} completed successfully")
            
            # Load and return key stats
            if results_json.exists():
                with open(results_json) as f:
                    data = json.load(f)
                return {
                    'success': True,
                    'area': data.get('room_area_sqm', 0),
                    'wall_segments': len(data.get('wall_segments', [])),
                    'openings': len(data.get('openings', [])),
                    'method': data.get('method', 'unknown')
                }
        else:
            print(f"❌ {script_path.stem} failed:")
            print(f"   stdout: {result.stdout[-200:]}")
            print(f"   stderr: {result.stderr[-200:]}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_path.stem} timed out after 5 minutes")
    except Exception as e:
        print(f"💥 {script_path.stem} crashed: {e}")
    
    return {'success': False, 'error': 'Failed to run'}

def main():
    # Test meshes
    small_mesh = Path("data/2026_01_13_14_47_59/export_refined.obj")
    large_mesh = Path("data/gdrive_sample/2026_02_09_19_03_38/export_refined.obj")
    
    # Approaches to test (corrected/improved versions)
    approaches = [
        ("research/v10_voxelization_projection_corrected.py", "results/v10_tests"),
        ("research/v11_normal_wall_segmentation_improved.py", "results/v11_tests"),
        ("research/v12_contour_depth_raster_corrected.py", "results/v12_tests"),
        ("research/v14_hybrid_walls_openings_corrected.py", "results/v14_tests"),
    ]
    
    results_summary = []
    
    for script_name, output_base in approaches:
        script_path = Path(script_name)
        
        if not script_path.exists():
            print(f"⚠️  Skipping {script_name} - file not found")
            continue
            
        # Test on small mesh
        small_result = run_approach(script_path, small_mesh, output_base, "test1_improved")
        small_result['mesh'] = 'small'
        small_result['approach'] = script_path.stem
        results_summary.append(small_result)
        
        # Test on large mesh  
        large_result = run_approach(script_path, large_mesh, output_base, "test2_improved")
        large_result['mesh'] = 'large'
        large_result['approach'] = script_path.stem
        results_summary.append(large_result)
    
    # Generate summary
    print(f"\n📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"Target: ~11.5 m², ~7 wall segments, 2-4 openings")
    print(f"=" * 60)
    
    for result in results_summary:
        if result['success']:
            mesh_type = result['mesh']
            approach = result['approach'].replace('_corrected', '').replace('_improved', '')
            area = result['area']
            walls = result['wall_segments']
            openings = result['openings']
            
            # Calculate accuracy vs target
            target_area = 11.5
            area_error = abs(area - target_area) / target_area * 100
            wall_error = abs(walls - 7) / 7 * 100
            opening_error = abs(openings - 3) / 3 * 100 if openings > 0 else 100
            
            print(f"{approach:20} ({mesh_type:5}): {area:5.1f}m² ({area_error:4.1f}% err), "
                  f"{walls:2d} walls ({wall_error:4.1f}% err), {openings:2d} openings")
        else:
            print(f"{result['approach']:20} ({result['mesh']:5}): FAILED")
    
    # Save summary to JSON
    summary_path = Path("results/comprehensive_test_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Full results saved to: {summary_path}")
    
    # Find best approach
    successful = [r for r in results_summary if r['success'] and r['mesh'] == 'large']
    if successful:
        def score_approach(result):
            """Score approach based on how close it is to target metrics"""
            area_score = 100 - min(100, abs(result['area'] - 11.5) / 11.5 * 100)
            wall_score = 100 - min(100, abs(result['wall_segments'] - 7) / 7 * 100)
            opening_score = 100 - min(100, abs(result['openings'] - 3) / 3 * 100) if result['openings'] > 0 else 0
            return (area_score + wall_score + opening_score) / 3
        
        best = max(successful, key=score_approach)
        best_score = score_approach(best)
        
        print(f"\n🏆 BEST APPROACH: {best['approach']}")
        print(f"   Score: {best_score:.1f}/100")
        print(f"   Results: {best['area']:.1f}m², {best['wall_segments']} walls, {best['openings']} openings")

if __name__ == "__main__":
    main()
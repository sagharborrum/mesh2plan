#!/usr/bin/env python3
"""
Create a comprehensive results summary from all improved test results
"""

import json
from pathlib import Path

def load_result(file_path):
    """Load a result JSON file"""
    try:
        with open(file_path) as f:
            data = json.load(f)
        return {
            'success': True,
            'area': data.get('room_area_sqm', 0),
            'wall_segments': len(data.get('wall_segments', [])),
            'openings': len(data.get('openings', [])),
            'method': data.get('method', 'unknown'),
            'file': file_path.name
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'file': file_path.name}

def main():
    results_dir = Path('results')
    
    # Find all improved result files 
    improved_files = list(results_dir.glob('*/test*_improved*.json'))
    improved_files.extend(list(results_dir.glob('*/test*_current*.json')))
    
    print(f"Found {len(improved_files)} result files")
    
    # Organize results
    summary = {}
    
    for file_path in improved_files:
        result = load_result(file_path)
        
        # Parse file path to get approach and test
        parts = file_path.stem.split('_')
        if 'test1' in file_path.name:
            mesh = 'small'
        elif 'test2' in file_path.name:
            mesh = 'large'
        else:
            mesh = 'unknown'
            
        approach = file_path.parent.name  # e.g., 'v11_tests' -> 'v11'
        
        key = (approach, mesh)
        if key not in summary:
            summary[key] = result
        elif result['success']:  # Prefer successful results
            summary[key] = result
    
    # Print summary
    print(f"\n📊 IMPROVED APPROACHES TEST RESULTS")
    print(f"=" * 70)
    print(f"Target: ~11.5 m², ~7 wall segments, 2-4 openings")
    print(f"=" * 70)
    print(f"{'Approach':12} {'Mesh':6} {'Area':8} {'Walls':6} {'Openings':9} {'Accuracy':10}")
    print(f"{'-' * 70}")
    
    results_list = []
    
    for (approach, mesh), result in sorted(summary.items()):
        if result['success']:
            area = result['area']
            walls = result['wall_segments']
            openings = result['openings']
            
            # Calculate accuracy vs target
            target_area = 11.5
            area_error = abs(area - target_area) / target_area * 100
            wall_error = abs(walls - 7) / 7 * 100
            opening_error = abs(openings - 3) / 3 * 100 if openings > 0 else 100
            
            overall_accuracy = 100 - (area_error + wall_error + opening_error) / 3
            
            print(f"{approach:12} {mesh:6} {area:6.1f}m² {walls:4}   {openings:7}   {overall_accuracy:6.1f}%")
            
            results_list.append({
                'approach': approach,
                'mesh': mesh,
                'area': area,
                'walls': walls,
                'openings': openings,
                'area_error': area_error,
                'wall_error': wall_error,
                'opening_error': opening_error,
                'overall_accuracy': overall_accuracy
            })
        else:
            print(f"{approach:12} {mesh:6} FAILED - {result.get('error', 'unknown')}")
    
    # Find best approaches
    print(f"\n🏆 TOP PERFORMERS (Large Mesh):")
    print(f"{'-' * 50}")
    
    large_mesh_results = [r for r in results_list if r['mesh'] == 'large']
    large_mesh_results.sort(key=lambda x: x['overall_accuracy'], reverse=True)
    
    for i, result in enumerate(large_mesh_results[:3]):
        print(f"{i+1}. {result['approach']:12} - {result['area']:5.1f}m², {result['walls']} walls, {result['openings']} openings "
              f"({result['overall_accuracy']:.1f}% accuracy)")
    
    # Save detailed summary
    summary_path = Path('results/improved_approaches_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {summary_path}")
    
    # Performance vs baseline
    if large_mesh_results:
        best = large_mesh_results[0]
        print(f"\n✨ BEST APPROACH: {best['approach']}")
        print(f"   Area: {best['area']:.1f} m² (target: 11.5 m², error: {best['area_error']:.1f}%)")
        print(f"   Wall segments: {best['walls']} (target: ~7, error: {best['wall_error']:.1f}%)")
        print(f"   Openings: {best['openings']} (target: 2-4, error: {best['opening_error']:.1f}%)")
        print(f"   Overall accuracy: {best['overall_accuracy']:.1f}%")
        
        if best['area_error'] < 20 and best['wall_error'] < 30 and best['openings'] >= 2:
            print(f"   🎯 This approach meets the target criteria!")
        else:
            print(f"   ⚠️  This approach needs further refinement")

if __name__ == "__main__":
    main()
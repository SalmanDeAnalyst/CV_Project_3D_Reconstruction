# diagnostic_check.py
"""
Quick diagnostic to check if BA is actually helping.
Run this to see if the fixes are working properly.
"""

import json
import os

def analyze_sfm_stats(output_dir='output/sfm_output'):
    """Analyze SfM statistics to check BA effectiveness."""
    
    stats_file = os.path.join(output_dir, 'statistics.json')
    
    if not os.path.exists(stats_file):
        print(f"❌ Stats file not found: {stats_file}")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("="*60)
    print("SfM RECONSTRUCTION DIAGNOSTICS")
    print("="*60)
    
    # Basic stats
    print(f"\n📊 Basic Stats:")
    print(f"   Cameras added: {stats['num_cameras']}")
    print(f"   3D points: {stats['num_points']}")
    
    # Reprojection errors
    if stats['reprojection_errors']:
        errors = stats['reprojection_errors']
        print(f"\n📐 Reprojection Errors:")
        print(f"   Mean: {sum(errors)/len(errors):.3f} pixels")
        print(f"   Min: {min(errors):.3f} pixels")
        print(f"   Max: {max(errors):.3f} pixels")
        
        if sum(errors)/len(errors) > 5.0:
            print("   ⚠️  HIGH! Should be < 3 pixels for good reconstruction")
        elif sum(errors)/len(errors) < 2.0:
            print("   ✅ EXCELLENT! Very accurate pose estimation")
        else:
            print("   ✅ Good quality")
    
    # Bundle Adjustment
    if stats['ba_improvements']:
        improvements = stats['ba_improvements']
        print(f"\n🔧 Bundle Adjustment:")
        print(f"   BA runs: {len(improvements)}")
        print(f"   Mean improvement: {sum(improvements)/len(improvements):.3f} pixels")
        print(f"   Total improvement: {sum(improvements):.3f} pixels")
        
        if sum(improvements)/len(improvements) < 0.5:
            print("   ⚠️  LOW IMPROVEMENT! Possible issues:")
            print("      - Observations not being tracked properly")
            print("      - Initial poses already very good")
            print("      - Not enough overlapping views")
        elif sum(improvements)/len(improvements) > 2.0:
            print("   ✅ EXCELLENT! BA is making significant corrections")
        else:
            print("   ✅ Good - BA is helping refine the reconstruction")
    else:
        print("\n🔧 Bundle Adjustment: NOT USED")
    
    print("\n" + "="*60)
    
    # Camera addition success rate
    if stats['num_cameras'] > 0:
        # Estimate how many images were processed vs added
        # This is approximate since we don't track failures explicitly
        print(f"\n📸 Camera Addition:")
        print(f"   Successfully added: {stats['num_cameras']} cameras")
        print(f"   Average points per camera: {stats['num_points'] / stats['num_cameras']:.1f}")
        
        if stats['num_points'] / stats['num_cameras'] < 100:
            print("   ⚠️  LOW! Should have 200-500 points per camera")
        elif stats['num_points'] / stats['num_cameras'] > 1000:
            print("   ⚠️  VERY HIGH! Might indicate duplicate points or issues")
        else:
            print("   ✅ Good point density")

if __name__ == "__main__":
    analyze_sfm_stats()
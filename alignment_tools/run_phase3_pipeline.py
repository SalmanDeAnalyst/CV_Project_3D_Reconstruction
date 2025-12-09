"""
run_phase3_pipeline.py - Complete Phase 3 Pipeline Orchestrator

Runs the complete alignment workflow:
1. Auto-align walls using corner overlap
2. Merge point clouds
3. Transform cameras
4. Build view graph

Usage:
    python run_phase3_pipeline.py --output-dir output --merged-dir output/merged
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*70)
    print(description)
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✅ {description} completed successfully")
    return True


def main():
    # Get script directory to find other scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(
        description="Complete Phase 3 alignment and merging pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                       help='Base output directory (contains wall_A/, wall_B/, etc.)')
    
    parser.add_argument('--merged-dir', '-m', type=str, required=True,
                       help='Directory for merged outputs')
    
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Visualize aligned point clouds (requires Open3D)')
    
    parser.add_argument('--min-shared', type=int, default=15,
                       help='Minimum shared points for view graph (default: 15)')
    
    parser.add_argument('--max-distance', type=float, default=15.0,
                       help='Maximum distance for view graph (default: 15.0)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PHASE 3: COMPLETE ALIGNMENT & MERGING PIPELINE")
    print("="*70)
    print(f"\nInput directory: {args.output_dir}")
    print(f"Merged output directory: {args.merged_dir}")
    
    # Create merged directory
    os.makedirs(args.merged_dir, exist_ok=True)
    
    # Define output paths
    transforms_path = os.path.join(args.merged_dir, 'transformations.json')
    merged_ply_path = os.path.join(args.merged_dir, 'merged_scene.ply')
    merged_cameras_path = os.path.join(args.merged_dir, 'all_cameras.json')
    view_graph_path = os.path.join(args.merged_dir, 'view_graph.json')
    
    # =================================================================
    # STEP 1: Auto-Align Walls
    # =================================================================
    
    cmd = [
        sys.executable, os.path.join(script_dir, 'auto_align_walls.py'),
        '--output-dir', args.output_dir,
        '--save-transforms', transforms_path
    ]
    
    if args.visualize:
        cmd.append('--visualize')
    
    run_command(cmd, "STEP 1: Auto-Align Walls")
    
    # =================================================================
    # STEP 2: Merge Point Clouds
    # =================================================================
    
    run_command(
        [sys.executable, os.path.join(script_dir, 'merge_clouds.py'),
         '--output-dir', args.output_dir,
         '--transforms', transforms_path,
         '--output', merged_ply_path],
        "STEP 2: Merge Point Clouds"
    )
    
    # =================================================================
    # STEP 3: Transform Cameras
    # =================================================================
    
    run_command(
        [sys.executable, os.path.join(script_dir, 'transform_cameras.py'),
         '--output-dir', args.output_dir,
         '--transforms', transforms_path,
         '--output', merged_cameras_path],
        "STEP 3: Transform Cameras"
    )
    
    # =================================================================
    # STEP 4: Build View Graph
    # =================================================================
    
    run_command(
        [sys.executable, os.path.join(script_dir, 'build_view_graph.py'),
         '--output-dir', args.output_dir,
         '--cameras', merged_cameras_path,
         '--output', view_graph_path,
         '--min-shared', str(args.min_shared),
         '--max-distance', str(args.max_distance),
         '--add-spatial-fallback'],
        "STEP 4: Build View Graph"
    )
    
    # =================================================================
    # DONE!
    # =================================================================
    
    print("\n" + "="*70)
    print("✅ PHASE 3 COMPLETE!")
    print("="*70)
    
    print("\n📦 Generated Files:")
    print(f"  ✅ Transformations: {transforms_path}")
    print(f"  ✅ Merged point cloud: {merged_ply_path}")
    print(f"  ✅ All cameras: {merged_cameras_path}")
    print(f"  ✅ View graph: {view_graph_path}")
    
    print("\n🎯 Next Steps:")
    print("  1. Copy these files to your Three.js web viewer:")
    print(f"     - {merged_ply_path} → web_viewer/data/")
    print(f"     - {merged_cameras_path} → web_viewer/data/")
    print(f"     - {view_graph_path} → web_viewer/data/")
    print("  2. Copy source images to web_viewer/images/")
    print("  3. Open web_viewer/index.html in browser")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
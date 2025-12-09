"""
complete_alignment_workflow.py - Complete Phase 3A Workflow

Runs all three tools in sequence:
1. Merge point clouds
2. Transform cameras
3. Build view graph

Usage:
    python complete_alignment_workflow.py --config alignment_config.json
"""

import json
import subprocess
import sys
import os
from pathlib import Path


def load_config(config_file):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*60)
    print(description)
    print("="*60)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Phase 3A alignment workflow")
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to alignment config JSON file')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PHASE 3A: COMPLETE ALIGNMENT WORKFLOW")
    print("="*60)
    
    # Load config
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    sections = config['sections']
    transforms = config.get('transforms', {})
    output = config['output']
    
    print(f"✓ Loaded config:")
    print(f"  Sections: {len(sections)}")
    print(f"  Transforms: {len(transforms)}")
    
    # Create output directory
    output_dir = str(Path(output['merged_ply']).parent)
    os.makedirs(output_dir, exist_ok=True)
    
    # =================================================================
    # STEP 1: Merge Point Clouds
    # =================================================================
    
    print("\n" + "="*60)
    print("STEP 1: MERGING POINT CLOUDS")
    print("="*60)
    
    # Save config for merge_clouds.py
    merge_config_file = f"{output_dir}/merge_config.json"
    with open(merge_config_file, 'w') as f:
        json.dump({
            'sections': sections,
            'transforms': transforms
        }, f, indent=2)
    
    run_command(
        ['python', 'merge_clouds.py', '--config', merge_config_file,
         '--output', output['merged_ply']],
        "Merge Point Clouds"
    )
    
    # =================================================================
    # STEP 2: Transform Cameras
    # =================================================================
    
    print("\n" + "="*60)
    print("STEP 2: TRANSFORMING CAMERA POSES")
    print("="*60)
    
    transformed_camera_files = []
    
    # First section is reference (no transformation)
    reference_section = sections[0]
    print(f"\n{reference_section['name']}: REFERENCE (no transformation needed)")
    
    # Copy reference cameras as-is
    import shutil
    reference_output = f"{output_dir}/cameras_{reference_section['name']}.json"
    shutil.copy(reference_section['cameras_path'], reference_output)
    transformed_camera_files.append(reference_output)
    print(f"✓ Copied: {reference_output}")
    
    # Transform other sections
    for section in sections[1:]:
        section_name = section['name']
        
        if section_name not in transforms:
            print(f"\n⚠️  Warning: No transformation for {section_name}, skipping")
            continue
        
        output_file = f"{output_dir}/cameras_{section_name}_transformed.json"
        
        run_command(
            ['python', 'transform_cameras.py',
             '--cameras', section['cameras_path'],
             '--transform', output['transformations'],
             '--section', section_name,
             '--output', output_file],
            f"Transform {section_name} Cameras"
        )
        
        transformed_camera_files.append(output_file)
    
    # Merge all transformed cameras
    print("\n" + "-"*60)
    print("Merging all transformed cameras...")
    print("-"*60)
    
    run_command(
        ['python', 'transform_cameras.py', '--merge',
         '--cameras'] + transformed_camera_files + 
        ['--output', output['merged_cameras']],
        "Merge All Cameras"
    )
    
    # =================================================================
    # STEP 3: Build View Graph
    # =================================================================
    
    print("\n" + "="*60)
    print("STEP 3: BUILDING VIEW GRAPH")
    print("="*60)
    
    observations_files = [section['observations_path'] for section in sections]
    
    view_graph_params = config.get('view_graph_params', {})
    
    cmd = [
        'python', 'build_view_graph.py',
        '--cameras', output['merged_cameras'],
        '--observations'] + observations_files + [
        '--output', output['view_graph'],
        '--min-shared', str(view_graph_params.get('min_shared_points', 20)),
        '--max-distance', str(view_graph_params.get('max_spatial_distance', 10.0))
    ]
    
    if view_graph_params.get('add_spatial_fallback', False):
        cmd.append('--add-spatial-fallback')
    
    run_command(cmd, "Build View Graph")
    
    # =================================================================
    # DONE!
    # =================================================================
    
    print("\n" + "="*60)
    print("✅ PHASE 3A COMPLETE!")
    print("="*60)
    
    print("\n📦 Generated Files:")
    print(f"  ✓ Merged point cloud: {output['merged_ply']}")
    print(f"  ✓ All cameras: {output['merged_cameras']}")
    print(f"  ✓ View graph: {output['view_graph']}")
    print(f"  ✓ Transformations: {output['transformations']}")
    
    print("\n🎯 Next Steps:")
    print("  1. Copy these files to your Three.js app")
    print("  2. Copy source images to web app's images/ folder")
    print("  3. Run Phase 3B (Three.js viewer implementation)")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
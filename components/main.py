# main.py

import os
import glob
import json
import numpy as np
from typing import Dict, List
from incremental_sfm import IncrementalSfM, natural_sort_key, remove_duplicate_images


class MultiWallSfMOrchestrator:
    """
    Orchestrates SfM reconstruction across multiple wall sections.
    Handles per-wall processing and metadata generation.
    """
    
    def __init__(self, base_data_dir: str, base_output_dir: str, K: np.ndarray):
        self.base_data_dir = base_data_dir
        self.base_output_dir = base_output_dir
        self.K = K
        
        # Wall configuration: image ranges and adjacencies
        self.wall_config = {
            'wall_A': {
                'image_range': (1, 9),
                'folder': 'wall_a',
                'adjacent_to': ['wall_B'],
                'corner_images': [9]  # Images showing corner with next wall
            },
            'wall_B': {
                'image_range': (10, 25),
                'folder': 'wall_b',
                'adjacent_to': ['wall_A', 'wall_C'],
                'corner_images': [10, 25]
            },
            'wall_C': {
                'image_range': (26, 31),
                'folder': 'wall_c',
                'adjacent_to': ['wall_B', 'wall_D'],
                'corner_images': [26, 31]
            },
            'wall_D': {
                'image_range': (32, 44),
                'folder': 'wall_d',
                'adjacent_to': ['wall_C', 'wall_E'],
                'corner_images': [32, 44]
            },
            'wall_E': {
                'image_range': (45, 73),
                'folder': 'wall_e',
                'adjacent_to': ['wall_D', 'wall_F'],
                'corner_images': [45, 73]
            },
            'wall_F': {
                'image_range': (74, 91),
                'folder': 'wall_f',
                'adjacent_to': ['wall_E', 'wall_G'],
                'corner_images': [74, 91]
            },
            'wall_G': {
                'image_range': (92, 96),
                'folder': 'wall_g',
                'adjacent_to': ['wall_F'],
                'corner_images': [92]
            }
        }
        
        os.makedirs(base_output_dir, exist_ok=True)
    
    def load_wall_images(self, wall_id: str) -> List[str]:
        """Load images for a specific wall from its folder."""
        config = self.wall_config[wall_id]
        wall_folder = os.path.join(self.base_data_dir, config['folder'])
        
        if not os.path.exists(wall_folder):
            print(f"❌ ERROR: Wall folder not found: {wall_folder}")
            return []
        
        # Load all image formats
        image_paths = (
            glob.glob(os.path.join(wall_folder, '*.jpeg')) +
            glob.glob(os.path.join(wall_folder, '*.jpg')) +
            glob.glob(os.path.join(wall_folder, '*.heic')) +
            glob.glob(os.path.join(wall_folder, '*.HEIC')) +
            glob.glob(os.path.join(wall_folder, '*.png'))
        )
        
        image_paths = sorted(image_paths, key=natural_sort_key)
        image_paths = remove_duplicate_images(image_paths)
        
        print(f"  Found {len(image_paths)} images in {config['folder']}/")
        
        # Validate image range
        start, end = config['image_range']
        expected_count = end - start + 1
        if len(image_paths) != expected_count:
            print(f"  ⚠️  Warning: Expected {expected_count} images, found {len(image_paths)}")
        
        return image_paths
    
    def save_wall_metadata(self, wall_id: str, sfm_stats: Dict):
        """Save metadata for a reconstructed wall section."""
        config = self.wall_config[wall_id]
        output_dir = os.path.join(self.base_output_dir, wall_id)
        
        metadata = {
            'wall_id': wall_id,
            'folder': config['folder'],
            'image_range': config['image_range'],
            'adjacent_to': config['adjacent_to'],
            'corner_images': config['corner_images'],
            'reconstruction_stats': {
                'num_cameras': sfm_stats.get('num_cameras', 0),
                'num_points': sfm_stats.get('num_points', 0),
                'mean_reprojection_error': float(np.mean(sfm_stats.get('reprojection_errors', [0])))
                    if sfm_stats.get('reprojection_errors') else 0.0
            }
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata: {metadata_path}")
    
    def reconstruct_wall(self, wall_id: str) -> bool:
        """Reconstruct a single wall section."""
        print("\n" + "="*70)
        print(f"RECONSTRUCTING: {wall_id}")
        print("="*70)
        
        # Load images
        image_paths = self.load_wall_images(wall_id)
        if len(image_paths) < 2:
            print(f"❌ Not enough images for {wall_id} (need at least 2)")
            return False
        
        # Setup output directory
        output_dir = os.path.join(self.base_output_dir, wall_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run SfM
        sfm = IncrementalSfM(
            K=self.K,
            ordered_image_paths=image_paths,
            output_dir=output_dir,
            use_bundle_adjustment=True,
            use_dense_stereo=False,
            ba_interval=5,  # Run BA every 5 cameras
            visualization_interval=10,
            feature_cache=None  # Extract fresh for each wall
        )
        
        success = sfm.reconstruct()
        
        if success:
            # Save metadata
            self.save_wall_metadata(wall_id, sfm.stats)
            print(f"✅ {wall_id} reconstruction complete!")
        else:
            print(f"❌ {wall_id} reconstruction failed!")
        
        return success
    
    def reconstruct_all_walls(self):
        """Reconstruct all wall sections sequentially."""
        print("\n" + "="*70)
        print("MULTI-WALL SfM RECONSTRUCTION PIPELINE")
        print("="*70)
        print(f"Base data directory: {self.base_data_dir}")
        print(f"Base output directory: {self.base_output_dir}")
        print(f"Walls to process: {len(self.wall_config)}")
        print("="*70)
        
        results = {}
        
        for wall_id in self.wall_config.keys():
            success = self.reconstruct_wall(wall_id)
            results[wall_id] = success
        
        # Summary
        print("\n" + "="*70)
        print("RECONSTRUCTION SUMMARY")
        print("="*70)
        
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        for wall_id, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{wall_id}: {status}")
        
        print(f"\nTotal: {successful}/{len(results)} walls reconstructed successfully")
        
        if failed > 0:
            print(f"⚠️  {failed} walls failed - check logs above")
        
        print("="*70)
        
        # Save overall summary
        summary_path = os.path.join(self.base_output_dir, 'reconstruction_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'total_walls': len(results),
                'successful': successful,
                'failed': failed,
                'results': {wall: ('success' if status else 'failed') 
                           for wall, status in results.items()}
            }, f, indent=2)
        
        print(f"\n✓ Saved summary: {summary_path}\n")


if __name__ == "__main__":
    # Get project root
    
    current_file = os.path.abspath(__file__)
    if 'components' in current_file:
        parent_dir = os.path.dirname(os.path.dirname(current_file))
    else:
        parent_dir = os.path.dirname(current_file)
    
    # Camera intrinsics
    K = np.array([
        [2184, 0, 1512],
        [0, 2184, 2016],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Paths
    base_data_dir = os.path.join(parent_dir, 'data', 'images')  # Contains wall_a/, wall_b/, etc.
    base_output_dir = os.path.join(parent_dir, 'output')
    
    # Verify data directory exists
    if not os.path.exists(base_data_dir):
        print(f"❌ ERROR: Data directory not found: {base_data_dir}")
        print("Expected structure:")
        print("  data/")
        print("    wall_a/  (images 1-9)")
        print("    wall_b/  (images 10-25)")
        print("    ...")
        exit(1)
    
    # Run reconstruction
    orchestrator = MultiWallSfMOrchestrator(base_data_dir, base_output_dir, K)
    orchestrator.reconstruct_all_walls()
    
    print("\n🎉 Multi-Wall SfM Pipeline Complete! 🎉")
    print(f"\nOutputs saved to: {base_output_dir}")
    print("  wall_A/")
    print("    ├── final_reconstruction.ply")
    print("    ├── camera_poses.json")
    print("    ├── observations.json  ← NEW!")
    print("    └── metadata.json      ← NEW!")
    print("  wall_B/")
    print("  ...")
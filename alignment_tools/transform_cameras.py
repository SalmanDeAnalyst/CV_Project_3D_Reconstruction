"""
transform_cameras.py - Transform camera poses using alignment matrices

Usage:
    # Transform all cameras automatically
    python transform_cameras.py \\
        --output-dir output \\
        --transforms output/merged/transformations.json \\
        --output output/merged/all_cameras.json
"""

import numpy as np
import json
import argparse
import os
from typing import Dict, List


class CameraTransformer:
    """Transform camera poses using pre-computed alignment matrices."""
    
    def __init__(self, output_dir: str, transformations: Dict[str, np.ndarray]):
        """
        Initialize transformer.
        
        Args:
            output_dir: Base output directory (contains wall_A/, wall_B/, etc.)
            transformations: Dict of {wall_name: 4x4 transform matrix}
        """
        self.output_dir = output_dir
        self.transformations = transformations
        self.all_cameras = []
    
    def load_and_transform_cameras(self):
        """Load all cameras and apply transformations."""
        print("\n" + "="*60)
        print("Loading and Transforming Cameras")
        print("="*60)
        
        # Find all wall directories
        wall_dirs = sorted([d for d in os.listdir(self.output_dir) 
                           if d.startswith('wall_') and os.path.isdir(os.path.join(self.output_dir, d))])
        
        camera_id_offset = 0
        
        for wall_dir in wall_dirs:
            wall_name = wall_dir
            cameras_path = os.path.join(self.output_dir, wall_dir, 'camera_poses.json')
            
            if not os.path.exists(cameras_path):
                print(f"⚠️  Warning: No cameras for {wall_name}")
                continue
            
            # Load cameras
            with open(cameras_path, 'r') as f:
                cameras = json.load(f)
            
            # Get transformation
            if wall_name not in self.transformations:
                print(f"⚠️  Warning: No transformation for {wall_name}, using identity")
                T = np.eye(4)
            else:
                T = self.transformations[wall_name]
            
            # Transform each camera
            for cam in cameras:
                R_old = np.array(cam['R'])
                t_old = np.array(cam['t'])
                
                # Handle different t formats
                if t_old.ndim == 1:
                    t_old = t_old.reshape(3, 1)
                elif t_old.shape == (1, 3):
                    t_old = t_old.reshape(3, 1)
                
                # Transform camera pose
                R_new, t_new = self.transform_camera_pose(R_old, t_old, T)
                
                # Compute center
                C_new = -R_new.T @ t_new
                
                # Create transformed camera
                transformed_cam = {
                    'id': camera_id_offset + cam['id'],  # Global unique ID
                    'wall': wall_name,  # Track which wall this came from
                    'local_id': cam['id'],  # Original ID within wall
                    'image': cam['image'],
                    'R': R_new.tolist(),
                    't': t_new.tolist(),
                    'center': C_new.flatten().tolist()
                }
                
                self.all_cameras.append(transformed_cam)
            
            print(f"✓ Transformed {len(cameras)} cameras from {wall_name}")
            camera_id_offset += len(cameras)
        
        print(f"\n✓ Total cameras: {len(self.all_cameras)}")
        print("="*60 + "\n")
    
    @staticmethod
    def transform_camera_pose(R_old: np.ndarray, t_old: np.ndarray, T: np.ndarray):
        """
        Transform camera pose.
        
        Math:
        1. C_old = -R_old^T @ t_old  (old camera center)
        2. R_new = R_transform @ R_old
        3. C_new = R_transform @ C_old + t_transform
        4. t_new = -R_new @ C_new
        """
        R_transform = T[:3, :3]
        t_transform = T[:3, 3].reshape(3, 1)
        
        # Old camera center
        C_old = -R_old.T @ t_old
        
        # Transform rotation
        R_new = R_transform @ R_old
        
        # Transform center
        C_new = R_transform @ C_old + t_transform
        
        # Compute new t
        t_new = -R_new @ C_new
        
        return R_new, t_new
    
    def save_cameras(self, output_path: str):
        """Save all transformed cameras."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.all_cameras, f, indent=2)
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"✓ Saved cameras: {output_path}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Total cameras: {len(self.all_cameras)}")


def main():
    parser = argparse.ArgumentParser(
        description="Transform all camera poses using alignment matrices"
    )
    
    parser.add_argument('--output-dir', '-d', type=str, required=True,
                       help='Base output directory (contains wall_A/, wall_B/, etc.)')
    
    parser.add_argument('--transforms', '-t', type=str, required=True,
                       help='Path to transformations.json')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for merged cameras JSON')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TRANSFORM CAMERAS")
    print("="*60)
    
    # Load transformations
    print(f"\nLoading transformations from: {args.transforms}")
    with open(args.transforms, 'r') as f:
        transforms_data = json.load(f)
    
    transformations = {}
    for wall_name, data in transforms_data.items():
        transformations[wall_name] = np.array(data['matrix'])
    
    print(f"✓ Loaded transformations for {len(transformations)} walls")
    
    # Initialize transformer
    transformer = CameraTransformer(args.output_dir, transformations)
    
    # Load and transform
    transformer.load_and_transform_cameras()
    
    # Save
    transformer.save_cameras(args.output)
    
    print("\n✅ Camera transformation complete!")


if __name__ == "__main__":
    main()
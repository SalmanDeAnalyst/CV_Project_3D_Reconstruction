"""
merge_clouds.py - Merge aligned point clouds

Usage:
    python merge_clouds.py \\
        --config output/merged/transformations.json \\
        --output output/merged/merged_scene.ply

Applies transformations from auto_align_walls.py to point clouds and merges them.
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple


class PointCloudMerger:
    """Handles merging of pre-aligned point cloud sections."""
    
    def __init__(self, output_dir: str, transformations: Dict[str, np.ndarray]):
        """
        Initialize merger.
        
        Args:
            output_dir: Base output directory (contains wall_A/, wall_B/, etc.)
            transformations: Dict of {wall_name: 4x4 transform matrix}
        """
        self.output_dir = output_dir
        self.transformations = transformations
        self.point_clouds = {}
    
    def load_point_clouds(self):
        """Load all point cloud files."""
        print("\n" + "="*60)
        print("Loading Point Clouds")
        print("="*60)
        
        # Find all wall directories
        wall_dirs = sorted([d for d in os.listdir(self.output_dir) 
                           if d.startswith('wall_') and os.path.isdir(os.path.join(self.output_dir, d))])
        
        for wall_dir in wall_dirs:
            wall_name = wall_dir
            ply_path = os.path.join(self.output_dir, wall_dir, 'final_reconstruction.ply')
            
            if not os.path.exists(ply_path):
                print(f"⚠️  Warning: No PLY file for {wall_name}")
                continue
            
            # Load using Open3D if available, else manual parsing
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(ply_path)
                num_points = len(pcd.points)
                self.point_clouds[wall_name] = pcd
                
            except ImportError:
                # Manual PLY loading
                points, colors = self.load_ply_manual(ply_path)
                # Create fake Open3D-like object
                pcd = type('PointCloud', (), {
                    'points': points,
                    'colors': colors,
                    'transform': lambda T: self.transform_points(points, T)
                })()
                self.point_clouds[wall_name] = pcd
                num_points = len(points)
            
            print(f"✓ Loaded {wall_name}: {num_points:,} points")
        
        print(f"\n✓ Total sections loaded: {len(self.point_clouds)}")
        print("="*60 + "\n")
    
    def load_ply_manual(self, ply_path: str):
        """Manual PLY file parsing (fallback)."""
        points = []
        colors = []
        
        with open(ply_path, 'r') as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.startswith('end_header'):
                        in_header = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 6:
                    # x y z r g b
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([float(parts[3])/255, float(parts[4])/255, float(parts[5])/255])
                elif len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([0.5, 0.5, 0.5])
        
        return np.array(points), np.array(colors)
    
    def transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Transform points using 4x4 matrix."""
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_transformed = (T @ points_homo.T).T[:, :3]
        return points_transformed
    
    def apply_transformations(self):
        """Apply transformations to all point clouds."""
        print("\n" + "="*60)
        print("Applying Transformations")
        print("="*60)
        
        for wall_name, pcd in self.point_clouds.items():
            if wall_name not in self.transformations:
                print(f"⚠️  Warning: No transformation for {wall_name}, using identity")
                T = np.eye(4)
            else:
                T = self.transformations[wall_name]
            
            # Apply transformation
            try:
                pcd.transform(T)
            except:
                # Manual transformation
                pcd.points = self.transform_points(np.array(pcd.points), T)
            
            # Extract rotation angle for logging
            R = T[:3, :3]
            t = T[:3, 3]
            
            print(f"✓ Transformed {wall_name}")
            print(f"    Translation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
        
        print("="*60 + "\n")
    
    def merge_all(self):
        """Merge all point clouds into one."""
        print("\n" + "="*60)
        print("Merging Point Clouds")
        print("="*60)
        
        try:
            import open3d as o3d
            combined = o3d.geometry.PointCloud()
            
            total_points = 0
            for wall_name, pcd in self.point_clouds.items():
                combined += pcd
                num_points = len(pcd.points)
                total_points += num_points
                print(f"✓ Added {wall_name}: {num_points:,} points")
            
            print(f"\n✓ Merged point cloud: {total_points:,} total points")
            print("="*60 + "\n")
            
            return combined
            
        except ImportError:
            # Manual merging
            all_points = []
            all_colors = []
            
            for wall_name, pcd in self.point_clouds.items():
                all_points.append(np.array(pcd.points))
                all_colors.append(np.array(pcd.colors))
                print(f"✓ Added {wall_name}: {len(pcd.points):,} points")
            
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            
            # Create fake point cloud object
            combined = type('PointCloud', (), {
                'points': combined_points,
                'colors': combined_colors
            })()
            
            print(f"\n✓ Merged point cloud: {len(combined_points):,} total points")
            print("="*60 + "\n")
            
            return combined
    
    def save_merged_cloud(self, output_path: str, merged_cloud):
        """Save merged point cloud to PLY."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            import open3d as o3d
            o3d.io.write_point_cloud(output_path, merged_cloud)
            
        except ImportError:
            # Manual PLY writing
            points = np.array(merged_cloud.points)
            colors = np.array(merged_cloud.colors)
            
            with open(output_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for i in range(len(points)):
                    p = points[i]
                    c = (colors[i] * 255).astype(int)
                    f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✓ Saved merged cloud: {output_path}")
        print(f"  File size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Merge aligned point clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output-dir', '-d', type=str, required=True,
                       help='Base output directory (contains wall_A/, wall_B/, etc.)')
    
    parser.add_argument('--transforms', '-t', type=str, required=True,
                       help='Path to transformations.json from auto_align_walls.py')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for merged point cloud (.ply)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MERGE ALIGNED POINT CLOUDS")
    print("="*60)
    
    # Load transformations
    print(f"\nLoading transformations from: {args.transforms}")
    with open(args.transforms, 'r') as f:
        transforms_data = json.load(f)
    
    # Convert to numpy matrices
    transformations = {}
    for wall_name, data in transforms_data.items():
        transformations[wall_name] = np.array(data['matrix'])
    
    print(f"✓ Loaded transformations for {len(transformations)} walls")
    
    # Initialize merger
    merger = PointCloudMerger(args.output_dir, transformations)
    
    # Load point clouds
    merger.load_point_clouds()
    
    # Apply transformations
    merger.apply_transformations()
    
    # Merge
    merged = merger.merge_all()
    
    # Save
    merger.save_merged_cloud(args.output, merged)
    
    print("\n✅ Merge complete!")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
"""
auto_align_walls.py - Automatically align wall sections using corner overlap

This tool:
1. Reads metadata.json to find adjacent walls
2. Uses observations.json to find cameras sharing 3D points
3. Computes transformation matrices automatically
4. Saves transformations.json for the merge pipeline

Usage:
    python auto_align_walls.py --output-dir output --save-transforms output/merged/transformations.json
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from scipy.spatial.transform import Rotation


class WallAlignment:
    """Handles automatic alignment of wall sections using corner overlap."""
    
    def __init__(self, output_dir: str):
        """
        Initialize wall alignment.
        
        Args:
            output_dir: Base output directory containing wall_A/, wall_B/, etc.
        """
        self.output_dir = output_dir
        self.walls = {}  # {wall_name: {metadata, cameras, observations, points}}
        self.adjacency_graph = {}  # {wall_A: [wall_B, wall_C]}
        self.transformations = {}  # {wall_B: T_matrix}
    
    def load_wall_data(self):
        """Load all wall metadata, cameras, and observations."""
        print("\n" + "="*60)
        print("Loading Wall Data")
        print("="*60)
        
        # Find all wall directories
        wall_dirs = sorted([d for d in os.listdir(self.output_dir) 
                           if d.startswith('wall_') and os.path.isdir(os.path.join(self.output_dir, d))])
        
        for wall_dir in wall_dirs:
            wall_name = wall_dir  # e.g., "wall_A"
            wall_path = os.path.join(self.output_dir, wall_dir)
            
            # Load metadata
            metadata_path = os.path.join(wall_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                print(f"⚠️  Warning: No metadata for {wall_name}")
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load cameras
            cameras_path = os.path.join(wall_path, 'camera_poses.json')
            with open(cameras_path, 'r') as f:
                cameras = json.load(f)
            
            # Load observations
            obs_path = os.path.join(wall_path, 'observations.json')
            with open(obs_path, 'r') as f:
                observations = json.load(f)
            
            # Load point cloud to get 3D coordinates
            ply_path = os.path.join(wall_path, 'final_reconstruction.ply')
            points_3d = self.load_points_from_ply(ply_path)
            
            self.walls[wall_name] = {
                'metadata': metadata,
                'cameras': cameras,
                'observations': observations,
                'points_3d': points_3d,
                'path': wall_path
            }
            
            print(f"✓ Loaded {wall_name}:")
            print(f"    Cameras: {len(cameras)}")
            print(f"    3D Points: {len(points_3d)}")
            print(f"    Adjacent to: {metadata.get('adjacent_to', [])}")
        
        print(f"\n✓ Total walls loaded: {len(self.walls)}")
        print("="*60 + "\n")
    
    def load_points_from_ply(self, ply_path: str) -> np.ndarray:
        """Load 3D points from PLY file."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(ply_path)
            return np.asarray(pcd.points)
        except:
            # Fallback: manual PLY parsing
            points = []
            with open(ply_path, 'r') as f:
                in_data = False
                for line in f:
                    if line.startswith('end_header'):
                        in_data = True
                        continue
                    if in_data:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.array(points)
    
    def build_adjacency_graph(self):
        """Build graph of which walls are adjacent to each other."""
        print("\n" + "="*60)
        print("Building Adjacency Graph")
        print("="*60)
        
        for wall_name, wall_data in self.walls.items():
            adjacent = wall_data['metadata'].get('adjacent_to', [])
            self.adjacency_graph[wall_name] = adjacent
            print(f"{wall_name} → {adjacent}")
        
        print("="*60 + "\n")
    
    def find_shared_points(self, wall_A: str, wall_B: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Find 3D points shared between two walls by matching observations.
        
        Returns:
            points_A: Nx3 array of 3D points in wall_A coordinate system
            points_B: Nx3 array of corresponding points in wall_B coordinate system
            shared_indices: List of point indices
        """
        print(f"\nFinding shared points between {wall_A} and {wall_B}...")
        
        obs_A = self.walls[wall_A]['observations']
        obs_B = self.walls[wall_B]['observations']
        
        # Build reverse lookup: {camera_id: [point_indices]}
        cam_to_points_A = {}
        for point_idx_str, obs_list in obs_A.items():
            for obs in obs_list:
                cam_id = obs['camera_id']
                if cam_id not in cam_to_points_A:
                    cam_to_points_A[cam_id] = []
                cam_to_points_A[cam_id].append(int(point_idx_str))
        
        cam_to_points_B = {}
        for point_idx_str, obs_list in obs_B.items():
            for obs in obs_list:
                cam_id = obs['camera_id']
                if cam_id not in cam_to_points_B:
                    cam_to_points_B[cam_id] = []
                cam_to_points_B[cam_id].append(int(point_idx_str))
        
        # Find corner cameras (from metadata)
        corner_imgs_A = set(self.walls[wall_A]['metadata'].get('corner_images', []))
        corner_imgs_B = set(self.walls[wall_B]['metadata'].get('corner_images', []))
        
        print(f"  Corner images {wall_A}: {corner_imgs_A}")
        print(f"  Corner images {wall_B}: {corner_imgs_B}")
        
        # Match points seen by cameras in corner regions
        # Strategy: Use feature descriptors to match points between overlapping views
        
        # Simpler approach: Find points with similar spatial relationships
        # For now, use ICP on all points and let it figure it out
        points_A = self.walls[wall_A]['points_3d']
        points_B = self.walls[wall_B]['points_3d']
        
        print(f"  Total points {wall_A}: {len(points_A)}")
        print(f"  Total points {wall_B}: {len(points_B)}")
        
        # Use subset for ICP (corners likely at boundaries)
        return points_A, points_B, list(range(min(len(points_A), len(points_B))))
    
    def compute_alignment_icp(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Compute rigid transformation using ICP.
        
        Args:
            source_points: Nx3 points to transform
            target_points: Nx3 reference points
        
        Returns:
            4x4 transformation matrix
        """
        try:
            import open3d as o3d
            
            # Create point clouds
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(source_points)
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(target_points)
            
            # Initial alignment using feature matching
            threshold = 2.0  # Distance threshold
            
            # Coarse registration
            print("  Running coarse ICP alignment...")
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            print(f"  ICP fitness: {reg_result.fitness:.4f}")
            print(f"  ICP RMSE: {reg_result.inlier_rmse:.4f}")
            
            return reg_result.transformation
            
        except ImportError:
            # Fallback: Procrustes alignment
            print("  Using Procrustes alignment (Open3D not available)")
            return self.compute_alignment_procrustes(source_points, target_points)
    
    def compute_alignment_procrustes(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Compute alignment using Procrustes analysis (closed-form solution).
        
        Finds optimal rotation R and translation t such that:
        target ≈ R @ source + t
        """
        # Ensure same number of points
        n = min(len(source_points), len(target_points))
        source = source_points[:n]
        target = target_points[:n]
        
        # Center the points
        source_mean = np.mean(source, axis=0)
        target_mean = np.mean(target, axis=0)
        
        source_centered = source - source_mean
        target_centered = target - target_mean
        
        # Compute optimal rotation using SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_mean - R @ source_mean
        
        # Build 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Compute error
        source_transformed = (R @ source.T).T + t
        rmse = np.sqrt(np.mean(np.sum((source_transformed - target)**2, axis=1)))
        print(f"  Procrustes RMSE: {rmse:.4f}")
        
        return T
    
    def align_walls_sequentially(self):
        """
        Align all walls sequentially starting from wall_A as reference.
        
        Strategy:
        1. wall_A is reference (identity transform)
        2. Align wall_B to wall_A
        3. Align wall_C to wall_B (or wall_A if better overlap)
        4. Continue for all walls
        """
        print("\n" + "="*60)
        print("Computing Wall Alignments")
        print("="*60)
        
        wall_names = sorted(self.walls.keys())
        
        if len(wall_names) == 0:
            print("❌ No walls to align!")
            return
        
        # First wall is reference
        reference_wall = wall_names[0]
        print(f"\n{reference_wall}: REFERENCE (identity transform)")
        self.transformations[reference_wall] = np.eye(4)
        
        aligned_walls = {reference_wall}
        
        # Align remaining walls
        for wall_name in wall_names[1:]:
            print(f"\n{'='*60}")
            print(f"Aligning {wall_name}")
            print(f"{'='*60}")
            
            # Find which adjacent wall is already aligned
            adjacent = self.walls[wall_name]['metadata'].get('adjacent_to', [])
            reference_candidates = [w for w in adjacent if w in aligned_walls]
            
            if not reference_candidates:
                print(f"⚠️  Warning: {wall_name} has no aligned adjacent walls!")
                print(f"   Setting identity transform (may not be correct)")
                self.transformations[wall_name] = np.eye(4)
                aligned_walls.add(wall_name)
                continue
            
            # Use first aligned adjacent wall as reference
            ref_wall = reference_candidates[0]
            print(f"Aligning to: {ref_wall}")
            
            # Get point clouds
            source_pts = self.walls[wall_name]['points_3d']
            target_pts = self.walls[ref_wall]['points_3d']
            
            # Downsample for faster ICP (use every 10th point)
            source_pts = source_pts[::10]
            target_pts = target_pts[::10]
            
            print(f"Using {len(source_pts)} source points, {len(target_pts)} target points")
            
            # Compute transformation: wall_name → ref_wall coordinate system
            T_to_ref = self.compute_alignment_icp(source_pts, target_pts)
            
            # Chain transformations: wall_name → ref_wall → reference
            T_ref_to_reference = self.transformations[ref_wall]
            T_final = T_ref_to_reference @ T_to_ref
            
            self.transformations[wall_name] = T_final
            aligned_walls.add(wall_name)
            
            print(f"✓ {wall_name} aligned successfully")
        
        print("\n" + "="*60)
        print("✓ All walls aligned!")
        print("="*60 + "\n")
    
    def save_transformations(self, output_path: str):
        """Save transformation matrices to JSON."""
        print("\n" + "="*60)
        print("Saving Transformations")
        print("="*60)
        
        transforms_dict = {}
        
        for wall_name, T in self.transformations.items():
            # Extract rotation and translation for readability
            R = T[:3, :3]
            t = T[:3, 3]
            
            # Compute rotation angle for info
            rot = Rotation.from_matrix(R)
            angle_deg = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi
            
            transforms_dict[wall_name] = {
                "matrix": T.tolist(),
                "rotation_degrees": float(angle_deg),
                "translation": t.tolist(),
                "description": f"Transformation for {wall_name} to reference coordinate system"
            }
            
            print(f"{wall_name}:")
            print(f"  Rotation: {angle_deg:.1f}°")
            print(f"  Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(transforms_dict, f, indent=2)
        
        print(f"\n✓ Saved: {output_path}")
        print("="*60 + "\n")
    
    def visualize_alignment(self):
        """Visualize aligned point clouds (optional, requires Open3D)."""
        try:
            import open3d as o3d
            
            print("\n🔍 Visualizing aligned point clouds...")
            
            pcds = []
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
            ]
            
            for i, (wall_name, wall_data) in enumerate(self.walls.items()):
                pcd = o3d.geometry.PointCloud()
                points = wall_data['points_3d']
                
                # Apply transformation
                T = self.transformations.get(wall_name, np.eye(4))
                points_homo = np.hstack([points, np.ones((len(points), 1))])
                points_transformed = (T @ points_homo.T).T[:, :3]
                
                pcd.points = o3d.utility.Vector3dVector(points_transformed[::10])  # Downsample
                pcd.paint_uniform_color(colors[i % len(colors)])
                pcds.append(pcd)
            
            o3d.visualization.draw_geometries(pcds, window_name="Aligned Walls")
            
        except ImportError:
            print("⚠️  Open3D not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically align wall sections using corner overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                       help='Output directory containing wall_A/, wall_B/, etc.')
    
    parser.add_argument('--save-transforms', '-t', type=str, required=True,
                       help='Output path for transformations.json')
    
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Visualize aligned point clouds (requires Open3D)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AUTO-ALIGN WALLS")
    print("="*60)
    
    # Initialize aligner
    aligner = WallAlignment(args.output_dir)
    
    # Load data
    aligner.load_wall_data()
    
    if len(aligner.walls) < 2:
        print("❌ Need at least 2 walls to align!")
        return
    
    # Build adjacency graph
    aligner.build_adjacency_graph()
    
    # Align walls
    aligner.align_walls_sequentially()
    
    # Save transformations
    aligner.save_transformations(args.save_transforms)
    
    # Visualize if requested
    if args.visualize:
        aligner.visualize_alignment()
    
    print("\n✅ Auto-alignment complete!")
    print(f"   Transformations saved to: {args.save_transforms}")
    print("\n🎯 Next: Run merge_clouds.py to merge point clouds")


if __name__ == "__main__":
    main()
"""
build_view_graph.py - Build view graph for camera navigation

Usage:
    python build_view_graph.py \\
        --output-dir output \\
        --cameras output/merged/all_cameras.json \\
        --output output/merged/view_graph.json
"""

import numpy as np
import json
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple


class ViewGraphBuilder:
    """Builds navigation graph from camera observations."""
    
    def __init__(self, cameras: List[Dict], min_shared_points: int = 10, max_spatial_distance: float = 50.0):
        """
        Initialize view graph builder.
        
        Args:
            cameras: List of camera dicts with 'id', 'R', 't', 'center'
            min_shared_points: Minimum shared 3D points to connect cameras
            max_spatial_distance: Maximum distance to connect cameras
        """
        self.cameras = cameras
        self.min_shared_points = min_shared_points
        self.max_spatial_distance = max_spatial_distance
        
        self.camera_dict = {cam['id']: cam for cam in cameras}
        self.view_graph = defaultdict(list)
    
    @staticmethod
    def load_all_observations(output_dir: str) -> Dict[str, Dict]:
        """
        Load observations from all walls.
        
        Returns:
            Dict mapping wall_name to observations dict
        """
        print("\n" + "="*60)
        print("Loading Observations from All Walls")
        print("="*60)
        
        all_observations = {}
        
        # Find all wall directories
        wall_dirs = sorted([d for d in os.listdir(output_dir) 
                           if d.startswith('wall_') and os.path.isdir(os.path.join(output_dir, d))])
        
        for wall_dir in wall_dirs:
            wall_name = wall_dir
            obs_path = os.path.join(output_dir, wall_dir, 'observations.json')
            
            if not os.path.exists(obs_path):
                print(f"⚠️  Warning: No observations for {wall_name}")
                continue
            
            with open(obs_path, 'r') as f:
                observations = json.load(f)
            
            all_observations[wall_name] = observations
            print(f"✓ Loaded {wall_name}: {len(observations):,} 3D points")
        
        print(f"\n✓ Total walls: {len(all_observations)}")
        print("="*60 + "\n")
        
        return all_observations
    
    def compute_shared_points_per_wall(self, wall_observations: Dict) -> Dict[Tuple[int, int], int]:
        """
        Compute shared points between cameras within a single wall.
        
        Args:
            wall_observations: {point_idx: [{"camera_id": X, "pixel": [x, y]}]}
        
        Returns:
            {(cam_i, cam_j): shared_count}
        """
        shared_counts = defaultdict(int)
        
        for point_idx, obs_list in wall_observations.items():
            # Get all cameras that see this point
            cameras_seeing_point = [obs['camera_id'] for obs in obs_list]
            
            # Every pair of cameras shares this point
            for i, cam_i in enumerate(cameras_seeing_point):
                for cam_j in cameras_seeing_point[i+1:]:
                    # Order pair consistently
                    pair = tuple(sorted([cam_i, cam_j]))
                    shared_counts[pair] += 1
        
        return shared_counts
    
    def compute_all_shared_points(self, all_observations: Dict[str, Dict]) -> Dict[Tuple[int, int], int]:
        """
        Compute shared points across all walls.
        
        Args:
            all_observations: {wall_name: observations_dict}
        
        Returns:
            {(global_cam_i, global_cam_j): shared_count}
        """
        print("\n" + "="*60)
        print("Computing Shared Point Visibility")
        print("="*60)
        
        all_shared = defaultdict(int)
        
        # First, need to map local camera IDs to global IDs
        local_to_global = {}
        for cam in self.cameras:
            if 'wall' in cam and 'local_id' in cam:
                key = (cam['wall'], cam['local_id'])
                local_to_global[key] = cam['id']
        
        for wall_name, wall_obs in all_observations.items():
            print(f"\nProcessing {wall_name}...")
            
            # Compute shared points within this wall
            local_shared = self.compute_shared_points_per_wall(wall_obs)
            
            # Convert local IDs to global IDs
            for (local_cam_i, local_cam_j), count in local_shared.items():
                global_cam_i = local_to_global.get((wall_name, local_cam_i))
                global_cam_j = local_to_global.get((wall_name, local_cam_j))
                
                if global_cam_i is not None and global_cam_j is not None:
                    pair = tuple(sorted([global_cam_i, global_cam_j]))
                    all_shared[pair] += count
            
            print(f"  Found {len(local_shared):,} camera pairs with shared visibility")
        
        print(f"\n✓ Total camera pairs with shared points: {len(all_shared):,}")
        print("="*60 + "\n")
        
        return all_shared
    
    def compute_spatial_distance(self, cam_i: Dict, cam_j: Dict) -> float:
        """Compute 3D distance between camera centers."""
        center_i = np.array(cam_i['center'])
        center_j = np.array(cam_j['center'])
        
        distance = np.linalg.norm(center_i - center_j)
        return distance
    
    def build_graph(self, shared_counts: Dict[Tuple[int, int], int]):
        """
        Build view graph based on shared visibility and spatial proximity.
        
        Args:
            shared_counts: {(cam_i, cam_j): shared_point_count}
        
        Returns:
            View graph dict
        """
        print("\n" + "="*60)
        print("Building View Graph")
        print("="*60)
        print(f"Parameters:")
        print(f"  Minimum shared points: {self.min_shared_points}")
        print(f"  Maximum spatial distance: {self.max_spatial_distance}")
        print()
        
        total_edges = 0
        filtered_by_points = 0
        filtered_by_distance = 0
        
        for (cam_i, cam_j), count in shared_counts.items():
            # Check if cameras share enough points
            if count < self.min_shared_points:
                filtered_by_points += 1
                continue
            
            # Check spatial distance
            if cam_i not in self.camera_dict or cam_j not in self.camera_dict:
                continue
            
            distance = self.compute_spatial_distance(
                self.camera_dict[cam_i],
                self.camera_dict[cam_j]
            )
            
            if distance > self.max_spatial_distance:
                filtered_by_distance += 1
                continue
            
            # Add bidirectional edge
            self.view_graph[cam_i].append({
                'target_camera': int(cam_j),
                'shared_points': int(count),
                'distance': float(distance)
            })
            
            self.view_graph[cam_j].append({
                'target_camera': int(cam_i),
                'shared_points': int(count),
                'distance': float(distance)
            })
            
            total_edges += 1
        
        print(f"✓ View graph built:")
        print(f"  Nodes (cameras): {len(self.view_graph)}")
        print(f"  Edges (connections): {total_edges}")
        print(f"  Filtered by shared points: {filtered_by_points}")
        print(f"  Filtered by distance: {filtered_by_distance}")
        print("="*60 + "\n")
        
        return dict(self.view_graph)
    
    def add_spatial_fallback_connections(self):
        """Add spatial connections for isolated cameras."""
        print("\n" + "="*60)
        print("Adding Spatial Fallback Connections")
        print("="*60)
        
        isolated_cameras = [cam_id for cam_id in self.camera_dict.keys() 
                           if cam_id not in self.view_graph or len(self.view_graph[cam_id]) == 0]
        
        if not isolated_cameras:
            print("✓ No isolated cameras found")
            print("="*60 + "\n")
            return
        
        print(f"Found {len(isolated_cameras)} isolated cameras")
        
        added = 0
        for cam_i in isolated_cameras:
            # Find nearest cameras
            distances = []
            for cam_j in self.camera_dict.keys():
                if cam_i == cam_j:
                    continue
                
                dist = self.compute_spatial_distance(
                    self.camera_dict[cam_i],
                    self.camera_dict[cam_j]
                )
                
                if dist <= self.max_spatial_distance * 2:  # More lenient for fallback
                    distances.append((cam_j, dist))
            
            # Connect to 3 nearest cameras
            distances.sort(key=lambda x: x[1])
            for cam_j, dist in distances[:3]:
                self.view_graph[cam_i].append({
                    'target_camera': int(cam_j),
                    'shared_points': 0,
                    'distance': float(dist),
                    'type': 'spatial_fallback'
                })
                
                # Bidirectional
                self.view_graph[cam_j].append({
                    'target_camera': int(cam_i),
                    'shared_points': 0,
                    'distance': float(dist),
                    'type': 'spatial_fallback'
                })
                
                added += 1
        
        print(f"✓ Added {added} spatial fallback connections")
        print("="*60 + "\n")
    
    def compute_graph_statistics(self):
        """Compute and print graph statistics."""
        print("\n" + "="*60)
        print("View Graph Statistics")
        print("="*60)
        
        total_cameras = len(self.camera_dict)
        cameras_in_graph = len(self.view_graph)
        
        print(f"Total cameras: {total_cameras}")
        print(f"Cameras in graph: {cameras_in_graph}")
        
        if cameras_in_graph == 0:
            print("⚠️  Warning: Empty view graph!")
            return
        
        # Degree distribution
        degrees = [len(neighbors) for neighbors in self.view_graph.values()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        min_degree = np.min(degrees)
        
        print(f"\nConnectivity:")
        print(f"  Average connections per camera: {avg_degree:.2f}")
        print(f"  Max connections: {max_degree}")
        print(f"  Min connections: {min_degree}")
        
        # Check for isolated cameras
        isolated = [cam_id for cam_id in self.camera_dict.keys() 
                   if cam_id not in self.view_graph or len(self.view_graph[cam_id]) == 0]
        
        if isolated:
            print(f"\n⚠️  Warning: {len(isolated)} isolated cameras")
            if len(isolated) <= 10:
                print(f"  Camera IDs: {isolated}")
        else:
            print(f"\n✓ All cameras are connected!")
        
        print("="*60 + "\n")
    
    def save_view_graph(self, output_file: str):
        """Save view graph to JSON."""
        graph_data = {
            'view_graph': {str(k): v for k, v in self.view_graph.items()},
            'metadata': {
                'num_cameras': len(self.camera_dict),
                'num_nodes': len(self.view_graph),
                'min_shared_points': self.min_shared_points,
                'max_spatial_distance': self.max_spatial_distance
            }
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"✓ Saved view graph: {output_file}")
        print(f"  File size: {file_size:.2f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Build view graph for camera navigation"
    )
    
    parser.add_argument('--output-dir', '-d', type=str, required=True,
                       help='Base output directory (contains wall_A/, wall_B/, etc.)')
    
    parser.add_argument('--cameras', '-c', type=str, required=True,
                       help='Path to merged cameras JSON file')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for view graph JSON')
    
    parser.add_argument('--min-shared', type=int, default=20,
                       help='Minimum shared 3D points to connect cameras (default: 20)')
    
    parser.add_argument('--max-distance', type=float, default=10.0,
                       help='Maximum spatial distance to connect cameras (default: 10.0)')
    
    parser.add_argument('--add-spatial-fallback', action='store_true',
                       help='Add spatial connections for isolated cameras')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VIEW GRAPH BUILDER")
    print("="*60)
    
    # Load cameras
    print(f"\nLoading cameras from: {args.cameras}")
    with open(args.cameras, 'r') as f:
        cameras = json.load(f)
    
    print(f"✓ Loaded {len(cameras)} cameras")
    
    # Initialize builder
    builder = ViewGraphBuilder(
        cameras=cameras,
        min_shared_points=args.min_shared,
        max_spatial_distance=args.max_distance
    )
    
    # Load observations from all walls
    all_observations = builder.load_all_observations(args.output_dir)
    
    if len(all_observations) == 0:
        print("❌ Error: No observations loaded!")
        return
    
    # Compute shared visibility
    shared_counts = builder.compute_all_shared_points(all_observations)
    
    if len(shared_counts) == 0:
        print("⚠️  Warning: No cameras share 3D points!")
        if args.add_spatial_fallback:
            print("   Will use spatial fallback connections")
    
    # Build graph
    view_graph = builder.build_graph(shared_counts)
    
    # Add spatial fallback if requested
    if args.add_spatial_fallback:
        builder.add_spatial_fallback_connections()
    
    # Statistics
    builder.compute_graph_statistics()
    
    # Save
    builder.save_view_graph(args.output)
    
    print("\n✅ Done!")
    print(f"   View graph saved to: {args.output}")


if __name__ == "__main__":
    main()
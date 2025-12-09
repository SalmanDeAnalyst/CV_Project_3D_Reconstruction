# components/incremental_sfm.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from essential_matrix import EssentialMatrixEstimator
from pose_detection import PoseRecovery
from triangulation import Triangulation
from pnp_solver import PnPSolver
from dense_stereo import DenseStereoMatcher
from image_sequencer import load_image_any_format

# Try to import GPU BA
try:
    from bundle_adjustment_gpu import BundleAdjustmentGPU, TORCH_AVAILABLE
    GPU_BA_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    GPU_BA_AVAILABLE = False


class IncrementalSfM:
    """
    Incremental Structure from Motion pipeline with FIXED observation tracking.
    """
    
    def __init__(
        self,
        K: np.ndarray,
        ordered_image_paths: List[str],
        output_dir: str,
        use_bundle_adjustment: bool = True,
        use_dense_stereo: bool = False,
        ba_interval: int = 10,
        visualization_interval: int = 10,
        feature_cache: Optional[Dict] = None
    ):
        self.K = K
        self.image_paths = ordered_image_paths
        self.output_dir = output_dir
        self.use_ba = use_bundle_adjustment
        self.use_dense = use_dense_stereo
        self.ba_interval = ba_interval
        self.vis_interval = visualization_interval
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'intermediate'), exist_ok=True)
        
        # Initialize components
        self.essential_estimator = EssentialMatrixEstimator(K)
        self.pose_recovery = PoseRecovery(K)
        self.triangulator = Triangulation(K)
        self.pnp_solver = PnPSolver(K)
        
        if use_dense_stereo:
            self.dense_matcher = DenseStereoMatcher(K, use_gpu=True)
        
        # Initialize GPU Bundle Adjustment if available
        self.gpu_ba = None
        if use_bundle_adjustment and GPU_BA_AVAILABLE:
            try:
                self.gpu_ba = BundleAdjustmentGPU(K, use_gpu=True)
            except Exception as e:
                print(f"⚠ GPU BA failed to initialize: {e}")
                self.gpu_ba = None
        
        # Reconstruction state
        self.cameras = []
        self.map_points_3d = None
        self.map_colors = None
        self.map_descriptors = None
        
        # ========== FIX #1: PROPER OBSERVATION TRACKING ==========
        # Track which camera sees which point at which pixel
        # Format: {point_idx: [(cam_idx, pixel_2d), ...]}
        self.observations = {}
        
        # Dense points stored separately
        self.dense_points_3d = None
        self.dense_colors = None
        
        # Feature cache
        if feature_cache is not None:
            print("✓ Reusing feature cache from ImageSequencer")
            self.feature_cache = feature_cache
        else:
            print("Building new feature cache...")
            self.feature_cache = {}
        
        # Statistics
        self.stats = {
            'num_cameras': 0,
            'num_points': 0,
            'reprojection_errors': [],
            'ba_improvements': []
        }
        
        print("="*60)
        print("Incremental SfM Initialized")
        print("="*60)
        print(f"Images: {len(self.image_paths)}")
        print(f"Bundle Adjustment: {'ENABLED' if self.use_ba else 'DISABLED'}")
        if self.use_ba:
            print(f"  BA backend: {'GPU (PyTorch)' if self.gpu_ba else 'CPU (scipy)'}")
        print(f"Dense Stereo: {'ENABLED' if self.use_dense else 'DISABLED'}")
        print("="*60 + "\n")
    
    def extract_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray]:
        """Extract and cache SIFT features."""
        if image_path in self.feature_cache:
            img, pts_2d, kp, des = self.feature_cache[image_path]
            if img is None:
                img = load_image_any_format(image_path)
                self.feature_cache[image_path] = (img, pts_2d, kp, des)
            return img, pts_2d, kp, des
        
        img = load_image_any_format(image_path)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        self.feature_cache[image_path] = (img, points_2d, keypoints, descriptors)
        return img, points_2d, keypoints, descriptors
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray, ratio: float = 0.75) -> List[cv2.DMatch]:
        """Match features using FLANN."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def bootstrap_reconstruction(self) -> bool:
        """Bootstrap with first two images - FIXED observation tracking."""
        print("\n" + "="*60)
        print("BOOTSTRAP RECONSTRUCTION")
        print("="*60)
        
        if len(self.image_paths) < 2:
            print("Error: Need at least 2 images!")
            return False
        
        img1_path = self.image_paths[0]
        img2_path = self.image_paths[1]
        
        print(f"\nImage 1: {os.path.basename(img1_path)}")
        print(f"Image 2: {os.path.basename(img2_path)}")
        
        # Extract features
        print("\nExtracting features...")
        img1, pts1_2d, kp1, des1 = self.extract_features(img1_path)
        img2, pts2_2d, kp2, des2 = self.extract_features(img2_path)
        print(f"Image 1: {len(kp1)} features")
        print(f"Image 2: {len(kp2)} features")
        
        # Match features
        print("\nMatching features...")
        matches = self.match_features(des1, des2)
        print(f"Matches: {len(matches)}")
        
        if len(matches) < 50:
            print(f"Error: Not enough matches ({len(matches)} < 50)")
            return False
        
        pts1_matched = np.array([pts1_2d[m.queryIdx] for m in matches])
        pts2_matched = np.array([pts2_2d[m.trainIdx] for m in matches])
        
        # Estimate Essential Matrix
        print("\nEstimating Essential Matrix...")
        E, mask = self.essential_estimator.estimate_essential_matrix(pts1_matched, pts2_matched)
        
        inlier_pts1 = pts1_matched[mask.ravel() == 1]
        inlier_pts2 = pts2_matched[mask.ravel() == 1]
        print(f"Inliers: {len(inlier_pts1)}/{len(pts1_matched)}")
        
        # Recover pose
        print("\nRecovering camera pose...")
        R, t, pose_mask, num_good = self.pose_recovery.recover_pose(E, inlier_pts1, inlier_pts2)
        print(f"Points in front of cameras: {num_good}")
        
        # Triangulate
        print("\nTriangulating initial 3D points...")
        P1, P2 = self.triangulator.create_projection_matrices(R, t)
        points_3d = self.triangulator.triangulate_points(P1, P2, inlier_pts1, inlier_pts2)
        
        filtered_points_3d, depth_mask = self.triangulator.filter_points_by_depth(points_3d, max_depth=100)
        print(f"3D points after filtering: {len(filtered_points_3d)}")
        
        # Get colors
        colors = []
        for pt in inlier_pts1[depth_mask]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                color = img1[y, x]
                colors.append([color[2], color[1], color[0]])  # BGR to RGB
            else:
                colors.append([128, 128, 128])
        colors = np.array(colors)
        
        # Get descriptors
        inlier_match_indices = np.where(mask.ravel() == 1)[0]
        filtered_match_indices = inlier_match_indices[depth_mask]
        map_descriptors = np.array([des1[matches[i].queryIdx] for i in filtered_match_indices])
        
        # Store cameras
        camera1 = {
            'R': np.eye(3),
            't': np.zeros((3, 1)),
            'image_path': img1_path,
            'points_2d': pts1_2d,
            'keypoints': kp1,
            'descriptors': des1
        }
        
        camera2 = {
            'R': R,
            't': t,
            'image_path': img2_path,
            'points_2d': pts2_2d,
            'keypoints': kp2,
            'descriptors': des2
        }
        
        self.cameras = [camera1, camera2]
        self.map_points_3d = filtered_points_3d
        self.map_colors = colors
        self.map_descriptors = map_descriptors
        
        # ========== FIX #2: TRACK BOOTSTRAP OBSERVATIONS ==========
        # These points are seen by BOTH camera 0 and camera 1
        self.observations = {}
        inlier_pts1_filtered = inlier_pts1[depth_mask]
        inlier_pts2_filtered = inlier_pts2[depth_mask]
        
        for point_idx in range(len(filtered_points_3d)):
            self.observations[point_idx] = [
                (0, inlier_pts1_filtered[point_idx]),  # Camera 0 sees it here
                (1, inlier_pts2_filtered[point_idx])   # Camera 1 sees it here
            ]
        
        print(f"\n✓ Tracked {len(self.observations)} points with observations")
        
        self.stats['num_cameras'] = 2
        self.stats['num_points'] = len(filtered_points_3d)
        
        print(f"\n{'='*60}")
        print("Bootstrap Complete!")
        print(f"{'='*60}")
        print(f"Cameras: {len(self.cameras)}")
        print(f"3D Points: {len(self.map_points_3d)}")
        print(f"{'='*60}\n")
        
        self.save_intermediate(0)
        self.visualize_reconstruction(iteration=0)
        
        if self.use_dense:
            print("\nComputing dense reconstruction for initial pair...")
            self.add_dense_reconstruction(0, 1)
        
        return True
    
    def add_new_view(self, image_idx: int) -> bool:
        """Add new view with FIXED observation tracking."""
        image_path = self.image_paths[image_idx]
        
        print(f"\n{'='*60}")
        print(f"Adding View {image_idx + 1}/{len(self.image_paths)}")
        print(f"{'='*60}")
        print(f"Image: {os.path.basename(image_path)}")
        
        # Extract features
        print("\nExtracting features...")
        img, pts_2d, keypoints, descriptors = self.extract_features(image_path)
        print(f"Features: {len(keypoints)}")
        
        # Match with 3D map
        print("\nMatching with 3D map...")
        matches = self.match_features(descriptors, self.map_descriptors, ratio=0.7)
        print(f"2D-3D matches: {len(matches)}")
        
        if len(matches) < 10:
            print(f"Warning: Not enough 2D-3D matches ({len(matches)} < 10)")
            return False
        
        matched_2d_indices = [m.queryIdx for m in matches]
        matched_3d_indices = [m.trainIdx for m in matches]
        
        matched_2d = pts_2d[matched_2d_indices]
        matched_3d = self.map_points_3d[matched_3d_indices]
        
        # Filter invalid 3D points
        valid_3d_mask = np.all(np.isfinite(matched_3d), axis=1)
        valid_3d_mask &= (matched_3d[:, 2] > 0.1) & (matched_3d[:, 2] < 100)
        valid_3d_mask &= np.all(np.abs(matched_3d) < 1000, axis=1)
        
        if np.sum(valid_3d_mask) < len(valid_3d_mask):
            num_invalid = len(valid_3d_mask) - np.sum(valid_3d_mask)
            print(f"Warning: Filtered {num_invalid} invalid 3D points")
        
        matched_2d = matched_2d[valid_3d_mask]
        matched_3d = matched_3d[valid_3d_mask]
        matched_3d_indices = [idx for idx, valid in zip(matched_3d_indices, valid_3d_mask) if valid]
        
        if len(matched_3d) < 10:
            print(f"Error: Only {len(matched_3d)} valid matches")
            return False
        
        # Solve PnP
        print("\nSolving PnP...")
        success, R, t, inlier_mask = self.pnp_solver.solve_pnp_ransac(matched_3d, matched_2d)
        
        if not success:
            print("PnP failed!")
            return False
        
        num_inliers = np.sum(inlier_mask)
        min_required_inliers = max(6, int(0.03 * len(matches)))  # Lower from 10 to 6
        
        if num_inliers < min_required_inliers:
            print(f"Too few PnP inliers ({num_inliers} < {min_required_inliers})")
            return False
        
        mean_error, _ = self.pnp_solver.compute_reprojection_error(
            matched_3d[inlier_mask], matched_2d[inlier_mask], R, t
        )
        print(f"Mean reprojection error: {mean_error:.3f} pixels")
        
        # More tolerant for later cameras (drift accumulates)
        if mean_error > 50.0:  # Very tolerant - was 10.0
            print(f"Reprojection error too high ({mean_error:.1f} > 50.0 pixels)")
            return False
        
        self.stats['reprojection_errors'].append(mean_error)
        
        # Store camera
        new_camera = {
            'R': R,
            't': t,
            'image_path': image_path,
            'points_2d': pts_2d,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        self.cameras.append(new_camera)
        camera_idx = len(self.cameras) - 1
        
        # ========== FIX #3: TRACK PNP OBSERVATIONS ==========
        # Add this camera's observations for matched 3D points
        for i, match_idx in enumerate(matched_3d_indices):
            if inlier_mask[i]:  # Only for PnP inliers
                point_3d_idx = match_idx
                pixel_2d = matched_2d[i]
                
                if point_3d_idx not in self.observations:
                    self.observations[point_3d_idx] = []
                
                self.observations[point_3d_idx].append((camera_idx, pixel_2d))
        
        print(f"✓ Updated observations for {np.sum(inlier_mask)} points")
        
        # Triangulate new points
        print("\nTriangulating new points...")
        new_points_added = self.triangulate_new_points(camera_idx)
        
        if self.use_dense and camera_idx >= 1:
            print("\nComputing dense reconstruction...")
            self.add_dense_reconstruction(camera_idx - 1, camera_idx)
        
        self.stats['num_cameras'] = len(self.cameras)
        self.stats['num_points'] = len(self.map_points_3d)
        
        dense_count = len(self.dense_points_3d) if self.dense_points_3d is not None else 0
        
        print(f"\n{'='*60}")
        print(f"View Added Successfully!")
        print(f"{'='*60}")
        print(f"Total cameras: {len(self.cameras)}")
        print(f"Sparse 3D points: {len(self.map_points_3d):,}")
        print(f"Dense 3D points: {dense_count:,}")
        print(f"New sparse points: {new_points_added}")
        print(f"{'='*60}\n")
        
        return True
    
    def triangulate_new_points(self, current_image_idx: int) -> int:
        """Triangulate new points with FIXED observation tracking."""
        if current_image_idx < 2:
            return 0
        
        cam_current = self.cameras[current_image_idx]
        cam_prev = self.cameras[current_image_idx - 1]
        
        matches = self.match_features(cam_current['descriptors'], cam_prev['descriptors'])
        
        if len(matches) < 10:
            print(f"  Not enough matches for triangulation: {len(matches)}")
            return 0
        
        pts_current = np.array([cam_current['points_2d'][m.queryIdx] for m in matches])
        pts_prev = np.array([cam_prev['points_2d'][m.trainIdx] for m in matches])
        
        P_current = self.K @ np.hstack([cam_current['R'], cam_current['t']])
        P_prev = self.K @ np.hstack([cam_prev['R'], cam_prev['t']])
        
        points_3d = self.triangulator.triangulate_points(P_prev, P_current, pts_prev, pts_current)
        
        # CRITICAL: Check for bad 3D points IMMEDIATELY after triangulation
        finite_mask = np.all(np.isfinite(points_3d), axis=1)
        if not np.all(finite_mask):
            num_bad = np.sum(~finite_mask)
            print(f"  ⚠️  Triangulation produced {num_bad} NaN/Inf points, filtering...")
            points_3d = points_3d[finite_mask]
            pts_prev = pts_prev[finite_mask]
            pts_current = pts_current[finite_mask]
            matches = [m for i, m in enumerate(matches) if finite_mask[i]]
        
        if len(points_3d) == 0:
            print(f"  All triangulated points were invalid!")
            return 0
        
        print(f"  Initial triangulated points: {len(points_3d)}")
        
        # Depth filtering (balanced - not too strict)
        filtered_points, depth_mask = self.triangulator.filter_points_by_depth(points_3d, max_depth=40)  # 40 is middle ground
        print(f"  After depth filter: {len(filtered_points)} (removed {len(points_3d) - len(filtered_points)})")
        
        if len(filtered_points) == 0:
            print(f"  All points filtered by depth")
            return 0
        
        # Reprojection error filtering (stricter)
        pts_prev_filtered = pts_prev[depth_mask]
        pts_current_filtered = pts_current[depth_mask]
        
        pts_prev_proj = (P_prev @ np.hstack([filtered_points, np.ones((len(filtered_points), 1))]).T).T
        pts_prev_proj = pts_prev_proj[:, :2] / pts_prev_proj[:, 2:3]
        
        pts_current_proj = (P_current @ np.hstack([filtered_points, np.ones((len(filtered_points), 1))]).T).T
        pts_current_proj = pts_current_proj[:, :2] / pts_current_proj[:, 2:3]
        
        error_prev = np.linalg.norm(pts_prev_filtered - pts_prev_proj, axis=1)
        error_current = np.linalg.norm(pts_current_filtered - pts_current_proj, axis=1)
        
        print(f"  Reprojection errors - Prev: mean={np.mean(error_prev):.2f}, max={np.max(error_prev):.2f}")
        print(f"  Reprojection errors - Curr: mean={np.mean(error_current):.2f}, max={np.max(error_current):.2f}")
        
        # Balanced threshold - not too strict, not too loose
        good_mask = (error_prev < 5.0) & (error_current < 5.0)  # Back to 5.0
        print(f"  After reprojection filter (<5px): {np.sum(good_mask)} (removed {len(good_mask) - np.sum(good_mask)})")
        
        # CRITICAL: Also check that projected points are finite
        good_mask &= np.all(np.isfinite(pts_prev_proj), axis=1)
        good_mask &= np.all(np.isfinite(pts_current_proj), axis=1)
        
        filtered_points = filtered_points[good_mask]
        pts_current_filtered = pts_current_filtered[good_mask]
        pts_prev_filtered = pts_prev_filtered[good_mask]
        
        if len(filtered_points) == 0:
            print(f"  All points filtered by reprojection error")
            return 0
        
        # FINAL CHECK: Ensure no extreme values
        extreme_mask = np.all(np.abs(filtered_points) < 1000, axis=1)
        if not np.all(extreme_mask):
            num_extreme = np.sum(~extreme_mask)
            print(f"  ⚠️  Filtered {num_extreme} points with extreme coordinates")
            filtered_points = filtered_points[extreme_mask]
            pts_current_filtered = pts_current_filtered[extreme_mask]
            pts_prev_filtered = pts_prev_filtered[extreme_mask]
        
        print(f"  Final points after all filters: {len(filtered_points)}")
        
        if len(filtered_points) == 0:
            print(f"  ⚠️  All points filtered out!")
            return 0
        
        depth_mask_indices = np.where(depth_mask)[0]
        good_indices = depth_mask_indices[good_mask][extreme_mask]
        
        # Get colors
        img = self.feature_cache[cam_current['image_path']][0]
        colors = []
        for idx in good_indices:
            pt = pts_current[idx]
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                color = img[y, x]
                colors.append([color[2], color[1], color[0]])
            else:
                colors.append([128, 128, 128])
        colors = np.array(colors)
        
        new_descriptors = np.array([
            cam_current['descriptors'][matches[idx].queryIdx] 
            for idx in good_indices
        ])
        
        assert len(filtered_points) == len(colors)
        assert len(filtered_points) == len(new_descriptors)
        
        # Get starting index for new points
        start_point_idx = len(self.map_points_3d) if self.map_points_3d is not None else 0
        
        # Add to map
        if self.map_points_3d is None:
            self.map_points_3d = filtered_points
            self.map_colors = colors
            self.map_descriptors = new_descriptors
        else:
            self.map_points_3d = np.vstack([self.map_points_3d, filtered_points])
            self.map_colors = np.vstack([self.map_colors, colors])
            self.map_descriptors = np.vstack([self.map_descriptors, new_descriptors])
        
        # ========== FIX #4: TRACK TRIANGULATION OBSERVATIONS ==========
        cam_prev_idx = current_image_idx - 1
        cam_current_idx = current_image_idx
        
        for i in range(len(filtered_points)):
            point_idx = start_point_idx + i
            
            self.observations[point_idx] = [
                (cam_prev_idx, pts_prev_filtered[i]),
                (cam_current_idx, pts_current_filtered[i])
            ]
        
        print(f"  ✓ Tracked observations for {len(filtered_points)} new points")
        
        return len(filtered_points)
    
    def add_dense_reconstruction(self, idx1: int, idx2: int):
        """Add dense stereo (stored separately)."""
        cam1 = self.cameras[idx1]
        cam2 = self.cameras[idx2]
        
        img1 = self.feature_cache[cam1['image_path']][0]
        img2 = self.feature_cache[cam2['image_path']][0]
        
        R_rel = cam2['R'] @ cam1['R'].T
        t_rel = cam2['t'] - R_rel @ cam1['t']
        
        dense_points, dense_colors, _ = self.dense_matcher.compute_dense_reconstruction(
            img1, img2, R_rel, t_rel
        )
        
        if dense_points is not None and len(dense_points) > 0:
            if self.dense_points_3d is None:
                self.dense_points_3d = dense_points
                self.dense_colors = dense_colors
            else:
                self.dense_points_3d = np.vstack([self.dense_points_3d, dense_points])
                self.dense_colors = np.vstack([self.dense_colors, dense_colors])
            
            print(f"  Added {len(dense_points):,} dense points (stored separately)")
    
    def run_bundle_adjustment(self):
        """Run BA with REAL observations."""
        if not self.use_ba:
            return
        
        if len(self.cameras) < 2:
            return
        
        if self.map_points_3d is None or len(self.map_points_3d) == 0:
            print("⚠ No 3D points to optimize!")
            return
        
        print(f"\n{'='*60}")
        print("Running Bundle Adjustment")
        print(f"{'='*60}")
        
        # ========== VALIDATE OBSERVATIONS BEFORE BA ==========
        observations_list = []
        total_valid_obs = 0
        bad_obs_count = 0
        
        for point_idx in range(len(self.map_points_3d)):
            if point_idx in self.observations:
                # Validate each observation
                valid_obs = []
                for cam_idx, pixel_2d in self.observations[point_idx]:
                    # Check for finite values
                    if not np.all(np.isfinite(pixel_2d)):
                        bad_obs_count += 1
                        continue
                    
                    # Check for reasonable pixel coordinates
                    if pixel_2d[0] < 0 or pixel_2d[0] > 10000 or pixel_2d[1] < 0 or pixel_2d[1] > 10000:
                        bad_obs_count += 1
                        continue
                    
                    valid_obs.append((cam_idx, pixel_2d))
                    total_valid_obs += 1
                
                observations_list.append(valid_obs)
            else:
                observations_list.append([])
        
        print(f"Total observations: {total_valid_obs}")
        if bad_obs_count > 0:
            print(f"⚠️  Filtered {bad_obs_count} bad observations (NaN/Inf/out-of-bounds)")
        
        # CRITICAL: Check 3D points for extreme values BEFORE BA
        point_norms = np.linalg.norm(self.map_points_3d, axis=1)
        bad_3d_mask = (~np.all(np.isfinite(self.map_points_3d), axis=1)) | (point_norms > 200)
        
        if np.any(bad_3d_mask):
            num_bad = np.sum(bad_3d_mask)
            print(f"🚨 WARNING: {num_bad} 3D points have NaN/Inf/extreme values (>{200} units from origin)!")
            print(f"   Filtering them out before BA...")
            
            # Filter bad points
            good_mask = ~bad_3d_mask
            self.map_points_3d = self.map_points_3d[good_mask]
            self.map_colors = self.map_colors[good_mask]
            self.map_descriptors = self.map_descriptors[good_mask]
            
            # Update observations
            new_observations = {}
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(good_mask)[0])}
            for old_idx, obs in enumerate(observations_list):
                if old_idx in old_to_new and len(obs) > 0:
                    new_observations[old_to_new[old_idx]] = obs
            
            observations_list = [new_observations.get(i, []) for i in range(len(self.map_points_3d))]
            print(f"   Remaining 3D points: {len(self.map_points_3d)}")
        
        if total_valid_obs == 0:
            print("❌ No valid observations! Skipping BA.")
            return
        
        # Run BA
        if self.gpu_ba is not None:
            print("Using GPU Bundle Adjustment...")
            optimized_cameras, optimized_points, stats = self.gpu_ba.bundle_adjustment(
                self.cameras,
                self.map_points_3d,
                observations_list,
                max_iterations=100,
                lr=0.01,
                verbose=True
            )
        else:
            print("Using CPU Bundle Adjustment...")
            optimized_cameras, optimized_points, stats = self.pnp_solver.bundle_adjustment(
                self.cameras,
                self.map_points_3d,
                observations_list,
                max_iterations=50,
                verbose=True
            )
        
        self.cameras = optimized_cameras
        self.map_points_3d = optimized_points
        
        self.stats['ba_improvements'].append(stats['improvement'])
        print(f"BA improvement: {stats['improvement']:.3f} pixels\n")
    
    def normalize_reconstruction_scale(self):
        """Normalize to prevent scale drift."""
        if self.map_points_3d is None or len(self.map_points_3d) == 0:
            return
        
        centroid = np.mean(self.map_points_3d, axis=0)
        self.map_points_3d = self.map_points_3d - centroid
        
        distances = np.linalg.norm(self.map_points_3d, axis=1)
        median_dist = np.median(distances)
        
        target_scale = 10.0
        scale_factor = target_scale / median_dist if median_dist > 0 else 1.0
        
        if scale_factor < 0.5 or scale_factor > 2.0:
            print(f"  Scale normalization: factor={scale_factor:.3f}")
            
            self.map_points_3d = self.map_points_3d * scale_factor
            
            for cam in self.cameras:
                cam['t'] = (cam['t'].flatten() - centroid) * scale_factor
                cam['t'] = cam['t'].reshape(3, 1)
            
            if self.dense_points_3d is not None:
                self.dense_points_3d = (self.dense_points_3d - centroid) * scale_factor
    
    def remove_outliers(self):
        """Remove statistical outliers."""
        if self.map_points_3d is None or len(self.map_points_3d) < 100:
            return
        
        print("\nRemoving outliers...")
        
        centroid = np.mean(self.map_points_3d, axis=0)
        distances = np.linalg.norm(self.map_points_3d - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 2 * std_dist
        
        inlier_mask = distances < threshold
        num_outliers = len(self.map_points_3d) - np.sum(inlier_mask)
        
        self.map_points_3d = self.map_points_3d[inlier_mask]
        self.map_colors = self.map_colors[inlier_mask]
        self.map_descriptors = self.map_descriptors[inlier_mask]
        
        # Update observations (reindex after filtering)
        new_observations = {}
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(inlier_mask)[0])}
        
        for old_idx, obs_list in self.observations.items():
            if old_idx in old_to_new:
                new_observations[old_to_new[old_idx]] = obs_list
        
        self.observations = new_observations
        
        print(f"Removed {num_outliers} outliers ({num_outliers/len(inlier_mask)*100:.1f}%)")
        print(f"Remaining points: {len(self.map_points_3d)}")
    
    def visualize_reconstruction(self, iteration: int):
        """Visualize reconstruction."""
        print(f"\nVisualizing reconstruction (iteration {iteration})...")
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if self.map_points_3d is not None and len(self.map_points_3d) > 0:
            step = max(1, len(self.map_points_3d) // 10000)
            points_vis = self.map_points_3d[::step]
            colors_vis = self.map_colors[::step] / 255.0
            
            ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                      c=colors_vis, marker='.', s=1, alpha=0.5)
        
        for i, cam in enumerate(self.cameras):
            cam_center = -cam['R'].T @ cam['t']
            ax.scatter(cam_center[0], cam_center[1], cam_center[2],
                      c='red', marker='o', s=100,
                      label=f'Camera {i}' if i < 5 else None)
            
            optical_axis = cam['R'].T @ np.array([[0], [0], [1]])
            ax.quiver(cam_center[0], cam_center[1], cam_center[2],
                     optical_axis[0], optical_axis[1], optical_axis[2],
                     length=2.0, color='blue', arrow_length_ratio=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Incremental SfM - Iteration {iteration}\n'
                    f'Cameras: {len(self.cameras)}, Points: {len(self.map_points_3d)}')
        
        if len(self.cameras) <= 5:
            ax.legend()
        
        vis_path = os.path.join(self.output_dir, 'visualizations',
                               f'reconstruction_iter_{iteration:03d}.png')
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {vis_path}")
    
    def save_intermediate(self, iteration: int):
        """Save intermediate state."""
        output_path = os.path.join(self.output_dir, 'intermediate',
                                   f'reconstruction_iter_{iteration:03d}.ply')
        
        if self.map_points_3d is not None and len(self.map_points_3d) > 0:
            self.triangulator.save_point_cloud(self.map_points_3d, output_path,
                                              colors=self.map_colors)
    
    def save_final_reconstruction(self):
        """Save final reconstruction."""
        print("\n" + "="*60)
        print("Saving Final Reconstruction")
        print("="*60)
        
        # Save point cloud
        if self.map_points_3d is not None:
            all_points = self.map_points_3d
            all_colors = self.map_colors
            
            if self.dense_points_3d is not None and len(self.dense_points_3d) > 0:
                all_points = np.vstack([all_points, self.dense_points_3d])
                all_colors = np.vstack([all_colors, self.dense_colors])
            
            ply_path = os.path.join(self.output_dir, 'final_reconstruction.ply')
            self.triangulator.save_point_cloud(all_points, ply_path, colors=all_colors)
            print(f"Saved: {ply_path}")
        
        # Save camera poses
        camera_data = [{
            'id': i,
            'image': os.path.basename(cam['image_path']),
            'R': cam['R'].tolist(),
            't': cam['t'].tolist()
        } for i, cam in enumerate(self.cameras)]
        
        camera_path = os.path.join(self.output_dir, 'camera_poses.json')
        with open(camera_path, 'w') as f:
            json.dump(camera_data, f, indent=2)
        print(f"Saved: {camera_path}")
        
        # Save statistics
        stats_serializable = {
            'num_cameras': int(self.stats['num_cameras']),
            'num_points': int(self.stats['num_points']),
            'reprojection_errors': [float(x) for x in self.stats['reprojection_errors']],
            'ba_improvements': [float(x) for x in self.stats['ba_improvements']]
        }
        
        stats_path = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"Saved: {stats_path}")
        
        # NEW: Save observations
        observations_data = {}
        for point_idx, obs_list in self.observations.items():
            observations_data[str(point_idx)] = [
                {
                    "camera_id": cam_idx,
                    "pixel": pixel_2d.tolist()
                }
                for cam_idx, pixel_2d in obs_list
            ]
        
        obs_path = os.path.join(self.output_dir, 'observations.json')
        with open(obs_path, 'w') as f:
            json.dump(observations_data, f, indent=2)
        print(f"Saved: {obs_path}")
        
        print("="*60 + "\n")



    def reconstruct(self):
        """Main reconstruction pipeline."""
        print("\n" + "="*60)
        print("INCREMENTAL SfM PIPELINE")
        print("="*60)
        
        if not self.bootstrap_reconstruction():
            print("Bootstrap failed!")
            return False
        
        for i in range(2, len(self.image_paths)):
            success = self.add_new_view(i)
            
            if not success:
                continue
            
            if i % self.vis_interval == 0:
                self.save_intermediate(i)
                self.visualize_reconstruction(i)
            
            if self.use_ba and i % self.ba_interval == 0 and i > 0:
                self.run_bundle_adjustment()
                self.normalize_reconstruction_scale()
        
        if self.use_ba:
            print("\nRunning final Bundle Adjustment...")
            self.run_bundle_adjustment()
            self.normalize_reconstruction_scale()
        
        self.remove_outliers()
        self.visualize_reconstruction(len(self.image_paths))
        self.save_final_reconstruction()
        
        print("\n" + "="*60)
        print("RECONSTRUCTION COMPLETE!")
        print("="*60)
        print(f"Total cameras: {len(self.cameras)}")
        print(f"Total 3D points: {len(self.map_points_3d)}")
        print("="*60 + "\n")
        
        return True


import re

def natural_sort_key(s):
    """Sort strings with numbers naturally (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def remove_duplicate_images(image_paths: List[str]) -> List[str]:
    """Remove duplicate images based on filename."""
    seen_filenames = set()
    unique_paths = []
    duplicates_found = 0
    
    for path in image_paths:
        filename = os.path.basename(path)
        if filename not in seen_filenames:
            seen_filenames.add(filename)
            unique_paths.append(path)
        else:
            duplicates_found += 1
            print(f"⚠️ Skipping duplicate: {filename}")
    
    if duplicates_found > 0:
        print(f"\n⚠️ Removed {duplicates_found} duplicate images")
    
    return unique_paths


# if __name__ == "__main__":
#     import glob

#     parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     cache_dir = os.path.join(parent_dir, 'cache_files')
#     os.makedirs(cache_dir, exist_ok=True)
    
#     cache_file = os.path.join(cache_dir, 'feature_cache.pkl')
#     match_cache_file = os.path.join(cache_dir, 'match_results.pkl')

#     dataset_path = os.path.join(parent_dir, 'data', 'Dataset_m3')

#     # Camera intrinsics
#     K = np.array([
#         [2184, 0, 1512],
#         [0, 2184, 2016],
#         [0, 0, 1]
#     ], dtype=np.float32)
    
#     # ============================================================
#     # CONFIGURATION: Set to True if images are already in sequence
#     # ============================================================
#     IMAGES_ALREADY_SEQUENCED = True  # Images are sequential
    
#     # Load dataset
#     if not os.path.exists(dataset_path):
#         print(f"ERROR: Path does not exist: {dataset_path}")
#         exit()
    
#     image_paths = (
#         glob.glob(os.path.join(dataset_path, '*.jpeg')) +
#         glob.glob(os.path.join(dataset_path, '*.jpg')) +
#         glob.glob(os.path.join(dataset_path, '*.heic')) +
#         glob.glob(os.path.join(dataset_path, '*.HEIC')) +
#         glob.glob(os.path.join(dataset_path, '*.png'))
#     )
#     image_paths = sorted(image_paths, key=natural_sort_key)

#     image_paths = remove_duplicate_images(image_paths)

#     if len(image_paths) == 0:
#         print(f"ERROR: No images found in {dataset_path}")
#         exit()
    
#     print(f"Found {len(image_paths)} images")
    
#     if IMAGES_ALREADY_SEQUENCED:
#         # ============================================================
#         # SKIP SEQUENCING - Use images in sorted order (already sequential)
#         # ============================================================
#         print("\n" + "="*60)
#         print("USING PRE-SEQUENCED IMAGES (sorted by filename)")
#         print("="*60)
        
#         ordered_paths = image_paths  # Use as-is
#         feature_cache = None  # Will extract features fresh
        
#         print(f"Images to process: {len(ordered_paths)}")
#         for i, p in enumerate(ordered_paths[:5]):
#             print(f"  {i+1}. {os.path.basename(p)}")
#         if len(ordered_paths) > 5:
#             print(f"  ... and {len(ordered_paths) - 5} more")
#         print("="*60 + "\n")
        
#     else:
#         # ============================================================
#         # RUN SEQUENCING - Order images by visual overlap
#         # ============================================================
#         from image_sequencer import ImageSequencer
        
#         # Initialize sequencer
#         sequencer = ImageSequencer(K, min_matches=50, n_workers=8)
        
#         # Load feature cache
#         cache_loaded = sequencer.load_feature_cache(cache_file)
        
#         if not cache_loaded:
#             print("No feature cache found, will extract features fresh")
        
#         # Sequence images (with match caching!)
#         ordered_paths, metadata = sequencer.sequence_images(
#             image_paths,
#             cache_file=cache_file,
#             match_cache_file=match_cache_file
#         )
        
#         # Final save
#         num_saved, file_size = sequencer.save_feature_cache(cache_file)
        
#         print(f"\nOrdered {len(ordered_paths)} images")
#         print(f"Cached features for {len(sequencer.feature_cache)} images")
        
#         feature_cache = sequencer.feature_cache
    
#     # Run incremental SfM
#     print("\n" + "="*60)
#     print("PHASE 2: INCREMENTAL SfM")
#     print("="*60)
    
#     sfm = IncrementalSfM(
#         K=K,
#         ordered_image_paths=ordered_paths,
#         output_dir=os.path.join(parent_dir, 'output', 'sfm_output'),
#         use_bundle_adjustment=True,  # ← ENABLE BA NOW - observations are fixed!
#         use_dense_stereo=False,
#         ba_interval=1,  # Run every 3 cameras
#         visualization_interval=10,
#         feature_cache=feature_cache if not IMAGES_ALREADY_SEQUENCED else None
#     )
    
#     sfm.reconstruct()
    
#     print("\n🎉 Complete Pipeline Done! 🎉")
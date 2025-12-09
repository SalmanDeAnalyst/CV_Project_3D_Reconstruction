# components/pnp_solver.py

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class PnPSolver:
    """
    Camera pose estimation from 2D-3D correspondences using PnP.
    Includes optional Bundle Adjustment for global optimization.
    """
    
    def __init__(self, K: np.ndarray):
        """
        Initialize PnP Solver.
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        print("="*60)
        print("PnP Solver Initialized")
        print("="*60)
        print(f"Camera intrinsics:\n{K}\n")
    
    def solve_pnp_ransac(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve PnP using RANSAC for robust pose estimation.
        
        Args:
            points_3d: Nx3 array of 3D points in world coordinates
            points_2d: Nx2 array of corresponding 2D points in image
            initial_guess: Optional (R, t) tuple for iterative refinement
        
        Returns:
            success: Whether PnP succeeded
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            inliers: Boolean mask of inlier correspondences
        """
        
        if len(points_3d) < 4:
            print(f"Warning: Need at least 4 points for PnP, got {len(points_3d)}")
            return False, None, None, None
        
        # Convert to correct format
        points_3d = points_3d.reshape(-1, 3).astype(np.float32)
        points_2d = points_2d.reshape(-1, 2).astype(np.float32)
        
        # Initial guess (optional)
        rvec_init = None
        tvec_init = None
        use_extrinsic_guess = False
        
        if initial_guess is not None:
            R_init, t_init = initial_guess
            rvec_init, _ = cv2.Rodrigues(R_init)
            tvec_init = t_init.reshape(3, 1)
            use_extrinsic_guess = True
        
        # Try multiple PnP methods for robustness
        pnp_methods = [
            (cv2.SOLVEPNP_EPNP, "EPNP"),
            (cv2.SOLVEPNP_P3P, "P3P"),
            (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
        ]
        
        success = False
        for method_flag, method_name in pnp_methods:
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=points_3d,
                    imagePoints=points_2d,
                    cameraMatrix=self.K,
                    distCoeffs=None,
                    rvec=rvec_init,
                    tvec=tvec_init,
                    useExtrinsicGuess=use_extrinsic_guess,
                    iterationsCount=2000,
                    reprojectionError=12.0,  # More lenient threshold
                    confidence=0.99,
                    flags=method_flag
                )
                
                if success and inliers is not None and len(inliers) >= 6:
                    # print(f"PnP succeeded with {method_name}")
                    break
            except cv2.error:
                continue
        
        if not success or inliers is None or len(inliers) < 6:
            print(f"PnP RANSAC failed! (tried {len(pnp_methods)} methods)")
            return False, None, None, None
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        
        # Create inlier mask
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        if inliers is not None:
            inlier_mask[inliers.flatten()] = True
        
        num_inliers = np.sum(inlier_mask)
        inlier_ratio = num_inliers / len(points_3d) * 100
        
        print(f"PnP RANSAC: {num_inliers}/{len(points_3d)} inliers ({inlier_ratio:.1f}%)")
        
        return success, R, t, inlier_mask
    
    def compute_reprojection_error(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute reprojection error for a set of 3D-2D correspondences.
        
        Args:
            points_3d: Nx3 array of 3D points
            points_2d: Nx2 array of 2D points
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        
        Returns:
            mean_error: Mean reprojection error in pixels
            errors: Per-point reprojection errors
        """
        
        # Project 3D points to 2D
        rvec, _ = cv2.Rodrigues(R)
        projected_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 3),
            rvec,
            t,
            self.K,
            None
        )
        projected_2d = projected_2d.reshape(-1, 2)
        
        # Compute errors
        errors = np.linalg.norm(points_2d - projected_2d, axis=1)
        mean_error = np.mean(errors)
        
        return mean_error, errors
    
    def bundle_adjustment(
        self,
        camera_params: List[Dict],
        points_3d: np.ndarray,
        observations: List[List[Tuple[int, int]]],
        max_iterations: int = 100,
        verbose: bool = True
    ) -> Tuple[List[Dict], np.ndarray, Dict]:
        """
        Global Bundle Adjustment using Levenberg-Marquardt.
        
        Refines all camera poses and 3D point positions to minimize
        total reprojection error across all observations.
        
        Args:
            camera_params: List of camera dicts with 'R', 't', 'points_2d'
            points_3d: Nx3 array of 3D map points
            observations: For each 3D point, list of (camera_idx, point_2d_idx) tuples
            max_iterations: Maximum optimization iterations
            verbose: Print progress
        
        Returns:
            optimized_cameras: Updated camera parameters
            optimized_points_3d: Updated 3D points
            stats: Optimization statistics
        """
        
        if verbose:
            print("\n" + "="*60)
            print("Bundle Adjustment")
            print("="*60)
            print(f"Cameras: {len(camera_params)}")
            print(f"3D points: {len(points_3d)}")
            print(f"Observations: {sum(len(obs) for obs in observations)}")
        
        # Pack parameters into single vector
        x0 = self._pack_ba_params(camera_params, points_3d)
        
        # Build observation structure
        n_cameras = len(camera_params)
        n_points = len(points_3d)
        
        # Compute initial error
        initial_error = self._compute_ba_residuals(
            x0, n_cameras, n_points, camera_params, observations
        )
        initial_rmse = np.sqrt(np.mean(initial_error**2))
        
        if verbose:
            print(f"Initial RMSE: {initial_rmse:.3f} pixels")
            print("\nOptimizing...")
        
        # Sparse Levenberg-Marquardt optimization
        result = least_squares(
            fun=self._compute_ba_residuals,
            x0=x0,
            args=(n_cameras, n_points, camera_params, observations),
            jac_sparsity=self._build_sparsity_structure(n_cameras, n_points, observations),
            verbose=2 if verbose else 0,
            max_nfev=max_iterations,
            ftol=1e-4,
            method='trf'
        )
        
        # Unpack optimized parameters
        optimized_cameras, optimized_points_3d = self._unpack_ba_params(
            result.x, n_cameras, n_points, camera_params
        )
        
        # Compute final error
        final_error = result.fun
        final_rmse = np.sqrt(np.mean(final_error**2))
        
        stats = {
            'initial_rmse': initial_rmse,
            'final_rmse': final_rmse,
            'iterations': result.nfev,
            'success': result.success,
            'improvement': initial_rmse - final_rmse
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Bundle Adjustment Complete")
            print(f"{'='*60}")
            print(f"Initial RMSE: {initial_rmse:.3f} pixels")
            print(f"Final RMSE: {final_rmse:.3f} pixels")
            print(f"Improvement: {stats['improvement']:.3f} pixels")
            print(f"Iterations: {stats['iterations']}")
            print(f"Success: {result.success}")
            print(f"{'='*60}\n")
        
        return optimized_cameras, optimized_points_3d, stats
    
    def _pack_ba_params(
        self,
        camera_params: List[Dict],
        points_3d: np.ndarray
    ) -> np.ndarray:
        """
        Pack camera poses and 3D points into single parameter vector.
        
        Camera parameters: [rvec1 (3), tvec1 (3), rvec2 (3), tvec2 (3), ...]
        3D points: [X1, Y1, Z1, X2, Y2, Z2, ...]
        """
        
        camera_vectors = []
        for cam in camera_params:
            rvec, _ = cv2.Rodrigues(cam['R'])
            camera_vectors.append(rvec.flatten())
            camera_vectors.append(cam['t'].flatten())
        
        camera_block = np.concatenate(camera_vectors)
        points_block = points_3d.flatten()
        
        return np.concatenate([camera_block, points_block])
    
    def _unpack_ba_params(
        self,
        x: np.ndarray,
        n_cameras: int,
        n_points: int,
        camera_params: List[Dict]
    ) -> Tuple[List[Dict], np.ndarray]:
        """Unpack parameter vector back to cameras and 3D points."""
        
        camera_block_size = n_cameras * 6  # 3 for rvec, 3 for tvec
        
        camera_block = x[:camera_block_size]
        points_block = x[camera_block_size:]
        
        # Unpack cameras
        optimized_cameras = []
        for i in range(n_cameras):
            idx = i * 6
            rvec = camera_block[idx:idx+3]
            tvec = camera_block[idx+3:idx+6]
            
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            
            optimized_cam = camera_params[i].copy()
            optimized_cam['R'] = R
            optimized_cam['t'] = t
            optimized_cameras.append(optimized_cam)
        
        # Unpack 3D points
        optimized_points_3d = points_block.reshape(n_points, 3)
        
        return optimized_cameras, optimized_points_3d
    
    def _compute_ba_residuals(
        self,
        x: np.ndarray,
        n_cameras: int,
        n_points: int,
        camera_params: List[Dict],
        observations: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Compute reprojection error residuals for Bundle Adjustment.
        
        Returns:
            residuals: Flattened array of reprojection errors (x and y for each observation)
        """
        
        # Unpack parameters
        cameras, points_3d = self._unpack_ba_params(x, n_cameras, n_points, camera_params)
        
        residuals = []
        
        # For each 3D point
        for point_idx, point_3d in enumerate(points_3d):
            # Get all observations of this point
            for cam_idx, point_2d_idx in observations[point_idx]:
                # Get camera
                cam = cameras[cam_idx]
                R = cam['R']
                t = cam['t']
                points_2d = cam['points_2d']
                
                # Observed 2D point
                observed_2d = points_2d[point_2d_idx]
                
                # Project 3D point to 2D
                rvec, _ = cv2.Rodrigues(R)
                projected_2d, _ = cv2.projectPoints(
                    point_3d.reshape(1, 3),
                    rvec,
                    t,
                    self.K,
                    None
                )
                projected_2d = projected_2d.reshape(2)
                
                # Residual (error)
                residual = observed_2d - projected_2d
                residuals.extend(residual)
        
        return np.array(residuals)
    
    def _build_sparsity_structure(
        self,
        n_cameras: int,
        n_points: int,
        observations: List[List[Tuple[int, int]]]
    ) -> lil_matrix:
        """
        Build sparse Jacobian structure for efficient optimization.
        
        Each observation creates dependencies between:
        - One camera's parameters (6 DOF)
        - One 3D point (3 DOF)
        """
        
        n_camera_params = n_cameras * 6
        n_point_params = n_points * 3
        n_params = n_camera_params + n_point_params
        
        # Count total observations
        n_observations = sum(len(obs) for obs in observations)
        n_residuals = n_observations * 2  # x and y for each observation
        
        # Build sparsity pattern
        A = lil_matrix((n_residuals, n_params), dtype=int)
        
        residual_idx = 0
        for point_idx, point_obs in enumerate(observations):
            for cam_idx, _ in point_obs:
                # Camera parameters influence this residual
                camera_param_start = cam_idx * 6
                A[residual_idx:residual_idx+2, camera_param_start:camera_param_start+6] = 1
                
                # 3D point parameters influence this residual
                point_param_start = n_camera_params + point_idx * 3
                A[residual_idx:residual_idx+2, point_param_start:point_param_start+3] = 1
                
                residual_idx += 2
        
        return A


if __name__ == "__main__":
    # Test PnP Solver
    
    # Camera intrinsics
    K = np.array([
        [2184, 0, 1512],
        [0, 2184, 2016],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create synthetic test data
    print("Creating synthetic test data...")
    
    # 3D points (random points in front of camera)
    np.random.seed(42)
    points_3d = np.random.rand(100, 3) * 5 + np.array([0, 0, 5])
    
    # True camera pose
    R_true = np.eye(3)
    t_true = np.array([[0], [0], [0]])
    
    # Project to 2D
    rvec, _ = cv2.Rodrigues(R_true)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, t_true, K, None)
    points_2d = points_2d.reshape(-1, 2)
    
    # Add noise
    points_2d += np.random.randn(*points_2d.shape) * 2.0
    
    print(f"Generated {len(points_3d)} 3D points")
    print(f"Projected to 2D with noise")
    
    # Test PnP
    solver = PnPSolver(K)
    success, R_est, t_est, inliers = solver.solve_pnp_ransac(points_3d, points_2d)
    
    if success:
        print(f"\nEstimated R:\n{R_est}")
        print(f"\nEstimated t:\n{t_est}")
        
        # Compute error
        mean_error, _ = solver.compute_reprojection_error(
            points_3d[inliers],
            points_2d[inliers],
            R_est,
            t_est
        )
        print(f"\nMean reprojection error: {mean_error:.3f} pixels")
    
    print("\nPnP Solver test complete!")
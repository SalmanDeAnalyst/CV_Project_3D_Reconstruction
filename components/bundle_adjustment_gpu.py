# components/bundle_adjustment_gpu.py
"""
GPU-accelerated Bundle Adjustment using PyTorch - FIXED to use pixel coordinates.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed. GPU BA not available.")


class BundleAdjustmentGPU:
    """
    GPU-accelerated Bundle Adjustment with FIXED observation handling.
    Now accepts pixel coordinates directly instead of indices.
    """
    
    def __init__(self, K: np.ndarray, use_gpu: bool = True):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.K = torch.tensor(K, dtype=torch.float32, device=self.device)
        
        print("="*60)
        print("GPU Bundle Adjustment Initialized")
        print("="*60)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)
    
    def rodrigues_to_rotation(self, rvec: torch.Tensor) -> torch.Tensor:
        """Convert Rodrigues vector to rotation matrix (batched)."""
        if rvec.dim() == 1:
            rvec = rvec.unsqueeze(0)
        
        theta = torch.norm(rvec, dim=1, keepdim=True).clamp(min=1e-8)
        r = rvec / theta
        
        zero = torch.zeros_like(r[:, 0])
        K = torch.stack([
            torch.stack([zero, -r[:, 2], r[:, 1]], dim=1),
            torch.stack([r[:, 2], zero, -r[:, 0]], dim=1),
            torch.stack([-r[:, 1], r[:, 0], zero], dim=1)
        ], dim=1)
        
        I = torch.eye(3, device=self.device).unsqueeze(0).expand(rvec.shape[0], -1, -1)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)
        
        R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
        
        return R.squeeze(0) if R.shape[0] == 1 else R
    
    def project_points(
        self,
        points_3d: torch.Tensor,
        rvecs: torch.Tensor,
        tvecs: torch.Tensor,
        camera_indices: torch.Tensor,
        point_indices: torch.Tensor
    ) -> torch.Tensor:
        """Project 3D points to 2D using camera poses."""
        rvec_obs = rvecs[camera_indices]
        tvec_obs = tvecs[camera_indices]
        points_obs = points_3d[point_indices]
        
        R_obs = self.rodrigues_to_rotation(rvec_obs)
        
        points_cam = torch.bmm(R_obs, points_obs.unsqueeze(-1)).squeeze(-1) + tvec_obs
        
        points_proj = torch.mm(self.K, points_cam.T).T
        points_2d = points_proj[:, :2] / points_proj[:, 2:3].clamp(min=1e-8)
        
        return points_2d
    
    def bundle_adjustment(
        self,
        camera_params: List[Dict],
        points_3d: np.ndarray,
        observations: List[List[Tuple[int, np.ndarray]]],  # ← FIXED: (cam_idx, pixel_2d)
        max_iterations: int = 100,
        lr: float = 0.01,
        verbose: bool = True
    ) -> Tuple[List[Dict], np.ndarray, Dict]:
        """
        Run GPU Bundle Adjustment with REAL observations.
        
        Args:
            camera_params: List of camera dicts with 'R', 't'
            points_3d: Nx3 array of 3D points
            observations: For each point, list of (cam_idx, pixel_2d) tuples
                         pixel_2d is np.array([x, y])
            max_iterations: Max optimization iterations
            lr: Learning rate
            verbose: Print progress
        
        Returns:
            optimized_cameras, optimized_points_3d, stats
        """
        import cv2
        
        if verbose:
            print("\n" + "="*60)
            print(f"GPU Bundle Adjustment ({self.device})")
            print("="*60)
            print(f"Cameras: {len(camera_params)}")
            print(f"3D points: {len(points_3d)}")
            total_obs = sum(len(obs) for obs in observations)
            print(f"Observations: {total_obs}")
        
        n_cameras = len(camera_params)
        n_points = len(points_3d)
        
        # Convert camera params to tensors
        rvecs = []
        tvecs = []
        for cam in camera_params:
            rvec, _ = cv2.Rodrigues(cam['R'])
            rvecs.append(rvec.flatten())
            tvecs.append(cam['t'].flatten())
        
        rvecs = torch.tensor(np.array(rvecs), dtype=torch.float32, 
                            device=self.device, requires_grad=True)
        tvecs = torch.tensor(np.array(tvecs), dtype=torch.float32, 
                            device=self.device, requires_grad=True)
        points_3d_t = torch.tensor(points_3d, dtype=torch.float32, 
                                   device=self.device, requires_grad=True)
        
        # ========== FIXED: BUILD OBSERVATION TENSORS FROM PIXEL COORDINATES ==========
        camera_indices = []
        point_indices = []
        observed_2d = []
        
        for point_idx, point_obs in enumerate(observations):
            for cam_idx, pixel_2d in point_obs:  # pixel_2d is np.array([x, y])
                camera_indices.append(cam_idx)
                point_indices.append(point_idx)
                observed_2d.append(pixel_2d)  # Direct pixel coordinates!
        
        camera_indices = torch.tensor(camera_indices, dtype=torch.long, device=self.device)
        point_indices = torch.tensor(point_indices, dtype=torch.long, device=self.device)
        observed_2d = torch.tensor(np.array(observed_2d), dtype=torch.float32, device=self.device)
        
        if verbose:
            print(f"Built observation tensors: {len(observed_2d)} observations")
        
        # Optimizer
        optimizer = torch.optim.Adam([rvecs, tvecs, points_3d_t], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Initial error
        with torch.no_grad():
            projected = self.project_points(points_3d_t, rvecs, tvecs, 
                                           camera_indices, point_indices)
            initial_error = torch.norm(observed_2d - projected, dim=1)
            initial_rmse = torch.sqrt(torch.mean(initial_error**2)).item()
            
            # DEBUG: Check for bad values
            if torch.any(~torch.isfinite(projected)):
                print(f"⚠️  WARNING: Non-finite values in projected points!")
                print(f"   NaN count: {torch.isnan(projected).sum().item()}")
                print(f"   Inf count: {torch.isinf(projected).sum().item()}")
            
            if torch.any(~torch.isfinite(observed_2d)):
                print(f"⚠️  WARNING: Non-finite values in observed points!")
            
            # DEBUG: Check for extreme values
            max_error = torch.max(initial_error).item()
            if max_error > 1000:
                print(f"🚨 EXTREME initial error detected: {max_error:.1f} pixels")
                print(f"   This indicates bad 3D points or wrong observations!")
        
        if verbose:
            print(f"Initial RMSE: {initial_rmse:.3f} pixels")
            if initial_rmse > 100:
                print(f"   🚨 CRITICAL: Initial RMSE is way too high!")
                print(f"   Expected: 2-10 pixels, Got: {initial_rmse:.1f}")
            print("\nOptimizing...")
        
        # Optimization loop
        best_rmse = initial_rmse
        patience_counter = 0
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            projected = self.project_points(points_3d_t, rvecs, tvecs, 
                                           camera_indices, point_indices)
            
            error = observed_2d - projected
            loss = torch.mean(error**2)
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            rmse = torch.sqrt(loss).item()
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}: RMSE = {rmse:.3f} pixels")
            
            # Early stopping
            if rmse < best_rmse - 0.01:
                best_rmse = rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 20:
                    if verbose:
                        print(f"  Early stopping at iteration {iteration + 1}")
                    break
        
        # Final error
        with torch.no_grad():
            projected = self.project_points(points_3d_t, rvecs, tvecs, 
                                           camera_indices, point_indices)
            final_error = torch.norm(observed_2d - projected, dim=1)
            final_rmse = torch.sqrt(torch.mean(final_error**2)).item()
        
        # Convert back to numpy
        rvecs_np = rvecs.detach().cpu().numpy()
        tvecs_np = tvecs.detach().cpu().numpy()
        points_3d_np = points_3d_t.detach().cpu().numpy()
        
        # Rebuild camera params
        optimized_cameras = []
        for i, cam in enumerate(camera_params):
            R, _ = cv2.Rodrigues(rvecs_np[i])
            optimized_cam = cam.copy()
            optimized_cam['R'] = R
            optimized_cam['t'] = tvecs_np[i].reshape(3, 1)
            optimized_cameras.append(optimized_cam)
        
        stats = {
            'initial_rmse': initial_rmse,
            'final_rmse': final_rmse,
            'iterations': iteration + 1,
            'success': final_rmse < initial_rmse,
            'improvement': initial_rmse - final_rmse
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"GPU Bundle Adjustment Complete")
            print(f"{'='*60}")
            print(f"Initial RMSE: {initial_rmse:.3f} pixels")
            print(f"Final RMSE: {final_rmse:.3f} pixels")
            print(f"Improvement: {stats['improvement']:.3f} pixels")
            print(f"Iterations: {stats['iterations']}")
            print(f"{'='*60}\n")
        
        return optimized_cameras, points_3d_np, stats
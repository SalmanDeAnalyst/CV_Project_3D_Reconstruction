# components/dense_stereo.py

import cv2
import numpy as np
from typing import Tuple, Optional

class DenseStereoMatcher:
    """
    Dense stereo reconstruction using Semi-Global Block Matching (SGBM).
    Supports GPU acceleration if available.
    """
    
    def __init__(self, K: np.ndarray, use_gpu: bool = True):
        """
        Initialize Dense Stereo Matcher.
        
        Args:
            K: 3x3 camera intrinsic matrix
            use_gpu: Use GPU acceleration if available
        """
        self.K = K
        
        # Check GPU availability
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.use_gpu:
            print("✓ Dense Stereo: GPU acceleration ENABLED")
        else:
            print("⚠ Dense Stereo: GPU not available, using CPU")
        
        print("="*60)
        print("Dense Stereo Matcher Initialized")
        print(f"Mode: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        print("="*60)
    
    def _rectify_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rectify stereo image pair."""
        
        h, w = img1.shape[:2]
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K, None,
            self.K, None,
            (w, h),
            R, t,
            alpha=0
        )
        
        # Compute rectification maps
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            self.K, None, R1, P1, (w, h), cv2.CV_32FC1
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            self.K, None, R2, P2, (w, h), cv2.CV_32FC1
        )
        
        # Apply rectification
        img1_rect = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)
        
        return img1_rect, img2_rect, Q
    
    def _compute_disparity_gpu(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> np.ndarray:
        """Compute disparity using GPU-accelerated stereo matching."""
        
        # Convert to grayscale
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left
            right_gray = right
        
        # Upload to GPU
        gpu_left = cv2.cuda_GpuMat()
        gpu_right = cv2.cuda_GpuMat()
        gpu_left.upload(left_gray)
        gpu_right.upload(right_gray)
        
        # Create GPU stereo matcher (StereoBM - faster on GPU than SGBM)
        stereo = cv2.cuda.createStereoBM(
            numDisparities=128,
            blockSize=19
        )
        
        # Compute disparity on GPU
        gpu_disparity = stereo.compute(gpu_left, gpu_right)
        
        # Download result
        disparity = gpu_disparity.download()
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def _compute_disparity_cpu(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> np.ndarray:
        """Compute disparity using CPU SGBM (slower but more accurate)."""
        
        # Convert to grayscale
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left
            right_gray = right
        
        # SGBM parameters
        min_disp = 0
        num_disp = 128
        block_size = 5
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        return disparity
    
    def _disparity_to_3d(
        self,
        disparity: np.ndarray,
        Q: np.ndarray,
        color_img: np.ndarray,
        max_depth: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert disparity map to 3D points with robust filtering."""
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # Robust filtering:
        # 1. Valid disparity (positive, non-zero)
        valid_disp = disparity > 1.0
        
        # 2. Finite 3D coordinates (no inf/nan)
        valid_finite = np.all(np.isfinite(points_3d), axis=2)
        
        # 3. Reasonable depth (positive and within range)
        valid_depth = (points_3d[:, :, 2] > 0.1) & (points_3d[:, :, 2] < max_depth)
        
        # 4. Not extreme X/Y values
        valid_xy = (np.abs(points_3d[:, :, 0]) < max_depth * 10) & (np.abs(points_3d[:, :, 1]) < max_depth * 10)
        
        # Combine all masks
        valid_mask = valid_disp & valid_finite & valid_depth & valid_xy
        
        points_3d_filtered = points_3d[valid_mask]
        
        # Get colors (BGR to RGB)
        colors_rgb = color_img[valid_mask]
        if len(colors_rgb.shape) == 2 and colors_rgb.shape[1] >= 3:
            colors_rgb = colors_rgb[:, [2, 1, 0]]  # BGR → RGB
        elif len(colors_rgb.shape) == 1:
            colors_rgb = np.column_stack([colors_rgb, colors_rgb, colors_rgb])
        
        return points_3d_filtered, colors_rgb
    
    def compute_dense_reconstruction(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute dense 3D reconstruction from stereo pair.
        
        Args:
            img1: First image
            img2: Second image
            R: Relative rotation matrix
            t: Relative translation vector
        
        Returns:
            points_3d: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
            disparity: Disparity map
        """
        
        print("  Rectifying stereo pair...")
        img1_rect, img2_rect, Q = self._rectify_images(img1, img2, R, t)
        
        print(f"  Computing disparity ({'GPU' if self.use_gpu else 'CPU'})...")
        
        # Choose GPU or CPU
        if self.use_gpu:
            disparity = self._compute_disparity_gpu(img1_rect, img2_rect)
        else:
            disparity = self._compute_disparity_cpu(img1_rect, img2_rect)
        
        print("  Converting to 3D points...")
        points_3d, colors = self._disparity_to_3d(disparity, Q, img1_rect)
        
        print(f"  ✓ Generated {len(points_3d):,} dense points")
        
        return points_3d, colors, disparity


# if __name__ == "__main__":
#     # Test GPU availability
#     import cv2
    
#     print("="*60)
#     print("Dense Stereo GPU Check")
#     print("="*60)
    
#     cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
#     print(f"CUDA-enabled devices: {cuda_devices}")
    
#     if cuda_devices > 0:
#         print("✓ GPU acceleration available!")
#     else:
#         print("⚠ No GPU support - using CPU")
#         print("\nTo enable GPU:")
#         print("1. Install CUDA toolkit")
#         print("2. Build opencv-contrib-python with CUDA")
#         print("   (or use pre-built: pip install opencv-contrib-python)")
    
#     print("="*60)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from components.feature_mapping import FeatureMapping
from components.essential_matrix import EssentialMatrixEstimator

class DenseStereoReconstruction:
    def __init__(self, K):
        """
        Initialize dense stereo reconstruction
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
    
    def compute_disparity(self, img1, img2, R, t):
        """
        Compute disparity map using stereo matching
        
        Args:
            img1, img2: Input images
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
        
        Returns:
            disparity: Disparity map
            Q: Reprojection matrix for 3D conversion
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K, None, self.K, None, 
            img1.shape[:2][::-1], R, t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # Must be divisible by 16
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Compute disparity
        disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
        
        return disparity, Q
    
    def disparity_to_3d(self, disparity, Q, img):
        """
        Convert disparity map to 3D points
        
        Args:
            disparity: Disparity map
            Q: Reprojection matrix
            img: Original image for colors
        
        Returns:
            points_3d: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
        """
        # Convert disparity to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # Get colors
        colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Filter valid points
        mask = disparity > disparity.min()
        
        points_3d_filtered = points_3d[mask]
        colors_filtered = colors[mask]
        
        return points_3d_filtered, colors_filtered
    
    def visualize_disparity(self, img1, img2, disparity):
        """
        Visualize disparity map
        
        Args:
            img1, img2: Original images
            disparity: Disparity map
        """
        # Create valid mask
        valid_mask = disparity > disparity.min()
        
        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original images
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image 1', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Image 2', fontsize=14)
        axes[0, 1].axis('off')
        
        # Disparity map
        im = axes[1, 0].imshow(disparity_normalized, cmap='jet')
        axes[1, 0].set_title('Disparity Map (Depth)\nBrighter = Closer', fontsize=14)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Valid depth mask
        axes[1, 1].imshow(valid_mask, cmap='gray')
        axes[1, 1].set_title(f'Valid Depth Points\n{valid_mask.sum():,} pixels '
                            f'({valid_mask.sum()/disparity.size*100:.1f}%)', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('dense_stereo_disparity.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Dense Stereo Matching Statistics:")
        print(f"{'='*60}")
        print(f"Total pixels: {disparity.size:,}")
        print(f"Valid depth pixels: {valid_mask.sum():,} ({valid_mask.sum()/disparity.size*100:.1f}%)")
        print(f"Disparity range: {disparity[valid_mask].min():.1f} to {disparity[valid_mask].max():.1f}")
        print(f"{'='*60}\n")
    
    def visualize_3d_points(self, points_3d, colors, sample_rate=10):
        """
        Visualize 3D point cloud
        
        Args:
            points_3d: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
            sample_rate: Sample every Nth point for performance
        """
        # Filter out inf and nan values first
        valid_mask = np.isfinite(points_3d).all(axis=1)
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]
        
        # Sample points
        points_sampled = points_3d[::sample_rate]
        colors_sampled = colors[::sample_rate] / 255.0
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points_sampled[:, 0], 
                  points_sampled[:, 1], 
                  points_sampled[:, 2],
                  c=colors_sampled, s=1, marker='.')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z (Depth)', fontsize=12)
        ax.set_title(f'Dense 3D Reconstruction\n{len(points_sampled):,} points (sampled from {len(points_3d):,})', 
                    fontsize=14)
        
        # Set equal aspect ratio
        max_range = np.array([
            points_sampled[:, 0].max() - points_sampled[:, 0].min(),
            points_sampled[:, 1].max() - points_sampled[:, 1].min(),
            points_sampled[:, 2].max() - points_sampled[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points_sampled[:, 0].max() + points_sampled[:, 0].min()) * 0.5
        mid_y = (points_sampled[:, 1].max() + points_sampled[:, 1].min()) * 0.5
        mid_z = (points_sampled[:, 2].max() + points_sampled[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig('dense_3d_reconstruction.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"3D Point Cloud Statistics:")
        print(f"Total points: {len(points_3d):,}")
        print(f"Displayed (sampled): {len(points_sampled):,}")
    
    def save_ply(self, points_3d, colors, filename='dense_reconstruction.ply'):
        """
        Save dense point cloud to PLY file
        
        Args:
            points_3d: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point, color in zip(points_3d, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} ")
                f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
        
        print(f"\nDense point cloud saved to {filename}")


if __name__ == "__main__":
    print("="*60)
    print("Dense Stereo Reconstruction Pipeline")
    print("="*60)
    
    # Load images
    print("\n1. Loading images...")
    img1 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\3D-Reconstruction-SFM\data\Dataset\image_1_study.jpeg')
    img2 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\3D-Reconstruction-SFM\data\Dataset\image_2_study.jpeg')
    
    if img1 is None or img2 is None:
        print("Error: Could not load images!")
        exit()
    
    print(f"   Image 1 shape: {img1.shape}")
    print(f"   Image 2 shape: {img2.shape}")
    
    # Camera intrinsic matrix
    K = np.array([
        [2184, 0, 1512],
        [0, 2184, 2016],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print("\n2. Detecting and matching SIFT features...")
    # Feature matching
    feature_mapping = FeatureMapping(img1, img2)
    kp1, des1, kp2, des2 = feature_mapping.detect_keypoints()
    good_matches = feature_mapping.match_features(kp1, des1, kp2, des2)
    pts1, pts2 = feature_mapping.get_matched_points(kp1, kp2, good_matches)
    print(f"   Total good matches: {len(good_matches)}")
    
    # Estimate Essential Matrix
    print("\n3. Estimating Essential Matrix...")
    estimator = EssentialMatrixEstimator(K)
    E, mask = estimator.estimate_essential_matrix(pts1, pts2)
    inlier_pts1, inlier_pts2 = estimator.get_inlier_matches(pts1, pts2, mask)
    
    print(f"   Inliers: {np.sum(mask)}/{len(pts1)} ({np.sum(mask)/len(pts1)*100:.2f}%)")
    
    # Recover pose
    print("\n4. Recovering camera pose...")
    _, R, t, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)
    print(f"   Rotation matrix shape: {R.shape}")
    print(f"   Translation vector shape: {t.shape}")
    
    # Dense stereo reconstruction
    print("\n5. Computing dense disparity map...")
    dense_recon = DenseStereoReconstruction(K)
    disparity, Q = dense_recon.compute_disparity(img1, img2, R, t)
    
    # Visualize disparity
    print("\n6. Visualizing disparity map...")
    dense_recon.visualize_disparity(img1, img2, disparity)
    
    # Convert to 3D
    print("\n7. Converting disparity to 3D points...")
    points_3d, colors = dense_recon.disparity_to_3d(disparity, Q, img1)
    print(f"   Generated {len(points_3d):,} 3D points")
    
    # Visualize 3D
    print("\n8. Visualizing 3D point cloud...")
    dense_recon.visualize_3d_points(points_3d, colors, sample_rate=5)
    
    # Save to PLY
    print("\n9. Saving point cloud...")
    dense_recon.save_ply(points_3d, colors, 'dense_reconstruction.ply')
    
    print("\n" + "="*60)
    print("Dense reconstruction complete!")
    print("="*60)
    print("\nOutput files:")
    print("  - dense_stereo_disparity.png (disparity visualization)")
    print("  - dense_3d_reconstruction.png (3D point cloud)")
    print("  - dense_reconstruction.ply (point cloud file)")
    print("\nYou can open the .ply file in MeshLab or CloudCompare to view the 3D model!")
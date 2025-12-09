import cv2
import numpy as np
from feature_mapping import FeatureMapping
from essential_matrix import EssentialMatrixEstimator
from pose_detection import PoseRecovery
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Triangulation:
    def __init__(self, K):
        """
        Initialize with camera intrinsic matrix K
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
    
    def create_projection_matrices(self, R, t):
        """
        Create projection matrices for both cameras
        
        Camera 1: P1 = K[I|0] (reference frame)
        Camera 2: P2 = K[R|t] (relative pose)
        
        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            
        Returns:
            P1: Projection matrix for camera 1 (3x4)
            P2: Projection matrix for camera 2 (3x4)
        """
        # Camera 1 is at origin (identity rotation, zero translation)
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Camera 2 has rotation R and translation t
        P2 = self.K @ np.hstack([R, t])
        
        return P1, P2
    
    def triangulate_points(self, P1, P2, pts1, pts2):
        """
        Triangulate 3D points from 2D correspondences
        
        Args:
            P1: Projection matrix for camera 1 (3x4)
            P2: Projection matrix for camera 2 (3x4)
            pts1: 2D points in image 1 (Nx2)
            pts2: 2D points in image 2 (Nx2)
            
        Returns:
            points_3d: Triangulated 3D points (Nx3)
        """
        # Transpose points for cv2.triangulatePoints (expects 2xN)
        pts1_T = pts1.T
        pts2_T = pts2.T
        
        # Triangulate
        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)
        
        # Convert from homogeneous (4xN) to 3D (Nx3)
        points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]
        points_3d = points_3d.T  # Transpose to Nx3
        
        return points_3d
    
    def filter_points_by_depth(self, points_3d, max_depth=100, min_depth=0.1):
        """
        Filter out invalid and outlier 3D points.
        
        Args:
            points_3d: 3D points (Nx3)
            max_depth: Maximum depth threshold
            min_depth: Minimum depth threshold (points too close are unstable)
            
        Returns:
            filtered_points: Valid 3D points within depth range
            mask: Boolean mask of valid points
        """
        # Check for inf/nan values
        valid_mask = np.all(np.isfinite(points_3d), axis=1)
        
        # Check depth (Z coordinate) - must be positive and within range
        depths = points_3d[:, 2]
        depth_mask = (depths > min_depth) & (depths < max_depth)
        
        # Also filter extreme X, Y coordinates
        xy_mask = (np.abs(points_3d[:, 0]) < max_depth * 10) & (np.abs(points_3d[:, 1]) < max_depth * 10)
        
        # Combine all masks
        mask = valid_mask & depth_mask & xy_mask
        filtered_points = points_3d[mask]
        
        return filtered_points, mask
    
    def save_point_cloud(self, points_3d, filename='point_cloud.ply', colors=None):
        """
        Save 3D points to PLY file
        
        Args:
            points_3d: 3D points (Nx3)
            filename: Output filename
            colors: Optional RGB colors for points (Nx3), values 0-255
        """
        with open(filename, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # Write points
            for i, point in enumerate(points_3d):
                if colors is not None:
                    f.write(f"{point[0]} {point[1]} {point[2]} {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")
                else:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        print(f"Point cloud saved to {filename}")



class PointCloudVisualizer:
    @staticmethod
    def visualize_3d(points_3d, colors=None, title="3D Point Cloud"):
        """
        Visualize 3D point cloud using matplotlib
        
        Args:
            points_3d: Nx3 array of 3D points
            colors: Optional Nx3 array of RGB colors (0-255)
            title: Plot title
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize colors if provided
        if colors is not None:
            colors = colors / 255.0  # matplotlib expects 0-1 range
        
        # Plot points
        ax.scatter(
            points_3d[:, 0], 
            points_3d[:, 1], 
            points_3d[:, 2],
            c=colors if colors is not None else 'blue',
            marker='.',
            s=1
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Equal aspect ratio
        max_range = np.array([
            points_3d[:, 0].max() - points_3d[:, 0].min(),
            points_3d[:, 1].max() - points_3d[:, 1].min(),
            points_3d[:, 2].max() - points_3d[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()

# # Update your main script:
# if __name__ == "__main__":
#     # Load images
#     img1 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\data\image_1_study.jpeg')
#     img2 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\data\image_2_study.jpeg')
    
#     # Feature matching
#     feature_mapping = FeatureMapping(img1, img2)
#     kp1, des1, kp2, des2 = feature_mapping.detect_keypoints()
#     good_matches = feature_mapping.match_features(kp1, des1, kp2, des2)
    
#     # Get matched points
#     pts1, pts2 = feature_mapping.get_matched_points(kp1, kp2, good_matches)
#     print(f"Total good matches: {len(good_matches)}")
    
#     # Define intrinsic matrix
#     K = np.array([
#         [2184, 0, 1512],
#         [0, 2184, 2016],
#         [0, 0, 1]
#     ], dtype=np.float32)
    
#     # Estimate Essential Matrix
#     estimator = EssentialMatrixEstimator(K)
#     E, mask = estimator.estimate_essential_matrix(pts1, pts2)
    
#     print(f"\nEssential Matrix:\n{E}")
#     print(f"Inliers: {np.sum(mask)}/{len(pts1)} ({np.sum(mask)/len(pts1)*100:.2f}%)")
    
#     # Get inlier points
#     inlier_pts1, inlier_pts2 = estimator.get_inlier_matches(pts1, pts2, mask)
    
#     # Recover pose
#     pose_recovery = PoseRecovery(K)
#     R, t, pose_mask, num_inliers = pose_recovery.recover_pose(E, inlier_pts1, inlier_pts2)
    
#     print(f"\nRotation Matrix:\n{R}")
#     print(f"Translation Vector:\n{t}")
#     print(f"Points in front of both cameras: {num_inliers}")
    
#     # Triangulate 3D points
#     triangulator = Triangulation(K)
#     P1, P2 = triangulator.create_projection_matrices(R, t)
    
#     print(f"\nProjection Matrix P1:\n{P1}")
#     print(f"Projection Matrix P2:\n{P2}")
    
#     # Triangulate all inlier points
#     points_3d = triangulator.triangulate_points(P1, P2, inlier_pts1, inlier_pts2)
#     print(f"\nTriangulated {len(points_3d)} 3D points")
    
#     # Filter points by depth
#     filtered_points, depth_mask = triangulator.filter_points_by_depth(points_3d, max_depth=50)
#     print(f"Points after depth filtering: {len(filtered_points)}")
    
#     # Optional: Get colors from image 1 for the points
#     colors = []
#     for pt in inlier_pts1[depth_mask]:
#         x, y = int(pt[0]), int(pt[1])
#         if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
#             color = img1[y, x]  # BGR
#             colors.append([color[2], color[1], color[0]])  # Convert to RGB
#         else:
#             colors.append([128, 128, 128])  # Gray for out of bounds
#     colors = np.array(colors)
    
#     # Save point cloud
#     triangulator.save_point_cloud(filtered_points, 'reconstruction.ply', colors=colors)
    
#     print("\n 3D reconstruction complete! Open 'reconstruction.ply' in MeshLab or CloudCompare to view.")
#     # Visualize
#     visualizer = PointCloudVisualizer()
#     visualizer.visualize_3d(filtered_points, colors=colors, title="Reconstructed 3D Scene")

import cv2
import numpy as np

class PoseRecovery:
    def __init__(self, K):
        """
        Initialize with camera intrinsic matrix K
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
    
    def recover_pose(self, E, pts1, pts2):
        """
        Recover camera pose (R, t) from Essential Matrix
        
        Args:
            E: Essential matrix (3x3)
            pts1: Inlier points from image 1 (Nx2)
            pts2: Inlier points from image 2 (Nx2)
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector (unit direction, not scale)
            mask: Inlier mask after cheirality check
            num_inliers: Number of points in front of both cameras
        """
        num_inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        return R, t, mask, num_inliers


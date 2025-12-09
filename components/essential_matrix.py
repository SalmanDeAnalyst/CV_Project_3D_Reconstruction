import cv2
import numpy as np
from feature_mapping import FeatureMapping

class EssentialMatrixEstimator:
    def __init__(self, K):
        """
        Initialize with camera intrinsic matrix K
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
    
    def estimate_essential_matrix(self, pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0):
        """
        Estimate Essential Matrix from matched points using RANSAC
        
        Args:
            pts1: Matched keypoints from image 1 (Nx2 array)
            pts2: Matched keypoints from image 2 (Nx2 array)
            method: Method for estimation (default: RANSAC)
            prob: Confidence level (default: 0.999)
            threshold: RANSAC threshold in pixels (default: 1.0)
        
        Returns:
            E: Essential matrix (3x3)
            mask: Inlier mask (Nx1)
        """
        # Estimate Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            self.K, 
            method=method,
            prob=prob,
            threshold=threshold
        )
        
        return E, mask
    
    def get_inlier_matches(self, pts1, pts2, mask):
        """
        Filter matches to keep only inliers
        
        Args:
            pts1: Points from image 1
            pts2: Points from image 2
            mask: Inlier mask from RANSAC
        
        Returns:
            inlier_pts1, inlier_pts2: Filtered point arrays
        """
        mask_bool = mask.ravel() == 1
        inlier_pts1 = pts1[mask_bool]
        inlier_pts2 = pts2[mask_bool]
        
        return inlier_pts1, inlier_pts2

if __name__ == "__main__":
    # Load images
    img1 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\data\image_1_study.jpeg')
    img2 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\data\image_2_study.jpeg')
    
    # Feature matching
    feature_mapping = FeatureMapping(img1, img2)
    kp1, des1, kp2, des2 = feature_mapping.detect_keypoints()
    good_matches = feature_mapping.match_features(kp1, des1, kp2, des2)
    
    # Get matched points
    pts1, pts2 = feature_mapping.get_matched_points(kp1, kp2, good_matches)
    print(f"Total good matches: {len(good_matches)}")
    
    # Define intrinsic matrix
    K = np.array([
        [2184, 0, 1512],
        [0, 2184, 2016],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Estimate Essential Matrix
    estimator = EssentialMatrixEstimator(K)
    E, mask = estimator.estimate_essential_matrix(pts1, pts2)
    
    print(f"\nEssential Matrix:\n{E}")
    print(f"\nInliers: {np.sum(mask)}/{len(pts1)}")
    print(f"Inlier ratio: {np.sum(mask)/len(pts1)*100:.2f}%")
    
    # Get inlier points only
    inlier_pts1, inlier_pts2 = estimator.get_inlier_matches(pts1, pts2, mask)
    print(f"\nInlier points shape: {inlier_pts1.shape}")
    
    # Optionally visualize matches
    img_matches = feature_mapping.draw_matches(img1, img2, kp1, kp2, good_matches)
    feature_mapping.save_matches(img_matches)

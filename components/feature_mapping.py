import cv2
import numpy as np

# Load images

class FeatureMapping:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def detect_keypoints(self):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)
        return kp1, des1, kp2, des2

    def match_features(self, kp1, des1, kp2, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.80 * n.distance:
                good_matches.append(m)
        return good_matches

    def draw_matches(self, img1, img2, kp1, kp2, good_matches):
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches

    def get_matched_points(self, kp1, kp2, good_matches):
        """
        Extract matched point coordinates from keypoints and matches
        
        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            good_matches: List of good matches
            
        Returns:
            pts1: Nx2 array of points from image 1
            pts2: Nx2 array of points from image 2
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return pts1, pts2

    def save_matches(self, img_matches):
        cv2.imwrite('matches.jpg', img_matches)
        cv2.imshow('Good Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     img1 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\data\image_1_study.jpeg')
#     img2 = cv2.imread(r'D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\data\image_2_study.jpeg')
#     feature_mapping = FeatureMapping(img1, img2)
#     kp1, des1, kp2, des2 = feature_mapping.detect_keypoints()
#     good_matches = feature_mapping.match_features(kp1, des1, kp2, des2)
#     img_matches = feature_mapping.draw_matches(img1, img2, kp1, kp2, good_matches)
#     feature_mapping.save_matches(img_matches)
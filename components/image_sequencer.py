# components/image_sequencer.py - OPTIMIZED WITH FLANN + INCREMENTAL SAVING

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import pillow_heif
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import itertools
import pickle

# Register HEIF opener
pillow_heif.register_heif_opener()


def load_image_any_format(image_path: str) -> np.ndarray:
    """Load image supporting HEIC, JPEG, PNG, etc."""
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ['.heic', '.heif']:
        img_pil = Image.open(image_path)
        img_rgb = np.array(img_pil.convert('RGB'))
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img


class ImageSequencer:
    """
    Fast multi-threaded image sequencing with FLANN matching.
    Orders unordered image dataset for incremental SfM.
    Supports incremental cache saving for crash recovery.
    """
    
    def __init__(self, K: np.ndarray, min_matches: int = 50, n_workers: int = 8):
        """
        Initialize Image Sequencer.
        
        Args:
            K: 3x3 camera intrinsic matrix
            min_matches: Minimum number of matches to consider valid overlap
            n_workers: Number of parallel workers
        """
        self.K = K
        self.min_matches = min_matches
        self.n_workers = n_workers
        self.feature_cache = {}
        self.match_matrix = {}
        
        # FLANN matcher setup (MUCH faster than BFMatcher)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        print("="*60)
        print("Image Sequencer Initialized (FLANN + Multi-threaded)")
        print("="*60)
        print(f"Matching: FLANN (Fast!)")
        print(f"Minimum matches threshold: {min_matches}")
        print(f"Parallel workers: {n_workers}")
        print(f"Supported formats: HEIC, JPEG, PNG, BMP, TIFF")
    
    def save_feature_cache(self, filepath: str = 'feature_cache.pkl'):
        """
        Save feature cache to disk for reuse.
        
        Args:
            filepath: Path to save cache file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert cache to serializable format
        cache_to_save = {}
        cache_snapshot = dict(self.feature_cache)
        for img_path, (img, pts_2d, keypoints, descriptors) in cache_snapshot.items():
            # Don't save full image (too large), just shape
            # Convert keypoints to tuples (cv2.KeyPoint not directly picklable)
            kp_data = [
                (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in keypoints
            ]
            
            cache_to_save[img_path] = {
                'image_shape': img.shape if img is not None else None,
                'points_2d': pts_2d,
                'keypoints_data': kp_data,
                'descriptors': descriptors
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_to_save, f)
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        return len(cache_to_save), file_size_mb
    
    def load_feature_cache(self, filepath: str = 'feature_cache.pkl') -> bool:
        """
        Load feature cache from disk.
        
        Args:
            filepath: Path to cache file
        
        Returns:
            success: Whether cache was loaded successfully
        """
        if not os.path.exists(filepath):
            print(f"Cache file not found: {filepath}")
            return False
        
        print(f"\nLoading feature cache from {filepath}...")
        
        with open(filepath, 'rb') as f:
            cache_loaded = pickle.load(f)
        
        # Convert back to runtime format
        for img_path, data in cache_loaded.items():
            # Reconstruct keypoints from tuples
            keypoints = [
                cv2.KeyPoint(
                    x=pt[0], y=pt[1],
                    size=size, angle=angle,
                    response=response, octave=octave,
                    class_id=class_id
                )
                for pt, size, angle, response, octave, class_id in data['keypoints_data']
            ]
            
            # Load image on demand (don't store in cache yet to save memory)
            img = None  # Placeholder
            
            self.feature_cache[img_path] = (
                img,
                data['points_2d'],
                keypoints,
                data['descriptors']
            )
        
        print(f"✓ Loaded {len(self.feature_cache)} image features")
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  File size: {file_size_mb:.2f} MB\n")
        return True
    
    def get_or_load_image(self, image_path: str) -> np.ndarray:
        """
        Get image from cache, loading it if needed.
        
        Args:
            image_path: Path to image
        
        Returns:
            img: Loaded image
        """
        if image_path in self.feature_cache:
            img, pts_2d, kp, des = self.feature_cache[image_path]
            
            # If image not loaded (was from disk cache), load it now
            if img is None:
                img = load_image_any_format(image_path)
                self.feature_cache[image_path] = (img, pts_2d, kp, des)
            
            return img
        else:
            # Not in cache, load fresh
            img, pts_2d, kp, des = self.load_and_detect_features(image_path)
            return img
    
    def load_and_detect_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray]:
        """Load image and detect SIFT features with caching."""
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]
        
        img = load_image_any_format(image_path)
        
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Convert keypoints to points_2d array
        points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        self.feature_cache[image_path] = (img, points_2d, keypoints, descriptors)
        
        return img, points_2d, keypoints, descriptors


    def save_match_results(self, filepath: str, ordered_indices: List[int], 
                          initial_pair: Tuple[int, int], num_matches: int):
        """Save match matrix + ordering."""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        match_results = {
            'ordered_indices': ordered_indices,
            'initial_pair': initial_pair,
            'initial_matches': num_matches,
            'match_matrix': self.match_matrix  # ← SAVE ALL MATCHES!
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(match_results, f)
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"\n✓ Match results saved: {filepath}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Cached {len(self.match_matrix)} pair matches")
        return file_size_mb


    def load_match_results(self, filepath: str) -> Optional[Tuple[List[int], Tuple[int, int], int]]:
        """Load match matrix + ordering."""
        
        if not os.path.exists(filepath):
            print(f"Match results not found: {filepath}")
            return None
        
        print(f"\nLoading match results from {filepath}...")
        
        with open(filepath, 'rb') as f:
            match_results = pickle.load(f)
        
        ordered_indices = match_results['ordered_indices']
        initial_pair = match_results['initial_pair']
        num_matches = match_results['initial_matches']
        
        # ← LOAD MATCH MATRIX!
        if 'match_matrix' in match_results:
            self.match_matrix = match_results['match_matrix']
            print(f"✓ Loaded {len(self.match_matrix)} cached pair matches")
        
        print(f"✓ Loaded ordering for {len(ordered_indices)} images")
        print(f"  Initial pair: {initial_pair}")
        print(f"  Initial matches: {num_matches}")
        
        return ordered_indices, initial_pair, num_matches


    def preload_all_features(self, image_paths: List[str], cache_file: Optional[str] = None):
        """
        Preload SIFT features for all images in parallel.
        Optionally save incrementally to disk for crash recovery.
        
        Args:
            image_paths: List of image paths
            cache_file: Optional path to save cache incrementally
        """
        print("\n" + "="*60)
        print("Preloading SIFT Features (Parallel)")
        print("="*60)
        print(f"Processing {len(image_paths)} images with {self.n_workers} workers...")
        if cache_file:
            print(f"Incremental saving to: {cache_file}")
        print()
        
        def load_features_wrapper(path):
            try:
                img, pts_2d, kp, des = self.load_and_detect_features(path)
                return (path, len(kp), True)
            except Exception as e:
                print(f"  Error loading {os.path.basename(path)}: {e}")
                return (path, 0, False)
        
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(load_features_wrapper, path) for path in image_paths]
            
            for future in tqdm(as_completed(futures), total=len(image_paths), desc="Loading features"):
                path, num_features, success = future.result()
                completed += 1
                
                if success:
                    print(f"  ✓ {os.path.basename(path)}: {num_features} features")
                else:
                    print(f"  ✗ {os.path.basename(path)}: FAILED")
                
                # INCREMENTAL SAVE every 10 images
                if cache_file and completed % 10 == 0:
                    num_saved, file_size = self.save_feature_cache(cache_file)
                    print(f"  💾 Cache saved: {num_saved} images ({file_size:.2f} MB) - Progress: {completed}/{len(image_paths)}")
        
        # Final save
        if cache_file:
            num_saved, file_size = self.save_feature_cache(cache_file)
            print(f"\n✓ Final cache saved: {num_saved} images ({file_size:.2f} MB)")
        
        print(f"\n{'='*60}")
        print(f"Feature extraction complete!")
        print(f"Cached {len(self.feature_cache)} images")
        print(f"{'='*60}\n")
    
    def compute_match_count_flann(self, img_path1: str, img_path2: str) -> int:
        """
        Compute number of good matches between two images using FLANN.
        MUCH faster than BFMatcher!
        """
        # Features should already be cached
        _, _, kp1, des1 = self.feature_cache[img_path1]
        _, _, kp2, des2 = self.feature_cache[img_path2]
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
        
        # FLANN matching
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return len(good_matches)
    
    def compute_match_pair(self, pair: Tuple[int, int], image_paths: List[str]) -> Tuple[int, int, int]:
        """Helper for parallel match computation."""
        i, j = pair
        num_matches = self.compute_match_count_flann(image_paths[i], image_paths[j])
        return i, j, num_matches
    
    def find_best_initial_pair(self, image_paths: List[str]) -> Tuple[int, int, int]:
        """Find best pair and SAVE all match counts."""
        
        print("\n" + "="*60)
        print("Finding Best Initial Pair (FLANN + Parallel)")
        print("="*60)
        
        n = len(image_paths)
        if n < 2:
            raise ValueError(f"Need at least 2 images, got {n}")
        
        pairs = list(itertools.combinations(range(n), 2))
        print(f"\nComputing matches for {len(pairs)} pairs...\n")
        
        best_pair = (0, 1)
        best_matches = 0
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self.compute_match_pair, pair, image_paths) for pair in pairs]
            
            for future in tqdm(as_completed(futures), total=len(pairs), desc="Matching pairs"):
                i, j, num_matches = future.result()
                
                # ← SAVE ALL MATCHES!
                self.match_matrix[(i, j)] = num_matches
                
                print(f"  {os.path.basename(image_paths[i])} <-> "
                      f"{os.path.basename(image_paths[j])}: {num_matches} matches")
                
                if num_matches > best_matches:
                    best_matches = num_matches
                    best_pair = (i, j)
        
        print(f"\n{'='*60}")
        print(f"Best Initial Pair Found!")
        print(f"{'='*60}")
        print(f"Image 1: {os.path.basename(image_paths[best_pair[0]])}")
        print(f"Image 2: {os.path.basename(image_paths[best_pair[1]])}")
        print(f"Matches: {best_matches}")
        print(f"{'='*60}\n")
        
        return best_pair[0], best_pair[1], best_matches
    
    def order_images_greedy(
        self,
        image_paths: List[str],
        initial_idx1: int,
        initial_idx2: int
    ) -> List[int]:
        """
        Order images using CHAIN ordering - each new image must match the LAST image.
        This ensures consecutive images have good overlap for incremental SfM.
        """
        
        print("\n" + "="*60)
        print("Ordering Images (CHAIN ordering for SfM)")
        print("="*60)
        
        n = len(image_paths)
        ordered = [initial_idx1, initial_idx2]
        remaining = set(range(n)) - {initial_idx1, initial_idx2}
        
        print(f"\nTotal images: {n}")
        print(f"Starting with: {os.path.basename(image_paths[initial_idx1])}, "
              f"{os.path.basename(image_paths[initial_idx2])}")
        print(f"Remaining: {len(remaining)}\n")
        
        iteration = 1
        stall_count = 0
        max_stalls = 3  # Allow fallback to any-match if chain breaks
        
        while remaining:
            best_next = -1
            best_matches = 0
            
            # CHAIN ORDERING: Match with LAST image in sequence (primary)
            last_idx = ordered[-1]
            
            for candidate_idx in remaining:
                pair = tuple(sorted([candidate_idx, last_idx]))
                num_matches = self.match_matrix.get(pair, 0)
                
                if num_matches > best_matches:
                    best_matches = num_matches
                    best_next = candidate_idx
            
            # If chain ordering fails, try matching with last few images
            if best_next < 0 or best_matches < self.min_matches:
                stall_count += 1
                
                if stall_count <= max_stalls:
                    print(f"  Chain broken! Trying last 3 images...")
                    
                    # Try matching with last 3 images
                    for candidate_idx in remaining:
                        for existing_idx in ordered[-3:]:
                            pair = tuple(sorted([candidate_idx, existing_idx]))
                            num_matches = self.match_matrix.get(pair, 0)
                            
                            if num_matches > best_matches:
                                best_matches = num_matches
                                best_next = candidate_idx
            
            if best_next >= 0 and best_matches >= self.min_matches:
                ordered.append(best_next)
                remaining.remove(best_next)
                stall_count = 0  # Reset stall counter on success
                
                print(f"[{iteration}] Added: {os.path.basename(image_paths[best_next])} "
                      f"({best_matches} matches with last image)")
                iteration += 1
            else:
                print(f"\nWarning: {len(remaining)} images have insufficient overlap")
                print(f"  Last image: {os.path.basename(image_paths[ordered[-1]])}")
                break
        
        print(f"\n{'='*60}")
        print(f"Image Ordering Complete (CHAIN)")
        print(f"{'='*60}")
        print(f"Ordered: {len(ordered)}/{n} images")
        print(f"{'='*60}\n")
        
        return ordered     

    def sequence_images(
        self, 
        image_paths: List[str],
        cache_file: Optional[str] = None,
        match_cache_file: Optional[str] = None  # ← NEW!
    ) -> Tuple[List[str], Dict]:
        """
        Complete image sequencing pipeline with FLANN + parallel processing.
        Supports incremental cache saving.
        
        Args:
            image_paths: List of image paths (unordered, any format)
            cache_file: Optional path to save feature cache incrementally
            match_cache_file: Optional path to save/load match results
        
        Returns:
            ordered_paths: List of image paths in reconstruction order
            metadata: Dictionary with sequencing info
        """
        
        if len(image_paths) == 0:
            raise ValueError("No images found! Check your dataset path.")
        
        print("\n" + "="*60)
        print("IMAGE SEQUENCING PIPELINE (FLANN + PARALLEL)")
        print("="*60)
        print(f"Total images: {len(image_paths)}\n")
        
        # Step 1: Preload all features in parallel (with incremental saving)
        self.preload_all_features(image_paths, cache_file=cache_file)
        
        # Step 2: Try to load MATCH MATRIX from cache (but re-run ordering!)
        if match_cache_file:
            cached_results = self.load_match_results(match_cache_file)
            if cached_results is not None:
                _, initial_pair, num_matches = cached_results
                idx1, idx2 = initial_pair
                
                print("\n✅ Loaded cached MATCH MATRIX! Re-running CHAIN ordering...")
                
                # RE-RUN ordering with the new chain algorithm using cached matches
                ordered_indices = self.order_images_greedy(image_paths, idx1, idx2)
                
                # Save updated ordering
                self.save_match_results(match_cache_file, ordered_indices, (idx1, idx2), num_matches)
                
                # Create ordered path list
                ordered_paths = [image_paths[i] for i in ordered_indices]
                
                # Metadata
                metadata = {
                    'total_images': len(image_paths),
                    'ordered_images': len(ordered_paths),
                    'initial_pair': initial_pair,
                    'initial_matches': num_matches,
                    'ordered_indices': ordered_indices
                }
                
                print("\n" + "="*60)
                print("SEQUENCING SUMMARY (CHAIN ORDERED)")
                print("="*60)
                print(f"Total images: {metadata['total_images']}")
                print(f"Successfully ordered: {metadata['ordered_images']}")
                print(f"Initial pair matches: {metadata['initial_matches']}")
                print("="*60 + "\n")
                
                return ordered_paths, metadata
        
        # Step 3: If no cache, do full matching
        print("\n⚠️ No match cache found. Running full matching pipeline...")
        
        # Find best initial pair (parallel FLANN matching)
        idx1, idx2, num_matches = self.find_best_initial_pair(image_paths)
        
        # Order remaining images (FLANN matching)
        ordered_indices = self.order_images_greedy(image_paths, idx1, idx2)
        
        # Save match results for next time!
        if match_cache_file:
            self.save_match_results(match_cache_file, ordered_indices, (idx1, idx2), num_matches)
        
        # Create ordered path list
        ordered_paths = [image_paths[i] for i in ordered_indices]
        
        # Metadata
        metadata = {
            'total_images': len(image_paths),
            'ordered_images': len(ordered_paths),
            'initial_pair': (idx1, idx2),
            'initial_matches': num_matches,
            'ordered_indices': ordered_indices
        }
        
        print("\n" + "="*60)
        print("SEQUENCING SUMMARY")
        print("="*60)
        print(f"Total images: {metadata['total_images']}")
        print(f"Successfully ordered: {metadata['ordered_images']}")
        print(f"Initial pair matches: {metadata['initial_matches']}")
        print("="*60 + "\n")
        
        return ordered_paths, metadata

# if __name__ == "__main__":
#     import glob

#     parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     cache_dir = os.path.join(parent_dir, 'cache_files')
#     os.makedirs(cache_dir, exist_ok=True)
#     cache_file = os.path.join(cache_dir, 'feature_cache.pkl')


#     dataset_path = os.path.join(parent_dir, 'data', 'Dataset')

#     # Camera intrinsics
#     K = np.array([
#         [2184, 0, 1512],
#         [0, 2184, 2016],
#         [0, 0, 1]
#     ], dtype=np.float32)
    
#     # Load dataset

    
#     # Check if path exists
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
#     image_paths = sorted(image_paths)
    
#     if len(image_paths) == 0:
#         print(f"ERROR: No images found in {dataset_path}")
#         print(f"Directory contents: {os.listdir(dataset_path)}")
#         exit()
    
#     print(f"Found {len(image_paths)} images:")
#     for path in image_paths[:10]:
#         print(f"  - {os.path.basename(path)}")
#     if len(image_paths) > 10:
#         print(f"  ... and {len(image_paths) - 10} more")
    

#   # Initialize sequencer
#     sequencer = ImageSequencer(K, min_matches=50, n_workers=8)
    
#     # Try to load existing cache
#     cache_loaded = sequencer.load_feature_cache(cache_file)
    
#     if not cache_loaded:
#         print("No cache found, will extract features fresh")
    
#     # Sequence images (will save incrementally during feature extraction!)
#     ordered_paths, metadata = sequencer.sequence_images(
#         image_paths,
#         cache_file=cache_file  # Incremental saving enabled!
#     )
    
#     # Final save (in case any new features were added)
#     num_saved, file_size = sequencer.save_feature_cache(cache_file)
    
#     print("\n" + "="*60)
#     print("FINAL ORDERED SEQUENCE")
#     print("="*60)
#     for i, path in enumerate(ordered_paths):
#         print(f"{i+1}. {os.path.basename(path)}")
    
#     print("\n" + "="*60)
#     print("Image sequencing complete!")
#     print(f"Feature cache saved to: {cache_file}")
#     print(f"Total features cached: {num_saved} images ({file_size:.2f} MB)")
#     print("="*60)

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

target_directory = r"D:\Salman Ahmed LUMS\Personal\LUMS\CV\PROJECT\3D-Reconstruction-SFM\data\Dataset"

filenames = os.listdir(target_directory)
list_of_image_paths = []
for file_name in filenames:
    full_path = os.path.join(target_directory, file_name)
    list_of_image_paths.append(full_path)

print(list_of_image_paths[:5])
test_paths = list_of_image_paths[2:7] # Get the 5 paths you want to test
for path in test_paths:
    try:
        img = load_image_any_format(path)
        print(f"✅ Loaded: {os.path.basename(path)} (Shape: {img.shape})")
    except ValueError as e:
        print(f"❌ FAILED: {os.path.basename(path)} ({e})")
    except Exception as e:
        print(f"❌ ERROR: {os.path.basename(path)} (Unexpected error: {e})")

print("--- Test Complete ---")
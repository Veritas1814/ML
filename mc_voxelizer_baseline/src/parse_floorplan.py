import cv2
import numpy as np
from skimage.morphology import skeletonize

def segment_floorplan(img_bgr, threshold: int = 180) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    bw = (bw > 0).astype(np.uint8)
    skel = skeletonize(bw.astype(bool))
    return skel

def downsample_mask(mask: np.ndarray, voxel_size: int = 1) -> np.ndarray:
    if voxel_size <= 1:
        return mask
    return mask[::voxel_size, ::voxel_size]

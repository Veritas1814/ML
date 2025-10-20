import numpy as np
import cv2
from skimage.morphology import remove_small_objects, skeletonize

def clean_mask(mask: np.ndarray,
               min_component: int = 120,
               close_k: int = 5,
               open_k: int = 0,
               do_skeleton: bool = False,
               thicken: int = 0) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)

    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    m_bool = remove_small_objects(m.astype(bool), min_size=max(1, min_component))
    m = m_bool.astype(np.uint8)

    if do_skeleton:
        m = skeletonize(m.astype(bool)).astype(np.uint8)

    if thicken > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (thicken, thicken))
        m = cv2.dilate(m, k, iterations=1)

    return (m > 0)

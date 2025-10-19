import cv2
import numpy as np

def clean_mask(mask, min_area=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            cleaned_mask[labels == i] = 1
    return cleaned_mask

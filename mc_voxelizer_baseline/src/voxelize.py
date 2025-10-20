import numpy as np
from typing import List, Tuple

def mask_to_voxels(mask: np.ndarray, height: int = 3, y_base: int = 0) -> List[Tuple[int,int,int]]:
    coords = []
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        x = int(c); z = int(r)
        for dy in range(height):
            coords.append((x, y_base + dy, z))
    return coords

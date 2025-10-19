import numpy as np
def mask_to_voxels(mask, height=10):
    h, w = mask.shape
    voxels = np.zeros((height, h, w), dtype=int)
    for z in range(height):
        voxels[z][mask == 1] = 1

    return voxels


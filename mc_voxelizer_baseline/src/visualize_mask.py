import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python visualize_mask.py <path_to_npy>")
    sys.exit(1)

path = sys.argv[1]
mask = np.load(path)

print("Shape:", mask.shape, "dtype:", mask.dtype, "min:", mask.min(), "max:", mask.max())

if mask.ndim == 3:
    mask = mask.sum(axis=-1)

plt.figure(figsize=(6,6))
plt.imshow(mask, cmap="viridis")
plt.title(path)
plt.axis("off")
plt.tight_layout()
plt.show()

import os
from PIL import Image
import numpy as np
from resplan_loader import load_resplan_pkl
from msd_loader import load_msd_csv
from vector_to_raster import rasterize_plan, render_floorplan_image

RAW_DIR = 'C:/Users/matvi/Code/ML/data/raw'
OUT_IMAGE_DIR = 'C:/Users/matvi/Code/ML/data/processed/images'
OUT_MASK_DIR = 'C:/Users/matvi/Code/ML/data/processed/masks'

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

print("Loading floorplans...")
# floorplans = load_resplan_pkl(os.path.join(RAW_DIR, 'ResPlan.pkl'))
floorplans = load_msd_csv(os.path.join(RAW_DIR, 'mds_V2_5.372k.csv'))
print(f"Loaded {len(floorplans)} floorplans")

for i, fp in enumerate(floorplans):
    try:
        print(f"Processing floorplan {i}")
        
        # Get the rendered image
        img = render_floorplan_image(fp)
        if img is None:
            print(f"Warning: Empty image for floorplan {i}")
            continue
            
        # If img is already a PIL Image, save directly
        if isinstance(img, Image.Image):
            img.save(os.path.join(OUT_IMAGE_DIR, f'{i:04d}.png'))
        else:
            # Convert numpy array to PIL Image
            if not isinstance(img, np.ndarray):
                print(f"Warning: Unexpected image type for floorplan {i}: {type(img)}")
                continue
                
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            
            img = Image.fromarray(img)
            img.save(os.path.join(OUT_IMAGE_DIR, f'{i:04d}.png'))

        # Convert polygons to class mask
        mask, _ = rasterize_plan(fp)  # 0=bg, 1=wall, 2=window, 3=door
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(OUT_MASK_DIR, f'{i:04d}.png'))
        
        print(f"Successfully processed floorplan {i}")
        
    except Exception as e:
        print(f"Error processing floorplan {i}: {str(e)}")
        continue

print('Dataset preparation completed!')

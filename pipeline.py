import os
import torch
from model.predict import predict_floorplan
from model.postprocessing import clean_mask
from reconstruction.mask_to_3d import mask_to_voxels
from reconstruction.schematic_exporter import export_schematic
from reconstruction.visualize_3d import visualize_voxels

def run_pipeline(image_path, model_path, output_dir='output', num_classes=4, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)

    print('[1/5] Running floorplan segmentation...')
    mask = predict_floorplan(image_path, model_path, num_classes, device=device)

    print('[2/5] Cleaning segmentation mask...')
    clean = clean_mask(mask)

    print('[3/5] Generating voxel representation...')
    voxels = mask_to_voxels(clean, height=10)

    print('[4/5] Exporting schematic...')
    schematic_path = os.path.join(output_dir, 'floorplan.schematic')
    export_schematic(voxels, schematic_path)

    print('[5/5] Visualizing 3D reconstruction...')
    visualize_voxels(voxels)

    print(f'Pipeline complete. Schematic saved to: {schematic_path}')

if __name__ == '__main__':
    image_path = 'floorplan.png'
    model_path = 'unet_floorplan.pth'
    run_pipeline(image_path, model_path)

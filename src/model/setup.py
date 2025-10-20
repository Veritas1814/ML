import os
import sys
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from train_model import train_model
from data_loader.floorplan_dataset import FloorplanDataset

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_dir = os.path.join(project_root, 'data', 'processed', 'images')
mask_dir = os.path.join(project_root, 'data', 'processed', 'masks')

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
else:
    device = torch.device('cpu')

if device.type == 'cuda':
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    print(f"Total GPU memory: {total_gb:.1f} GB, Free: {free_gb:.1f} GB")

    batch_size = 2
    gradient_accumulation_steps = 2
    num_epochs = 5
    print(f"Using batch_size={batch_size}, accumulation_steps={gradient_accumulation_steps}")

else:
    batch_size = 1
    gradient_accumulation_steps = 1
    num_epochs = 5


def main():
    print(f"Using device: {device} | batch_size: {batch_size} | accumulation_steps: {gradient_accumulation_steps}")
    train_dataset = FloorplanDataset(image_dir, mask_dir)
    val_dataset = FloorplanDataset(image_dir, mask_dir)

    train_model(
        train_dataset,
        val_dataset,
        num_classes=4,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=True
    )


if __name__ == "__main__":
    main()


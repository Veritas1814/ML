import torch
import cv2
import numpy as np
from model.unet import UNet

def predict_floorplan(image_path, model_path, num_classes, device='cuda'):
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    return mask
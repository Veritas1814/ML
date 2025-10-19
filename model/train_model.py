import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from unet import UNet
from torch.amp import GradScaler, autocast

def train_model(train_dataset, val_dataset, num_classes, num_epochs=30, 
                lr=1e-3, batch_size=8, gradient_accumulation_steps=1, mixed_precision=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Optimized DataLoader ---
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers, persistent_workers=True
    )

    scaler = GradScaler(enabled=mixed_precision)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        for i, (images, masks) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=mixed_precision):
                outputs = model(images)
                loss = criterion(outputs, masks) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * gradient_accumulation_steps

            if i % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}")

        # Validate every 3 epochs
        if (epoch + 1) % 3 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    with autocast(device_type='cuda', enabled=mixed_precision):
                        outputs = model(images)
                        val_loss += criterion(outputs, masks).item()
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), 'unet_floorplan.pth')
    return model

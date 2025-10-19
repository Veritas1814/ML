import albumentations as A
def get_training_augmentations(image_size=512):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
    ])

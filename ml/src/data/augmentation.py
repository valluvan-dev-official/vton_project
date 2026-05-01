"""Data augmentation transforms for VTON training."""
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class SynchronizedAugment:
    """
    Apply the same random flip/crop to person, garment, and result
    so spatial alignment is preserved.
    """

    def __init__(self, image_size: int = 512, flip_prob: float = 0.5, crop_scale: tuple = (0.85, 1.0)):
        self.size = image_size
        self.flip_prob = flip_prob
        self.crop_scale = crop_scale

    def __call__(self, person: Image.Image, garment: Image.Image, result: Image.Image):
        # Random horizontal flip
        if random.random() < self.flip_prob:
            person = TF.hflip(person)
            garment = TF.hflip(garment)
            result = TF.hflip(result)

        # Random resized crop (same params for all)
        i, j, h, w = T.RandomResizedCrop.get_params(
            person, scale=self.crop_scale, ratio=(0.9, 1.1)
        )
        person = TF.resized_crop(person, i, j, h, w, (self.size, self.size))
        garment = TF.resized_crop(garment, i, j, h, w, (self.size, self.size))
        result = TF.resized_crop(result, i, j, h, w, (self.size, self.size))

        return person, garment, result


def get_train_transforms(image_size: int = 512) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_val_transforms(image_size: int = 512) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

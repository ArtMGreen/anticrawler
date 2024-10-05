import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset:

    def __init__(self, image_paths, targets, resize = None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(always_apply = True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):

        image = Image.open(self.image_paths[item]).convert("RGB")
        label = self.targets[item]
        if self.resize:
            image = image.resize((self.resize[1], self.resize[0]), resample = Image.BILINEAR)
        image = np.array(image)
        image = self.aug(image = image)["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype = torch.float),
            "target": torch.tensor(label, dtype = torch.long)
        }
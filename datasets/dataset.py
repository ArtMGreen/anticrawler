import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype

import os


class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.labels = [self._get_label_from_filename(img) for img in os.listdir(image_dir)]

    def _get_label_from_filename(self, filename):
        label = filename.split('.')[0]  # Assuming the label is in the filename
        encoded_label = []
        for char in label:
            if char.isdigit():
                encoded_label.append(ord(char) - ord('0'))  # Encode digits as 0-9
            elif char.isalpha():
                encoded_label.append(ord(char.upper()) - ord('A') + 10)  # Encode letters as 10-35
        return encoded_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], mode=ImageReadMode.RGB)
        image = convert_image_dtype(image, dtype=torch.float)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
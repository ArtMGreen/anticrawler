import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype

import os
import random

class CaptchaDataset(Dataset):
    def __init__(self, image_dir=None, image_paths=None, transform=None):
        self.transform = transform
        if image_dir:
            self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        elif image_paths:
            self.image_paths = image_paths

        self.labels = [self._get_label_from_filename(img) for img in self.image_paths]

    @staticmethod
    def _get_label_from_filename(filename):
        label = filename.split('/')[-1].split('.')[0]  # Assuming the label is in the filename
        encoded_label = []
        for char in label:
            if char.isdigit():
                encoded_label.append(ord(char) - ord('0'))  # Encode digits as 0-9
            elif char.isalpha():
                encoded_label.append(ord(char.upper()) - ord('A') + 10)  # Encode letters as 10-35
        return encoded_label

    def train_test_split(self, ratio:float=0.2) -> (Dataset, Dataset):
        imgs = list(self.image_paths)
        random.shuffle(imgs)

        n = len(self)
        train_size = n - int(n * ratio)

        train_paths = imgs[:train_size]
        test_paths = imgs[train_size:]

        train = CaptchaDataset(image_paths=train_paths, transform=self.transform)
        test = CaptchaDataset(image_paths=test_paths, transform=self.transform)

        return train, test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], mode=ImageReadMode.RGB)
        image = convert_image_dtype(image, dtype=torch.float)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
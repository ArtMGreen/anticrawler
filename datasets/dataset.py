import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype

import os
import random

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        if image_dir:
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

    def train_test_split(self, ratio:float=0.2) -> (Dataset, Dataset):
        train = CaptchaDataset(None)
        test = CaptchaDataset(None)
        train.image_dir = self.image_dir
        test.image_dir = self.image_dir

        imgs = list(zip(self.image_paths, self.labels))
        random.shuffle(imgs)

        n = len(self)
        train_size = n - int(n * ratio)

        train.image_paths = []
        train.labels = []
        for item in imgs[:train_size]:
            train.image_paths.append(item[0])
            train.labels.append(item[1])

        test.image_paths = []
        test.labels = []
        for item in imgs[train_size:]:
            test.image_paths.append(item[0])
            test.labels.append(item[1])

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
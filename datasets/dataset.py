import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype


class CaptchaDataset(Dataset):
    def __init__(self, image_dir=None, image_paths=None, transform=None):
        """
        Args:
            image_dir (str/list[str]): Directory or a list of directories containing the images.
            image_paths (list[str]): List of individual image file paths.
            transform: Optional transformations to apply to the images.
        """
        self.transform = transform

        if image_dir:
            self.image_paths = list()
            if not isinstance(image_dir, list):
                image_dir = [image_dir]
            for single_dir in image_dir:
                self.image_paths.extend([os.path.join(single_dir, img) for img in os.listdir(single_dir)])

        elif image_paths:
            self.image_paths = image_paths
        else:
            raise ValueError("Either 'image_dir' or 'image_paths' must be provided.")

        self.labels = [self._encode_label(self._get_label_string_from_filename(path)) for path in self.image_paths]

    @staticmethod
    def _get_label_string_from_filename(filename):
        """
        Extract the raw label string from the filename - the part of the filename before the first '.'.

        Args:
            filename (str): Full path or filename.

        Returns:
            str: Label as a string.
        """
        return os.path.splitext(os.path.basename(filename))[0]

    @staticmethod
    def _encode_label(label):
        """
        Encode the label string into a list of integers.

        Characters are encoded as:
            - Digits ('0'-'9') -> 0-9
            - Letters ('A'-'Z', case-insensitive) -> 10-35

        Args:
            label (str): Raw label string.

        Returns:
            list[int]: Encoded label as a list of integers.
        """
        return [
            (ord(char) - ord('0')) if char.isdigit() else (ord(char.upper()) - ord('A') + 10)
            for char in label
        ]

    def train_test_split(self, test_ratio=0.2):
        """
        Split the dataset into two (train and test) subsets.

        Args:
            test_ratio (float): Proportion of the second set.

        Returns:
            tuple[CaptchaDataset, CaptchaDataset]: Two subsets of the dataset.
        """
        shuffled_paths = self.image_paths[:]
        random.shuffle(shuffled_paths)

        split_idx = int(len(shuffled_paths) * (1 - test_ratio))
        train_paths = shuffled_paths[:split_idx]
        test_paths = shuffled_paths[split_idx:]

        train_dataset = CaptchaDataset(image_paths=train_paths, transform=self.transform)
        test_dataset = CaptchaDataset(image_paths=test_paths, transform=self.transform)
        return train_dataset, test_dataset

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """
        Load and preprocess an image given its file path.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = read_image(path, mode=ImageReadMode.RGB)
        image = convert_image_dtype(image, dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The image and label tensors.
        """
        image = self._load_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

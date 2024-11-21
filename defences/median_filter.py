import torch
from torch import nn
import numpy as np


class MedianFilter(nn.Module):
    def __init__(self, kernel_size):
        super(MedianFilter, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, image):
        ch, height, width = image.size()
        image_filtered = torch.zeros_like(image)
        kernel_half_size = self.kernel_size // 2

        for i in range(height):
            for j in range(width):
                filter_x_start = np.clip(j - kernel_half_size, 0, width-1)
                filter_x_end = np.clip(j + kernel_half_size + 1, 0, width-1)
                filter_y_start = np.clip(i - kernel_half_size, 0, height-1)
                filter_y_end = np.clip(i + kernel_half_size + 1, 0, height-1)
                roi = image[:, filter_y_start:filter_y_end, filter_x_start:filter_x_end]
                image_filtered[:, i, j] = torch.Tensor(np.median(roi, axis=(1, 2)))

        return image_filtered

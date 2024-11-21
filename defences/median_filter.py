import cv2
import numpy as np
from torch import nn
import torch

class MedianFilter(nn.Module):
    def __init__(self, kernel_size):
        super(MedianFilter, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        self.kernel_size = kernel_size

    def forward(self, image):
        img_np = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

        filtered_np = np.stack(
            [cv2.medianBlur(img_np[:, :, c], self.kernel_size) for c in range(img_np.shape[2])],
            axis=2
        )

        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return filtered_tensor
import cv2
from torch import nn
import torch


class Thresholding(nn.Module):
    def __init__(self):
        super(Thresholding, self).__init__()

    def forward(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
        # Use a bimodal image as an input.
        # Optimal threshold value is determined automatically.
        otsu_threshold, image_result = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return torch.tensor(image_result)
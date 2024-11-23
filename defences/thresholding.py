import cv2
import numpy as np
from torch import nn
import torch
import torchvision.transforms.v2 as transforms
from defences.io_utils import load_image, save_image


class Thresholding(nn.Module):
    def __init__(self):
        super(Thresholding, self).__init__()

    def forward(self, image):
        img_gray = transforms.Grayscale()(image)
        img_np = (img_gray.squeeze().cpu().numpy() * 255).astype(np.uint8)
        # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
        # Use a bimodal image as an input.
        # Optimal threshold value is determined automatically.
        otsu_threshold, image_result = cv2.threshold(
            img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        binary_img = torch.from_numpy(image_result).float() / 255.0
        return torch.stack([binary_img, binary_img, binary_img], dim=0)


def thresholding_defend_image_from_path(input_path, save_path):
    res_tensor = load_image(input_path, Thresholding())
    save_image(res_tensor, save_path)
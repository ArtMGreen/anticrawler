import torch
from torch import nn
import numpy as np
from defences.io_utils import _load_image, _save_image


class GradientTransform(nn.Module):
    def forward(self, image):
        image = image.unsqueeze(0)

        sobel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]] * 3)
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(sobel_x).float().unsqueeze(0), requires_grad=False)
        G_x = conv1(image)

        sobel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]] * 3)
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(sobel_y).float().unsqueeze(0), requires_grad=False)
        G_y = conv2(image)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2)).squeeze(0).squeeze(0)
        G = torch.clamp((G - G.min()) / G.max(), 0, 1)  # to [0, 1]

        return torch.stack([G, G, G], dim=0)


def gradient_transform_defend_image_from_path(input_path, save_path):
    res_tensor = _load_image(input_path, GradientTransform())
    _save_image(res_tensor, save_path)
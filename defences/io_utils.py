from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype, to_pil_image
from torch import float as float_dtype
import os


def load_image(path, transform):
    """
    Load and preprocess an image given its file path.

    Args:
        path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = read_image(path, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=float_dtype)
    image = transform(image)
    return image


def save_image(img_tensor, path):
    """
    Save an image tensor into the file path.

    Args:
        img_tensor (torch.Tensor): An image tensor to save.
        path (str): Path to the image file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adv_image_pil = to_pil_image(img_tensor.squeeze(0).detach().cpu())
    adv_image_pil.save(path)
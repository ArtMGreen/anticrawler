from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype, to_pil_image
from torch import float as float_dtype
# from torchvision.transforms import Compose
import os


def _load_image(path, transform):
    """
    Load and preprocess an image given its file path.

    Args:
        path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = read_image(path, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=float_dtype)
    # composed_transform = Compose([transform])
    # image = composed_transform(image)
    image = transform(image)
    return image


def _save_image(img_tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adv_image_pil = to_pil_image(img_tensor.squeeze(0).detach().cpu())
    adv_image_pil.save(path)
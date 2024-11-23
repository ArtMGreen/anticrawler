from defences.median_filter import median_filter_defend_image_from_path as median_filter
from defences.thresholding import thresholding_defend_image_from_path as thresholding
from defences.gradient_transform import gradient_transform_defend_image_from_path as gradient_transform
from defences.identity import Identity
import torchvision.transforms.v2 as transforms
from defences.io_utils import load_image, save_image


def gaussian_blur(input_path, save_path):
    res_tensor = load_image(input_path, transforms.GaussianBlur(kernel_size=5, sigma=1))
    save_image(res_tensor, save_path)


def grayscale(input_path, save_path):
    res_tensor = load_image(input_path, transforms.Grayscale(num_output_channels=3))
    save_image(res_tensor, save_path)


def gaussian_noise(input_path, save_path):
    res_tensor = load_image(input_path, transforms.GaussianNoise())
    save_image(res_tensor, save_path)


def normalize(input_path, save_path):
    res_tensor = load_image(input_path, transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    save_image(res_tensor, save_path)
import os
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import to_pil_image

from tqdm import tqdm

from datasets.dataset import CaptchaDataset
from models.resnet_captcha_model_definition import ResNetCaptchaModel, differentiable_predict


def PGD_attack_image(model, image, label, device, epsilon=1.0, alpha=0.01, num_iter=40, save_path=None):
    model.eval()
    image = image.to(device)
    label = label.to(device)
    original_image = image.clone().detach()

    for i in range(num_iter):
        # Apply gradient step
        prediction, dL_dx = differentiable_predict(model, image, label, device)
        image = image + alpha * torch.sign(dL_dx)

        # Clamp image to maintain within eps bounds and normalize
        image = torch.clamp(image, original_image - epsilon, original_image + epsilon)
        image = torch.clamp(image, 0, 1).detach()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_pil = to_pil_image(image)
        image_pil.save(save_path)

    return image

def PGD_attack_dataset(model, dataset, attacked_directory, device, epsilon=1.0, alpha=0.01, num_iter=40):
    for idx in tqdm(range(len(dataset)), total=len(dataset), leave=True, desc="Attacking via PGD"):
        image, label = dataset[idx]
        image_path = dataset.image_paths[idx]
        save_path = os.path.join(attacked_directory, os.path.basename(image_path))
        PGD_attack_image(model, image, label, device, epsilon=epsilon, alpha=alpha, num_iter=num_iter, save_path=save_path)

if __name__ == "__main__":
    image_directory = '../datasets/fournierp_captcha-version-2-images'
    attacked_directory = '../datasets/PGD_attacked'

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetCaptchaModel().to(device)
    checkpoint = torch.load("../models/captcha_resnet50.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_to_attack = CaptchaDataset(image_dir=image_directory, transform=None)
    PGD_attack_dataset(model, dataset_to_attack, attacked_directory, device, epsilon=0.05, alpha=0.01, num_iter=10)

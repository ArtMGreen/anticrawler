import os
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import to_pil_image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.dataset import CaptchaDataset
from models.resnet_captcha_model_definition import ResNetCaptchaModel, differentiable_predict


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def FGSM_attack_image(model, image, label, device, epsilon=1.0, save_path=None):
    image = image.to(device)

    prediction, dL_dx = differentiable_predict(model, image, label, device)
    evil_noise = torch.sign(dL_dx)
    evil_image = image + epsilon * evil_noise
    evil_image = torch.clamp(evil_image, 0, 1).detach()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        evil_image_pil = to_pil_image(evil_image)
        evil_image_pil.save(save_path)

    return evil_image


def FGSM_attack_dataset(model, dataset, attacked_directory, device, epsilon=1.0):
    for idx in tqdm(range(len(dataset)), total=len(dataset), leave=True, desc="Attacking via FGSM"):
        image, label = dataset[idx]
        image_path = dataset.image_paths[idx]
        save_path = os.path.join(attacked_directory, os.path.basename(image_path))
        FGSM_attack_image(model, image, label, device, epsilon=epsilon, save_path=save_path)


if __name__ == "__main__":
    image_directory = '../datasets/fournierp_captcha-version-2-images'
    attacked_directory = '../datasets/FGSM_attacked'
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # -- deprecated | transforms.ToTensor(),
        # -- redundant | transforms.ToImage(),
        # -- redundant | transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetCaptchaModel().to(device)
    checkpoint = torch.load("../models/captcha_resnet50.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_to_attack = CaptchaDataset(image_dir=image_directory, transform=None)
    FGSM_attack_dataset(model, dataset_to_attack, attacked_directory, device, epsilon=0.05)
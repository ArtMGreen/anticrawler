import os
import random

import torch
from torch import nn
import torchvision.transforms.v2 as transforms

import numpy as np
from tqdm import tqdm

from models.resnet_captcha_model_definition import ResNetCaptchaModel, differentiable_predict


CHAR_TYPES_NUM = 36  # Assumption: 10 digits + 26 letters
CAPTCHA_LENGTH = 5  # Assumption: CAPTCHA length is 5
image_directory = '../datasets/fournierp_captcha-version-2-images'
attacked_directory = '../datasets/FGSM_attacked'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetCaptchaModel(CHAR_TYPES_NUM, CAPTCHA_LENGTH).to(device)
checkpoint = torch.load("../models/captcha_resnet50.pth", weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fvis

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = Fvis.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # -- deprecated | transforms.ToTensor(),
    # -- redundant | transforms.ToImage(),
    # -- redundant | transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

criterion = nn.CrossEntropyLoss()

from torchvision.transforms.v2.functional import convert_image_dtype, to_pil_image
from torchvision.io import read_image, ImageReadMode

def evil_noise_fgsm(model, path_to_img, save_path, device, criterion, transform=None, epsilon=0.0):
    prediction, dL_dx = differentiable_predict(model, path_to_img, device, criterion, transform)
    evil_noise = torch.sign(dL_dx)

    image = read_image(path_to_img, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=torch.float)
    image = image.to(device)

    evil_image = image + epsilon * evil_noise

    evil_image = torch.clamp(evil_image, 0, 1).detach().cpu()
    evil_image_pil = to_pil_image(evil_image)
    evil_image_pil.save(save_path)

for imgname in tqdm(os.listdir(image_directory), total=len(os.listdir(image_directory)), leave=True):
    evil_noise_fgsm(model, image_directory+'/'+imgname,
                    attacked_directory+'/'+imgname, device, criterion, transform, epsilon=0.1)
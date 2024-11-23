import os
import torch
from torchvision.transforms.v2.functional import to_pil_image

from tqdm import tqdm

from datasets.dataset import CaptchaDataset
from models.resnet_captcha_model_definition import ResNetCaptchaModel

def CW_attack_image(model, image, label, device, c=0.5, lr=0.01, num_iter=100, kappa=0, save_path=None):
    model.eval()
    image = image.to(device).unsqueeze(0)
    label = label.to(device).unsqueeze(0)
    # chosen = torch.tensor([0] * model.num_chars, dtype=torch.long, device=device).unsqueeze(0)

    # Inverted change of variables, see below
    w = torch.arctanh((image * 2 - 1) * 0.99999).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for i in range(num_iter):
        # Change of variables from original paper
        adv_image = 0.5 * (torch.tanh(w) + 1)
        outputs = model(adv_image)  # Shape: [1, 5, 36]

        label_onehot = torch.zeros_like(outputs).scatter_(2, label.unsqueeze(2), 1)

        f = - torch.max((outputs - kappa) * (1 - label_onehot), dim=2)[0] + outputs.gather(2, label.unsqueeze(2)).squeeze(2) + kappa
        misclassification_loss = torch.sum(c * torch.clamp(f, min=0))
        l2_loss = torch.sum((adv_image - image) ** 2)
        # print("Incorrect Logits:", torch.max((outputs - kappa) * (1 - label_onehot), dim=2)[0])
        # print("True Logits:", outputs.gather(2, label.unsqueeze(2)).squeeze(2))
        # print("Loss Term f:", f)
        #
        # print(misclassification_loss.item())
        # print(l2_loss.item())
        loss = misclassification_loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        adv_image_pil = to_pil_image(adv_image.squeeze(0).detach().cpu())
        adv_image_pil.save(save_path)

    return adv_image.squeeze(0)


def CW_attack_image_from_path(model, input_path, save_path, device='cpu', c=1, lr=0.01, num_iter=15, kappa=0):
    """For purely illustrative purposes of the web app; no transforms available"""
    d = CaptchaDataset(image_paths=[input_path])
    image, label = d[0]
    CW_attack_image(model, image, label, device, c=c, lr=lr, num_iter=num_iter, kappa=kappa, save_path=save_path)


def CW_attack_dataset(model, dataset, attacked_directory, device, c=1, lr=0.01, num_iter=100, kappa=0):
    for idx in tqdm(range(len(dataset)), total=len(dataset), leave=True, desc="Attacking via CW"):
        image, label = dataset[idx]
        image_path = dataset.image_paths[idx]
        save_path = os.path.join(attacked_directory, os.path.basename(image_path))
        CW_attack_image(model, image, label, device, c=c, lr=lr, num_iter=num_iter, kappa=kappa, save_path=save_path)


if __name__ == "__main__":
    image_directory = '../datasets/fournierp_captcha-version-2-images'
    test_directory = '../datasets/test_set'
    attacked_directory = '../datasets/CW_attacked'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetCaptchaModel().to(device)
    checkpoint = torch.load("../models/captcha_resnet50.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_to_attack = CaptchaDataset(image_dir=test_directory, transform=None)
    CW_attack_dataset(model, dataset_to_attack, attacked_directory, device, c=1, lr=0.05, num_iter=20, kappa=1)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

import torchvision.transforms.v2 as transforms

from datasets.dataset import CaptchaDataset
from models.resnet_captcha_model_definition import ResNetCaptchaModel
from models.resnet_captcha_model_definition import train_epoch
from models.resnet_captcha_model_definition import eval_epoch
from models.resnet_captcha_model_definition import predict

batch_size = 32
learning_rate = 0.001
num_epochs = 10

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
image_directory = os.path.join(ROOT_DIR, 'datasets', 'fournierp_captcha-version-2-images')
FGSM_attacked_directory = os.path.join(ROOT_DIR, 'datasets', 'FGSM_attacked')
PGD_attacked_directory = os.path.join(ROOT_DIR, 'datasets', 'PGD_attacked')
CW_attacked_directory = os.path.join(ROOT_DIR, 'datasets', 'CW_attacked')

used_directory = CW_attacked_directory
img_filename = 'gwnm6.png'
training, evaluation, prediction = False, True, False


def train_run(model, dataset: CaptchaDataset, optimizer, device):
    train_dataset, test_dataset = dataset.train_test_split()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
        eval_epoch(model, val_loader, device, epoch + 1)

    torch.save(model.state_dict(), 'captcha_resnet50.pth')


def eval_run(model, dataset: CaptchaDataset, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    avg_loss, accuracy = eval_epoch(model, dataloader, device)
    print(f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")


def main(training=False, evaluation=False, prediction=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # -- deprecated | transforms.ToTensor(),
        # -- redundant | transforms.ToImage(),
        # -- redundant | transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = ResNetCaptchaModel().to(device)

    dataset = CaptchaDataset(image_dir=used_directory, transform=None)

    if training:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_run(model, dataset, optimizer, device)

    if evaluation:
        checkpoint = torch.load("captcha_resnet50.pth", weights_only=True)
        model.load_state_dict(checkpoint)
        eval_run(model, dataset, device)

    if prediction:
        checkpoint = torch.load("captcha_resnet50.pth", weights_only=True)
        model.load_state_dict(checkpoint)

        path_to_img = used_directory + '/' + img_filename
        predicted_captcha = predict(model, path_to_img, device, transform)
        print(f"Predicted CAPTCHA: {predicted_captcha}")


if __name__ == "__main__":
    main(training, evaluation, prediction)

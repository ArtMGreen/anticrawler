import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as transforms

from datasets.dataset import CaptchaDataset
from resnet_captcha_model_definition import ResNetCaptchaModel
from resnet_captcha_model_definition import train_epoch
from resnet_captcha_model_definition import eval_epoch
from resnet_captcha_model_definition import predict


CHAR_TYPES_NUM = 36  # Assumption: 10 digits + 26 letters
CAPTCHA_LENGTH = 5  # Assumption: CAPTCHA length is 5
batch_size = 32
learning_rate = 0.001
num_epochs = 10

image_directory = '../datasets/fournierp_captcha-version-2-images'
img_filename = '2en7g.png'
training, evaluation, prediction = True, True, True


def train_run(model, image_dir, transform, criterion, optimizer, device):
    dataset = CaptchaDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch+1)
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

    torch.save(model.state_dict(), 'captcha_resnet50.pth')


def eval_run(model, image_dir, transform, criterion, device):
    dataset = CaptchaDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    avg_loss, accuracy = eval_epoch(model, dataloader, criterion, device)
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

    model = ResNetCaptchaModel(CHAR_TYPES_NUM, CAPTCHA_LENGTH).to(device)

    if training:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_run(model, image_directory, transform, criterion, optimizer, device)

    if evaluation:
        criterion = nn.CrossEntropyLoss()
        checkpoint = torch.load("captcha_resnet50.pth", weights_only=True)
        model.load_state_dict(checkpoint)
        eval_run(model, image_directory, transform, criterion, device)

    if prediction:
        checkpoint = torch.load("captcha_resnet50.pth", weights_only=True)
        model.load_state_dict(checkpoint)

        path_to_img = image_directory + '/' + img_filename
        predicted_captcha = predict(model, path_to_img, device, transform)
        print(f"Predicted CAPTCHA: {predicted_captcha}")


if __name__ == "__main__":
    main(training, evaluation, prediction)

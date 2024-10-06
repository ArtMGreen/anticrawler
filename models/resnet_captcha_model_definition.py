import torch
from torch import nn

import torchvision.models as models
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype

from tqdm import tqdm


class ResNetCaptchaModel(nn.Module):
    '''https://medium.com/swlh/solving-captchas-using-resnet-50-without-using-ocr-3bdfbd0004a4'''
    def __init__(self, num_classes_per_char, num_chars):
        super(ResNetCaptchaModel, self).__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Modify the final layer to output num_classes_per_char * num_chars
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes_per_char * num_chars)
        self.num_classes_per_char = num_classes_per_char
        self.num_chars = num_chars

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, self.num_chars,
                   self.num_classes_per_char)  # Reshape to (batch_size, num_chars, num_classes_per_char)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch_num}, train", leave=True)
    model.train()
    train_loss = 0.0
    iter_num = 0
    for images, labels in loop:
        iter_num += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # (batch_size, num_chars, num_classes_per_char)

        loss = 0
        for i in range(model.num_chars):
            loss += criterion(outputs[:, i, :], labels[:, i])  # Calculate loss for each character
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss/iter_num})

    return train_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, device, epoch_num=0):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    iter_num = 0
    loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch_num}, evaluation", leave=True)

    with torch.no_grad():
        for images, labels in loop:
            iter_num += 1
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # (batch_size, num_chars, num_classes_per_char)

            loss = 0
            correct = 0
            for i in range(model.num_chars):
                loss += criterion(outputs[:, i, :], labels[:, i])
                _, predicted = torch.max(outputs[:, i, :], 1)
                correct += (predicted == labels[:, i]).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_chars += labels.size(0) * model.num_chars

            loop.set_postfix({"avg_loss": total_loss/iter_num, "accuracy": total_correct/total_chars})

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_chars  # Accuracy at character level
    return avg_loss, accuracy


def predict(model, path_to_img, device, transform=None):
    model.eval()
    # image = Image.open(path_to_img).convert('RGB')
    image = read_image(path_to_img, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=torch.float)

    if transform:
        image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)  # (1, num_chars, num_classes_per_char)

    predicted_chars = []
    for i in range(model.num_chars):
        _, predicted = torch.max(outputs[:, i, :], 1)
        predicted_chars.append(predicted.item())

    # Convert indices back to characters (0-9 and A-Z)
    captcha_text = ''.join([chr(c + ord('0')) if c < 10 else chr(c - 10 + ord('A')) for c in predicted_chars])

    return captcha_text
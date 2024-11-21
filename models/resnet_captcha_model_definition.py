import torch
from torch import nn

import torchvision.models as models
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import convert_image_dtype

from tqdm import tqdm

from datasets.dataset import CaptchaDataset


class ResNetCaptchaModel(nn.Module):
    '''https://medium.com/swlh/solving-captchas-using-resnet-50-without-using-ocr-3bdfbd0004a4'''
    def __init__(self,
                 num_classes_per_char = 36,  # Assumption: 10 digits + 26 letters
                 num_chars = 5,              # Assumption: CAPTCHA length is 5:
                 criterion = None            # nn.CrossEntropyLoss()
                ):
        super(ResNetCaptchaModel, self).__init__()

        self.num_classes_per_char = num_classes_per_char
        self.num_chars = num_chars
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Modify the final layer to output num_classes_per_char * num_chars
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes_per_char * num_chars)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, self.num_chars,
                   self.num_classes_per_char)  # Reshape to (batch_size, num_chars, num_classes_per_char)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)  # (batch_size, num_chars, num_classes_per_char)
        loss = 0
        for i in range(self.num_chars):
            loss += self.criterion(outputs[:, i, :], labels[:, i])  # Calculate loss for each character
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)  # (batch_size, num_chars, num_classes_per_char)
        predicted_classes = torch.argmax(outputs, dim=2)  # (batch_size, num_chars)
        correct = (predicted_classes == labels).sum().item()

        loss = 0
        for i in range(self.num_chars):
            loss += self.criterion(outputs[:, i, :], labels[:, i])
        return loss, correct

    def inference(self, images, labels=None):
        outputs = self(images)  # (batch_size, num_chars, num_classes_per_char)
        predicted_texts = list()
        predicted_classes = torch.argmax(outputs, dim=2)  # (batch_size, num_chars)
        for prediction in predicted_classes:
            captcha_text = ''.join([chr(c + ord('0')) if c < 10 else chr(c - 10 + ord('A')) for c in prediction])
            predicted_texts.append(captcha_text)

        if labels is not None:
            correct = (predicted_classes == labels).sum().item()
            loss = 0
            for i in range(self.num_chars):
                loss += self.criterion(outputs[:, i, :], labels[:, i])
            return loss, correct, predicted_texts
        else:
            return predicted_texts


def train_epoch(model, dataloader, optimizer, device, epoch_num):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num}, train", leave=True)
    model.train()
    train_loss = 0.0
    for batch_idx, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # outputs = model(images)  # (batch_size, num_chars, num_classes_per_char)

        loss = model.training_step((images, labels), batch_idx)
        # loss = 0
        # for i in range(model.num_chars):
        #     loss += model.criterion(outputs[:, i, :], labels[:, i])  # Calculate loss for each character
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss/(batch_idx+1)})

    return train_loss / len(dataloader)


def eval_epoch(model, dataloader, device, epoch_msg=""):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=epoch_msg, leave=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            # outputs = model(images)  # (batch_size, num_chars, num_classes_per_char)

            loss, correct = model.validation_step((images, labels), batch_idx)
            # loss = 0
            # correct = 0
            # for i in range(model.num_chars):
            #     loss += model.criterion(outputs[:, i, :], labels[:, i])
            #     _, predicted = torch.max(outputs[:, i, :], 1)
            #     correct += (predicted == labels[:, i]).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_chars += labels.size(0) * model.num_chars

            loop.set_postfix({"avg_loss": total_loss/(batch_idx+1), "accuracy": total_correct/total_chars})

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_chars  # Accuracy at character level
    return avg_loss, accuracy


def predict(model, path_to_img, device, transform=None):
    model.eval()
    # image = Image.open(path_to_img).convert('RGB')
    image = read_image(path_to_img, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=torch.float)

    if transform is not None:
        image = transform(image)
    image = image.to(device).unsqueeze(0)  # Add batch dimension (1, C, H, W)

    with torch.no_grad():
    #     outputs = model(image)  # (1, num_chars, num_classes_per_char)
    #
    # predicted_chars = []
    # for i in range(model.num_chars):
    #     _, predicted = torch.max(outputs[:, i, :], 1)
    #     predicted_chars.append(predicted.item())
        text = model.inference(image)[0]

    # # Convert indices back to characters (0-9 and A-Z)
    # captcha_text = ''.join([chr(c + ord('0')) if c < 10 else chr(c - 10 + ord('A')) for c in predicted_chars])

    return text

def differentiable_predict(model, image, label, device):
    model.eval()
    image.requires_grad = True
    image, label = image.to(device).unsqueeze(0), label.to(device).unsqueeze(0)
    loss, correct, texts = model.inference(image, label)
    captcha_text = texts[0]

    model.zero_grad()
    image.retain_grad()
    loss.backward()

    return captcha_text, image.grad.squeeze(0)


def differentiable_predict_from_path(model, path_to_img, device, transform=None):
    # image = Image.open(path_to_img).convert('RGB')
    image = read_image(path_to_img, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=torch.float)
    if transform is not None:
        image = transform(image)
    # image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W) - will be done in differential_predict()

    label_str = CaptchaDataset._get_label_string_from_filename(path_to_img)
    label = torch.tensor(CaptchaDataset._encode_label(label_str), dtype=torch.long)
    # label = label.unsqueeze(0) - will be done in differential_predict()

    captcha_text, gradient = differentiable_predict(model, image, label, device)

    return captcha_text, gradient
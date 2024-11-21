import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

import torchvision.transforms.v2 as transforms

from datasets.dataset import CaptchaDataset

from defences.identity import Identity
from defences.gradient_transform import GradientTransform
from defences.median_filter import MedianFilter
from defences.thresholding import Thresholding

from models.resnet_captcha_model_definition import ResNetCaptchaModel
from models.resnet_captcha_model_definition import train_epoch
from models.resnet_captcha_model_definition import eval_epoch
from models.resnet_captcha_model_definition import predict

batch_size = 32
learning_rate = 0.001
num_epochs = 10

# the directories for all flavors of the dataset
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
image_directory = os.path.join(ROOT_DIR, 'datasets', 'fournierp_captcha-version-2-images')
FGSM_attacked_directory = os.path.join(ROOT_DIR, 'datasets', 'FGSM_attacked')
PGD_attacked_directory = os.path.join(ROOT_DIR, 'datasets', 'PGD_attacked')
CW_attacked_directory = os.path.join(ROOT_DIR, 'datasets', 'CW_attacked')
# if multiple directories specified:
#                 in training - they shall appear in a single joint dataset
#    in evaluation/prediction - the run shall be repeated on each one separately
used_directories = [image_directory, FGSM_attacked_directory, PGD_attacked_directory, CW_attacked_directory]

# the files for all versions of the model
vanilla_model = "captcha_resnet50.pth"
normalized_images_model = "normalized_images_resnet50.pth"
adversarially_trained_model = "adversarial_trained_resnet50.pth"
# only one model can be specified
used_model = vanilla_model

# all the transformations we test. If multiple specified:
#      in training and prediction - they shall be composed in a single one
#                   in evaluation - the run shall be repeated on each one separately
used_transforms = (
    Identity(),
    MedianFilter(kernel_size=3),
    MedianFilter(kernel_size=5),
    transforms.GaussianBlur(kernel_size=3, sigma=1),
    transforms.GaussianBlur(kernel_size=5, sigma=1),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianNoise(),
    Thresholding(),
    # GradientTransform(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)

img_filename = 'gwnm6.png'
training, evaluation, prediction = False, True, False


def train_run(model, dataset: CaptchaDataset, optimizer, device):
    train_dataset, test_dataset = dataset.train_test_split()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
        eval_epoch(model, val_loader, device, epoch_msg=f"Epoch {epoch + 1}, evaluation")

    torch.save(model.state_dict(), used_model)


def eval_run(model, dataset: CaptchaDataset, device, epoch_msg=""):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    avg_loss, accuracy = eval_epoch(model, dataloader, device, epoch_msg)
    # print(f"Accuracy: {accuracy:.5f}, Loss: {avg_loss:.4f}")


def main(training=False, evaluation=False, prediction=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    composed_transform = transforms.Compose(used_transforms)

    model = ResNetCaptchaModel().to(device)

    if training:
        dataset = CaptchaDataset(image_dir=used_directories, transform=composed_transform)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        train_run(model, dataset, optimizer, device)

    if evaluation:
        checkpoint = torch.load(used_model, weights_only=True)
        model.load_state_dict(checkpoint)
        for directory in used_directories:
            print("Using dataset:", directory.split('/')[-1])
            for transform in used_transforms:
                dataset = CaptchaDataset(image_dir=directory, transform=transform)
                eval_run(model, dataset, device, epoch_msg=f"Using transformation: {transform.__class__.__name__}")

    if prediction:
        checkpoint = torch.load(used_model, weights_only=True)
        model.load_state_dict(checkpoint)

        for directory in used_directories:
            path_to_img = directory + '/' + img_filename
            predicted_captcha = predict(model, path_to_img, device, transform=composed_transform)
            print(f"Predicted CAPTCHA ({directory.split('/')[-1]}): {predicted_captcha}")


if __name__ == "__main__":
    main(training, evaluation, prediction)

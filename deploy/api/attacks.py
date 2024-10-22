import torch
from torch.nn import CrossEntropyLoss, MSELoss, Tanh
from torch.optim import Adam
from torchvision.transforms.v2.functional import convert_image_dtype, to_pil_image
from models.resnet_captcha_model_definition import differentiable_predict
from torchvision.io import read_image, ImageReadMode

CHAR_TYPES_NUM = 36  # Assumption: 10 digits + 26 letters
CAPTCHA_LENGTH = 5  # Assumption: CAPTCHA length is 5


def fgsm(model, path_to_img, save_path, device='cpu', epsilon=0.0):
    prediction, dL_dx = differentiable_predict(
        model, path_to_img,
        device, criterion=CrossEntropyLoss(),
        transform=None
    )
    evil_noise = torch.sign(dL_dx)

    image = read_image(path_to_img, mode=ImageReadMode.RGB)
    image = convert_image_dtype(image, dtype=torch.float)
    image = image.to(device)

    evil_image = image + epsilon * evil_noise

    evil_image = torch.clamp(evil_image, 0, 1).detach().cpu()
    evil_image_pil = to_pil_image(evil_image)
    evil_image_pil.save(save_path)


def pgd(model, path_to_img, save_path, device='cpu'):
    def attack(model, images, labels, eps=0.03, alpha=0.01, iters=40):
        # Add a batch dimension to the images tensor
        images = images.unsqueeze(0).to(device)
        labels = labels.to(device)
        original_images = images.clone().detach()

        # Set the images to require gradients
        images.requires_grad = True

        for i in range(iters):
            outputs = model(images)
            loss = CrossEntropyLoss()(outputs.view(-1, CHAR_TYPES_NUM), labels.view(-1))

            model.zero_grad()
            loss.backward()

            # Update the images based on gradient information
            images = images + alpha * images.grad.sign()

            # Clamp images to maintain within eps bounds and normalize
            eta = torch.clamp(images - original_images, min=-eps, max=eps)
            images = torch.clamp(original_images + eta, min=0, max=1).detach_()

            images.requires_grad = True

        return images

    # Load and transform the image
    original_image = read_image(path_to_img, mode=ImageReadMode.RGB)
    original_image = convert_image_dtype(original_image, dtype=torch.float)

    # Perform PGD attack
    labels = torch.randint(0, CHAR_TYPES_NUM, (1, CAPTCHA_LENGTH)).to(device)  # Dummy labels for loss computation
    adv_image = attack(model, original_image, labels)

    adv_image = torch.clamp(adv_image, 0, 1).detach().cpu()
    adv_img_pil = to_pil_image(adv_image.squeeze(0))
    adv_img_pil.save(save_path)


def cw(model, path_to_img, save_path, device='cuda'):
    def cw_l2(model, loss_fn, X, y):
        def f(delta, X, y, model, loss_fn):
            outputs = model(X)
            return -loss_fn(outputs.view(-1, CHAR_TYPES_NUM), y.view(-1))

        X = X.unsqueeze(0).to(device)
        y = y.to(device)

        # Change of variables from original paper
        w = torch.zeros_like(X, requires_grad=True, device=device)

        optimizer = Adam([w], lr=lr)
        loss_d = MSELoss(reduction='sum')
        loss_f = torch.sum

        for i in range(num_iter):
            # Change of variables from original paper
            delta = 1 / 2 * (Tanh()(w) + 1) - X

            # minimizing both negative loss of target and distance of attack
            overall_loss = loss_d(delta + X, X) + loss_f(c * f(delta, X, y, model, loss_fn))

            overall_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return delta.detach()

    num_iter = 1000
    c = 1e-3
    lr = 0.01

    # Load and transform the image
    original_image = read_image(path_to_img, mode=ImageReadMode.RGB)
    original_image = convert_image_dtype(original_image, dtype=torch.float).to(device)

    # Perform C&W attack
    labels = torch.randint(0, CHAR_TYPES_NUM, (1, CAPTCHA_LENGTH)).to(device)
    adv_input = cw_l2(model, CrossEntropyLoss(), original_image, labels).to(device)
    adv_image = original_image + adv_input

    # Ensure the adversarial image is valid (clamp it to valid pixel values)
    adv_image = torch.clamp(adv_image, 0, 1).detach().cpu()
    adv_img_pil = to_pil_image(adv_image.squeeze(0))
    adv_img_pil.save(save_path)
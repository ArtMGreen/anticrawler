from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
from models.resnet_captcha_model_definition import predict, ResNetCaptchaModel
import torch
from torchvision.transforms import Normalize, Compose
import attacks

CHAR_TYPES_NUM = 36  # Assumption: 10 digits + 26 letters
CAPTCHA_LENGTH = 5  # Assumption: CAPTCHA length is 5

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
IMAGES_DIR = os.path.join(ROOT_DIR, 'deploy', 'images')

# Model checkpoint path ==========================================================

# for docker run
# CHKPT_DIR = os.path.join(os.getcwd(), 'models', 'captcha_resnet50.pth')

# for local run
CHKPT_DIR = os.path.join(ROOT_DIR, 'models', 'captcha_resnet50.pth')

# ================================================================================

MODEL = ResNetCaptchaModel(CHAR_TYPES_NUM, CAPTCHA_LENGTH)
MODEL.load_state_dict(torch.load(CHKPT_DIR, weights_only=True, map_location=torch.device('cpu')))

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
MODEL.to(DEVICE)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(os.path.join(IMAGES_DIR,file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


@app.get("/attack/")
def attack(method: str, filename: str):
    new_filename = method + '_' + filename
    src_path = os.path.join(IMAGES_DIR, filename)
    dst_path = os.path.join(IMAGES_DIR, new_filename)

    if method == "FGSM":
        attacks.fgsm(MODEL, src_path, dst_path, device=DEVICE)
    elif method == "PGD":
        attacks.pgd(MODEL, src_path, dst_path, device=DEVICE)
    elif method == "CW":
        attacks.cw(MODEL, src_path, dst_path, device=DEVICE)
    else:
        raise HTTPException(status_code=401, detail="Method Not Allowed")

    return {"filename": new_filename}


@app.get("/defend/")
def defend(method: str, filename: str):
    new_filename = method + '_' + filename
    src_path = os.path.join(IMAGES_DIR, filename)
    dst_path = os.path.join(IMAGES_DIR, new_filename)

    #TODO IMPLEMENT

    return {"filename": new_filename}


@app.get("/inference/")
def inference(filename: str):
    transform = Compose([
        # transforms.Resize((224, 224)),
        # -- deprecated | transforms.ToTensor(),
        # -- redundant | transforms.ToImage(),
        # -- redundant | transforms.ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    prediction = predict(MODEL, os.path.join(IMAGES_DIR, filename), DEVICE, transform)
    return {"label": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host='0.0.0.0')

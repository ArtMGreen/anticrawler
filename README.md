# Anticrawler :space_invader:

**Adversarial attacks and defenses for CAPTCHA recognition model.**

(small note for graders: come to `models/resnet_captcha_recognizer.py`, we used it as a grand-control script to train and evaluate our models. For even better experience after training, follow the `Installation` section below to run the overviewing app you had seen during the presentation)

## Project Overview
This repository explores different adversarial attacks and defensive methods applied to CAPTCHA recognition models. The goal is to test and implement various attack techniques (such as FGSM, PGD, C&W) and evaluate the effectiveness of different defensive strategies (including adversarial training, image transformations, and more).

## Repository Structure :open_file_folder:	

The project is organized as follows:


- `attacks/`  :radioactive: contains scripts and resources for generating adversarial attacks on models, including methods like FGSM, PGD, and C&W.


- `datasets/` contains the process of constructing a dataset from multiple image directories, including the various datasets used for training the model.


- `defences/` - scripts and resources for implementing defensive techniques against adversarial attacks. These include image transformation methods such as grayscale, thresholding, and taking gradients.


- `deploy/` contains deployment-related resources, including the frontend for deploying the model.


- `models/` includes code for training, testing, and dividing data for the model used in the project.


- `notebooks/` - Jupyter notebooks of original experiments for data exploration, model training, and testing.


## Installation :wrench:

1. Clone the repository
   ```git clone https://github.com/ArtMGreen/anticrawler.git```
2. Navigate to the project directory:
   ```cd anticrawler```
3. To run the entire app using Docker, just run this command in terminal:
   ```docker compose up```.
   
   To run the app locally:
   
    a) Install the required dependencies:
     ```pip install -r requirements.txt```
   
    b) Run the backend app `main.py`.
   
    c) Run the frontend app through this command:
      ```streamlit run app.py --server.port=8501```.
   
   In both cases, the app will be accessible through your browser at `localhost:8501`.

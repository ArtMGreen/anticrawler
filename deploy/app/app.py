import streamlit as st
import os
import requests

# Directory for images
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
IMAGES_DIR = os.path.join(ROOT_DIR,'images')

# API endpoints ==================================================================

# for local run
UPLOAD_URL = "http://localhost:8000/upload"
ATTACK_URL = "http://localhost:8000/attack/"
DEFEND_URL = "http://localhost:8000/defend/"
PREDICT_URL = "http://localhost:8000/inference/"

# for docker run
# UPLOAD_URL = "http://fastapi:8000/upload"
# ATTACK_URL = "http://fastapi:8000/attack/"
# DEFEND_URL = "http://fastapi:8000/defend/"
# PREDICT_URL = "http://fastapi:8000/inference/"

# ================================================================================

# Fetch available images from the directory
def get_available_images():
    return [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]

# Upload image to FastAPI server
def upload_image(image):
    files = {'file': image}
    response = requests.post(UPLOAD_URL, files=files)
    return response.ok

# Attack image
def attack_image(method, filename):
    response = requests.get(f"{ATTACK_URL}?method={method}&filename={filename}")
    return response.ok, response.json().get("filename")

# Defend image
def defend_image(method, filename):
    response = requests.get(f"{DEFEND_URL}?method={method}&filename={filename}")
    return response.ok, response.json().get("filename")

# Predict image label
def predict_image(filename):
    response = requests.get(f"{PREDICT_URL}?filename={filename}")
    if response.ok:
        return response.json().get("label")
    return "Error"

# Initialize selected image in session state if not set
if 'selected_image' not in st.session_state:
    st.session_state['selected_image'] = None

# Sidebar: Available images
st.sidebar.title("Available images")
image_files = get_available_images()
selected_image = st.sidebar.selectbox("Choose an image", image_files)

# Set selected image in session state
if selected_image:
    st.session_state['selected_image'] = selected_image

# Upload section
uploaded_file = st.sidebar.file_uploader("Upload new image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    if upload_image(uploaded_file):
        st.sidebar.success("Image uploaded successfully!")
        # Set the uploaded file as the selected image
        st.session_state['selected_image'] = uploaded_file.name
    else:
        st.sidebar.error("Failed to upload image.")

# Main section: Show selected image
st.subheader("Selected image")
if st.session_state['selected_image']:
    st.image(os.path.join(IMAGES_DIR, st.session_state['selected_image']),
             caption=f"Image name: {st.session_state['selected_image']}",
             use_column_width=True)

# Prediction section
if st.button("Predict"):
    label = predict_image(st.session_state['selected_image'])
    st.write(f"Predicted label: {label}")

# Attack section
st.subheader("Attack")
attack_method = st.selectbox("Select attack method", ["FGSM", "PGD", "CW"])
if st.button("Apply Attack"):
    ok, filename = attack_image(attack_method, st.session_state['selected_image'])
    if ok:
        st.success(f"Attack applied: {attack_method}")
        # Show the attacked file
        st.image(os.path.join(IMAGES_DIR, filename), caption=f"Attacked image: {filename}",use_column_width=True)
    else:
        st.error("Failed to apply attack.")

# Defense section
st.subheader("Defense")
defense_method = st.selectbox("Select defense method", [
    "MEDIAN_FILTER", "THRESHOLDING", "GRADIENT_TRANSFORM",
    "GAUSSIAN_BLUR", "GRAYSCALE", "GAUSSIAN_NOISE", "NORMALIZE"
])
if st.button("Apply Defense"):
    ok, filename = defend_image(defense_method, st.session_state['selected_image'])
    if ok:
        st.success(f"Defense applied: {defense_method}")
        # Set the defended file as the selected image
        # Show the defended image
        st.image(os.path.join(IMAGES_DIR, filename), caption=f"Defended image: {filename}",use_column_width=True)
    else:
        st.error("Failed to apply defense.")


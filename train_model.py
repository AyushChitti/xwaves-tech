import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="X-ray Pneumonia Classifier", layout="centered")

st.title("🩺 Chest X-ray Pneumonia Classifier")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Use ImageNet mean/std
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    with st.spinner("Analyzing X-ray..."):
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        label = "PNEUMO

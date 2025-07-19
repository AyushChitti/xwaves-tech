import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    return model

model = load_model()

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title("🩺 Chest X-ray Pneumonia Classifier")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        input_tensor = preprocess_image(image).to(device)
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
        label = "PNEUMONIA" if prediction.item() == 1 else "NORMAL"
        st.success(f"Prediction: **{label}**")

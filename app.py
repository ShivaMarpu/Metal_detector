import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os
import gdown

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
class_labels = ['Aluminium', 'Brass', 'Copper']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Model file path
MODEL_PATH = "best_vgg16_model.pth"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found! Downloading the model...")

    # Google Drive file ID (change this to your own file ID)
    file_id = '1T3ggpQNrb8qA25DBPmVi0wuUchjZgsWv'
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Use gdown to download the file
    gdown.download(url, MODEL_PATH, quiet=False)
    
    st.success(f"Model downloaded successfully to {MODEL_PATH}")

# Load model
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 3)

# Use weights_only=False to ensure compatibility with PyTorch 2.6 changes
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.to(device)
model.eval()

# Metal properties
metal_properties = {
    "Aluminium": {
        "Colour": "Silvery-white",
        "Density (g/cm³)": 2.70,
        "Melting Point (°C)": 660.3,
        "Boiling Point (°C)": 2519,
        "Thermal Conductivity (W/m·K)": 237.0,
        "Corrosion Resistance": "High",
        "Surface Roughness (µm)": 0.20,
        "Application": "Aerospace, Automotive, Packaging"
    },
    "Copper": {
        "Colour": "Reddish-brown",
        "Density (g/cm³)": 8.96,
        "Melting Point (°C)": 1084.6,
        "Boiling Point (°C)": 2562,
        "Thermal Conductivity (W/m·K)": 401.0,
        "Corrosion Resistance": "Medium",
        "Surface Roughness (µm)": 0.15,
        "Application": "Electrical Wiring, Plumbing, Coins"
    },
    "Brass": {
        "Colour": "Yellowish",
        "Density (g/cm³)": 8.40,
        "Melting Point (°C)": 900.0,
        "Boiling Point (°C)": 1700,
        "Thermal Conductivity (W/m·K)": 109.0,
        "Corrosion Resistance": "Medium",
        "Surface Roughness (µm)": 0.18,
        "Application": "Plumbing, Musical Instruments, Decorative Items"
    }
}

# Predict function
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

# Streamlit UI
st.title("Metal Classifier 🔍")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Predict
    prediction = predict_image(image)
    properties = metal_properties[prediction]
    
    # Output
    st.success(f"**Predicted Metal: {prediction}**")
    
    st.subheader("Properties:")
    for key, value in properties.items():
        st.write(f"**{key}:** {value}")

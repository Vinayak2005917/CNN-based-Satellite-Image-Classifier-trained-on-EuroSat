import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNN64x64
import matplotlib.pyplot as plt
import io
import pandas as pd
import os
import random

# ----------------------
# Paths and transforms
# ----------------------
path = r"C:\Users\vk200\.cache\kagglehub\datasets\apollo2506\eurosat-dataset\versions\6"
test_csv_path = os.path.join(path, "EuroSAT/test.csv")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ----------------------
# Load model
# ----------------------
model = CNN64x64(num_classes=10)
model.load_state_dict(torch.load("eurosat_cnn64x64.pth", map_location=torch.device('cpu')))
model.eval()

class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegatation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

st.header("EuroSat Satellite Image Classification")
st.write("This CNN model is trained on only the EuroSAT dataset, which has a scale of 10m/px. "
"since it is so specific we cannot expect it to generalize well to other datasets. " 
"The Following are some examples of images from the test part of the dataset:")

st.markdown("---")

# ----------------------
# Load test CSV and pick 25 random images
# ----------------------
test_df = pd.read_csv(test_csv_path)
num_images = 25
random_indices = random.sample(range(len(test_df)), num_images)
selected_img_path = st.session_state.get('selected_img_path', None)
selected_class = st.session_state.get('selected_class', None)


cols = st.columns(5)

if not selected_img_path:
    # Display images in a 5x5 grid with buttons
    st.subheader("Select an image from the test set")
    for i, idx in enumerate(random_indices):
        img_path = os.path.join(path, "EuroSAT", test_df.iloc[idx, 1])
        class_label = test_df.iloc[idx, 3]
        img = Image.open(img_path).convert("RGB")
        col = cols[i % 5]
        col.image(img, width=180)  # Increased image size
        if col.button(f"Select", key=f"img_{i}"):
            st.session_state['selected_img_path'] = img_path
            st.session_state['selected_class'] = class_label
            st.rerun()

# ----------------------
# Prediction
# ----------------------
elif selected_img_path:
    # Go Back button
    if st.button("Go Back", key="go_back"):
        st.session_state['selected_img_path'] = None
        st.session_state['selected_class'] = None
        st.rerun()
    img = Image.open(selected_img_path).convert("RGB")
    tensor_img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1).squeeze()
        pred_class = torch.argmax(outputs, dim=1).item()

    st.image(img, caption=f"Selected Image: {selected_class}", width=350)
    st.header(f"Predicted class: {class_names[pred_class]}")
    if selected_class == class_names[pred_class]:
        st.success("Correct Prediction!")
    else:
        st.error(f"Incorrect Prediction! Actual: {selected_class}")

    # Probability bar chart with percentages
    fig, ax = plt.subplots(figsize=(10, 5))  # Wider figure for better fit
    bars = ax.bar(class_names, [p.item()*100 for p in probs])
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Class Probability Distribution")
    ax.set_xticklabels(class_names, rotation=45, ha='right')

    # Add percentage labels on top of bars
    for bar, p in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{p.item()*100:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()  # Ensure everything fits
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    st.write("Probability Distribution")
    plt.close(fig)

st.markdown("---")

st.markdown("# About")
st.markdown("#### Hyperparameters")
st.write("learning rate : 0.001")
st.write("number of epochs : 2")
st.write("number of classes : 10")
st.write(" ")
st.write(" ")



st.markdown("#### Model structure")
st.write("Convolutional layers 1 (in channel = 3 , out channel = 32, kernel size = 3)")
st.write("Convolutional layers 2 (in channel = 32 , out channel = 64, kernel size = 3)")
st.write("Max Pooling layers 1")
st.write("Convolutional layers 3 (in channel = 64 , out channel = 128, kernel size = 3)")
st.write("Convolutional layers 4 (in channel = 128 , out channel = 256, kernel size = 3)")
st.write("Max Pooling layers 2")
st.write("Fully connected layers 1")
st.write("Fully connected layers 2")
st.write("ReLU activation")
st.write(" ")
st.write(" ")


st.markdown("#### Model training")
st.write("at 2 epochs, the current model is at 76% accuracy")
st.write("you can train the model on your own from [here](https://github.com/Vinayak2005917/CNN-based-Satellite-Image-Classifier-trained-on-EuroSat) or wait for me to train one at a higher epoch")


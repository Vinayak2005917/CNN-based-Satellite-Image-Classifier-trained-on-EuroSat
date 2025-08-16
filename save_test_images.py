import os
import pandas as pd
from PIL import Image


# Path to EuroSAT dataset and test.csv
path = r"C:\Users\vk200\.cache\kagglehub\datasets\apollo2506\eurosat-dataset\versions\6"
test_csv_path = os.path.join(path, "EuroSAT", "test.csv")
# Save test_images in the current project directory
output_dir = os.path.join(os.getcwd(), "test_images")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load test CSV
test_df = pd.read_csv(test_csv_path)

for idx in range(len(test_df)):
    img_rel_path = test_df.iloc[idx, 1]  # image filename
    img_path = os.path.join(path, "EuroSAT", img_rel_path)
    img = Image.open(img_path).convert("RGB")
    # Save image to output_dir with original filename
    img.save(os.path.join(output_dir, os.path.basename(img_rel_path)))

print(f"Saved {len(test_df)} images to {output_dir}")

# Satellite Image Classification (EuroSAT)

This project is a deep learning-based satellite image classifier trained on the EuroSAT dataset. It uses a custom CNN architecture to classify images into 10 land use and land cover classes. The project provides both a Streamlit web app (`main.py` or `app.py`) and a Jupyter notebook (`test.ipynb`) for experimentation and demonstration.

## Project Structure

-  `app.py`: Streamlit app for interactive image classification and visualization.
- `model.py`: Contains the CNN model definition (`CNN64x64`).
- `train.ipynb`: Jupyter notebook for traing, tuning and testing the model, running predictions, and visualizing results.
- `eurosat_cnn64x64.pth`: Pretrained model weights.
- `requirements.txt`: Python dependencies.

## Getting Started

Go to : https://cnn-based-satellite-image-classifier-trained-on-eurosat-jebxjw.streamlit.app/

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```powershell
streamlit run app.py
```

- Select an image from the grid to classify.
- View the predicted class and probability distribution.
- Use the "Go Back" button to return to the image grid.

### 3. Use the Jupyter Notebook

Open `test.ipynb` in Jupyter or VS Code to:
- Load and test the model on sample images.
- Visualize predictions and probability distributions.
- Experiment with custom images.

## Model Details

- The model is defined in `model.py` as `CNN64x64`.
- It expects RGB images of size 64x64.
- The model is trained to classify images into the following classes:
  - AnnualCrop
  - Forest
  - HerbaceousVegetation
  - Highway
  - Industrial
  - Pasture
  - PermanentCrop
  - Residential
  - River
  - SeaLake

## Dataset

- EuroSAT dataset (https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- The test set is referenced via a CSV file in the Streamlit app.

## How It Works

- Images are loaded and preprocessed using torchvision transforms.
- The model predicts class probabilities for each image.
- Results are visualized as bar charts and images in the app/notebook.

## Customization

- You can replace `uploaded_image.jpg` with your own image for testing.
- Modify `model.py` to experiment with different architectures.

## License

This project is for educational and research purposes. Please refer to the EuroSAT dataset license for data usage.

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import models, transforms
import os
import kagglehub
import io
import json
import shutil
import subprocess
import random

app = FastAPI()

# Template configuration
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Dataset and model directories
dataset_dir = "new-plant-diseases-dataset"
model_dir = "model/checkpoints"
model_path = f"{model_dir}/plant_disease_model.pth"
class_names_path = f"{model_dir}/class_names.json"

def check_and_download_dataset():
    # Check if dataset exists; if not, download it
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("Dataset not found. Downloading from Kaggle...")
        path = kagglehub.dataset_download("emmarex/plantdisease")
        print("Dataset downloaded at:", path)
        
        # Copy downloaded dataset to project directory
        shutil.copytree(path, dataset_dir)
        print(f"Dataset copied to project directory: {dataset_dir}")

        # Define directories for train and validation sets
        plantvillage_dir = os.path.join(dataset_dir, 'PlantVillage')
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val')

        # Create train and val directories if they don't exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Split each class directory into train and val sets
        for class_dir in os.listdir(plantvillage_dir):
            class_path = os.path.join(plantvillage_dir, class_dir)
            if os.path.isdir(class_path):
                # Create class-specific directories in train and val
                train_class_dir = os.path.join(train_dir, class_dir)
                val_class_dir = os.path.join(val_dir, class_dir)
                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)

                # List all images in the class directory and shuffle
                images = os.listdir(class_path)
                random.shuffle(images)
                
                # Split 80% for train and 20% for val
                train_count = int(0.8 * len(images))
                train_images = images[:train_count]
                val_images = images[train_count:]

                # Move images to train and val directories
                for img in train_images:
                    shutil.move(os.path.join(class_path, img), train_class_dir)
                for img in val_images:
                    shutil.move(os.path.join(class_path, img), val_class_dir)

        print(f"Dataset structured into train and val directories at {dataset_dir}")

        # Train model after dataset download and reorganization
        train_model_if_needed()
    else:
        print("Dataset found in project directory:", dataset_dir)

def load_model():
    """Function to load the model and class names on CPU."""
    global model, class_names
    if os.path.exists(model_path) and os.path.exists(class_names_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model on CPU
        model.eval()
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        print("Model and class names loaded successfully.")
    else:
        model = None
        class_names = []
        print("Model not found. Please train the model using train.py before running the application.")

def train_model_if_needed():
    # Check if model exists; if not, run train.py
    if not os.path.exists(model_path) or not os.path.exists(class_names_path):
        print("Model not found. Training the model...")
        subprocess.run(["python3", "app/train.py"])
        print("Model training completed.")
        # Reload the model after training
        load_model()
    else:
        print("Model already trained and ready for use.")

# Verify and download dataset on application startup
@app.on_event("startup")
def on_startup():
    check_and_download_dataset()
    load_model()  # Load model initially if already trained

# Main endpoint
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint to process the image and predict disease
@app.post("/predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    # Read and process the image
    image = Image.open(io.BytesIO(await file.read()))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to('cpu')  # Ensure tensor is on CPU

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        disease_name = class_names[predicted.item()]  # Map `predicted` to class name

    return templates.TemplateResponse("result.html", {
        "request": request,
        "disease_name": disease_name
    })

# Endpoint to clean up the dataset if needed
@app.get("/clean_dataset")
async def clean_dataset():
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        return {"status": "Dataset successfully deleted"}
    else:
        return {"status": "Dataset was already empty"}

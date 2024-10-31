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
import uuid
import json

app = FastAPI()

# Template configuration
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Dataset and model directories
dataset_dir = "new-plant-diseases-dataset"
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
model_dir = "model/checkpoints"
model_path = f"{model_dir}/plant_disease_model.pth"
class_names_path = f"{model_dir}/class_names.json"
class_info_path = f"{model_dir}/class_info.json"
upload_dir = "app/static/uploads"

# Initialize global variables for model, class names, and class info
model = None
class_names = []
class_info = {}

# Ensure upload directory exists
os.makedirs(upload_dir, exist_ok=True)

# Load the analyses from the JSON file
def save_analysis_to_json(data):
    file_path = os.path.join(upload_dir, "analyses.json")
    # Cargar datos existentes
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            analyses = json.load(file)
    else:
        analyses = []

    # Añadir el nuevo análisis y guardar el archivo JSON
    analyses.append(data)
    with open(file_path, "w") as file:
        json.dump(analyses, file, indent=4)

def check_and_download_dataset():
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("Dataset not found. Downloading from Kaggle...")
        path = kagglehub.dataset_download("emmarex/plantdisease")
        print("Dataset downloaded at:", path)
        
        shutil.copytree(path, dataset_dir)
        print(f"Dataset copied to project directory: {dataset_dir}")

        plantvillage_dir = os.path.join(dataset_dir, 'PlantVillage')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for class_dir in os.listdir(plantvillage_dir):
            class_path = os.path.join(plantvillage_dir, class_dir)
            if os.path.isdir(class_path):
                train_class_dir = os.path.join(train_dir, class_dir)
                val_class_dir = os.path.join(val_dir, class_dir)
                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)

                images = os.listdir(class_path)
                random.shuffle(images)
                train_count = int(0.8 * len(images))
                train_images = images[:train_count]
                val_images = images[train_count:]

                for img in train_images:
                    shutil.move(os.path.join(class_path, img), train_class_dir)
                for img in val_images:
                    shutil.move(os.path.join(class_path, img), val_class_dir)

        print(f"Dataset structured into train and val directories at {dataset_dir}")

def load_model():
    global model, class_names, class_info
    if os.path.exists(model_path) and os.path.exists(class_names_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model on CPU
        model.eval()
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        print("Model and class names loaded successfully.")
    else:
        print("Model not found. Please train the model using train.py before running the application.")

    # Load class info if available
    if os.path.exists(class_info_path):
        with open(class_info_path, "r") as f:
            class_info = json.load(f)
        print("Class information loaded successfully.")
    else:
        class_info = {}
        print("Class information not found. Please ensure class_info.json is in the model directory.")

def train_model_if_needed():
    if not os.path.exists(model_path) or not os.path.exists(class_names_path):
        print("Model not found. Training the model...")
        subprocess.run(["python3", "app/train.py"])
        print("Model training completed.")
        load_model()
    else:
        print("Model already trained and ready for use.")

@app.on_event("startup")
def on_startup():
    check_and_download_dataset()
    load_model()  # Ensure model and class info are loaded at startup
    train_model_if_needed()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    # Save uploaded image to display in the results page
    image_id = str(uuid.uuid4())
    image_path = os.path.join(upload_dir, f"{image_id}.png")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process the image
    image = Image.open(image_path)
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
        class_name = class_names[predicted.item()]

    # Retrieve class information
    info = class_info.get(class_name, {
        "display_name": class_name,
        "description": "No description available for this class.",
        "solution": "No solution available for this class."
    })

    # Save analysis data to JSON file
    analysis_data = {
        "image_url": f"/static/uploads/{image_id}.png",
        "display_name": info["display_name"],
        "description": info["description"],
        "solution": info["solution"]
    }
    save_analysis_to_json(analysis_data)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "display_name": info["display_name"],
        "description": info["description"],
        "solution": info["solution"],
        "image_url": f"/static/uploads/{image_id}.png"  # Send image path to the template
    })

@app.get("/clean_dataset")
async def clean_dataset():
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        return {"status": "Dataset successfully deleted"}
    else:
        return {"status": "Dataset was already empty"}

@app.get("/gallery", response_class=HTMLResponse)
async def get_gallery(request: Request):
    # Load analyses from JSON file
    file_path = os.path.join(upload_dir, "analyses.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            analyses = json.load(file)
    else:
        analyses = []

    # Render the gallery page with the analyses
    return templates.TemplateResponse("gallery.html", {
        "request": request,
        "analyses": analyses
    })
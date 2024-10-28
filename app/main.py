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
        path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset") # https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
        print("Dataset downloaded at:", path)
    else:
        print("Dataset found in directory:", dataset_dir)

# Verify and download dataset on application startup
@app.on_event("startup")
def on_startup():
    check_and_download_dataset()

# Load model and class names if available
if os.path.exists(model_path) and os.path.exists(class_names_path):
    model = torch.load(model_path)
    model.eval()
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
else:
    # Show error message and skip model loading
    model = None
    class_names = []
    print("Model not found. Please train the model using train.py before running the application.")

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
    image_tensor = transform(image).unsqueeze(0)

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

# Plant Disease Detection V2

The **Plant Disease Detection V2** is a machine learning model that uses deep learning to detect plant diseases in images. It is designed to be a fast and efficient tool for identifying plant diseases in images, making it a valuable asset for farmers, gardeners, and anyone else who needs to monitor their plants.

The model is trained on a large dataset of plant images, which includes images of various plant diseases. It uses convolutional neural networks (CNNs) to classify the images based on their content. The model is able to identify plant diseases with high accuracy, making it a valuable tool for identifying plant diseases in images.

This project was born from the [Plant Disease Detection project](https://github.com/joakimvivas/plant-disease-detector) and is a continuation of that project. The goal of this project is to improve the model's performance and make it more efficient.

## Running the project locally (How to Run)

1. Create the Python virtual environment

```sh
python3 -m venv plant-disease-detection
```

```sh
source plant-disease-detection/bin/activate
```

2. Install dependencies:

It is recommended, first, upgrade pip:
```sh
pip install --upgrade pip
```

Install dependencies/requirements:
```sh
pip install -r requirements.txt
```

3. Execute the following command:

```sh
uvicorn app.main:app --reload --host 0.0.0.0 --port 3000
```

4. You should see an output similar to:

```
INFO:     Uvicorn running on http://127.0.0.1:3000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXXX] using WatchFiles
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Licensing

All packages in this repository are open-source software and licensed under the [MIT License](https://github.com/joakimvivas/marco-bot/blob/main/LICENSE). By contributing in this repository, you agree to release your code under this license as well.

Let's build the future of **Plant Disease Detection V2** development together! 🤖🚀
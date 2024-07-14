# Tennis Face Recognition App

This project is a Face Recognition App specifically designed for recognizing tennis players. The app uses two different methods for face recognition: `compare_faces` from the `face_recognition` library and a Deep Learning model based on FaceNet. The app includes tools for downloading images from Google Images, preprocessing and filtering outliers, model training, prediction, and deployment using Docker and Flask API.

![Tennis Face Recognition image](static/tennis_face_recognition_wide.jpeg)

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Loader](#data-loader)
- [Image Preprocessing](#image-preprocessing)
- [Model Training](#model-training)
- [Model Predictor](#model-predictor)
- [Deployment](#deployment)
- [Screenshots](#screenshots)


## Demo



## Features

- Download images from tennis players using Google Images crawler;
- Preprocess images and filtering out outliers and images with multiple faces;
- Train models using two different methods: `compare_faces` and a Deep Learning model using transfer learning based on FaceNet;
- Predict faces using the trained models;
- Deploy the app using Docker and Flask with a web interface.

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/)
- [Docker](https://www.docker.com/)

### Step-by-Step Guide

1. **Clone the repository:**
    ```sh
    git clone https://github.com/iglesiascaio/tennis_face_recognition.git
    cd tennis_face_recognition
    ```

2. **Create and activate the Conda environment:**
    ```sh
    make create-env
    ```

3. **Activate the environment and install dependencies:**
    ```sh
    . ./activate
    ```

## Usage

### Data Loader

The data loader uses a Google Images crawler to download images of tennis players. You can customize the players and the number of images in the `data_loader.py` file. You can also choose the players to download images using the `config/data-loader.yaml` file. 

```sh
python runner/data_loader.py
```


### Image Preprocessing

Preprocess images and filter out outliers, images with multiple faces, and other unwanted data. The preprocessing script can be found in `image_preprocessing.py`.

```sh
python runner/image_preprocessing.py
```


### Training the Model

To train the face recognition models, you can use the provided script with Click for command-line argument parsing. The training process involves two main components: computing distances between face encodings and training a deep learning model based on FaceNet. We also use the `ImageDataGenerator` from TensorFlow in order to dynamically generate images that avoid overfitting. 

To train the models, use a command like this one:

```sh
python train.py --data-dir ./data/preprocessed_images --model-save-path ./model/face_recognition_model_all --encodings-save-path ./model/face_encodings.pkl --img-size 160,160 --batch-size 32 --epochs 100
```

Parameters
- --data-dir: Directory containing preprocessed images (default: ./data/preprocessed_images).
- --model-save-path: Path to save the trained model (default: ./model/face_recognition_model_all).
- --encodings-save-path: Path to save the face encodings (default: ./model/face_encodings.pkl).
- --img-size: Desired size for resizing the images (default: 160,160).
- --batch-size: Batch size for training (default: 32).
- --epochs: Number of epochs for training (default: 100).


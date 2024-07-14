# Tennis Face Recognition App

This project is a face recognition app specifically designed for recognizing tennis players. The app uses two different methods for face recognition: `compare_faces` from the `face_recognition` library and a deep learning model based on FaceNet. The app includes tools for downloading images, preprocessing, model training, prediction, and deployment using Docker and Flask.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Loader](#data-loader)
- [Image Preprocessing](#image-preprocessing)
- [Model Training](#model-training)
- [Model Predictor](#model-predictor)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Demo](#demo)

## Features

- Download images using Google Images crawler
- Preprocess images by filtering out outliers and images with multiple faces
- Train models using two different methods: `compare_faces` and a deep learning model based on FaceNet
- Predict faces using the trained models
- Deploy the app using Docker and Flask with a clean web interface

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

The data loader uses a Google Images crawler to download images of tennis players. You can customize the players and the number of images in the `data_loader.py` file.

### Image Preprocessing

Preprocess images to filter out outliers, images with multiple faces, and other unwanted data. The preprocessing script can be found in `preprocessing.py`.

### Model Training

Train the face recognition models using two different methods:
1. `compare_faces` from the `face_recognition` library.
2. A deep learning model based on FaceNet using `ImageDataGenerator`.

Run the training scripts using:
```sh
python train_compare_faces.py
python train_facenet.py

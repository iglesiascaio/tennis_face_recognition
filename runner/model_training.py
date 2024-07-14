import os
import logging
import numpy as np
from PIL import Image
import face_recognition
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras_facenet import FaceNet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from abc import ABC, abstractmethod
import click
from utils.utils import TupleParamType

TUPLE = TupleParamType()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a color formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger("FaceRecognitionTrainer")
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists. If it does not exist, create it.

    :param directory: The directory path to check and create if necessary.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


class TrainerInterface(ABC):
    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass


class FaceEncodingTrainer(TrainerInterface):
    def __init__(self, data_dir: str, save_path: str, img_size: tuple = (160, 160)):
        """
        Initialize the FaceEncodingTrainer with directories and parameters.

        :param data_dir: Directory containing the preprocessed images.
        :param save_path: Path to save the face encodings.
        :param img_size: Desired size for resizing the images.
        """
        self.data_dir = data_dir
        self.save_path = save_path
        self.img_size = img_size

    def load_data(self) -> dict:
        """
        Load images, detect faces, and compute face encodings.

        :return: Dictionary containing labels and their corresponding encodings.
        """
        logger.info("Loading data and computing encodings...")
        encodings = {label: [] for label in os.listdir(self.data_dir)}

        for label in tqdm(os.listdir(self.data_dir)):
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for img_name in os.listdir(label_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(label_dir, img_name)
                    img = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(img)
                    if face_locations:
                        face_encoding = face_recognition.face_encodings(img)[0]
                        encodings[label].append(face_encoding)

        return encodings

    def train(self) -> None:
        """
        Compute and save average face encodings.
        """
        encodings = self.load_data()
        self.average_encodings = {
            label: np.mean(encodings[label], axis=0) for label in encodings
        }

    def save(self) -> None:
        """
        Save the computed face encodings.
        """
        ensure_dir(os.path.dirname(self.save_path))
        with open(self.save_path, "wb") as f:
            pickle.dump(self.average_encodings, f)

        logger.info(f"Average face encodings saved to {self.save_path}")


class ImageModelTrainer(TrainerInterface):
    def __init__(
        self,
        data_dir: str,
        model_save_path: str,
        img_size: tuple = (160, 160),
        batch_size: int = 32,
        epochs: int = 100,
    ):
        """
        Initialize the ImageModelTrainer with directories and parameters.

        :param data_dir: Directory containing the preprocessed images.
        :param model_save_path: Path to save the trained model.
        :param img_size: Desired size for resizing the images.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        """
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def load_data(self) -> tuple:
        """
        Load images and labels from the specified directory.

        :return: Numpy arrays for images and labels, and a dictionary mapping labels to indices.
        """
        logger.info("Loading data...")
        images = []
        labels = []
        label_dict = {label: idx for idx, label in enumerate(os.listdir(self.data_dir))}

        for label in tqdm(os.listdir(self.data_dir)):
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for img_name in os.listdir(label_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(label_dir, img_name)
                    img = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(img)
                    if face_locations:
                        top, right, bottom, left = face_locations[0]
                        face_image = img[top:bottom, left:right]
                        pil_image = Image.fromarray(face_image).convert("RGB")
                        pil_image = pil_image.resize(self.img_size)
                        img_array = np.array(pil_image)
                        images.append(img_array)
                        labels.append(label_dict[label])

        images = np.array(images)
        labels = np.array(labels)
        logger.info(f"Loaded {len(images)} images and corresponding labels")

        return images, labels, label_dict

    def prepare_data(self) -> tuple:
        """
        Prepare data for training and validation.

        :return: Training and validation data generators, and the label dictionary.
        """
        logger.info("Preparing data...")
        images, labels, label_dict = self.load_data()

        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        logger.info(
            f"Split data into {len(X_train)} training samples and {len(X_val)} validation samples"
        )

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow(
            X_train, y_train, batch_size=self.batch_size
        )
        val_generator = val_datagen.flow(X_val, y_val, batch_size=self.batch_size)

        return train_generator, val_generator, label_dict

    def build_model(self, num_classes: int, learning_rate: float = 0.0001) -> None:
        """
        Build and compile the CNN model using FaceNet as the base.

        :param num_classes: Number of classes for the output layer.
        :param learning_rate: Learning rate for the optimizer.
        """
        facenet_model = FaceNet()
        base_model = facenet_model.model

        for layer in base_model.layers[:-30]:
            layer.trainable = False  # Freeze all layers except the last 30

        x = base_model.output
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation="softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self) -> None:
        """
        Train the model with the prepared data.
        """
        train_generator, val_generator, label_dict = self.prepare_data()
        num_classes = len(label_dict)
        self.build_model(num_classes)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
        ]

        logger.info("Starting training...")
        self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=val_generator,
            callbacks=callbacks,
        )
        logger.info("Training completed")

    def save(self) -> None:
        """
        Save the trained model to the specified path.
        """
        ensure_dir(os.path.dirname(self.model_save_path))
        tf.saved_model.save(self.model, self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")

    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set.
        """
        _, val_generator, _ = self.prepare_data()
        logger.info("Evaluating model on validation data...")
        val_loss, val_accuracy = self.model.evaluate(val_generator)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")


@click.command()
@click.option(
    "--data-dir",
    default="./data/preprocessed_images",
    help="Directory containing preprocessed images.",
)
@click.option(
    "--model-save-path",
    default="./model/face_recognition_model_all",
    help="Path to save the trained model.",
)
@click.option(
    "--encodings-save-path",
    default="./model/face_encodings.pkl",
    help="Path to save the face encodings.",
)
@click.option(
    "--img-size",
    type=TUPLE,
    default="160,160",
    help="Desired size for resizing the images.",
)
@click.option("--batch-size", default=32, help="Batch size for training.")
@click.option("--epochs", default=100, help="Number of epochs for training.")
def main(
    data_dir: str,
    model_save_path: str,
    encodings_save_path: str,
    img_size: tuple,
    batch_size: int,
    epochs: int,
) -> None:
    # Compute and save average face encodings
    encoding_trainer = FaceEncodingTrainer(data_dir, encodings_save_path, img_size)
    encoding_trainer.train()
    encoding_trainer.save()

    # Train and save the ML model
    model_trainer = ImageModelTrainer(
        data_dir, model_save_path, img_size, batch_size, epochs
    )
    model_trainer.train()
    model_trainer.save()
    model_trainer.evaluate()


if __name__ == "__main__":
    main()

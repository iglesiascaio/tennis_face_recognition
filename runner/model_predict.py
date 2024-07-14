import os
import numpy as np
import re
import logging
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import face_recognition
import pickle
import colorlog
from abc import ABC, abstractmethod
import click

# Set up logging for PlayerPredictor
logger = logging.getLogger("PlayerPredictor")
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)

    # Create a color formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "green",
        },
    )

    # Create a handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

# Disable debug messages from PIL and other libraries
logging.getLogger("PIL").setLevel(logging.WARNING)


class LoaderInterface(ABC):
    @abstractmethod
    def load(self):
        pass


class ModelLoader(LoaderInterface):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self) -> tf.keras.Model:
        """
        Load the saved model from the specified path.

        :return: Loaded TensorFlow model.
        """
        logger.info(f"Loading model from {self.model_path}")
        model = tf.saved_model.load(self.model_path)
        logger.info("Model loaded successfully")
        return model


class FaceEncodingLoader(LoaderInterface):
    def __init__(self, encodings_path: str):
        self.encodings_path = encodings_path

    def load(self) -> dict:
        """
        Load the precomputed average face encodings from the specified path.

        :return: Dictionary of average face encodings.
        """
        logger.info(f"Loading encodings from {self.encodings_path}")
        with open(self.encodings_path, "rb") as f:
            encodings = pickle.load(f)
        logger.info("Encodings loaded successfully")
        return encodings


class FaceDetector:
    @staticmethod
    def detect_and_preprocess_face(img_path: str, img_size: tuple) -> np.ndarray:
        """
        Detect and preprocess the face in the image for prediction.

        :param img_path: Path to the image file.
        :param img_size: Desired size for resizing the images.
        :return: Preprocessed image array or None if no face is detected.
        """
        # Load the image file
        img = face_recognition.load_image_file(img_path)

        # Detect face locations
        face_locations = face_recognition.face_locations(img)
        if face_locations:
            # Get the coordinates of the first face found
            top, right, bottom, left = face_locations[0]

            # Crop and resize the face image
            face_image = img[top:bottom, left:right]
            pil_image = Image.fromarray(face_image).resize(img_size)
            img_array = np.array(pil_image)

            # Expand dimensions to fit model input and preprocess the image
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        else:
            logger.error(f"No face detected in the image {img_path}")
            return None


class PlayerPredictor:
    def __init__(
        self,
        model_loader: LoaderInterface,
        encoding_loader: LoaderInterface,
        img_size: tuple = (160, 160),
    ):
        """
        Initialize the PlayerPredictor with model loader, encoding loader, and image size.

        :param model_loader: Instance of LoaderInterface to load the model.
        :param encoding_loader: Instance of LoaderInterface to load the face encodings.
        :param img_size: Desired size for resizing the images.
        """
        self.model = model_loader.load()
        self.average_encodings = encoding_loader.load()
        self.img_size = img_size

    @staticmethod
    def convert_to_native(obj):
        """
        Convert NumPy data types to native Python types for JSON serialization.

        :param obj: Object to convert.
        :return: Converted object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    @staticmethod
    def is_valid_image_name(image_name: str) -> bool:
        """
        Check if the image name follows the pattern of a number followed by .jpg, .png, or .jpeg.

        :param image_name: Name of the image file.
        :return: True if the image name is valid, False otherwise.
        """
        pattern = r".+\.(jpg|png|jpeg)$"
        return bool(re.match(pattern, image_name))

    def compare_faces(self, img_path: str) -> tuple:
        """
        Compare the face encoding of the given image with the average encodings of each player.

        :param img_path: Path to the image file.
        :return: Tuple of predicted player name and distance.
        """
        # Load the image file
        img = face_recognition.load_image_file(img_path)

        # Detect face locations
        face_locations = face_recognition.face_locations(img)
        if face_locations:
            # Get the coordinates of the first face found
            top, right, bottom, left = face_locations[0]

            # Crop the face image
            face_image = img[top:bottom, left:right]

            # Compute face encoding
            face_encoding = face_recognition.face_encodings(np.array(face_image))[0]

            # Compare the face encoding with the average encodings
            distances = {
                player: np.linalg.norm(face_encoding - avg_encoding)
                for player, avg_encoding in self.average_encodings.items()
            }

            # Find the player with the minimum distance
            predicted_player = min(distances, key=distances.get)
            distance = distances[predicted_player]

            return predicted_player, distance
        else:
            logger.error(f"No face detected in the image {img_path}")
            return None, None

    def predict_image(self, img_path: str, method: str = "combined") -> dict:
        """
        Predict the player in the given image using the specified method.

        :param img_path: Path to the image file.
        :param method: Method to use for prediction ("ml", "face_comparison", "combined").
        :return: Dictionary with predictions.
        """
        logger.info(f"Predicting player for image {img_path} using method {method}")

        # Detect and preprocess the face
        img_array = FaceDetector.detect_and_preprocess_face(img_path, self.img_size)
        if img_array is None:
            return None

        results = {}

        # Predict using ML model
        if method in ["ml", "combined"]:
            predictions = self.model.signatures["serving_default"](
                tf.constant(img_array)
            )["output_0"]
            predicted_class_ml = np.argmax(predictions, axis=1)[0]
            probability_ml = np.max(predictions)
            results["ML Prediction"] = (
                list(self.average_encodings.keys())[predicted_class_ml],
                self.convert_to_native(probability_ml),
            )

        # Predict using face comparison
        if method in ["face_comparison", "combined"]:
            predicted_class_fc, distance_fc = self.compare_faces(img_path)
            results["Face Comparison"] = (
                predicted_class_fc,
                self.convert_to_native(distance_fc),
            )

        # Combine results if method is "combined"
        if (
            method == "combined"
            and "ML Prediction" in results
            and "Face Comparison" in results
        ):
            if results["ML Prediction"][0] != results["Face Comparison"][0]:
                combined_prediction = (
                    results["ML Prediction"]
                    if results["ML Prediction"][1]
                    > (1 / (1 + results["Face Comparison"][1]))
                    else results["Face Comparison"]
                )
                results["Combined Prediction"] = combined_prediction
                logger.info(
                    f"Combined prediction chose {combined_prediction[0]} with confidence/distance {self.convert_to_native(combined_prediction[1]):.2f}"
                )
            else:
                results["Combined Prediction"] = results["ML Prediction"]
                logger.info(f"Both methods agree on {results['ML Prediction'][0]}")

        return results

    def evaluate_predictions(self, to_predict_path: str) -> None:
        """
        Run predictions on images in the specified directory.

        :param to_predict_path: Directory containing images to predict.
        """
        count_good_prediction = 0
        total_count = 0
        for image_name in os.listdir(to_predict_path):
            if self.is_valid_image_name(image_name):
                total_count += 1
                results = self.predict_image(
                    os.path.join(to_predict_path, image_name), method="combined"
                )
                if results:
                    combined_prediction = results.get("Combined Prediction")
                    real_player = (
                        image_name.split("_")[0] + "_" + image_name.split("_")[1]
                    )
                    if real_player == combined_prediction[0]:
                        count_good_prediction += 1
                        logger.critical(
                            f"Awesome prediction! Real player was indeed {real_player}"
                        )
                    else:
                        logger.warning(
                            f"Wrong prediction. Real player was {real_player}"
                        )
                    print("-" * 100)
                else:
                    logger.error(f"Failed to predict for image {image_name}")
        accuracy = count_good_prediction / total_count if total_count else 0
        logger.info(f"Final Accuracy: {accuracy: .2%}")


@click.command()
@click.option(
    "--model-path",
    default="./model/face_recognition_model_all",
    help="Path to the saved model.",
)
@click.option(
    "--encodings-path",
    default="./model/face_encodings.pkl",
    help="Path to the saved face encodings.",
)
@click.option(
    "--to-predict-path",
    default="./data/to_predict",
    help="Directory containing images to predict.",
)
@click.option(
    "--img-size",
    type=(int, int),
    default=(160, 160),
    help="Desired size for resizing the images.",
)
def main(
    model_path: str, encodings_path: str, to_predict_path: str, img_size: tuple
) -> None:
    """
    Run the PlayerPredictor to evaluate predictions on a set of images.

    :param model_path: Path to the saved model.
    :param encodings_path: Path to the saved face encodings.
    :param to_predict_path: Directory containing images to predict.
    :param img_size: Desired size for resizing the images.
    """
    model_loader = ModelLoader(model_path)
    encoding_loader = FaceEncodingLoader(encodings_path)
    predictor = PlayerPredictor(model_loader, encoding_loader, img_size)
    predictor.evaluate_predictions(to_predict_path)


if __name__ == "__main__":
    main()

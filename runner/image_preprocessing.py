import os
import logging
import colorlog
from PIL import Image
import face_recognition
import numpy as np
from shutil import move
import re
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a color formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

# Create a handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger("ImagePreprocessor")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class ImagePreprocessor:
    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        size: tuple = (128, 128),
        threshold: float = 0.5,
    ):
        """
        Initialize the ImagePreprocessor with directories and parameters.

        :param image_dir: Directory containing the downloaded images.
        :param output_dir: Directory to save the preprocessed images.
        :param size: Desired size for resizing the images.
        :param threshold: Threshold for face distance to detect outliers.
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.size = size
        self.threshold = threshold
        self.create_directory(output_dir)

    @staticmethod
    def create_directory(directory: str) -> None:
        """Create directory if it does not exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def is_valid_image_name(image_name: str) -> bool:
        """Check if the image name follows the pattern of a number followed by .jpg, .png, or .jpeg."""
        pattern = r"^\d+\.(jpg|png|jpeg)$"
        return bool(re.match(pattern, image_name))

    @staticmethod
    def is_valid_player_name(player_name: str) -> bool:
        """Check if the image name follows the pattern of a number followed by .jpg, .png, or .jpeg."""
        pattern = r"^[A-Z]\w+_\w+$"
        return bool(re.match(pattern, player_name))

    def collect_valid_images(self, player_dir: str, trash_dir: str) -> list:
        """
        Collect valid images with face encodings from the player's directory.
        Move images with multiple faces or no faces to the trash directory.

        :param player_dir: Directory containing the player's images.
        :param trash_dir: Directory to move invalid images.
        :return: List of valid images and their face encodings.
        """
        valid_images = []

        for image_name in os.listdir(player_dir):
            if not self.is_valid_image_name(image_name):
                continue

            image_path = os.path.join(player_dir, image_name)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error opening {image_path}: {e}")
                move(image_path, os.path.join(trash_dir, image_name))
                continue

            image_array = np.array(image)
            face_locations = face_recognition.face_locations(image_array)
            face_encodings = face_recognition.face_encodings(
                image_array, face_locations
            )

            if len(face_encodings) == 1:
                valid_images.append((image, image_name, image_path, face_encodings[0]))
                # print("Found valid image!")
            else:
                move(image_path, os.path.join(trash_dir, image_name))
                logger.warning(
                    f"Moved {image_name} to trash (multiple/no faces detected)."
                )
                print(f"{len(valid_images)} valid images found until now...")

        return valid_images

    def filter_outliers(self, valid_images: list, trash_dir: str) -> list:
        """
        Filter out images with face encodings that are outliers based on the distance threshold.

        :param valid_images: List of valid image details and their face encodings.
        :param trash_dir: Directory to move outlier images.
        :return: List of non-outlier image details.
        """
        face_encodings_list = [encoding for _, _, _, encoding in valid_images]
        mean_encoding = np.mean(face_encodings_list, axis=0)
        distances = face_recognition.face_distance(face_encodings_list, mean_encoding)
        non_outliers = []

        for (image, image_name, image_path, _), distance in zip(
            valid_images, distances
        ):
            if distance > self.threshold:
                move(image_path, os.path.join(trash_dir, image_name))
                logger.warning(f"Moved {image_name} to trash (outlier detected).")
            else:
                non_outliers.append((image, image_name, image_path))

        return non_outliers

    def preprocess_and_save(
        self, image: Image, output_player_dir: str, image_name: str
    ) -> None:
        """
        Preprocess the image and save it to the specified directory.

        :param image: PIL Image object.
        :param output_player_dir: Directory to save the preprocessed image.
        :param image_name: Name of the current image.
        """
        image = image.resize(self.size)
        image.save(os.path.join(output_player_dir, image_name))
        logger.info(f"Processed and saved {image_name}")

    def preprocess_images(self) -> None:
        """
        Preprocess images by resizing and converting to RGB format. Moves images with multiple faces or outliers to a trash folder.
        """
        for player in tqdm(os.listdir(self.image_dir)):
            if not self.is_valid_player_name(player):
                continue
            player_dir = os.path.join(self.image_dir, player)
            output_player_dir = os.path.join(self.output_dir, player)
            trash_dir = os.path.join(output_player_dir, "trash")

            self.create_directory(output_player_dir)
            self.create_directory(trash_dir)

            logger.info(f"Processing player: {player}")
            valid_images = self.collect_valid_images(player_dir, trash_dir)
            non_outliers = self.filter_outliers(valid_images, trash_dir)

            for image, image_name, image_path in non_outliers:
                self.preprocess_and_save(image, output_player_dir, image_name)


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(
        "./data/player_images", "./data/preprocessed_images"
    )
    preprocessor.preprocess_images()

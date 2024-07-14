import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import logging
import colorlog
from runner.model_predict import PlayerPredictor, ModelLoader, FaceEncodingLoader

# Set up logging for Flask app
logging.basicConfig(level=logging.DEBUG)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levellevel)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger("FlaskApp")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model and encodings
model_loader = ModelLoader("./model/face_recognition_model_all")
encoding_loader = FaceEncodingLoader("./model/face_encodings.pkl")
predictor = PlayerPredictor(model_loader, encoding_loader)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the player in the uploaded image using the specified method.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if file and PlayerPredictor.is_valid_image_name(file.filename):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        method = request.form.get("method", "combined")
        results = predictor.predict_image(file_path, method)
        if results:
            player_name = results.get("Combined Prediction")[0]
            probability = results.get("Combined Prediction")[1] * 100
            if probability < 65 or results.get("Face Comparison")[1] > 0.6:
                return render_template(
                    "result.html",
                    player_name=None,
                    probability=None,
                    file_name=file.filename,
                )
            return render_template(
                "result.html",
                player_name=player_name,
                probability=probability,
                file_name=file.filename,
            )
        else:
            return jsonify({"error": "Prediction failed"}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

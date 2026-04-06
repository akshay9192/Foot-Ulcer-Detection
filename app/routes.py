import os

import pandas as pd
from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from .image_model import load_image_model, predict_image
from .model import load_models, make_predictions

main = Blueprint("main", __name__)

UPLOAD_DIR = "app/uploads"
REQUIRED_FIELDS = ["field1", "field2", "field3", "field4", "field5", "field6"]
ALLOWED_CSV_EXTENSIONS = {"csv"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

models = load_models()
image_model = load_image_model()
os.makedirs(UPLOAD_DIR, exist_ok=True)


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    if "csv_file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["csv_file"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        file.save(file_path)
        df = pd.read_csv(file_path)

        if not set(REQUIRED_FIELDS).issubset(df.columns):
            return jsonify({"error": "Missing required fields", "required_fields": REQUIRED_FIELDS}), 400

        input_data = df[REQUIRED_FIELDS].astype(float)
        predictions = make_predictions(models, input_data)
        return jsonify(predictions)

    except Exception as exc:
        return jsonify({"error": f"Error processing file: {exc}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@main.route("/predict-image", methods=["POST"])
def predict_image_route():
    if "image_file" not in request.files:
        return jsonify({"error": "No image file part"}), 400

    file = request.files["image_file"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Invalid image type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        file.save(file_path)
        with open(file_path, "rb") as image_stream:
            result = predict_image(image_model, image_stream)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Error processing image: {exc}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

from flask import Blueprint, render_template, request, jsonify
from .model import load_models, make_predictions
import pandas as pd
import os
from werkzeug.utils import secure_filename

main = Blueprint("main", __name__)

# Load models when the app starts
models = load_models()

# Ensure the 'uploads' directory exists
if not os.path.exists('app/uploads'):
    os.makedirs('app/uploads')

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict():
    if 'csv_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['csv_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("app/uploads", filename)

    try:
        file.save(file_path)

        # Load CSV into DataFrame
        df = pd.read_csv(file_path)

        # Validate required fields
        required_fields = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6']
        if not set(required_fields).issubset(df.columns):
            return jsonify({"error": "Missing required fields in the CSV file"}), 400

        # Extract and format input data
        input_data = df[required_fields].astype(float).values  # Ensure 2D ndarray

        print("Shape of input_data before reshaping:", input_data.shape)

        # Ensure input is 2D
        if input_data.ndim == 3:
            input_data = input_data.reshape(input_data.shape[1], input_data.shape[2])
        elif input_data.ndim != 2:
            raise ValueError(f"Unexpected input shape: {input_data.shape}")

        print("Shape of input_data after reshaping:", input_data.shape)

        # Make predictions
        predictions = make_predictions(models, input_data)

        # Remove uploaded file
        os.remove(file_path)

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": f"Error processing the file: {e}"}), 500


# Helper function to check allowed file types
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

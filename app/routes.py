from flask import Blueprint, render_template, request, jsonify
from .model import load_models, make_predictions

main = Blueprint("main", __name__)

# Load models when the app starts
models = load_models()

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    input_data = request.form.to_dict()
    
    # Convert input data to the required format (if needed)
    predictions = make_predictions(models, input_data)
    
    return jsonify(predictions)

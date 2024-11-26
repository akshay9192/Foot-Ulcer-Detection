from flask import Blueprint, request, jsonify, render_template
from .model import train_model, predict

app_routes = Blueprint('app_routes', __name__)

# Index route
@app_routes.route('/')
def index():
    return render_template('index.html')

# Train the model
@app_routes.route('/train', methods=['POST'])
def train():
    result = train_model('data/modified_pressure_data.csv')
    return jsonify(result)

# Predict route
@app_routes.route('/predict', methods=['POST'])
def predict_ulcer():
    input_data = request.json.get('input_data', [])
    if len(input_data) != 6:
        return jsonify({"error": "Invalid input data. Six pressure values are required."}), 400
    
    result = predict(input_data)
    return jsonify(result)

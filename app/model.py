import pickle
import pandas as pd

def load_models():
    # Load all models into a dictionary
    models = {
        "Random Forest": pickle.load(open("models/rf_model.pkl", "rb")),
        "SVM": pickle.load(open("models/svm_model.pkl", "rb")),
        "Logistic Regression": pickle.load(open("models/logreg_model.pkl", "rb")),
        "Gradient Boosting": pickle.load(open("models/gb_model.pkl", "rb")),
        "KNN": pickle.load(open("models/knn_model.pkl", "rb")),
        "Neural Network": pickle.load(open("models/mlp_model.pkl", "rb"))
    }
    return models

def make_predictions(models, input_data):
    # Convert input data into a DataFrame
    features = pd.DataFrame([input_data])
    features = features.astype(float)

    # Make predictions with each model
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(features)
        predictions[model_name] = pred.tolist()
    
    return predictions

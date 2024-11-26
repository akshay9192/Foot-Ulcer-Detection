import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

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
    # Preprocess the input data (e.g., handle missing values, scaling)
    features = pd.DataFrame([input_data])
    features = features.astype(float)
    
    # If scaling is needed, normalize the features the same way as during training (if you used StandardScaler)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Make predictions with each model
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(features_scaled)
        predictions[model_name] = pred.tolist()
    
    return predictions

def train_models():
    # Placeholder: Implement actual model training code here
    df = pd.read_csv("data/dataset.csv")  # Load the dataset
    X = df.drop("target", axis=1)  # Drop the target column
    y = df["target"]  # Target column

    # Handle any missing values, if needed (e.g., impute or drop)
    X = X.fillna(X.mean())  # Simple imputation with mean

    # If scaling is needed during training, scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit and transform the data

    # Train models and save them
    rf_model = RandomForestClassifier()
    rf_model.fit(X_scaled, y)
    pickle.dump(rf_model, open("models/rf_model.pkl", "wb"))

    svm_model = SVC()
    svm_model.fit(X_scaled, y)
    pickle.dump(svm_model, open("models/svm_model.pkl", "wb"))

    logreg_model = LogisticRegression()
    logreg_model.fit(X_scaled, y)
    pickle.dump(logreg_model, open("models/logreg_model.pkl", "wb"))

    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_scaled, y)
    pickle.dump(gb_model, open("models/gb_model.pkl", "wb"))

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_scaled, y)
    pickle.dump(knn_model, open("models/knn_model.pkl", "wb"))

    mlp_model = MLPClassifier()
    mlp_model.fit(X_scaled, y)
    pickle.dump(mlp_model, open("models/mlp_model.pkl", "wb"))

    return "Models are trained and saved!"

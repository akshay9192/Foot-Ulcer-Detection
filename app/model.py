import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train the model
def train_model(data_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Feature columns and target
    feature_cols = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6']
    target_col = 'label'  # Add a 'label' column (0 or 1)

    X = data[feature_cols]
    y = data[target_col]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open('models/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Evaluate the model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return {"accuracy": accuracy}

# Predict using the model
def predict(input_data):
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([input_data])
    return {"prediction": int(prediction[0])}

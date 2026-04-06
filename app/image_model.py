import pickle
from pathlib import Path
import numpy as np

IMAGE_MODEL_PATH = Path("models/image_clf.pkl")

def load_image_model(model_path: Path = IMAGE_MODEL_PATH):
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)

def preprocess_image(image_file, image_size=(128, 128)):
    from PIL import Image
    image = Image.open(image_file).convert("RGB")
    image = image.resize(image_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)

def predict_image(image_model, image_file):
    if image_model is None:
        raise ValueError("Image model not found. Train scripts/train_image_classifier.py first.")
    x = preprocess_image(image_file)
    pred = image_model.predict(x)[0]
    prob = float(image_model.predict_proba(x)[0][1]) if hasattr(image_model, "predict_proba") else None
    return {"prediction": int(pred), "ulcer_probability": prob}

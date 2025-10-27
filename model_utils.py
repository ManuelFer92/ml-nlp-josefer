# model_utils.py
import re
import joblib
from pathlib import Path

# --- Función de preprocesamiento global ---
def preprocessing_min(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Rutas del modelo ---
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
BUNDLE_PATH = MODELS_DIR / "model-latest.pkl"

# --- Guardado y carga ---
def save_model_bundle(bundle):
    joblib.dump(bundle, BUNDLE_PATH, compress=("xz", 3))

def load_model_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("models/model-latest.pkl not found. Run training first.")
    return joblib.load(BUNDLE_PATH)

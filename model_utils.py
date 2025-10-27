import joblib
import re
from pathlib import Path

def preprocessing_min(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
BUNDLE_PATH = MODELS_DIR / "model-latest.pkl"

def save_model_bundle(bundle):
    joblib.dump(bundle, BUNDLE_PATH)

def load_model_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("models/model-latest.pkl not found. Run train.py first.")
    return joblib.load(BUNDLE_PATH)

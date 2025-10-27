import pandas as pd
import joblib, json, re, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from model_utils import save_model_bundle, preprocessing_min

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emotions.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model-latest.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- ENTRENAMIENTO ---
def train_and_eval(sample_fraction: float = 0.4, use_logreg: bool = True):
    """
    Entrena el modelo de emociones de forma optimizada.
    Parámetros:
      sample_fraction : fracción del dataset a usar (0.4 = 40%)
      use_logreg      : True usa LogisticRegression, False usa RandomForest
    """
    print(f"🔹 Training Emotion Classifier ({'LogReg' if use_logreg else 'RF'})...")
    print(f"📦 Using {sample_fraction*100:.0f}% of data for training")

    df = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("El CSV debe tener columnas 'text' y 'label'.")

    # Usar solo una fracción del dataset
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # --- Modelo ---
    if use_logreg:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocessing_min,
            max_features=1000,
            ngram_range=(1, 1)
        )),
        ("clf", model)
    ])

    # --- Entrenamiento ---
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    print(f"✅ Accuracy: {acc:.4f}")

    # --- Guardado ---
    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {"model": pipeline, "target_names": sorted(df["label"].unique().tolist())}
    save_model_bundle(bundle)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"💾 Modelo guardado en: {MODEL_PATH}")
    print(f"📊 Métricas guardadas en: {METRICS_PATH}")

    return {"accuracy": acc, "samples_used": len(df), "metrics": metrics}

if __name__ == "__main__":
    result = train_and_eval(sample_fraction=0.4, use_logreg=True)
    print("\n🔍 Training Summary:")
    print(json.dumps(result["metrics"], indent=2))

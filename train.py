import pandas as pd
import joblib, json, re, os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from model_utils import save_model_bundle, preprocessing_min

# ============================================================
# CONFIGURACIÓN
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emotions.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model-latest.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def train_and_eval(
    sample_fraction: float = 1.0,
    use_logreg: bool = False,
    max_features: int = 8000,
):
    """
    Entrena un modelo de emociones (TF-IDF + LogReg o RandomForest)
    y ajusta hiperparámetros básicos para mejorar rendimiento.
    """
    print(f"🔹 Training Emotion Classifier ({'LogReg' if use_logreg else 'RF'})")
    print(f"📦 Using {sample_fraction*100:.0f}% of dataset")

    # ------------------------------------------------------------
    # CARGA Y PREPROCESAMIENTO
    # ------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("El CSV debe tener columnas 'text' y 'label'.")

    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    # 70 / 15 / 15
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    print(f"📊 Split -> Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # ------------------------------------------------------------
    # SELECCIÓN DE MODELO
    # ------------------------------------------------------------
    if use_logreg:
        model = LogisticRegression(max_iter=1500, solver="liblinear", C=2.0)
    else:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocessing_min,
            max_features=max_features,
            ngram_range=(1, 2)
        )),
        ("clf", model)
    ])

    # ------------------------------------------------------------
    # BÚSQUEDA DE HIPERPARÁMETROS (solo para RF)
    # ------------------------------------------------------------
    if not use_logreg:
        param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [15, 25, None],
        }
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1,
            scoring="accuracy",
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        print(f"🏆 Best Params: {grid.best_params_}")
    else:
        pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------
    # VALIDACIÓN Y TEST
    # ------------------------------------------------------------
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    metrics = classification_report(y_test, y_test_pred, output_dict=True)

    print(f"📈 Validation Accuracy: {val_acc:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    # ------------------------------------------------------------
    # GUARDADO
    # ------------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {"model": pipeline, "target_names": sorted(df["label"].unique().tolist())}
    joblib.dump(bundle, MODEL_PATH, compress=("xz", 3))
    save_model_bundle(bundle)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"💾 Model Size: {size_mb:.2f} MB  |  Path: {MODEL_PATH}")
    print(f"📊 Metrics saved at: {METRICS_PATH}")

    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "size_mb": size_mb,
        "metrics": metrics,
    }

# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    result = train_and_eval(
        sample_fraction=1.0,
        use_logreg=False,      # usa RandomForest optimizado
        max_features=8000
    )
    print("\n🔍 Training Summary:")
    print(json.dumps({
        "Validation_Accuracy": result["val_acc"],
        "Test_Accuracy": result["test_acc"],
        "Model_Size_MB": result["size_mb"],
    }, indent=2))

import gradio as gr
from model_utils import load_model_bundle
import numpy as np

# --- Carga el modelo guardado ---
bundle = load_model_bundle()
MODEL = bundle["model"]
TARGET_NAMES = bundle["target_names"]

def predict_emotion(text):
    """Predice la emoción y devuelve probabilidades."""
    preds = MODEL.predict([text])[0]
    probs = MODEL.predict_proba([text])[0]
    result = {TARGET_NAMES[i]: float(probs[i]) for i in range(len(TARGET_NAMES))}
    return result

# --- Interfaz Gradio ---
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(label="✍️ Ingresa un texto o titular de noticia"),
    outputs=gr.Label(num_top_classes=3),
    title="🧠 Emotion Classifier (TF-IDF + RandomForest)",
    description="Clasificador de emociones basado en TF-IDF + RandomForest entrenado con dataset anotado.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()

---
title: NLP Emotion Classifier
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.37.2"
app_file: app.py
pinned: false
license: mit
---

# 🧠 NLP Emotion Classifier

Clasificador de emociones basado en **TF-IDF + Logistic Regression**, entrenado sobre un dataset anotado de titulares y textos cortos.

---

## 🚀 Demo

Escribe una frase o titular para analizar su emoción predominante.

Ejemplo:
> *"The team celebrated their victory with joy and pride."*

---

## ⚙️ Tecnologías

- **Python 3.10+**
- **Scikit-learn** (TF-IDF + Logistic Regression)
- **Gradio** (interfaz)
- **Hugging Face Spaces** (despliegue automático desde GitHub)
- **GitHub Actions** (CI/CD integrado)

---

## 🧩 Cómo usarlo localmente

```bash
git clone https://github.com/ManuelFer92/ml-nlp-josefer.git
cd ml-nlp-josefer
pip install -r requirements.txt
python app.py

---
title: Emotion Classifier
emoji: 💬
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🧠 Emotion Classifier

Este proyecto implementa un clasificador de emociones basado en **TF-IDF + RandomForest**, integrado con **CI/CD de GitHub Actions** y desplegado automáticamente en **Hugging Face Spaces**.

**Workflow:**
- `ci.yml`: Ejecuta pruebas automáticas.
- `train.yml`: Entrena el modelo periódicamente.
- `deploy.yml`: Sincroniza los cambios con Hugging Face.

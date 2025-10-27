from train import train_and_eval

def test_training_runs():
    """
    Prueba rápida para verificar que el modelo entrene correctamente
    y alcance una precisión mínima razonable.
    """
    result = train_and_eval(sample_fraction=0.2, use_logreg=False, max_features=2000)
    print(result)
    
    # Validar que las métricas clave existen
    assert "val_acc" in result and "test_acc" in result, "Faltan métricas de accuracy"
    
    # Asegurarse de que los valores sean razonables (> 0.5 en test)
    assert result["test_acc"] > 0.5, f"Accuracy demasiado bajo: {result['test_acc']}"
    
    # Validar que el modelo pese menos de 50 MB
    assert result["size_mb"] < 50, f"Modelo demasiado grande: {result['size_mb']} MB"

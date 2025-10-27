from train import train_and_eval

def test_training_runs():
    metrics = train_and_eval()
    assert metrics["accuracy"] > 0.5

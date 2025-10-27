from train import train_and_eval

def test_training_runs():
    metrics = train_and_eval(sample_fraction=0.2)
    assert metrics["accuracy"] > 0.5

from pipeline.load_data import load_data
from pipeline.train import train_model
from pipeline.evaluate import evaluate_model


def test_data_shape():
    X, y = load_data()
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 4


def test_model_accuracy():
    model, X_test, y_test = train_model()
    acc = evaluate_model(model, X_test, y_test)
    assert acc > 0.8
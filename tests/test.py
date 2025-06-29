from pipeline.load_data import load_data
from pipeline.train import train_model
import unittest
from pipeline.evaluate import evaluate_model


class TestEvaluation(unittest.TestCase):
    def test_score_range(self):
        model, X_test, y_test = train_model()
        score = evaluate_model(model, X_test, y_test)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_model_accuracy(self):
        model, X_test, y_test = train_model()
        acc = evaluate_model(model, X_test, y_test)
        self.assertGreater(acc, 0.8)

if __name__ == '__main__':
    unittest.main()


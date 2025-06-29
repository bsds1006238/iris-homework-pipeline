from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from load_data import load_data
import joblib

def train_model(save_path="model.pkl"):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
import pandas as pd


def load_data():
    df = pd.read_csv("data/iris.csv")
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    Y = df["species"]
    print(X.head())
    print(Y.head())
    return X,Y

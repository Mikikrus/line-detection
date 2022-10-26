from tensorflow import keras
import numpy as np

def load_model(path: str) -> keras.Model:
    model = keras.models.load_model(path)
    return model


def predict(model: keras.Model, X: np.ndarray, y: np.ndarray) -> float:
    return model.evaluate(X, y)
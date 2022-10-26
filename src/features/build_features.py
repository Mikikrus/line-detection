import numpy as np


def preprocess(dataset: zip) -> (np.ndarray, np.ndarray):
    X, y = dataset
    return np.array(X), np.array(y) / 64

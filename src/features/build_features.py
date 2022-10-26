import numpy as np


def preprocess(dataset):
    X, y = dataset
    return np.array(X), np.array(y) / 64

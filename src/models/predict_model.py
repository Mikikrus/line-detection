from tensorflow import keras


def load_model(path):
    model = keras.models.load_model(path)
    return model


def predict(model, X, y):
    return model.evaluate(X, y)
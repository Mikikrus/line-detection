from tensorflow import keras
import numpy as np


def make_model() -> keras.Model:
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(keras.layers.AveragePooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.AveragePooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(4))
    return model


def compile_and_train(model: keras.Model, X: np.ndarray, y:np.ndarray) -> (keras.Model, keras.callbacks.History):
    model.compile(loss="mean_absolute_error", optimizer="Adam", metrics=[])
    checkpoint_filepath = './tmp/checkpoint'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    # fit model
    history = model.fit(X, y, batch_size=256, validation_split=0.2, epochs=100, verbose=1,
                        callbacks=[checkpoint_callback, es])
    model.load_weights(checkpoint_filepath)
    return model, history
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from src.data.make_dataset import generate_image

def show_training_history(history: keras.callbacks.History) -> None:
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def show_examples(model: keras.Model, n: int) -> None:
    canvas = np.full([64, 64, 1], 1, dtype=np.uint8)
    fig, axs = plt.subplots(n, n)
    fig.set_size_inches(15, 15)
    for i in range(n):
        for j in range(n):
            im, true = generate_image(canvas)
            im = np.expand_dims(im, axis=0)
            res = model.predict(im, verbose=0)
            res = np.around(np.array((res[0] * 64))).astype(int)
            #         print(true, res)
            axs[i, j].axis('off')
            axs[i, j].set_xlim([0, 64])
            axs[i, j].set_ylim([0, 64])
            axs[i, j].plot(true[0::2], true[1::2], c='k')
            axs[i, j].plot(res[0::2], res[1::2], c='r')
            axs[i, j].scatter(res[0::2], res[1::2], c='r')
    plt.show()
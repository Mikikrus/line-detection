import numpy as np
import random
import cv2


def generate_image(canvas: np.ndarray) -> np.ndarray:
    x = (random.randint(0, 64), random.randint(0, 64))
    y = (random.randint(0, 64), random.randint(0, 64))
    return cv2.line(canvas.copy(), x, y, False, 1), np.array(sorted([x, y])).flatten()


def generate_dataset(n: int) -> zip:
    canvas = np.full([64,64, 1], 1, dtype=np.uint8)
    dataset = (generate_image(canvas) for _ in range(n))
    return zip(*dataset)

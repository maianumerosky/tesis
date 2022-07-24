import numpy as np


def preprocess(images):
    scaled_images = _scale(images)
    return _stretch(scaled_images)


def _stretch(images):
    amount_of_images = images.shape[0]
    width = images.shape[1]
    if len(images.shape) > 2:
        height = images.shape[2]
    else:
        height = 1
    amount_pixels = width * height
    return np.reshape(images, (amount_of_images, amount_pixels))


def _scale(images):
    return images / 10000


def flatten_last_dims(image):
    return np.reshape(image, (image.shape[0], -1))
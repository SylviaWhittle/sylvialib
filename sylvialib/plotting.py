"""Useful plotting functions"""

import numpy as np
import matplotlib.pyplot as plt


def imshow(image: np.ndarray, size=(8, 8)):
    """Plot an image"""
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.show()


def plot_gallery(images: list[np.ndarray], n_cols=4):
    """Plot a list of images in a grid."""

    n_images = len(images)
    n_rows = np.ceil(n_images / n_cols)

    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(20 * n_rows, 20))
    for index, image in enumerate(images):
        axes[int(index // n_cols), index % n_cols].imshow(image)

    plt.show()

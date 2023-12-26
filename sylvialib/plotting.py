"""Useful plotting functions"""

from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def imshow(
    image: np.ndarray,
    size=(8, 8),
    title: str = "",
    cmap: Union[str, matplotlib.colors.Colormap] = None,
):
    """Plot an image"""
    plt.figure(figsize=size)
    if cmap:
        plt.imshow(image, cmap=cmap)
    plt.imshow(image)
    plt.title(title)
    plt.show()


def plot_gallery(images: list[np.ndarray], n_cols=4, title=None):
    """Plot a list of images in a grid."""

    n_images = len(images)
    n_rows = np.ceil(n_images / n_cols)

    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(20 * n_rows, 20))
    for index, image in enumerate(images):
        axes[int(index // n_cols), index % n_cols].imshow(image)

    if title:
        plt.title(title)

    plt.show()

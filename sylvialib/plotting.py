"""Useful plotting functions"""

import numpy as np
import matplotlib.pyplot as plt


def imshow(image: np.ndarray, size=(8, 8)):
    """Plot an image"""
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.show()

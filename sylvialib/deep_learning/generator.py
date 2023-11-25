"""A generator for deep learning models"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# An image generator that loads images as they are needed
def image_generator(
    original_image_dir: Path,
    mask_dir: Path,
    image_indexes: list,
    batch_size: int = 4,
    file_type: str = ".npy",
):
    """A generator that yields batches of images and ground truth masks.

    Parameters
    ----------
    original_image_dir : Path
        The directory containing the original images.
    mask_dir : Path
        The directory containing the ground truth masks.
    image_indexes : list
        A list of the indices of the images to be loaded.
    batch_size : int, optional
        The number of images to be loaded per batch. The default is 4.
    file_type : str, optional
        The file type of the images. The default is ".npy".

    Yields
    ------
    batch_x : np.ndarray
        A batch of images.
    batch_y : np.ndarray
        A batch of ground truth masks.

    """

    while True:
        # Select files (paths/indices) for the batch
        batch_image_indexes = np.random.choice(a=image_indexes, size=batch_size)
        batch_input = []
        batch_output = []

        # Load the image and ground truth
        for index in batch_image_indexes:
            # Load the training image
            if file_type == ".npy":
                image = np.load(original_image_dir / f"image_{index}.npy")
            elif file_type == ".png":
                image = cv2.imread(str(original_image_dir / f"image_{index}.png"), 0)
            else:
                raise ValueError("File type must be either .npy or .png")
            # Rescale the image to 512x512
            image = Image.fromarray(image)
            image = image.resize((512, 512))
            image = np.array(image)
            # Normalise the image
            image = image - np.min(image)
            image = image / np.max(image)

            # Load the ground truth
            if file_type == ".npy":
                ground_truth = np.load(mask_dir / f"mask_{index}.npy")
            elif file_type == ".png":
                ground_truth = cv2.imread(str(mask_dir / f"mask_{index}.png"), 0)
            else:
                raise ValueError("File type must be either .npy or .png")
            ground_truth = np.array(ground_truth)
            # Force the ground truth to be boolean
            ground_truth = ground_truth.astype(bool)
            # Rescale the image to 512x512
            ground_truth = Image.fromarray(ground_truth.astype(np.uint8))
            ground_truth = ground_truth.resize((512, 512))
            ground_truth = np.array(ground_truth).astype(int)

            # Augment the image and ground truth
            # Flip the images 50% of the time
            if random.choice([0, 1]) == 1:
                image = np.flip(image, axis=1)
                ground_truth = np.flip(ground_truth, axis=1)
            # Rotate the images by either 0, 90, 180, or 270 degrees
            rotation = random.choice([0, 1, 2, 3])
            image = np.rot90(image, rotation)
            ground_truth = np.rot90(ground_truth, rotation)

            # Add the image and ground truth to the batch
            batch_input.append(image)
            batch_output.append(ground_truth)

        # Force the batch to be numpy arrays
        batch_x = np.array(batch_input).astype(np.float32)
        batch_y = np.array(batch_output).astype(np.float32)

        yield (batch_x, batch_y)

"""A generator for deep learning models"""

import cv2
import numpy as np
import random
import re
from PIL import Image


# An image generator that loads images as they are needed
def image_generator(MASK_DIR, ORIGINAL_IMAGE_DIR, image_indexes, batch_size=4):
    while True:
        # Select files (paths/indices) for the batch
        batch_image_indexes = np.random.choice(a=image_indexes, size=batch_size)
        batch_input = []
        batch_output = []

        # Load the image and ground truth
        for index in batch_image_indexes:
            # Find the index as the only number in the filename
            # index = re.search(r"\d+", image_path)
            # print(index)
            image = cv2.imread(str(ORIGINAL_IMAGE_DIR / f"training_image_{index}.png"), 0)
            image = Image.fromarray(image)
            image = image.resize((512, 512))
            image = np.array(image)
            # Normalise the image
            image = image - np.min(image)
            image = image / np.max(image)

            # ground_truth = np.load(MASK_DIR / f"mask_array_{index}.npy")
            ground_truth = cv2.imread(str(MASK_DIR / f"mask_{index}.png"), 0)
            ground_truth = Image.fromarray(ground_truth)
            ground_truth = np.array(ground_truth)
            ground_truth = ground_truth.astype(bool)
            ground_truth = Image.fromarray(ground_truth.astype(np.uint8))
            ground_truth = ground_truth.resize((512, 512))
            ground_truth = np.array(ground_truth).astype(int)

            # Augment the images
            # Flip the images 50% of the time
            if random.choice([0, 1]) == 1:
                image = np.flip(image, axis=1)
                ground_truth = np.flip(ground_truth, axis=1)
            # Rotate the images by either 0, 90, 180, or 270 degrees
            rotation = random.choice([0, 1, 2, 3])
            image = np.rot90(image, rotation)
            ground_truth = np.rot90(ground_truth, rotation)

            batch_input.append(image)
            batch_output.append(ground_truth)

        batch_x = np.array(batch_input).astype(np.float32)
        batch_y = np.array(batch_output).astype(np.float32)

        yield (batch_x, batch_y)

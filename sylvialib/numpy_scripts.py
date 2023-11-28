"""Scripts for various numpy operations."""

from typing import List, Tuple

import numpy as np


def coordinate_in_array(coordinate: np.ndarray[(int, int)], array: np.ndarray[Tuple]) -> bool:
    """Check if a coordinate is in an array."""

    array_view = array.view([("", array.dtype)] * array.shape[1])
    coordinate_view = coordinate.view([("", coordinate.dtype)] * coordinate.shape[0])

    common_coordinates = np.in1d(array_view, coordinate_view)

    # If common coordinates is not empty, then the coordinate is in the array
    if common_coordinates.any():
        return True
    return False


def find_touching_pixels(image: np.ndarray) -> np.ndarray:
    """Take an image with three labels: background 0, object 1, and object 2,
    and return the pixels where object 1 are adjacent to object 2.
    """

    # Get the pixels where the image is labelled as object 1
    object_1_pixels = np.where(image == 1)
    # Get the pixels where the image is labelled as object 2
    object_2_pixels = np.where(image == 2)

    # Get the coordinates of the object 1 pixels
    object_1_coordinates = np.array(list(zip(object_1_pixels[0], object_1_pixels[1])))
    # Get the coordinates of the object 2 pixels
    object_2_coordinates = np.array(list(zip(object_2_pixels[0], object_2_pixels[1])))

    # Get the coordinates of the pixels where object 1 and object 2 are adjacent
    touching_pixels = []
    for object_1_coordinate in object_1_coordinates:
        # Get the adjacent pixels
        adjacent_pixels = np.array(
            [
                [object_1_coordinate[0] - 1, object_1_coordinate[1]],
                [object_1_coordinate[0] + 1, object_1_coordinate[1]],
                [object_1_coordinate[0], object_1_coordinate[1] - 1],
                [object_1_coordinate[0], object_1_coordinate[1] + 1],
            ]
        )

        # Check if any of the adjacent pixels are in the object 2 coordinates
        for adjacent_pixel in adjacent_pixels:
            if coordinate_in_array(adjacent_pixel, object_2_coordinates):
                touching_pixels.append(object_1_coordinate)

    # Convert the touching pixels to a numpy array
    return np.array(touching_pixels)


def create_2d_array_from_string(string: str) -> np.ndarray:
    """Create a 2d numpy array from grid in the form of a string. This is useful for creating
    custom images and masks to use in testing.

    Notes:
    - The string should be composed of rows with single integer values separated by spaces.
    - Each row should be separated by a newline character.


    Example:
    --------
    ```
    >>> string = "
    ... 0 0 0 1 0 0 0
    ... 0 0 1 1 1 0 0
    ... 0 1 1 1 1 1 0
    ... 0 0 0 0 0 0 0
    ... 0 0 0 0 0 0 0
    ... "
    >>> array = create_2d_array_from_string(string)
    >>> print(array)
    [[0 0 0 1 0 0 0]
     [0 0 1 1 1 0 0]
     [0 1 1 1 1 1 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]
    ```

    Parameters
    ----------
    string: str
        A string representing a 2d grid of values.

    Returns:
    --------
    np.ndarray
        A 2d numpy array.
    """

    # Take a string representing a 2d grid of values and convert it to a 2d numpy array
    # For example:
    # string = """
    # 0 0 0 1 0 0 0
    # 0 0 1 1 1 0 0
    # 0 1 1 1 1 1 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # """

    # Split string into rows
    rows: List[str] = string.split("\n")

    # Remove empty rows
    rows = [row for row in rows if row != ""]
    # Remove rows that are just all spaces
    rows = [row for row in rows if not row.isspace()]
    # Remove leading and trailing spaces
    rows = [row.strip() for row in rows]

    # Convert to integers
    integer_rows = [[int(i) for i in row.split()] for row in rows]

    # Convert to numpy array
    array = np.array(integer_rows)

    return array

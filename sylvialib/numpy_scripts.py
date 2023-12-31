"""Scripts for various numpy operations."""

from typing import List, Tuple

import numpy as np

from scipy.ndimage import binary_dilation


def coordinate_in_array(
    coordinate: np.ndarray[(int, int)], array: np.ndarray[Tuple]
) -> bool:
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

    Notes:
    - Does not count diagonals as adjacent.
    - Returns the pixels in the first object that are adjacent to the second object and
    not the other way around.
    """

    # Get a mask for 1s
    mask_1 = image == 1
    # Get a mask for 2s
    mask_2 = image == 2

    # Get the pixels where the masks are adjacent
    # Dilate mask 1
    dilated_mask_2 = binary_dilation(mask_2)

    # Get the pixels where the dilated mask 1 overlaps with mask 2
    touching_pixels = np.logical_and(dilated_mask_2, mask_1)

    return touching_pixels


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


def detect_overlap(mask_1: np.ndarray, mask_2: np.ndarray):
    """Detect if two masks overlap and return the overlapping image"""

    return np.logical_and(mask_1, mask_2).any()

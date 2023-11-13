"""Scripts for various numpy operations."""

import numpy as np


def create_2d_array_from_string(string) -> np.ndarray:
    """Create a 2d numpy array from grid in the form of a string. This is useful for creating
    custom images and masks to use in testing.

    Parameters
    ----------
    string: str
        A string representing a 2d grid of values.

    Returns:
    --------
    np.ndarray
        A 2d numpy array.
    """

    # Split the string into rows
    rows = string.split("\n")

    # Split each row into individual integer values
    rows = [row.split() for row in rows]

    # Convert the values to integers
    rows = [[int(value) for value in row] for row in rows]

    # Convert the list of lists to a numpy array
    array = np.array(rows)

    return array

"""Scripts for various numpy operations."""

from typing import List

import numpy as np


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

"""Scripts for various numpy operations."""

from typing import List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from scipy.ndimage import binary_dilation


def coordinate_in_array(coordinate: np.ndarray[(int, int)], array: np.ndarray[Tuple]) -> bool:
    """Check if a coordinate is in an array."""

    return (coordinate == array).all(axis=1).any()


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


def calculate_curvature_from_points(x_points, y_points, error=0.1, k=4):
    """Calculate the curvature for a set of points"""
    # Check that the number of points is the same for both x and y
    if x_points.shape[0] != y_points.shape[0]:
        raise ValueError(
            "x_points and y_points must have the same number of points."
            f"x_points has {x_points.shape[0]} points and y_points has {y_points.shape[0]} points."
        )

    # Weight the values so less weight is given to points with higher error
    # K is the order of the spline to use. Increasing this increases the smoothness of the spline. A
    # value of 1 is linear interpolation, 2 is quadratic, 3 is cubic, etc. 4 is used to ensure the
    # spline is smooth enough to differentiate to the second derivative.
    # t is the independent variable that monotically increases with the data, similar to how
    # we use a dummy x variable in plotting calculations.

    # Disable pylint warning about snake case variable names for this function
    # pylint: disable=invalid-name
    t = np.arange(x_points.shape[0])
    weight_values = 1 / np.sqrt(error * np.ones_like(x_points))
    fx = UnivariateSpline(t, x_points, k=k, w=weight_values)
    fy = UnivariateSpline(t, y_points, k=k, w=weight_values)

    spline_x = fx(t)
    spline_y = fy(t)

    dx = fx.derivative(1)(t)
    dx2 = fx.derivative(2)(t)
    dy = fy.derivative(1)(t)
    dy2 = fy.derivative(2)(t)
    curvatures = (dx * dy2 - dy * dx2) / np.power(dx**2 + dy**2, 3 / 2)
    return curvatures, spline_x, spline_y

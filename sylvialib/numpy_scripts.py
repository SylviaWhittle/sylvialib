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


def calculate_curvature_periodic_boundary(x_points, y_points, error=0.1, periods=2, k=4):
    """Take a set of points that form a loop and calculate the curvature. Uses periodic
    boundary conditions, so the first and last points are connected. This reduces the error
    in the curvature calculation at the boundaries.

    Parameters
    ----------
    x_points: np.ndarray
        1D numpy array of x coordinates of the points.
    y_points: np.ndarray
        1D numpy array of y coordinates of the points.
    error: float
        Error in the points. Used to weight the points in the spline calculation.
    periods: int
        Number of times to repeat the points either side of the original points to reduce
        the error at the boundaries.

    Returns
    -------
    curvature: np.ndarray
        1D numpy array of the curvature for each point.
    """

    # Check that the number of points is the same for both x and y
    if x_points.shape[0] != y_points.shape[0]:
        raise ValueError(
            "x_points and y_points must have the same number of points."
            f"x_points has {x_points.shape[0]} points and y_points has {y_points.shape[0]} points."
        )

    # Repeat the points either side of the original points to reduce the error at the boundaries
    extended_points_x = np.copy(x_points)
    extended_points_y = np.copy(y_points)
    for _ in range(periods * 2):
        extended_points_x = np.append(extended_points_x, x_points)
        extended_points_y = np.append(extended_points_y, y_points)

    # Calculate the curvature
    extended_curvature, spline_x, spline_y = calculate_curvature_from_points(
        extended_points_x, extended_points_y, error=error, k=k
    )

    # Return only the original points
    return (
        extended_curvature[x_points.shape[0] * int(periods / 2) : x_points.shape[0] * int((periods / 2) + 1)],
        spline_x[x_points.shape[0] * int(periods / 2) : x_points.shape[0] * int((periods / 2) + 1)],
        spline_y[x_points.shape[0] * int(periods / 2) : x_points.shape[0] * int((periods / 2) + 1)],
    )


def turn_spline_path_into_pixel_map(array: np.ndarray):
    # Convert the spline to a pixelated trace 1 pixel thick

    # Create a map of pixels
    pixel_map = np.zeros((int(np.max(array) + 1), int(np.max(array) + 1)), dtype=int)
    pixelated_path = np.empty((0, 2), dtype=int)

    def check_is_touching(coordinate, original_coordinate):
        if np.abs(coordinate[0] - original_coordinate[0]) <= 1 and np.abs(coordinate[1] - original_coordinate[1]) <= 1:
            return True
        else:
            return False

    # Convert the array to integers and remove duplicates
    integer_array = np.array(array, dtype=int)
    removed_duplicates = []
    for index in range(len(integer_array)):
        coordinate = integer_array[index]
        if index > 0:
            if np.array_equal(coordinate, integer_array[index - 1]):
                continue

        removed_duplicates.append(coordinate)
    integer_array = np.array(removed_duplicates)

    last_coordinate = None
    for index in range(len(integer_array)):
        coordinate = integer_array[index]

        # If the coordinate is a repeat, skip it
        if index > 0:
            if np.array_equal(coordinate, integer_array[index - 1]):
                continue
        if index == 0:
            pixel_map[coordinate[0], coordinate[1]] = 1
            last_coordinate = coordinate
        elif index == len(integer_array) - 1:
            pixel_map[coordinate[0], coordinate[1]] = 1
            last_coordinate = coordinate
            break
        else:
            # Check if the coordinate after this one is touching the coordinate before this one
            # and if so, skip this pixel
            if check_is_touching(integer_array[index + 1], last_coordinate):
                continue
            else:
                # Add the coordinate to the pixel map and the pixelated path
                pixel_map[int(coordinate[0]), int(coordinate[1])] = 1
                pixelated_path = np.vstack((pixelated_path, coordinate.reshape(1, 2)))
                last_coordinate = coordinate

    return pixel_map, pixelated_path

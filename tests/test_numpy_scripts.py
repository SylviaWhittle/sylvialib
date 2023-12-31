"""Test the functions in the numpy_scripts module"""

import numpy as np

from sylvialib.numpy_scripts import create_2d_array_from_string, find_touching_pixels


def test_find_touching_pixels():
    """Test the find_touching_pixels function"""

    image = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 2, 2, 0, 1],
            [0, 1, 1, 1, 0, 0, 1],
        ]
    )

    touching_pixels = find_touching_pixels(image)

    assert np.array_equal(
        touching_pixels,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
            ]
        ),
    )


def test_create_2d_array_from_string():
    """Test the create_2d_array_from_string function"""

    string = """
    0 0 0 1 0 0 0
    0 0 1 1 1 0 0
    0 1 1 1 1 1 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    """

    array = create_2d_array_from_string(string)

    print("-------")
    print(array)

    assert np.array_equal(
        array,
        np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )

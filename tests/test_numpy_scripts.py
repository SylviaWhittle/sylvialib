"""Test the functions in the numpy_scripts module"""

import numpy as np

from sylvialib.numpy_scripts import create_2d_array_from_string


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

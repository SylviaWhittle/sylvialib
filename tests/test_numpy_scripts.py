"""Test the functions in the numpy_scripts module"""

import pytest

import numpy as np


from sylvialib.numpy_scripts import (
    create_2d_array_from_string,
    find_touching_pixels,
    coordinate_in_array,
    signed_angle_between_vectors,
)


def test_coordinate_in_array():
    """Test the coordinate_in_array function"""

    array = np.array(
        [
            [3, 5],
            [2, 4],
            [1, 3],
            [9, 8],
            [7, 2],
        ]
    )

    assert coordinate_in_array(array, [2, 4])
    assert not coordinate_in_array(array, [1, 9])


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


@pytest.mark.parametrize(
    ("vector1", "vector2", "expected_angle"),
    [
        pytest.param(np.array([1, 0]), np.array([0, 1]), -np.pi / 2, id="-90 degrees"),
        pytest.param(np.array([0, 1]), np.array([1, 0]), np.pi / 2, id="-90 degrees"),
        pytest.param(np.array([1, 0]), np.array([1, 0]), 0, id="0 degrees"),
        pytest.param(np.array([0, 1]), np.array([0, 1]), 0, id="0 degrees"),
    ],
)
def test_signed_angle_between_vectors(vector1, vector2, expected_angle) -> None:
    """Test the signed_angle_between_vectors function"""

    angle = signed_angle_between_vectors(vector1, vector2)

    assert angle == expected_angle

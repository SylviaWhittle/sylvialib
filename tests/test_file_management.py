"""Test the file management scripts"""

from pathlib import Path

import numpy as np

from sylvialib.deep_learning.file_management import (
    rename_files_alphabetical,
    rename_files_numerical,
)


def test_rename_files_alphabetical(tmp_path: Path):
    """Test the rename_files function"""

    # Create files in the temporary directory
    file_names = [
        "labelling_task_b_5.npy",
        "labelling_task_a_8.npy",
        "labelling_task_d_22.npy",
        "labelling_task_c_25.npy",
    ]

    # Create the files, where the contents are the original index to check that the files are
    # renamed correctly
    for i, file_name in enumerate(file_names):
        # Create the file with i in the file
        file_path = tmp_path / file_name
        np.save(file_path, i)

    # Rename the files
    rename_files_alphabetical(tmp_path, "npy", "mask")

    # Check that the files have been renamed
    expected_file_names = ["mask_0.npy", "mask_1.npy", "mask_2.npy", "mask_3.npy"]
    actual_file_names = sorted([file.name for file in tmp_path.glob("*.npy")])

    expected_contents_order = [1, 0, 3, 2]
    # Get the contents of each of the files to check that they have been renamed correctly
    actual_contents_order = []
    for file_name in actual_file_names:
        file_path = tmp_path / file_name
        contents = np.load(file_path)
        print(f"Contents of {file_name}: {contents}")
        list_contents = contents.tolist()
        actual_contents_order.append(list_contents)

    assert expected_file_names == actual_file_names
    assert expected_contents_order == actual_contents_order
    print(actual_contents_order, expected_contents_order)


def test_rename_files_numerical(tmp_path: Path):
    """Test the rename_files_numerical function"""

    # Create files in the temporary directory
    file_names = [
        "labelling_task_b_5.npy",
        "labelling_task_a_8.npy",
        "labelling_task_d_22.npy",
        "labelling_task_c_25.npy",
    ]

    # Create the files, where the contents are the original index to check that the files are
    # renamed correctly
    for i, file_name in enumerate(file_names):
        # Create the file with i in the file
        file_path = tmp_path / file_name
        np.save(file_path, i)

    # Rename the files
    rename_files_numerical(tmp_path, "npy", "mask")

    # Check that the files have been renamed
    expected_file_names = ["mask_0.npy", "mask_1.npy", "mask_2.npy", "mask_3.npy"]
    actual_file_names = sorted([file.name for file in tmp_path.glob("*.npy")])

    expected_contents_order = [0, 1, 2, 3]
    # Get the contents of each of the files to check that they have been renamed correctly
    actual_contents_order = []
    for file_name in actual_file_names:
        file_path = tmp_path / file_name
        contents = np.load(file_path)
        print(f"Contents of {file_name}: {contents}")
        list_contents = contents.tolist()
        actual_contents_order.append(list_contents)

    assert expected_file_names == actual_file_names
    assert expected_contents_order == actual_contents_order
    print(actual_contents_order, expected_contents_order)

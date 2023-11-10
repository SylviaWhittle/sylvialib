"""Test the file management scripts"""

from pathlib import Path
from sylvialib.deep_learning.file_management import rename_files


def test_rename_files(tmp_path: Path):
    """Test the rename_files function"""

    # Create files in the temporary directory
    file_names = [
        "labbeling_task_5.npy",
        "labelling_task_8.npy",
        "labelling_task_22.npy",
        "labelling_task_25.npy",
    ]

    for file_name in file_names:
        file_path = tmp_path / file_name
        file_path.touch()

    # Rename the files
    rename_files(tmp_path, "npy", "mask")

    # Check that the files have been renamed
    expected_file_names = ["mask_0.npy", "mask_1.npy", "mask_2.npy", "mask_3.npy"]
    actual_file_names = sorted([file.name for file in tmp_path.glob("*.npy")])

    print(actual_file_names)

    assert expected_file_names == actual_file_names

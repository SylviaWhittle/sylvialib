"""File management scripts"""

from pathlib import Path
import sys


def rename_files_alphabetical(path: Path, file_ext: str, file_name_base: str):
    """Renames all files in a directory to a given filename and extension, with index i where i
    is replaced with the number of the file as it appears in the directory after being sorted
    alphabetically. This is useful for renaming files that have been labelled by software to
    something more useful.
    """

    # Ensure that the path is a directory
    if not path.is_dir():
        print("Path is not a directory")
        sys.exit()

    # Ensure that the file extension starts with a period
    if file_ext[0] != ".":
        file_ext = "." + file_ext

    i = 0

    files = list(path.glob("*" + file_ext))
    files.sort()

    if len(files) == 0:
        print("No files found")
        sys.exit()
    else:
        # First pass, rename to temporary name to avoid overwriting
        for i, file in enumerate(files):
            temp_filename = path / f"temp_{i}{file_ext}"
            print(f"Renaming {file.name} to {temp_filename}")
            file.rename(temp_filename)

        # Second pass, rename to desired name
        temp_files = list(path.glob("*" + file_ext))
        temp_files.sort()
        for i, file in enumerate(temp_files):
            new_filename = path / f"{file_name_base}_{i}{file_ext}"
            print(f"Renaming {file.name} to {new_filename}")
            file.rename(new_filename)


def file_rename_numerical(path: Path, file_ext: str, file_name_base: str):
    """Renames all files in a directory. Sorts the files in order of numbers that appear
    in the existing file names.

    Eg: [image_1.png, image_5.png, image_8.png] -> [image_0.png, image_1.png, image_2.png]
    """

    # Ensure that the path is a directory
    if not path.is_dir():
        print("Path is not a directory")
        sys.exit()

    # Ensure that the file extension starts with a period
    if file_ext[0] != ".":
        file_ext = "." + file_ext

    i = 0

    # Get all files in the directory
    files = list(path.glob("*" + file_ext))
    # Sort the files by the number that appears in the filename
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f.name))))

    if len(files) == 0:
        print("No files found")
        sys.exit()
    else:
        # First pass, rename to temporary name to avoid overwriting
        for i, file in enumerate(files):
            temp_filename = path / f"temp_{i}{file_ext}"
            print(f"Renaming {file.name} to {temp_filename}")
            file.rename(temp_filename)

        # Rename the files
        temp_files = list(path.glob("*" + file_ext))
        temp_files.sort(key=lambda f: int("".join(filter(str.isdigit, f.name))))

        for i, file in enumerate(temp_files):
            new_filename = path / f"{file_name_base}_{i}{file_ext}"
            print(f"Renaming {file.name} to {new_filename}")
            file.rename(new_filename)

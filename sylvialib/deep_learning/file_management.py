"""File management scripts"""

from pathlib import Path
import sys


def rename_files(path: Path, file_ext: str, file_name_base: str):
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
        for file in files:
            new_filename = path / f"{file_name_base}_{i}{file_ext}"
            print(f"Renaming {file.name} to {new_filename}")
            file.rename(new_filename)
            i += 1

import warnings
from pathlib import Path

import dill


def save_object(obj: object, filepath: str | Path, overwrite: bool = False) -> None:
    """Uses the `dill` library to save almost any python objects into file.

    This is basically a thin wrapper around `dill.dump` that performs file exists check to prevent accidental overwrites.
    """
    filepath = Path(filepath)

    if filepath.suffix == "":
        filepath = filepath.parent / (filepath.name + ".pkl")

    if filepath.exists():
        if not overwrite:
            raise FileExistsError(
                f"{filepath} already exists. To overwrite, set overwrite=True."
            )
        else:
            warnings.warn(
                f"{filepath} already exists. Overwriting the file as overwrite is set to True."
            )

    with open(filepath, "wb") as file:
        dill.dump(obj, file)


def load_object(filepath: str | Path) -> object:
    """Uses the `dill` library to saved object file.

    This is basically a thin wrapper around `dill.load` to complement `write_object`."""
    with open(Path(filepath), "rb") as file:
        return dill.load(file)

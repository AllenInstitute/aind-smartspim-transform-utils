""" Init package """


from pathlib import Path

__version__ = "1.0.0"


# where files will be downloaded if needed
base_dir = Path.home() / ".transform_utils" / "transform_utils"
base_dir.mkdir(parents=True, exist_ok=True)
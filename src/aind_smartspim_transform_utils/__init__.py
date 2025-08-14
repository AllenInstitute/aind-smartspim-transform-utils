""" Init package """

import os
from pathlib import Path

__version__ = "1.0.0"


# where files will be downloaded if needed
base_dir = Path.home() / ".transform_utils" / "transform_utils"
base_dir.mkdir(parents=True, exist_ok=True)

# this code would install in scratch if on CO but dont know if it is good for local

# if os.path.exists(Path.home() / "capsule"):
#    base_dir = Path.home() / 'capsule' / 'scratch' / "transform_utils"
# else:
#    base_dir = Path.home() / ".transform_utils" / "transform_utils"

# base_dir.mkdir(parents=True, exist_ok=True)

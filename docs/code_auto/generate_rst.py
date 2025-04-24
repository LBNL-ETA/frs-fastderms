import os
import sys
from sphinx.cmd.build import build_main

from pathlib import Path

path_to_here = Path(__file__).parent
path_to_repo = path_to_here / ".." / ".." / "src"
path_to_repo = path_to_repo.resolve()
sys.path.insert(0, str(path_to_repo))

output_folder = (path_to_here / ".." / "code").resolve()

# Ensure output directory exists
output_folder.mkdir(parents=True, exist_ok=True)

build_args = [
    "-b",
    "rst",  # Use restbuilder
    "-c",
    str(path_to_here),  # Use local conf.py in code_auto/
    str(path_to_here),  # Source directory
    str(output_folder),  # Output directory
]
result = build_main(build_args)

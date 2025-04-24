import sys
from pathlib import Path

path_to_here = Path(__file__).parent
path_to_repo = path_to_here / ".." / "src"
path_to_repo = path_to_repo.resolve()
sys.path.insert(0, str(path_to_repo))

master_doc = "_overview"

extensions = ["sphinx.ext.autodoc"]
extensions += ["sphinx_substitution_extensions"]
extensions += ["sphinx_design"]
extensions += ["sphinxcontrib.restbuilder"]

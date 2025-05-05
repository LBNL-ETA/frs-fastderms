# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

path_to_here = Path(__file__).parent
path_to_repo = path_to_here / ".." / "src"
path_to_repo = path_to_repo.resolve()
sys.path.insert(0, str(path_to_repo))

# -- Project information -----------------------------------------------------
master_doc = "index"

project = "Flexible Resource Scheduler (FRS) for FAST-DERMS"
copyright = "Copyright 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved"
author = "_"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc"]
extensions += ["sphinx_substitution_extensions"]
extensions += ["sphinx_design"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["code_auto/*", "code_auto/**/*"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Prolog ------------------------------------------------------------------

rst_prolog = """
.. |frs_repo| replace:: https://github.com/LBNL-ETA/frs-fastderms
.. |frs_shared_data_folder| replace:: https://doi.org/10.5281/zenodo.15344843
"""

# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "GADD"
copyright = f"{datetime.now().year}, GADD Contributors"
author = "GADD Contributors"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "qiskit_sphinx_theme",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_iVar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Templates path
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
html_theme = "qiskit-ecosystem"

html_context = {
    "display_github": True,
    "github_user": "qiskit-community",
    "github_repo": "gadd",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# NBSphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Math rendering
mathjax3_config = {
    "tex": {
        "inlineMath": [[", "], ["\\(", "\\)"]],
        "displayMath": [["$", "$"], ["\\[", "\\]"]],
    }
}

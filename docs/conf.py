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
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
project = "TextAttack"
copyright = "2020, UVA QData Lab"
author = "UVA QData Lab"

# The full version, including alpha/beta/rc tags
release = "0.2.14"

# Set master doc to `index.rst`.
master_doc = "index"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    # Enable .ipynb doc files
    "nbsphinx",
    # Enable .md doc files
    "recommonmark",
]
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Mock expensive textattack imports. Docs imports are in `docs/requirements.txt`.
autodoc_mock_imports = []

# Output file base name for HTML help builder.
htmlhelp_basename = "textattack_doc"
html_theme_options = {
    "logo_only": False,
    "style_nav_header_background": "transparent",
    "analytics_id": "UA-88637452-2",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Path to favicon.
html_favicon = "favicon.png"

# Don't show module names in front of class names.
add_module_names = True

# Sort members by group
autodoc_member_order = "groupwise"

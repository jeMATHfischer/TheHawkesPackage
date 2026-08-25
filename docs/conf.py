"""Sphinx configuration for the-hawkes-package."""

import importlib.metadata as md

project = "the-hawkes-package"
author = "Jens Fischer"
copyright = "2019-2026, Jens Fischer"
release = md.version("the-hawkes-package")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- API documentation ------------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": True,
}

# autosummary generates the per-member pages; numpydoc's own class-member table
# would duplicate them.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Notebooks --------------------------------------------------------------

# Executing on every build makes the examples a regression test: combined with
# `sphinx-build -W`, an API change that breaks an example fails the build.
nb_execution_mode = "cache"
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_cache_path = "_build/.jupyter_cache"

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]

# -- HTML -------------------------------------------------------------------

html_theme = "furo"
html_title = f"the-hawkes-package {release}"
html_static_path = []

# Warnings are errors in CI, so nothing is suppressed here by default.
suppress_warnings = []

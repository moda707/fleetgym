"""Sphinx configuration of the FleetGym documentation (build: ``sphinx-build -W docs/site out``)."""

from importlib.metadata import version as _version

project = "FleetGym"
author = "Mohsen Dastpak"
copyright = "2026, Mohsen Dastpak"
release = _version("fleetgym")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
default_role = "py:obj"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
typehints_fully_qualified = False
always_document_param_types = False
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_title = "FleetGym"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/moda707/fleetgym",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "header_links_before_dropdown": 6,
}
html_context = {"default_mode": "light"}

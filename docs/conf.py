"""Sphinx configuration for the biodb docs site."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

project = "biodb"
author = "Brian Schilder"
copyright = "2026, Brian Schilder"

try:
    release = _pkg_version("biodb")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_show_sourcelink = False
html_baseurl = "https://bschilder.github.io/biodb/"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#0d1117",
}
html_context = {
    "display_github": True,
    "github_user": "bschilder",
    "github_repo": "biodb",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

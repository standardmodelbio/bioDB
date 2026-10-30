"""Sphinx configuration for the biodb docs site."""

from __future__ import annotations

import inspect
import json
from importlib.metadata import version as _pkg_version
from pathlib import Path
from shutil import copyfile

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
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"
templates_path = ["_templates"]
# ``superpowers/`` holds internal design specs + implementation plans, not
# public site content — keep them out of the autosummary/toctree sweep.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "superpowers/**",
    "forensics/**",
    "team-updates/**",
]

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
# Render dataclass "Attributes" sections (e.g. biodb.gtr.GTRTest) as :ivar:
# info fields rather than standalone py:attribute directives — otherwise they
# collide with autodoc's own field documentation and emit duplicate-object
# warnings.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_show_sourcelink = False
html_baseurl = "https://standardmodel.bio/bioDB/"
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
    "github_user": "standardmodelbio",
    "github_repo": "bioDB",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# Publish the saved, reviewed outputs. Offline execution is a separate CI check;
# building the site must never start live API queries or bulk downloads.
nb_execution_mode = "off"


def _copy_tutorials(app):
    """Keep notebooks in one canonical location while making them Sphinx sources."""
    source = Path(app.confdir).parent / "tutorials"
    target = Path(app.srcdir) / "_generated" / "tutorials"
    target.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(source.glob("*.ipynb"))
    for stale in target.glob("*.ipynb"):
        if not (source / stale.name).exists():
            stale.unlink()
    for notebook in notebooks:
        destination = target / notebook.name
        if not destination.exists() or destination.read_bytes() != notebook.read_bytes():
            copyfile(notebook, destination)
    index = (source / "README.md").read_text()
    index += "\n\n```{toctree}\n:hidden:\n:maxdepth: 1\n\n"
    index += "\n".join(notebook.stem for notebook in notebooks) + "\n```\n"
    destination = target / "index.md"
    if not destination.exists() or destination.read_text() != index:
        destination.write_text(index)


def _notebook_download(app, docname, source):
    """Add a download link to rendered notebooks without modifying their files."""
    if not docname.startswith("_generated/tutorials/") or docname.endswith("/index"):
        return
    notebook = json.loads(source[0])
    filename = Path(docname).name + ".ipynb"
    notebook["cells"].insert(
        1,
        {
            "cell_type": "markdown",
            "id": "biodb-site-download",
            "metadata": {},
            "source": f"{{download}}`Download this notebook <{filename}>`",
        },
    )
    source[0] = json.dumps(notebook)


def _tutorial_source_link(app, pagename, templatename, context, doctree):
    if pagename.startswith("_generated/tutorials/"):
        filename = Path(pagename).name
        filename = "README.md" if filename == "index" else filename + ".ipynb"
        context["meta"] = dict(context.get("meta") or {})
        context["meta"]["github_url"] = (
            "https://github.com/standardmodelbio/bioDB/blob/main/tutorials/" + filename
        )


def _skip_foreign_members(app, what, name, obj, skip, options):
    """Document biodb re-exports, but not imported dependencies such as tqdm."""
    module = obj.__name__ if inspect.ismodule(obj) else getattr(obj, "__module__", "")
    # These shared result types are re-exported by multiple readers. Keep their
    # full class documentation at gwas_catalog, the canonical acquisition page.
    if (
        module == "biodb._acquisition"
        and name in {"IngestError", "IngestResult"}
        and app.env.docname != "_generated/biodb.gwas_catalog"
    ):
        return True
    if (inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj)) and (
        module and module != "biodb" and not module.startswith("biodb.")
    ):
        return True
    return None


def setup(app):
    app.connect("builder-inited", _copy_tutorials)
    app.connect("source-read", _notebook_download)
    app.connect("html-page-context", _tutorial_source_link)
    app.connect("autodoc-skip-member", _skip_foreign_members)

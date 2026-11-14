# Configuration file for the Sphinx documentation builder.

from tagarnina import __version__

project = "tagarnina"
copyright = "2026, Guillermo Navas-Palencia"
author = "Guillermo Navas-Palencia"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
numpydoc_show_class_members = False
templates_path = ["_templates"]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "download_all_examples": False,
    "within_subsection_order": "ExampleTitleSortKey",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "github_url": "https://github.com/guillermo-navas-palencia/tagarnina",
}

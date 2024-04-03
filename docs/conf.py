# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath('..'))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = 'GeometricKernels'
copyright = '2022-2024, the GeometricKernels Contributors'
author = 'The GeometricKernels Contributors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
]

# autoapi
extensions.append("autoapi.extension")
autoapi_dirs = ["../geometric_kernels"]
autodoc_typehints = 'description'
autodoc_default_options = {"special-members": "__init__",}
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_python_class_content = "class"  # we handle __init__ and __new__ below
autoapi_member_order = "groupwise"
# ignore these files to suppress warning multiple dispatch
autoapi_ignore = [f'**/lab_extras/{b}**' for b in ["torch", "jax", "tensorflow", "numpy"]]
autoapi_options = [
    "members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]

# Never skip __init__ or __new__
def never_skip_init_or_new(app, what, name, obj, would_skip, options):
    if "__init__" in name or "__new__":
        return False
    return would_skip
def setup(sphinx):
    sphinx.connect("autoapi-skip-member", never_skip_init_or_new)


# Add any paths that contain templates here, relative to this directory.
# The templates are as in https://stackoverflow.com/a/62613202
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/bootstrap_namespaced.css',
]

html_js_files = [
    'js/bootstrap.min.js',
]

# Theme-specific options. See theme docs for more info
html_context = {
  'display_github': True,
  'github_user': 'GPflow',
  'github_repo': 'GeometricKernels',
  'github_version': 'main/docs/'
}

# For sphinx_math_dollar (see https://www.sympy.org/sphinx-math-dollar/)

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

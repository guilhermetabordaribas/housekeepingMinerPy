# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'housekeepingMinerPy'
copyright = '2024, Guilherme Taborda Ribas'
author = 'Guilherme Taborda Ribas'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']#,'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = ['sknetwork','umap','false_discovery_control','scipy.stats','hdbscan','pygad',
'boruta','statannotations','adjustText',
'pandas','anndata','sklearn','scipy','matplotlib','seaborn','numpy']#


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'#'sphinx_rtd_theme'#
html_static_path = ['_static']
# html_theme_options = {
#     'logo_only': True,
# }
html_logo = "logo.png"

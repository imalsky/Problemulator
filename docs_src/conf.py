"""Sphinx configuration for the Problemulator documentation site."""

project = "Problemulator"
author = "imalsky"
copyright = "2026, imalsky"

extensions = []
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
html_static_path = []
html_title = "Problemulator documentation"
html_show_sourcelink = False

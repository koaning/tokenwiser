from setuptools import setup, find_packages

base_packages = [
    "scikit-learn>=1.0.0",
]

dev_packages = [
    "ruff",
    "pytest",
]

docs_packages = [
    "mkdocs-material",
    "mkdocstrings-python",
]


setup(
    name="tokenwiser",
    version="0.3.0",
    packages=find_packages(exclude=["notebooks"]),
    install_requires=base_packages,
    extras_require={"dev": dev_packages + docs_packages, "docs": docs_packages},
)

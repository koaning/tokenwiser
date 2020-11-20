from setuptools import setup, find_packages

base_packages = ["gensim>=3.8.3", "jellyfish>=0.8.2", "Pyphen>=0.10.0", "scikit-learn>=0.23.2"]

dev_packages = ["flake8>=3.6.0", "pytest>=4.0.2", "jupyter>=1.0.0", "jupyterlab>=0.35.4"]


setup(
    name="tokenwiser",
    version="0.1.0",
    packages=find_packages(exclude=['notebooks']),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages
    }
)

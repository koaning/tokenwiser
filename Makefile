ruff:
	ruff tokenwiser/ --fix

test:
	pytest

check: ruff test

install:
	python -m pip install -e ".[dev]"
	python -m pip install -e .

install-test:
	python -m pip install -e ".[test]"
	python -m pip install -e ".[all]"

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

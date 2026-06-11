develop:
	maturin develop

release:
	maturin develop --release

test:
	pytest -q

check: develop test

wheel:
	maturin build --release

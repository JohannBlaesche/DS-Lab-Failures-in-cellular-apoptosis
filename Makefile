# adapted from https://stackoverflow.com/questions/33839018/activate-virtualenv-in-makefile
all: install install/rapids run

install: venv
	. venv/bin/activate && pip3 install -e ".[dev]"

install/rapids: venv
	. venv/bin/activate && pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* cuml-cu12==24.6.*

venv:
	test -d venv || python3 -m venv venv

run:
	. venv/bin/activate && apopfail

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

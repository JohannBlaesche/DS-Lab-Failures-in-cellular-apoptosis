# adapted from https://stackoverflow.com/questions/33839018/activate-virtualenv-in-makefile
all: install run

install: venv
	. venv/bin/activate && pip install -e ".[dev]"
	pre-commit install

venv:
	test -d venv || python -m venv venv

run:
	. venv/bin/activate && dmgpred

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

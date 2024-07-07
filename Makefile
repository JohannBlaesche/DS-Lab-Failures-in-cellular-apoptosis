# adapted from https://stackoverflow.com/questions/33839018/activate-virtualenv-in-makefile
all: install run

install: venv
	. venv/bin/activate && pip3 install -e ".[dev]"

venv:
	test -d venv || python3 -m venv venv

run:
	. venv/bin/activate && afp --refit

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

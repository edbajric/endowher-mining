# Simple Makefile for common tasks

install:
	pip install -r requirements.txt

lint:
	flake8 src/

test:
	pytest tests/

run-all:
	python scripts/run_all.py

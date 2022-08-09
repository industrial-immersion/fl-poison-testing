#!/usr/bin/env bash

isort *.py && \
black *.py && \
flake8 *.py && \
mypy *.py && \
echo All passed!

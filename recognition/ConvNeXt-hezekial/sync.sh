#!/usr/bin/env bash

rsync --exclude .venv --exclude /assets --exclude __pycache__ --exclude README.MD --exclude .git -P -a . rangpur:$(basename "$PWD")/

#!/usr/bin/env bash

rsync --exclude .venv --exclude README.MD --exclude .git -P -a . rangpur:$(basename "$PWD")/

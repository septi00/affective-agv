#!/usr/bin/env bash
# Build the image once, then use `run.sh` to get a shell.
set -e
docker build -t affective-galactic ./dockerenv

#!/usr/bin/env bash
# Open a shell in the Nav2+Galactic env with the workspace mounted.
# Same container name so a second terminal can `docker exec` into it.
set -e
docker run -it --rm --name affective \
  -v "$HOME/dis_ws:/dis_ws" \
  affective-galactic bash

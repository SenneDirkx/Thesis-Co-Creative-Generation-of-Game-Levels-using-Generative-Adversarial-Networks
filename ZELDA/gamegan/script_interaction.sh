#!/bin/zsh
pipenv shell
python generate_random_latent.py 10
# doe some level editing
python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 10 1000 1000 250 2
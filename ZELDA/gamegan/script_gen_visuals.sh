#!/bin/zsh
pipenv shell
python generate_random_latent.py 64
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16
python json_to_image.py level_output.json og_image.png
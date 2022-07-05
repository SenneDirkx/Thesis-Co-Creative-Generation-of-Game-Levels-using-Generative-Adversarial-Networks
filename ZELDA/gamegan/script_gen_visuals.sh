#!/bin/zsh
pipenv shell
python generate_random_latent.py copypaste/latent_input.txt 64
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 copypaste/latent_input.txt copypaste/level_output.json
python json_to_image.py copypaste/level_output.json copypaste/og_image.png
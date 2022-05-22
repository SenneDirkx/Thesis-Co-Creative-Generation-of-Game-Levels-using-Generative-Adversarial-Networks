#!/bin/zsh
pipenv shell
python generate_random_latent.py 16
python main.py ./rewritten_generator.pth 10 4 11 16
python json_to_image.py rewritten_image.png
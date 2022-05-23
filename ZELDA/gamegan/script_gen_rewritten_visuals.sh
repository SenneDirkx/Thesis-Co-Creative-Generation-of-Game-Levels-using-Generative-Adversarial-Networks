#!/bin/zsh
pipenv shell
python main.py ./rewritten_generator.pth 10 4 11 16
python json_to_image.py level_output.json rewritten_image.png
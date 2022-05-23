#!/bin/zsh
pipenv shell
python editor.py
python json_to_image.py edited_level_output.json edited_image.png
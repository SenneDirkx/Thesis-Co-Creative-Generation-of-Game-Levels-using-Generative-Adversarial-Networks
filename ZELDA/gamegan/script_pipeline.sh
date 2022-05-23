#!/bin/zsh
pipenv shell

echo "generating latent input"
python generate_random_latent.py latent_input.txt 64

echo "creating og levels"
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 latent_input.txt level_output.json
python json_to_image.py level_output.json og_image.png

echo "editing levels"
python generate_random_latent.py interact_input.txt 10
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 interact_input.txt interact_level_output.json
python editor.py interact_level_output.json edited_level_output.json
python json_to_image.py edited_level_output.json edited_image.png

echo "performing interaction"
python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 10 5000 25000 512 interact_input.txt edited_level_output.json

echo "creating rewritten levels"
python main.py ./rewritten_generator.pth 10 4 11 16 latent_input.txt rewritten_level_output.json
python json_to_image.py rewritten_level_output.json rewritten_image.png
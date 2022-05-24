#!/bin/zsh
pipenv shell
echo "Diversity estimate:"
python generate_random_latent.py latent_input.txt 100
python main.py ./rewritten_generator.pth 10 4 11 16 latent_input.txt level_output.json
python diversity.py 1000